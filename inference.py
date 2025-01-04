import cv2
import numpy as np
import torch
import csv
from ultralytics import YOLO
from boxmot import DeepOCSORT
from pathlib import Path
import time
import threading
import queue
import logging
from collections import deque
import mediapipe as mp
import contextlib
import io
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import pytemi as temi
from threading import Lock  # 確保引入 Lock
import os
import requests
from model import Transformer_rev_Complex, CNN_LSTM, CBiLSTM, GRUClassifier, CRNN  # 確保導入所有需要的模型
import json
import psycopg2
# Initialize Google logging
from absl import logging as absl_logging
import subprocess
absl_logging.set_verbosity(absl_logging.INFO)
from moviepy.editor import VideoFileClip
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('boxmot').setLevel(logging.WARNING)
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# TEMI robot configuration
TEMI_SERIAL = "00120495065"
# MQTT_HOST = 'stevetw.serv00.net'
MQTT_HOST = '128.204.223.100'
MQTT_PORT = 1883
MQTT_USERNAME = 'steve'
MQTT_PASSWORD = '062028633'

# 設定目標裝置的 IP 和 Port
TARGET_DEVICE_IP = '100.72.105.61'  # 替換為目標裝置的 IP
TARGET_DEVICE_PORT = '5000'         # 替換為目標裝置的 Port

# TARGET_DEVICE_IP = '100.68.188.72'  # 替換為目標裝置的 IP
# TARGET_DEVICE_PORT = '6000'         # 替換為目標裝置的 Port

# Connect to TEMI robot
try:
    mqtt_client = temi.connect(MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD)
    robot = temi.Robot(mqtt_client, TEMI_SERIAL)
    logging.info("Connected to TEMI robot.")
except Exception as e:
    logging.error(f"Failed to connect to TEMI robot: {e}")
    robot = None

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# Load YOLO model
try:
    yolo_model = YOLO('best_m.pt', verbose=False)
    yolo_model.to(device)
    yolo_model.conf = 0.6
    logging.info("YOLO model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    yolo_model = None

# Initialize DeepOCSORT tracker
try:
    tracker = DeepOCSORT(
        model_weights=Path('osnet_x0_25_msmt17.pt'),
        device=0 if device == 'cuda' else 'cpu',
        fp16=(device == 'cuda'),
        cmc=False,
    )
    logging.info("DeepOCSORT tracker initialized.")
except Exception as e:
    logging.error(f"Failed to initialize DeepOCSORT: {e}")
    tracker = None

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize gesture recognition model

# gesture_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# gesture_model = Transformer_rev_Complex(99, 9, 32, 2).to(gesture_device)
# try:
#     gesture_model.load_state_dict(torch.load('model/numpy/gesture_Federated.pth', map_location=gesture_device))
#     gesture_model.eval()
#     logging.info("Gesture recognition model loaded successfully.")
# except Exception as e:
#     logging.error(f"Failed to load gesture recognition model: {e}")

# Gesture recognition parameters
class_name = {
    0: 'Forward',
    1: 'Backward',
    2: 'Stop',
    3: 'Fire',
    4: 'Follow',
    5: '0 o\'clock direction',
    6: '3 o\'clock direction',
    7: '6 o\'clock direction',
    8: '9 o\'clock direction'
}
actions = ['Forward', 'Backward', 'Stop', 'Fire', 'Follow', '0 o\'clock direction', '3 o\'clock direction',
           '6 o\'clock direction', '9 o\'clock direction']
seq_length = 30
seq = deque(maxlen=seq_length)
action_seq = deque(maxlen=9)
last_action = None
last_gesture_time = 0
gesture_interval = 3  # seconds
gesture_lock = threading.Lock()

# 新增變量
fire_gesture_interval = 5  # 秒
last_fire_gesture_time = 0
last_detection_confidence = 0.0
last_detection_confidence_lock = Lock()
confidence_threshold_for_fire = 0.6  # 設定允許執行 Fire 手勢的最低置信度閾值
command_send_interval = 2  # 最小指令發送間隔（秒）
last_command_sent_time = 0
command_send_lock = Lock()
target_detected = False  # 新增變量，用於指示是否檢測到目標
target_detected_lock = Lock()
fire_gesture_duration = 5  # 秒，瞄準或檢測後的持續時間
shot_interval = 10.0  # 冷卻時間（秒）
# 在全局變量中新增 last_target_class_id
last_target_class_id = None
confidence_queue = deque(maxlen=10)  # 用于存储最近十帧的 confidence 值
confidence_threshold = 0.6  # 'Fire' 手势允许的 confidence 阈值
angle_confidence_threshold = 0.6  # 传输角度指令的 confidence 阈值
# 行動佇列和鎖
action_queue = queue.Queue()
action_queue_lock = Lock()
data_buffer = []
buffer_lock = threading.Lock()
save_interval = 60  # 每60秒保存一次
last_save_time = time.time()
# 定義 CSV 標題（99 個特徵 + 1 個標籤）
feature_headers = [f'feature{i}' for i in range(99)]
label_header = 'label'
headers = feature_headers + [label_header]
# Global variables
cap_yolo = None
cap_gesture = None
camera_opened_yolo = False
camera_opened_gesture = False
paused = False
recording = False
frame_count = 0
frame_count_gesture = 0
processed_targets = {}
processed_classes = {}  # New dictionary to track cooldown per class
processed_targets_lock = threading.Lock()
shot_fired_time = None
current_target_class_id = None  # 定義 current_target_class_id
# 全局變量
gesture_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gesture_model = None
num_classes = 9
current_neural_network = "Transformer"

# Variables for controlling 'Fire' gesture usage
fire_gesture_allowed = False
fire_gesture_start_time = 0

# Angle related variables
angle_history = []
history_length = 5
angle_threshold = 1.0
fov = 55  # Adjust according to your camera
last_turn_time = 0
last_shot_time = 0
rotation_interval = 1.0

# CSV file names
gt_filename = 'ground_truth.csv'
tr_filename = 'tracking_results.csv'
gesture_inference_data = './static/gesture_inference_data.csv'


# Initialize CSV headers
with open(gt_filename, mode='w', newline='') as gt_file, open(tr_filename, mode='w', newline='') as tr_file, open(gesture_inference_data, mode='w', newline='') as gesture_file:
    gt_writer = csv.writer(gt_file)
    tr_writer = csv.writer(tr_file)
    gesture_writer = csv.writer(gesture_file)
    gt_writer.writerow(['frame', 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id'])
    tr_writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'class_id', 'confidence'])
    gesture_writer.writerow(headers + ['time'])  # 添加時間戳欄位

# Queues
frame_queue_yolo = queue.Queue(maxsize=50)
gt_queue = queue.Queue()
tr_queue = queue.Queue()
gesture_queue = queue.Queue()
record_queue_yolo = queue.Queue(maxsize=500)  # Increased size from 200 to 500
record_queue_gesture = queue.Queue(maxsize=500)  # Increased size from 50 to 100

# Locks
latest_frame_yolo = None
latest_frame_gesture = None
gesture_model = None
current_neural_network = None
training_thread = None
latest_frame_lock_yolo = threading.Lock()
latest_frame_lock_gesture = threading.Lock()
cap_lock_yolo = threading.Lock()
cap_lock_gesture = threading.Lock()
record_lock = threading.Lock()
gesture_model_lock = threading.Lock()
# 使用鎖來保護共享變量的訪問
progress_lock = threading.Lock()
metrics_lock = threading.Lock()
hyperparameters_lock = threading.Lock()


# FPS calculation variables
processing_fps_counter = 0
processing_fps_timer = time.time()
processing_current_fps = 0.0

# Thread events
processing_event = threading.Event()
gesture_event = threading.Event()
exit_event = threading.Event()
stop_training_event = threading.Event()

# Gesture recognition control variable
gesture_paused = False
gesture_paused_lock = threading.Lock()

# 全局變量來存儲超參數、訓練進度和訓練指標
hyperparameters = {}
training_progress = {
    "epoch": 0,
    "batch": 0,
    "loss": 0.0,
    "progress": 0,
    "accuracy": 0.0
}
training_metrics = {
    "epoch": 0,
    "avg_train_loss": 0.0,
    "Train_accuracy": 0.0,
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "confusion": []
}

# 数据库连接参数
db_host = "localhost"
db_name = "gesture"
db_user = "t1204"
db_password = "t1204"
table_name = "gesture_1_test"  # 假设表名为 'gesture_1_test'
# 全局变量
current_video_time = 0.0
current_video_time_lock = threading.Lock()
# 在全局范围内初始化 start_time 和 current_video_time
start_time = None
YOLO_CAMERA_SOURCE = 'rtmp://100.70.26.103:1935/live/stream'
# YOLO_CAMERA_SOURCE =0
GESTURE_CAMERA_SOURCE = 'rtmp://100.100.87.86:1935/live/stream'  # 可以是文件路徑或攝影機ID
# GESTURE_CAMERA_SOURCE ='rtmp://100.70.26.103:1935/live/stream'
# Define a helper function to emit messages in a non-blocking way

@app.route("/test1")
def page1():
    return render_template('test1.html')  # HTML for page 1

@app.route("/test2")
def page2():
    return render_template('test2.html')  # HTML for page 2

@app.route("/test3")
def page3():
    return render_template('test3.html')


# 初始化模型函式
def initialize_gesture_model(username='default', neural_network='Transformer'):
    global num_classes, current_neural_network, gesture_model
    try:
        current_neural_network = neural_network

        default_num_classes = 9

        # 讀取手勢數量
        gestures_file = f'./data/{username}/gestures.json'
        if os.path.exists(gestures_file):
            with open(gestures_file, 'r') as f:
                data = json.load(f)
            gestures = data.get('gestures', [])
            if gestures:
                # 計算最大索引並調整 num_classes
                max_index = max(g.get('index', -1) for g in gestures)
                if max_index + 1 > default_num_classes:
                    num_classes = max_index + 1
                else:
                    num_classes = default_num_classes
            else:
                num_classes = default_num_classes
            logging.info(f"Number of gestures for user '{username}': {num_classes}")
        else:
            num_classes = default_num_classes
            logging.info(f"Gestures file for user '{username}' not found. Using default num_classes={default_num_classes}.")

        # 根據神經網絡架構初始化模型
        gesture_model = initialize_model(current_neural_network, num_classes)
        if gesture_model is None:
            logging.error("Failed to initialize gesture recognition model.")
            return False

        # 載入模型權重
        model_path = f'./model/{username}/{current_neural_network}/latest.pth'  # 假設權重文件命名為 latest.pth
        if os.path.exists(model_path):
            gesture_model.load_state_dict(torch.load(model_path, map_location=gesture_device))
            gesture_model.eval()
            logging.info("Gesture recognition model loaded successfully.")
        else:
            gesture_model = None  # 設置為 None，因為模型文件不存在
            logging.info(f"Model weights file not found at {model_path}. Gesture model is not loaded.")

        return True

    except Exception as e:
        logging.error(f"Error initializing gesture recognition model: {e}")
        return False

def initialize_model(neural_network, num_classes):
    try:
        if neural_network == "Transformer":
            model = Transformer_rev_Complex(99, num_classes, 32, 2).to(gesture_device)
        elif neural_network == "CNN_LSTM":
            model = CNN_LSTM(99, num_classes, 64, 32).to(gesture_device)
        elif neural_network == "CBiLSTM":
            model = CBiLSTM(99, 64, 32).to(gesture_device)
        elif neural_network == "GRUClassifier":
            model = GRUClassifier(99, 64, 32, num_classes).to(gesture_device)
        elif neural_network == "CRNN":
            model = CRNN(99, 64, 32).to(gesture_device)
        else:
            app.logger.error(f'Unknown neural network architecture: {neural_network}')
            return None
        return model
    except Exception as e:
        app.logger.error(f'Error initializing model: {e}')
        return None


# 返回 data 目錄下的所有使用者列表
@app.route('/get_users', methods=['GET', 'POST'])
def get_users():
    """
    返回 data 目錄下的所有使用者列表。
    """
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        users = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
        return jsonify({'users': users}), 200
    except Exception as e:
        app.logger.error(f'Error in /get_users: {e}')
        return jsonify({'error': 'Internal Server Error'}), 500

# 获取用户的手势列表
@app.route('/get_gestures/<username>')
def get_gestures(username):
    gestures_file = f'./data/{username}/gestures.json'
    if os.path.exists(gestures_file):
        with open(gestures_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"message": "Gestures file not found"}), 404

# 获取用户的模型列表
@app.route('/get_user_models/<username>/<neural_network>')
def get_user_models(username, neural_network):
    models_dir = f'./model/{username}/{neural_network}'
    if os.path.exists(models_dir):
        models = os.listdir(models_dir)
        return jsonify(models=models)
    else:
        return jsonify(models=[])

# 获取所有模型
@app.route('/get_models')
def get_models():
    models_dir = 'model'
    if os.path.exists(models_dir):
        models = os.listdir(models_dir)
        return jsonify(models=models)
    else:
        return jsonify(models=[])

# 新增的 update_model 路由
@app.route('/update_model', methods=['POST', 'GET'])
def update_model():
    """
    接收前端發送的模型更新請求，從目標裝置下載模型並重新命名為 latest.pth，
    然後重新初始化手勢識別模型。
    """
    global gesture_model_initialized
    try:
        data = request.get_json()
        if not data:
            logging.error("No JSON payload received.")
            return jsonify({'error': 'No JSON payload received.'}), 400

        username = data.get('username')
        neural_network = data.get('neural_network')
        model_name = data.get('model')

        if not all([username, neural_network, model_name]):
            logging.error("Missing parameters in the request.")
            return jsonify({'error': 'Missing parameters'}), 400

        # 確保使用者和神經網絡的目錄存在
        local_model_dir = os.path.join('model', username, neural_network)
        os.makedirs(local_model_dir, exist_ok=True)

        # 構建目標裝置的模型下載 URL
        # 假設目標裝置有一個端點 /get_model/<username>/<neural_network>/<model_name> 來提供模型下載
        target_url = f'http://{TARGET_DEVICE_IP}:{TARGET_DEVICE_PORT}/get_model/{username}/{neural_network}/{model_name}'
        logging.info(f"Fetching model from URL: {target_url}")

        # 發送 GET 請求到目標裝置以下載模型文件
        try:
            response = requests.get(target_url, stream=True, timeout=30)
            logging.info(f"Received response from target device: {response.status_code}")
            if response.status_code == 200:
                # 定義本地模型存儲路徑
                local_model_path = os.path.join(local_model_dir, 'latest.pth')  # 重新命名為 latest.pth

                # 寫入下載的模型文件
                with open(local_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logging.info(f"Model '{model_name}' for user '{username}' and neural network '{neural_network}' downloaded successfully from {TARGET_DEVICE_IP}:{TARGET_DEVICE_PORT} and saved as 'latest.pth'.")

                # 重新初始化手勢模型
                success = initialize_gesture_model(username, neural_network)
                if not success:
                    logging.error("Failed to reinitialize gesture recognition model after update.")
                    return jsonify({'error': 'Model updated but failed to initialize model.'}), 500

                gesture_model_initialized = True
                return jsonify({'message': 'Model updated and loaded successfully.'}), 200
            else:
                logging.error(f'Failed to download model. Status code: {response.status_code}, Response: {response.text}')
                return jsonify({'error': 'Failed to download model from target device'}), 500
        except requests.exceptions.RequestException as e:
            logging.error(f'Error downloading model from target device: {e}')
            return jsonify({'error': 'Failed to download model from target device'}), 500

    except Exception as e:
        app.logger.error(f'Error in /update_model: {e}')
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/update_csv', methods=['POST'])
def update_csv():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input"}), 400

        time = data['time']
        label = data['label']
        time_float = float(time.replace(" sec", ""))
        print(f"Time: {time_float}, Label: {label}")

        # Read CSV
        csv_data = []
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            for row in reader:
                csv_data.append(row)

        # Find matching rows (assuming time is in the last column)
        updated = False
        for row in csv_data:
            try:
                if float(row[-1]) == time_float:
                    row[-2] = str(label)  # 修改為倒數第二列
                    updated = True
            except ValueError:
                continue  # 如果轉換失敗，跳過該行

        # 在 update_csv 函数中
        if not updated:
            return jsonify({"error": "Time not found in CSV"}), 404
        # 只在找到匹配时才写回 CSV
        else:
            with open(csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(csv_data)

        return jsonify({"message": "Success"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
def get_video_duration(video_path):
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration  # 以秒为单位
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()
        return duration
    except Exception as e:
        logging.error(f"Failed to get video duration: {e}")
        return None
@app.route('/process-csv', methods=['GET'])
def process_csv():
    """
    将 CSV 文件中的数据处理并插入到数据库中，同时删除 label 为 99 的无效行。
    """
    try:
        # 获取查询参数中的用户名
        username = request.args.get('username', None)
        if not username:
            return jsonify({'error': 'Username is required'}), 400

        # 读取 CSV 文件并获取标题
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # 检查是否存在 'label' 和 'time' 列
            if 'label' not in headers or 'time' not in headers:
                return jsonify({'error': "CSV file must contain 'label' and 'time' columns"}), 400

            # 过滤掉 label 为 99 的行
            valid_rows = []
            invalid_row_count = 0
            for row in reader:
                try:
                    label_value = int(float(row['label'].strip()))
                    if label_value != 99:
                        valid_rows.append(row)
                    else:
                        invalid_row_count += 1
                except ValueError:
                    # 如果 label 无法转换为整数，视为无效行
                    invalid_row_count += 1

        if not valid_rows:
            return jsonify({'error': 'No valid data to upload after removing label 99'}), 400

        # 清理列名，确保它们是有效的 SQL 列名
        sanitized_headers = [header.strip().replace(' ', '_').lower() for header in headers]
        num_columns = len(sanitized_headers) + 1  # +1 为 username 列

        conn = psycopg2.connect(host=db_host, dbname=db_name, user=db_user, password=db_password)
        cur = conn.cursor()

        # 检查表格是否存在，如果不存在则创建
        cur.execute("SELECT EXISTS(SELECT FROM pg_tables WHERE tablename = %s);", (table_name,))
        table_exists = cur.fetchone()[0]
        if not table_exists:
            # 创建表格，包含 username 列
            create_table_query = f"CREATE TABLE {table_name} (" + ", ".join(
                [f"{header} FLOAT" for header in sanitized_headers if header not in ['label', 'time']] +
                [f"label INTEGER", f"time FLOAT", "username TEXT"]) + ");"
            cur.execute(create_table_query)

        # 插入数据到表格
        insert_query = f"INSERT INTO {table_name} VALUES (" + ", ".join(['%s'] * num_columns) + ")"
        for row in valid_rows:
            row_values = []
            for header in headers:
                value = row[header].strip()
                if header == 'label':
                    try:
                        # 先将字符串转换为浮点数，再转换为整数
                        row_values.append(int(float(value)))
                    except ValueError:
                        row_values.append(None)  # 或者根据需求处理
                elif header == 'time':
                    try:
                        row_values.append(float(value))
                    except ValueError:
                        row_values.append(None)  # 或者根据需求处理
                else:
                    try:
                        row_values.append(float(value))
                    except ValueError:
                        row_values.append(None)  # 或者根据需求处理
            row_values.append(username)  # 添加 username 到每行的最后

            cur.execute(insert_query, tuple(row_values))

        # 提交更改并关闭连接
        conn.commit()
        cur.close()
        conn.close()

        response = {'success': 'CSV file processed and data inserted into database'}
        if invalid_row_count > 0:
            response['warning'] = f'{invalid_row_count} invalid rows with label 99 were skipped.'

        return jsonify(response), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/add_gesture', methods=['POST'])
def add_gesture():
    """
    為特定使用者新增手勢。
    """
    global class_name, actions
    try:
        data = request.get_json()
        if not data or 'name' not in data or 'username' not in data:
            return jsonify({"error": "Invalid input"}), 400

        gesture_name = data['name'].strip()
        username = data['username'].strip()
        if not gesture_name:
            return jsonify({"error": "Gesture name cannot be empty"}), 400
        if not username:
            return jsonify({"error": "Username cannot be empty"}), 400

        # 確保不是 default 模式
        if username == 'default':
            return jsonify({"error": "Cannot add gestures to default mode"}), 400

        # 创建 'data' 目录（如果不存在）
        if not os.path.exists('data'):
            os.makedirs('data')

        # 在 'data' 目录下创建以用户名为名的子目录
        user_dir = os.path.join('data', username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        # 读取现有的手势信息，如果文件存在
        gestures_file = os.path.join(user_dir, 'gestures.json')
        if os.path.exists(gestures_file):
            with open(gestures_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            gestures_list = user_data.get('gestures', [])
            existing_indices = [gesture['index'] for gesture in gestures_list]

            # 检查手势名称是否已存在
            existing_gesture_names = [gesture['name'] for gesture in gestures_list]
            if gesture_name in existing_gesture_names:
                return jsonify({"error": "Gesture name already exists"}), 400
        else:
            user_data = {'username': username, 'gestures': []}
            gestures_list = []
            existing_indices = []

        # 获取新的手势索引，确保索引唯一并且不与默认手势冲突（不包含99）
        existing_class_indices = [k for k in class_name.keys() if k < 99]
        all_existing_indices = existing_class_indices + existing_indices
        new_index = max(all_existing_indices) + 1 if all_existing_indices else 0

        # 添加新的手势到用户的手势列表
        new_gesture = {'name': gesture_name, 'index': new_index}
        gestures_list.append(new_gesture)
        user_data['gestures'] = gestures_list

        # 保存用户数据到 gestures.json
        with open(gestures_file, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, ensure_ascii=False, indent=4)

        # 更新全局的 actions 字典（如果需要）
        # actions.append(gesture_name)  # 假设 actions 是全局手势列表

        return jsonify({"message": "Gesture added", "label": new_index}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

csv_file_path = "./static/gesture_inference_data.csv"
@app.route('/delete_csv', methods=['POST'])
def delete_csv():
    """
    刪除 CSV 文件中的特定時間點的資料。
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input"}), 400

        time = data['time']
        time_float = float(time.replace(" sec", ""))
        print(f"Time: {time_float}")

        # Read CSV
        csv_data = []
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            for row in reader:
                if len(row) < 2:  # 確保至少有兩列
                    continue
                try:
                    row_time = float(row[-1])
                except ValueError:
                    continue
                if row_time != time_float:
                    csv_data.append(row)

        # Save updated CSV
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)

        return jsonify({"message": "Success"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/add_user', methods=['POST'])
def add_user():
    """
    新增一個使用者。
    """
    try:
        data = request.get_json()
        if not data or 'username' not in data:
            return jsonify({"error": "Invalid input: 'username' is required"}), 400

        username = data['username'].strip()
        if not username:
            return jsonify({"error": "Username cannot be empty"}), 400

        if username == 'default':
            return jsonify({"error": "Cannot add user with name 'default'"}), 400

        user_dir = os.path.join('data', username)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            # 初始化 gestures.json
            gestures_file = os.path.join(user_dir, 'gestures.json')
            with open(gestures_file, 'w', encoding='utf-8') as f:
                json.dump({'username': username, 'gestures': []}, f, ensure_ascii=False, indent=4)
            return jsonify({"message": "User added successfully"}), 200
        else:
            return jsonify({"error": "User already exists"}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500



























def emit_message(event, data):
    try:
        socketio.emit(event, data)
    except Exception as e:
        logging.error(f"Failed to emit message '{event}': {e}")
# Start threads
def start_threads():
    global writer_thread, capture_thread_yolo, processing_thread_yolo, record_thread, capture_thread_gesture, action_handler_thread, gesture_data_thread
    writer_thread = threading.Thread(target=csv_writer_thread_func, daemon=True)
    writer_thread.start()
    logging.info("CSV writer thread started.")

    capture_thread_yolo = threading.Thread(target=capture_thread_func_yolo, daemon=True)
    capture_thread_yolo.start()
    logging.info("YOLO capture thread started.")

    processing_thread_yolo = threading.Thread(target=processing_thread_func_yolo, daemon=True)
    processing_thread_yolo.start()
    logging.info("YOLO processing thread started.")

    record_thread = threading.Thread(target=record_thread_func, daemon=True)
    record_thread.start()
    logging.info("Consolidated recording thread started.")

    capture_thread_gesture = threading.Thread(target=frame_reader_gesture, daemon=True)
    capture_thread_gesture.start()
    logging.info("Gesture capture and processing thread started.")

    action_handler_thread = threading.Thread(target=action_handler_thread_func, daemon=True)
    action_handler_thread.start()
    logging.info("Action handler thread started.")

    gesture_data_thread = threading.Thread(target=gesture_data_handler_thread_func, daemon=True)
    gesture_data_thread.start()
    logging.info("Gesture data handler thread started.")

def csv_writer_thread_func():
    global recording
    try:
        with open(gt_filename, mode='a', newline='') as gt_file, \
             open(tr_filename, mode='a', newline='') as tr_file:
            
            gt_writer = csv.writer(gt_file)
            tr_writer = csv.writer(tr_file)
            
            while not exit_event.is_set():
                if recording:
                    try:
                        # 儲存 ground truth 資料
                        gt_data = gt_queue.get(timeout=1)
                        gt_writer.writerow(gt_data)
                        gt_file.flush()
                        logging.debug(f"Saved ground truth data: {gt_data}")
                    except queue.Empty:
                        pass
                    except Exception as e:
                        logging.error(f"Error writing ground truth data: {e}")
                    
                    try:
                        # 儲存 tracking 資料
                        tr_data = tr_queue.get(timeout=1)
                        tr_writer.writerow(tr_data)
                        tr_file.flush()
                        logging.debug(f"Saved tracking data: {tr_data}")
                    except queue.Empty:
                        pass
                    except Exception as e:
                        logging.error(f"Error writing tracking data: {e}")
                else:
                    time.sleep(0.1)
    except Exception as e:
        logging.error(f"Failed to open CSV files for writing: {e}")

@app.route('/')
def index():
    return render_template('index_combined.html')

@app.route('/video_feed_yolo')
def video_feed_yolo():
    return Response(gen_frames_yolo(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_gesture')
def video_feed_gesture():
    return Response(gen_frames_gesture(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames_yolo():
    global latest_frame_yolo, processing_current_fps, height, width
    black_frame_yolo = np.zeros((height if 'height' in globals() else 480, width if 'width' in globals() else 640, 3), dtype=np.uint8)
    while not exit_event.is_set():
        with latest_frame_lock_yolo:
            frame = latest_frame_yolo.copy() if latest_frame_yolo is not None else None
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except GeneratorExit:
                break
            except Exception as e:
                logging.error(f"Error in gen_frames_yolo: {e}")
                break
        else:
            # Send black frame
            ret, buffer = cv2.imencode('.jpg', black_frame_yolo)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except GeneratorExit:
                break
            except Exception as e:
                logging.error(f"Error in gen_frames_yolo: {e}")
                break

def gen_frames_gesture():
    global latest_frame_gesture
    black_frame_gesture = np.zeros((480, 640, 3), dtype=np.uint8)
    while not exit_event.is_set():
        with latest_frame_lock_gesture:
            frame = latest_frame_gesture.copy() if latest_frame_gesture is not None else None
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except GeneratorExit:
                break
            except Exception as e:
                logging.error(f"Error in gen_frames_gesture: {e}")
                break
        else:
            # Send black frame
            ret, buffer = cv2.imencode('.jpg', black_frame_gesture)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except GeneratorExit:
                break
            except Exception as e:
                logging.error(f"Error in gen_frames_gesture: {e}")
                break
        time.sleep(0.01)

# Add a variable to track whether the gesture model has been initialized
gesture_model_initialized = False

@socketio.on('initialize_camera')
def handle_initialize_camera():
    global cap_yolo, cap_gesture, camera_opened_yolo, camera_opened_gesture, height, width, center_x, center_y, processing_current_fps, recording, start_time, current_video_time
    # Check if the gesture model has been initialized
    if not gesture_model_initialized:
        emit('message', 'Please update the model before initializing cameras.')
        return
    # Initialize YOLO camera
    with cap_lock_yolo:
        if not camera_opened_yolo:
            cap_yolo = cv2.VideoCapture(YOLO_CAMERA_SOURCE)
            if not cap_yolo.isOpened():
                emit('message', 'Failed to open the YOLO camera stream')
                logging.error("Failed to open YOLO camera stream during initialization.")
                return

            ret, frame = cap_yolo.read()
            if not ret:
                emit('message', 'Failed to read from the YOLO camera')
                logging.error("Failed to read frame from YOLO camera during initialization.")
                cap_yolo.release()
                cap_yolo = None
                return

            # Rotate the frame 90 degrees clockwise if needed
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (640, 480))

            height, width, _ = frame.shape
            center_x = width // 2
            center_y = height // 2
            camera_opened_yolo = True
            processing_event.set()
            processing_current_fps = 0.0

            try:
                if not frame_queue_yolo.empty():
                    frame_queue_yolo.get_nowait()
                frame_queue_yolo.put_nowait(frame)
                logging.info("Initial frame put into YOLO frame_queue.")
            except queue.Full:
                logging.warning("YOLO frame queue is full during camera initialization. Dropping frame.")
        else:
            emit('message', 'YOLO camera is already initialized')
            logging.info("Attempted to initialize YOLO camera, but it is already initialized.")

    # Initialize gesture recognition camera
    with cap_lock_gesture:
        if not camera_opened_gesture:
            cap_gesture = cv2.VideoCapture(GESTURE_CAMERA_SOURCE)
            if not cap_gesture.isOpened():
                emit('message', 'Failed to open the gesture camera stream')
                logging.error("Failed to open gesture camera stream during initialization.")
                return

            ret2, frame2 = cap_gesture.read()
            frame2 = cv2.flip(frame2, 1)
                    # Resize the frame if needed
            frame2 = cv2.resize(frame2, (640, 480))
            if not ret2:
                emit('message', 'Failed to read from the gesture camera')
                logging.error("Failed to read frame from gesture camera during initialization.")
                cap_gesture.release()
                cap_gesture = None
                return

            camera_opened_gesture = True
            gesture_event.set()
            logging.info("Gesture camera initialized.")
        else:
            emit('message', 'Gesture camera is already initialized')
            logging.info("Attempted to initialize gesture camera, but it is already initialized.")

    emit('message', 'Cameras initialized')
    logging.info("Cameras initialized.")

    # 設置起始時間
    start_time = time.time()
    with current_video_time_lock:
        current_video_time = 0.0
    recording = True  # 自动开始录影
    logging.info(f"Recording started at {start_time}")

@socketio.on('stop_camera')
def handle_stop_camera():
    global camera_opened_yolo, cap_yolo, camera_opened_gesture, cap_gesture
    global processing_current_fps, recording
    
    processing_event.clear()
    gesture_event.clear()
    processing_current_fps = 0.0

    with cap_lock_yolo:
        if camera_opened_yolo:
            if cap_yolo is not None:
                cap_yolo.release()
                cap_yolo = None
            camera_opened_yolo = False
            logging.info("YOLO camera released and closed.")
    with cap_lock_gesture:
        if camera_opened_gesture:
            if cap_gesture is not None:
                cap_gesture.release()
                cap_gesture = None
            camera_opened_gesture = False
            logging.info("Gesture camera released and closed.")

    # 清空 gesture_queue 以防止继续处理数据
    while not gesture_queue.empty():
        try:
            gesture_queue.get_nowait()
        except queue.Empty:
            break

    # Clear data queues
    while not gt_queue.empty():
        gt_queue.get()
    while not tr_queue.empty():
        tr_queue.get()
    while not gesture_queue.empty():
        gesture_queue.get()

    recording = False  # Ensure recording is stopped

    emit('message', 'Cameras stopped')
    logging.info("Cameras stopped.")
    recording = False  # 自动停止录影

    # # 定义源视频文件和目标视频文件路径
    # input_filename = 'gesture_output.mp4'  # 根据实际录制的文件名修改
    # input_path = os.path.join('static', input_filename)  # 'video' 目录下

    # # 定义转换后的文件名和路径
    # output_filename_webm = f'gesture_output.webm'
    # output_path_webm = os.path.join('static', output_filename_webm)

    # os.makedirs(os.path.dirname(output_path_webm), exist_ok=True)


    # threading.Thread(target=convert_video, args=(input_path, output_path_webm, 'webm'), daemon=True).start()
    # logging.error(f"Input video file does not exist: {input_path}")

# @socketio.on('start_recording')
# def handle_start_recording():
#     global recording, frame_count, frame_count_gesture, start_time
#     if not recording and (camera_opened_yolo or camera_opened_gesture):
#         recording = True
#         frame_count = 0
#         frame_count_gesture = 0
#         start_time = time.time()  # 初始化開始錄製的時間
#         emit('message', 'Recording started...')
#         logging.info("Recording started.")
#     else:
#         emit('message', 'Already recording or cameras not opened')
#         logging.info("Attempted to start recording, but already recording or cameras not opened.")

# @socketio.on('stop_recording')
# def handle_stop_recording():
#     global recording, frame_count, frame_count_gesture
#     if recording:
#         recording = False
#         emit('message', 'Recording stopped.')
#         logging.info("Recording ended.")
#     else:
#         emit('message', 'Recording was not started')
#         logging.info("Attempted to stop recording, but recording was not started.")

@socketio.on('pause')
def handle_pause():
    global paused
    paused = not paused
    status = 'Paused' if paused else 'Resumed'
    emit('message', f'Recording {status}')
    logging.info(f"Recording {status}.")

def capture_thread_func_yolo():
    global cap_yolo, camera_opened_yolo, latest_frame_yolo
    while not exit_event.is_set():
        if processing_event.is_set() and camera_opened_yolo:
            with cap_lock_yolo:
                if cap_yolo is not None:
                    ret, frame = cap_yolo.read()
                else:
                    ret = False
            if not ret:
                socketio.emit('message', 'Failed to read frame from YOLO camera')
                processing_event.clear()
                with cap_lock_yolo:
                    if cap_yolo is not None:
                        cap_yolo.release()
                        cap_yolo = None
                    camera_opened_yolo = False
                with latest_frame_lock_yolo:
                    latest_frame_yolo = None  # Set to None on disconnection
                logging.error("Failed to read frame from YOLO camera. Camera closed.")
                break
            try:
                if not frame_queue_yolo.empty():
                    frame_queue_yolo.get_nowait()
                # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame_queue_yolo.put_nowait(frame)
            except queue.Full:
                logging.warning("YOLO frame queue is full. Dropping frame.")
        else:
            time.sleep(0.01)

def smooth_angle_func(previous_angle, current_angle, alpha=0.5):
    if previous_angle is None:
        return current_angle
    return alpha * current_angle + (1 - alpha) * previous_angle

def is_angle_stable(angle_history, threshold=1.0, required_stable_frames=3):
    if len(angle_history) < required_stable_frames:
        return False
    recent_angles = angle_history[-required_stable_frames:]
    max_angle = max(recent_angles)
    min_angle = min(recent_angles)
    return (max_angle - min_angle) <= threshold

def processing_thread_func_yolo():
    global frame_count, angle_history, recording, tracked_objects, center_x, center_y, latest_frame_yolo, shot_fired_time
    global processing_fps_counter, processing_fps_timer, processing_current_fps
    global last_turn_time, last_shot_time
    global gesture_paused
    global processed_classes
    global target_detected
    global fire_gesture_allowed, fire_gesture_start_time, fire_gesture_duration
    global current_target_class_id
    global last_target_class_id  # 确保可以在函数内使用
    global confidence_queue  # 新增
    global confidence_threshold  # 新增
    global angle_confidence_threshold  # 新增

    # 定义需要排除的类别名称
    excluded_class_names = ['white soldier', 'blue soldier', 'green soldier']  # 替换为您要排除的类别名称
    # 将排除的类别名称转换为类别ID
    excluded_class_ids = [yolo_model.names.index(name) for name in excluded_class_names if name in yolo_model.names]
    logging.info(f"Excluded class IDs: {excluded_class_ids}")

    while not exit_event.is_set():
        if processing_event.is_set() and camera_opened_yolo:
            try:
                frame = frame_queue_yolo.get(timeout=1)
            except queue.Empty:
                continue

            if not paused:
                frame_count += 1
                try:
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        results = yolo_model(frame, stream=False)
                except Exception as e:
                    logging.error(f"YOLO inference error: {e}")
                    continue

                # FPS 计算
                processing_fps_counter += 1
                elapsed_time = time.time() - processing_fps_timer
                if elapsed_time >= 1.0:
                    processing_current_fps = processing_fps_counter / elapsed_time
                    processing_fps_counter = 0
                    processing_fps_timer = time.time()

                detections = []
                highest_confidence_detection = None

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())

                            # 如果类别ID在排除列表中，则跳过
                            if class_id in excluded_class_ids:
                                logging.info(f"Excluded detection: {yolo_model.names[class_id]} (ID: {class_id})")
                                continue

                            # 检查是否在冷却中
                            current_time = time.time()
                            if class_id in processed_classes:
                                last_shot_time = processed_classes[class_id]
                                if (current_time - last_shot_time) < shot_interval:
                                    logging.info(f"Class ID {class_id} is in cooldown. Skipping detection.")
                                    continue

                            detection = [x1, y1, x2, y2, confidence, class_id]
                            detections.append(detection)
                            if highest_confidence_detection is None or confidence > highest_confidence_detection[4]:
                                highest_confidence_detection = detection
                        except Exception as e:
                            logging.warning(f"Detection parsing error: {e}")
                            continue

                detections = np.array(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)

                if highest_confidence_detection is not None:
                    confidence = float(highest_confidence_detection[4])
                    # 将 confidence 加入队列
                    confidence_queue.append(confidence)
                    if len(confidence_queue) == confidence_queue.maxlen and all(c > confidence_threshold for c in confidence_queue):
                        # ��队列已满且所有 confidence 值都大于阈值时，���许 'Fire' 手势
                        if recording:
                            gt_queue.put([frame_count] + highest_confidence_detection)
                        class_id = int(highest_confidence_detection[5])
                        label = yolo_model.names[class_id]
                        socketio.emit('target_info', {'label': label, 'confidence': confidence})
                        with processed_targets_lock:
                            fire_gesture_allowed = True
                            fire_gesture_start_time = time.time()
                            current_target_class_id = class_id
                            logging.info("Fire gesture allowed due to ten consecutive detections with high confidence.")
                        with target_detected_lock:
                            target_detected = True
                    else:
                        # 不允许 'Fire' 手势
                        socketio.emit('target_info', {'label': 'N/A', 'confidence': 0.0})
                        fire_gesture_allowed = False
                        logging.info("Fire gesture not allowed due to insufficient confidence over ten frames.")
                else:
                    # 如果没有检测到，清空队列
                    confidence_queue.clear()
                    socketio.emit('target_info', {'label': 'N/A', 'confidence': 0.0})
                    fire_gesture_allowed = False

                # 更新追踪器
                try:
                    if len(detections) > 0:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            tracked_objects = tracker.update(detections, frame)
                    else:
                        tracked_objects = tracker.update(np.empty((0, 6)), frame)
                except Exception as e:
                    logging.error(f"DeepOCSORT tracking error: {e}")
                    tracked_objects = np.empty((0, 7))

                with target_detected_lock:
                    target_detected = False  # 重置 target_detected

                if tracked_objects.size > 0:
                    for obj in tracked_objects:
                        if len(obj) >= 7:
                            x1, y1, x2, y2, track_id, confidence, class_id = obj[:7]
                            class_id = int(class_id)
                            current_time = time.time()

                            with processed_targets_lock:
                                if class_id in processed_classes:
                                    last_shot_time = processed_classes[class_id]
                                    if current_time - last_shot_time < shot_interval:
                                        continue  # 跳过这个类别，避免继续处理和绘制框
                                else:
                                    pass  # 没有冷却，继续处理

                            # 绘制所有检测框
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{yolo_model.names[class_id]} ID:{int(track_id)} {confidence:.2f}",
                                        (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # 更新 CSV
                            if recording:
                                tr_queue.put([frame_count, track_id, int(x1), int(y1), int(x2), int(y2), class_id, float(confidence)])

                    # 仅处理最高信心的追踪物体
                    try:
                        highest_tracked_object = max(tracked_objects, key=lambda x: x[5])
                    except ValueError:
                        highest_tracked_object = None

                    if highest_tracked_object is not None and len(highest_tracked_object) >= 7:
                        x1, y1, x2, y2, track_id, confidence, class_id = highest_tracked_object[:7]
                        class_id = int(class_id)
                        confidence = float(confidence)  # 确保 confidence 为浮点数
                        current_time = time.time()

                        # 新增对 confidence 值的检查
                        if confidence >= angle_confidence_threshold:
                            with processed_targets_lock:
                                if class_id in processed_classes and current_time - processed_classes[class_id] < shot_interval:
                                    # 跳过处理此类别
                                    pass
                                else:
                                    # 处理目标
                                    cx = (int(x1) + int(x2)) // 2
                                    cy = (int(y1) + int(y2)) // 2

                                    # 计算 delta_x
                                    delta_x = cx - center_x

                                    # 根据摄像机 FOV 计算角度
                                    angle_per_pixel = fov / width
                                    current_angle = delta_x * angle_per_pixel

                                    # 平滑角度
                                    smoothed_angle = smooth_angle_func(angle_history[-1] if angle_history else None, current_angle, alpha=0.5)
                                    angle_history.append(smoothed_angle)
                                    if len(angle_history) > history_length:
                                        angle_history.pop(0)

                                    # 确保 average_angle 被定义
                                    average_angle = smoothed_angle

                                    # 检查角度是否稳定
                                    if is_angle_stable(angle_history, threshold=angle_threshold, required_stable_frames=5):
                                        average_angle = sum(angle_history[-history_length:]) / len(angle_history[-history_length:])
                                        logging.info(f"Average angle: {average_angle:.2f} degrees")

                                        # 发送角度信息到前端
                                        socketio.emit('angle_update', {'angle': average_angle})

                                        # 发送转向指令
                                        if (current_time - last_turn_time) >= rotation_interval:
                                            desired_angle = max(-6, min(6, float(round(average_angle, 2))))

                                            try:
                                                if robot:
                                                    robot.turn_by(-desired_angle)
                                                    socketio.emit('aiming_status', f'Aiming... Angle: {desired_angle} degrees')
                                                    logging.info(f"Sent turn command: {desired_angle} degrees")
                                                    last_turn_time = current_time
                                            except Exception as e:
                                                logging.error(f"Robot turn command error: {e}")
                                                continue

                                        # 瞄准完成后，允许 Fire 手势在 fire_gesture_duration 秒内
                                        if -3 <= average_angle <= 3:
                                            fire_gesture_allowed = True
                                            fire_gesture_start_time = time.time()
                                            last_target_class_id = current_target_class_id  # 保存当前的目标类别 ID
                                            logging.info("Fire gesture allowed due to successful aiming.")
                                            # 显示 "Start Shooting!" 文字
                                            cv2.putText(frame, "Start Shooting!", (center_x - 100, center_y),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
                                            robot.tts("Finish aiming target!")
                                            # robot.stop()
                                            if fire_gesture_allowed:
                                                if time.time() - fire_gesture_start_time > fire_gesture_duration:
                                                    fire_gesture_allowed = False
                                            # 启动计时器，在 fire_gesture_duration 秒后重置 fire_gesture_allowed 并开始冷却
                                            threading.Thread(target=reset_fire_gesture_after_duration, args=(fire_gesture_duration, last_target_class_id), daemon=True).start()

                                        # 在框架上显示角度信息
                                        cv2.putText(frame, f"Angle: {round(float(average_angle), 2)}", (10, 60),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        else:
                            logging.info(f"Confidence {confidence:.2f} below angle command threshold. Skipping angle command.")

                else:
                    with gesture_paused_lock:
                        gesture_paused = False  # 恢复手势识别
                    target_detected = False  # 没有检测到目标
                
                # Check if 'Fire' gesture allowed time has expired (for aiming state)
                

                # 绘制中心十字线
                cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 2)
                cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 2)

                # 显示 FPS
                cv2.putText(frame, f"FPS: {processing_current_fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # 更新最新帧供前端显示
                with latest_frame_lock_yolo:
                    latest_frame_yolo = frame.copy()

                # 将处理后的帧加入录像队列
                if recording:
                    try:
                        record_queue_yolo.put(frame.copy(), timeout=0.5)  # Changed from put_nowait to put with timeout
                    except queue.Full:
                        logging.warning("Record queue for YOLO is still full after waiting. Dropping frame.")
            else:
                processing_current_fps = 0.0
                time.sleep(0.01)

def reset_fire_gesture_after_duration(duration, class_id):
    global fire_gesture_allowed, processed_classes, last_target_class_id
    time.sleep(duration)
    fire_gesture_allowed = False
    if class_id is not None:
        with processed_targets_lock:
            processed_classes[class_id] = time.time()
            logging.info(f"Class ID {class_id} added to cooldown.")
        last_target_class_id = None
    logging.info("Fire gesture permission expired and cooldown started.")


def record_thread_func():
    global recording, frame_writer_yolo, frame_writer_gesture, width, height, frame_number, current_video_time
    frame_writer_yolo = None
    frame_writer_gesture = None
    actual_fps = 20.0
    # check video folder
    if not os.path.exists('./video'):
        os.makedirs('./video')
    frame_number = 0  # 初始化帧编号
    while not exit_event.is_set():
        if recording:
            # Initialize YOLO video writer
            if frame_writer_yolo is None and 'width' in globals() and 'height' in globals():
                with record_lock:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        frame_writer_yolo = cv2.VideoWriter('./video/yolo_output.avi', fourcc, actual_fps, (width, height))
                        logging.info("YOLO VideoWriter initialized.")
                    except Exception as e:
                        logging.error(f"Failed to initialize YOLO VideoWriter: {e}")
                        recording = False
                        continue
            # Initialize gesture recognition video writer
            if frame_writer_gesture is None:
                with record_lock:
                    try:
                        frame_writer_gesture = cv2.VideoWriter('static/gesture_output.webm', cv2.VideoWriter_fourcc(*'VP80'), 30, (640, 480))
                        logging.info("Gesture VideoWriter initialized.")
                    except Exception as e:
                        logging.error(f"Failed to initialize Gesture VideoWriter: {e}")
                        recording = False
                        continue

            # Synchronously write YOLO and Gesture frames
            try:
                frame_yolo = record_queue_yolo.get(timeout=1)
            except queue.Empty:
                logging.warning("YOLO record queue is empty.")
                frame_yolo = np.zeros((height, width, 3), dtype=np.uint8)  # Placeholder black frame

            try:
                frame_gesture = record_queue_gesture.get(timeout=1)
            except queue.Empty:
                logging.warning("Gesture record queue is empty.")
                frame_gesture = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder black frame

            with record_lock:
                try:
                    frame_writer_yolo.write(frame_yolo)
                    logging.debug("YOLO frame written to file.")
                except Exception as e:
                    logging.error(f"Recording thread YOLO error: {e}")

                try:
                    frame_writer_gesture.write(frame_gesture)
                    logging.debug("Gesture frame written to file.")
                except Exception as e:
                    logging.error(f"Recording thread Gesture error: {e}")

            # **更新当前视频时间为实际时间**
            with current_video_time_lock:
                current_video_time = time.time() - start_time
                logging.debug(f"Updated video_time: {current_video_time}")

            frame_number += 1  # 增加帧编号

        else:
            with record_lock:
                if frame_writer_yolo is not None:
                    frame_writer_yolo.release()
                    frame_writer_yolo = None
                    logging.info("YOLO VideoWriter released.")
                if frame_writer_gesture is not None:
                    frame_writer_gesture.release()
                    frame_writer_gesture = None
                    logging.info("Gesture VideoWriter released.")
            time.sleep(0.01)

def frame_reader_gesture():
    global cap_gesture, paused, seq, recording, gesture_event, latest_frame_gesture, camera_opened_gesture
    global last_action, last_gesture_time, gesture_paused
    global frame_count_gesture
    global target_detected
    global fire_gesture_allowed, fire_gesture_start_time, fire_gesture_duration
    global processed_classes
    global current_target_class_id  # 確保可以在函數內使用

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0
    action_display_duration = 1
    action_to_display = None
    action_display_time = 0
    action_lock = threading.Lock()

    # 定義 'unknown' 手勢類別
    UNKNOWN_LABEL_IDX = 99  # 假設 99 是未使用的標籤
    class_name[UNKNOWN_LABEL_IDX] = 'unknown'  # 確保 class_name 包含 'unknown'
    actions.append('unknown')  # 確保 actions 列表包含 'unknown'

    # 增加幀緩衝區
    frame_buffer = deque(maxlen=5)

    while not exit_event.is_set():
        if gesture_event.is_set() and camera_opened_gesture:
            with cap_lock_gesture:
                if cap_gesture is not None:
                    ret, frame = cap_gesture.read()
                else:
                    ret = False
            if not ret:
                logging.error("Failed to read frame from gesture camera.")
                gesture_event.clear()
                with cap_lock_gesture:
                    if cap_gesture is not None:
                        cap_gesture.release()
                        cap_gesture = None
                    camera_opened_gesture = False
                with latest_frame_lock_gesture:
                    latest_frame_gesture = None  # Set to None on disconnection
                break

            frame_buffer.append(frame)

            # 使用最新的幀進行處理
            if frame_buffer:
                current_time = time.time()
                timestamp = round(current_time - start_time, 3)
                img = cv2.resize(frame_buffer[-1], (640, 480))
                img = cv2.flip(img, 1)

                if not paused:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = hands.process(img_rgb)

                    with action_lock:
                        if action_to_display and time.time() - action_display_time > action_display_duration:
                            action_to_display = None

                    if result.multi_hand_landmarks is not None:
                        for res in result.multi_hand_landmarks:
                            joint = np.zeros((21, 4))
                            for j, lm in enumerate(res.landmark):
                                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                            v1 = joint[
                                [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                            v2 = joint[
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                            v = v2 - v1
                            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                            angle = np.arccos(np.einsum('nt,nt->n',
                                                        v[
                                                            [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                                                             16, 17, 18], :],
                                                        v[
                                                            [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                                                             17, 18, 19], :]))
                            angle = np.degrees(angle)
                            d = np.concatenate([joint.flatten(), angle])
                            seq.append(d)

                            if len(seq) == seq_length:
                                test_data = np.array(seq)
                                input_data = torch.FloatTensor(test_data.reshape(1, seq_length, -1)).to(device)
                                with torch.no_grad():
                                    y_pred = gesture_model(input_data)
                                values, indices = torch.max(y_pred.data, dim=1, keepdim=True)
                                label_idx = indices.item()
                                confidence = values.item()
                                class_label = class_name.get(label_idx, None) if confidence >= 0.9 else None

                                if confidence < 0.8:
                                    # 處理未知手勢
                                    class_label = 'unknown'
                                    label_idx = UNKNOWN_LABEL_IDX

                                if class_label and class_label != 'unknown':
                                    action = actions[label_idx]

                                    # 根據 fire_gesture_allowed 狀態控制手勢處理
                                    if fire_gesture_allowed:
                                        # 只允許 "Fire" 手勢
                                        if action == 'Fire':
                                            action_seq.append(action)
                                        else:
                                            continue  # 忽略其他手勢
                                    else:
                                        # 忽略 "Fire" ���勢，允許其他手勢
                                        if action != 'Fire':
                                            action_seq.append(action)
                                        else:
                                            continue  # 忽略 "Fire" 手勢

                                    if len(action_seq) >= 9 and list(action_seq) == [action] * 9:
                                        current_time = time.time()
                                        if (current_time - last_gesture_time >= gesture_interval) or (action == 'Stop'):
                                            last_gesture_time = current_time
                                            last_action = action
                                            with action_lock:
                                                action_to_display = action
                                                action_display_time = time.time()
                                            try:
                                                action_queue.put_nowait(action)
                                                logging.info(f"Action queued: {action}")
                                            except queue.Full:
                                                logging.warning("Action queue is full. Dropping action.")

                                            # 發送手勢資訊到前端
                                            socketio.emit('gesture_info', {'gesture': action, 'confidence': confidence})

                                # 處理未知手勢
                                if class_label == 'unknown':
                                    # 顯示問號在畫面上
                                    cv2.putText(img, '?', org=(int(img.shape[1] / 2), int(img.shape[0] / 2)),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0,
                                                color=(0, 0, 255), thickness=3)

                                    # 將 'unknown' 標籤加入標籤資料中
                                    labeled_data = np.concatenate([d, [UNKNOWN_LABEL_IDX,timestamp]])
                                    assert len(labeled_data) == 101, f"Expected 100 elements, got {len(labeled_data)}"
                                    if recording:
                                        try:
                                            gesture_queue.put_nowait(labeled_data.tolist())
                                            # logging.info(f"Queued unknown gesture inference data: {labeled_data.tolist()}")
                                        except queue.Full:
                                            logging.warning("Gesture queue is full. Dropping data.")

                                    # 發送未知手勢資訊到前端
                                    socketio.emit('gesture_info', {'gesture': 'unknown', 'confidence': confidence})

                                # 繪製手部標記
                                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                                # 顯示動作文字
                                wrist = res.landmark[0]
                                img_h, img_w, _ = img.shape
                                x = int(wrist.x * img_w)
                                y = int(wrist.y * img_h)

                                with action_lock:
                                    if action_to_display:
                                        cv2.putText(img, text=action_to_display,
                                                    org=(x, y - 20),
                                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                                                    color=(0, 255, 0),
                                                    thickness=2)

                                # 儲存推理資料和標籤（99 個特徵 + 1 個標籤）
                                if confidence >= 0.8 and class_label != 'unknown':
                                    labeled_data = np.concatenate([d, [label_idx,timestamp]])
                                    assert len(labeled_data) == 101, f"Expected 100 elements, got {len(labeled_data)}"
                                    if recording:
                                        try:
                                            gesture_queue.put_nowait(labeled_data.tolist())
                                            # logging.info(f"Queued gesture inference data: {labeled_data.tolist()}")
                                        except queue.Full:
                                            logging.warning("Gesture queue is full. Dropping data.")

                            else:
                                # 沒有偵測到手部，清除動作顯示
                                with action_lock:
                                    if action_to_display and time.time() - action_display_time > action_display_duration:
                                        action_to_display = None

                # FPS 計算
                fps_counter += 1
                elapsed_time = time.time() - fps_timer
                if elapsed_time >= 1.0:
                    current_fps = fps_counter / elapsed_time
                    fps_counter = 0
                    fps_timer = time.time()

                cv2.putText(img, text=f"FPS: {current_fps:.2f}",
                            org=(10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                            color=(0, 255, 0),
                            thickness=2)

                # 更新最新的畫面供前端顯示
                with latest_frame_lock_gesture:
                    latest_frame_gesture = img.copy()

                # 將處理後的畫面加入錄製隊列
                if recording:
                    frame_count_gesture += 1
                    try:
                        record_queue_gesture.put_nowait(img.copy())
                    except queue.Full:
                        logging.warning("Record queue for gesture is full. Dropping frame.")

            time.sleep(0.01)
        else:
            time.sleep(0.01)

def execute_action(action):
    global last_command_sent_time, last_shot_time, last_fire_gesture_time
    current_time = time.time()
    with command_send_lock:
        if action == 'Fire':
            if current_time - last_fire_gesture_time >= fire_gesture_interval:
                if mqtt_client is None:
                    logging.warning("MQTT client not initialized. Cannot publish Fire command.")
                    return
                try:
                    mqtt_client.publish('servo', 'move', qos=1)
                    last_fire_gesture_time = current_time  # Update the last fire time
                    logging.info("Executed action: Fire")
                except Exception as e:
                    logging.error(f"Failed to execute Fire action: {e}")
            else:
                logging.info("Fire gesture interval not reached. Skipping Fire command.")
        else:
            if current_time - last_command_sent_time < command_send_interval:
                logging.info(f"Command '{action}' not sent due to command send interval.")
                return
            else:
                last_command_sent_time = current_time
            if action == 'Stop' and robot:
                try:
                    robot.stop()
                    logging.info("Executed action: Stop")
                except Exception as e:
                    logging.error(f"Failed to execute Stop action: {e}")
            elif action == 'Forward' and robot:
                try:
                    robot.forward(0.3)
                    logging.info("Executed action: Forward")
                except Exception as e:
                    logging.error(f"Failed to execute Forward action: {e}")
            elif action == 'Backward' and robot:
                try:
                    robot.backward(0.3)
                    logging.info("Executed action: Backward")
                except Exception as e:
                    logging.error(f"Failed to execute Backward action: {e}")
            elif action == 'Follow' and robot:
                try:
                    robot.follow()
                    logging.info("Executed action: Follow")
                except Exception as e:
                    logging.error(f"Failed to execute Follow action: {e}")
            elif action == '0 o\'clock direction' and robot:
                try:
                    robot.turn_by(0)
                    logging.info("Executed action: 0 o'clock direction")
                except Exception as e:
                    logging.error(f"Failed to execute 0 o'clock direction action: {e}")
            elif action == '3 o\'clock direction' and robot:
                try:
                    robot.turn_by(90)
                    logging.info("Executed action: 3 o'clock direction")
                except Exception as e:
                    logging.error(f"Failed to execute 3 o'clock direction action: {e}")
            elif action == '6 o\'clock direction' and robot:
                try:
                    robot.turn_by(180)
                    logging.info("Executed action: 6 o'clock direction")
                except Exception as e:
                    logging.error(f"Failed to execute 6 o'clock direction action: {e}")
            elif action == '9 o\'clock direction' and robot:
                try:
                    robot.turn_by(270)
                    logging.info("Executed action: 9 o'clock direction")
                except Exception as e:
                    logging.error(f"Failed to execute 9 o'clock direction action: {e}")

def publish_servo_command(command):
    if mqtt_client is None:
        logging.warning("MQTT client not initialized. Cannot publish command.")
        return

    def _publish():
        try:
            mqtt_client.publish('servo', command, qos=1)
            logging.info(f"Published servo command: {command}")
        except Exception as e:
            logging.error(f"Failed to publish servo command: {e}")
    threading.Thread(target=_publish, daemon=True).start()

def action_handler_thread_func():
    while not exit_event.is_set():
        try:
            action = action_queue.get(timeout=1)
            execute_action(action)
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Action handler error: {e}")

# 新增手勢數據處理執行緒函數
def gesture_data_handler_thread_func():
    global exit_event, recording
    gesture_buffer = []
    buffer_size = 10  # 增加缓冲区大小
    last_write_time = time.time()
    write_interval = 0.1  # 写入间隔（秒）
    max_allowed_diff = 0.5  # 增加允许的最大时间差，调整为更大值

    while not exit_event.is_set():
        if recording:
            try:
                if recording:  # 只在录影时处理数据
                    gesture_data = gesture_queue.get(timeout=0.1)
                    
                    # 假设 gesture_data 包含特征、标签和时间戳
                    features = gesture_data[:-2]
                    label = gesture_data[-2]
                    timestamp = gesture_data[-1]
                    
                    # 获取当前视频帧���时间戳
                    with current_video_time_lock:
                        video_time = current_video_time

                    # 比较手势时间戳和视频时间戳是否匹配
                    time_diff = abs(timestamp - video_time)
                    
                    # Log warning if timestamp difference exceeds threshold
                    if time_diff > max_allowed_diff:
                        logging.warning(f"Timestamp mismatch: gesture={timestamp}, video={video_time}, diff={time_diff}")
                    
                    # Always write data to buffer
                    csv_row = features + [label] + [timestamp]
                    gesture_buffer.append(csv_row)

                    # 当缓冲区满或达到写入间隔时写入文件
                    if len(gesture_buffer) >= buffer_size or (time.time() - last_write_time) >= write_interval:
                        with open(gesture_inference_data, mode='a', newline='') as gesture_file:
                            gesture_writer = csv.writer(gesture_file)
                            gesture_writer.writerows(gesture_buffer)
                        gesture_buffer.clear()
                        last_write_time = time.time()

            except queue.Empty:
                # 如果缓冲区有数据但队列为空，也进行写入
                if gesture_buffer and recording:
                    with open(gesture_inference_data, mode='a', newline='') as gesture_file:
                        gesture_writer = csv.writer(gesture_file)
                        gesture_writer.writerows(gesture_buffer)
                    gesture_buffer.clear()
                continue
            except Exception as e:
                logging.error(f"Error in gesture data handler: {e}")
            time.sleep(0.01)
        else:
            time.sleep(0.1)
if __name__ == '__main__':
    try:
        start_threads()
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting...")
    finally:
        exit_event.set()
        with cap_lock_yolo:
            if cap_yolo is not None:
                cap_yolo.release()
                logging.info("YOLO camera released during shutdown.")
        with cap_lock_gesture:
            if cap_gesture is not None:
                cap_gesture.release()
                logging.info("Gesture camera released during shutdown.")
        cv2.destroyAllWindows()
        logging.info("Program terminated gracefully.")
