import time
from flask import Flask, request, render_template
from flask import jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import threading
import subprocess
import signal
import csv
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Event
from model_1 import Transformer_rev_Complex, GRUClassifier, CRNN, CNN_LSTM, CBiLSTM
from flask import Flask, send_from_directory, jsonify, request
import torch
import cv2
from threading import Semaphore
import base64
import queue
from collections import deque
import psutil
import psycopg2
import pytemi as temi

TEMI_SERIAL = "00120495065"
MQTT_HOST = 'stevetw.serv00.net'
MQTT_PORT = 1883
MQTT_USERNAME = 'steve'
MQTT_PASSWORD = '062028633'
mqtt_client = temi.connect(MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD)
robot = temi.Robot(mqtt_client, TEMI_SERIAL)

# 以下变量根据您的需求进行设置
num = 9  # 初始手势类别数量
csv_file_path = "./static/test.csv"
executor = ThreadPoolExecutor(max_workers=20)  # Thread pool executor for handling image processing
max_threads = 20
task_semaphore = Semaphore(max_threads)
global_frame_id = 0  # Global frame ID counter
out = None  # Video writer object
recording_thread = None  # Recording thread object
recording_queue = queue.Queue()  # Recording queue

video_write_lock = Lock()
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources={r"/*": {"origins": "*"}})
process = None
is_training_complete = False

# 数据库连接参数
db_host = "100.68.188.72"
db_name = "gesture"
db_user = "t1204"
db_password = "t1204"
table_name = "gesture_1_test"  # 假设表名为 'gesture_1_test'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
if not os.path.exists('model.json'):
    model_name = "Transformer"
    with open("model.json", "w") as f:
        json.dump(model_name, f)
else:
    try:
        with open("model.json", "r") as f:
            model_name = json.load(f)
    except json.JSONDecodeError:
        model_name = "Transformer"
        with open("model.json", "w") as f:
            json.dump(model_name, f)

# Define models
def load_model(model_name, num_classes,dropout=0.3):
    if model_name == "Transformer":
        return Transformer_rev_Complex(99, num_classes, 32, 2,dropout).to(device)
    elif model_name == "CNN_LSTM":
        return CNN_LSTM(99, num_classes, 64, 32,dropout).to(device)
    elif model_name == "CBiLSTM":
        return CBiLSTM(99, 64, 32,dropout).to(device)
    elif model_name == "GRUClassifier":
        return GRUClassifier(99, 64, 32, num_classes,dropout).to(device)
    elif model_name == "CRNN":
        return CRNN(99, 64, 32,dropout).to(device)

# Load the default model
model = load_model(model_name, num)
model.load_state_dict(torch.load('model/Transformer_final.pth', map_location=device))
model.eval()

# Initialize class_name and actions
if os.path.exists('class_name.json'):
    with open('class_name.json', 'r', encoding='utf-8') as f:
        class_name = json.load(f)
    # Convert keys to integers if necessary
    class_name = {int(k): v for k, v in class_name.items()}
else:
    class_name = {
        0: 'Forward',
        1: 'Backward',
        2: 'Stop',
        3: 'Fire',
        4: 'Follow',
        5: '0 oclock direction',
        6: '3 oclock direction',
        7: '6 oclock direction',
        8: '9 oclock direction',
        99: '?'  # Added unknown label
    }
    # Save the initial class_name to file
    with open('class_name.json', 'w', encoding='utf-8') as f:
        json.dump(class_name, f, ensure_ascii=False, indent=4)
actions = list(class_name.values())


cumulative_delay = 0  # To keep track of any cumulative delay
fps = 30
is_recording = True  # 初始狀態為正在錄影
stop_event = Event()
recording_start_time = None
out = None
total_time = 0.0
st = time.time()
frame_queue = deque(maxlen=300)

# 全局变量，用于控制训练线程和停止事件
training_thread = None
stop_training_event = threading.Event()
# 使用鎖來保護共享變量的訪問
progress_lock = threading.Lock()
metrics_lock = threading.Lock()
hyperparameters_lock = threading.Lock()

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

def publish_servo_command(command):
    print(f"Publishing command: {command}")
    mqtt_client.publish('servo', command, qos=0)

keep_sending = False
command_thread = None

@app.route('/control', methods=['GET'])
def control():
    global keep_sending, command_thread

    command = request.args.get('command')
    print(f"Received command: {command}")
    if not command:
        return "No command sent."

    # Single execution for Fire
    if command == 'Fire':
        publish_servo_command('move')
        return "Fire command executed once."

    if command == 'stop':
        keep_sending = False
        if command_thread and command_thread.is_alive():
            command_thread.join()
        return "Stopped the command loop."

    # Stop any previous loop
    keep_sending = False
    if command_thread and command_thread.is_alive():
        command_thread.join()

    # Start a new loop
    keep_sending = True
    command_thread = threading.Thread(target=loop_commands, args=(command,))
    command_thread.start()
    return f"Loop started with command: {command}"

def loop_commands(cmd):
    while keep_sending:
        if cmd.startswith('Rotate '):
            try:
                angle = float(cmd.split(' ')[1])
                robot.turn_by(angle)
            except:
                pass
        elif cmd == 'Forward':
            robot.forward(1)
        elif cmd == 'Stop':
            robot.stop()
        elif cmd == 'Turn Left':
            robot.turn_by(-30)
        elif cmd == 'Turn Right':
            robot.turn_by(30)
        elif cmd == 'Follow':
            robot.follow()
        elif cmd == 'Backward':
            robot.backward(1)
        time.sleep(1)




@app.route('/set_hyperparameters', methods=['POST'])
def set_hyperparameters():
    """
    接收前端發送的超參數，並存儲在內存中的全局變量 `hyperparameters`。
    同時，根據用戶名和神經網絡架構創建必要的目錄。
    """
    global hyperparameters

    data = request.get_json()
    if not data:
        return jsonify({'message': 'No data provided'}), 400

    username = data.get('username')
    print(username)
    neural_network = data.get('neuralNetwork')

    if not username or not neural_network:
        return jsonify({'message': 'Username and Neural Network are required'}), 400

    # 使用鎖來保護超參數的設置
    with hyperparameters_lock:
        hyperparameters = {
            "learningRate": data.get("learningRate", 0.001),
            "epoch": data.get("epoch", 10),
            "batchSize": data.get("batchSize", 32),
            "lossFunction": data.get("lossFunction", "CrossEntropyLoss"),
            "optimizer": data.get("optimizer", "adam"),
            "neuralNetwork": neural_network,
            "username": username,
            "model_name": data.get("model_name", "train_scratch")  # 默認為從頭訓練
        }

    # 創建必要的目錄
    model_dir = f'./model/{username}/{neural_network}'
    os.makedirs(model_dir, exist_ok=True)

    return jsonify({'message': 'Hyperparameters set successfully'}), 200



@app.route('/set_model', methods=['POST'])
def set_model():
    global model_instance
    model_name = request.form['model']
    model_instance = load_model(model_name)
    with open("model.json", "w") as f:
        json.dump(model_name, f)
    return jsonify({'status': 'Model updated', 'model': model_name})

def process_image(data):
    global frame_count, recording_start_time, fps, total_time, st
    global is_recording, hands, mp_drawing, mp_hands, model, device
    global seq, seq_length, data_buffer, actions, class_name
    global action_seq, out, video_write_lock

    if not is_recording:
        return np.zeros((480, 640, 3), dtype=np.uint8)  # 返回一張黑色圖像

    # 解碼圖像資料
    try:
        sbuf = base64.b64decode(data.split(',')[1])
        pimg = np.frombuffer(sbuf, dtype=np.uint8)
        frame = cv2.imdecode(pimg, flags=1)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    if frame is None:
        print("Failed to decode image")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # 圖像預處理
    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    confidence = 0.0
    label_idx = 99  # 預設未知手勢索引為 99
    action = '?'

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            # 提取關節點資料
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # 計算角度特徵
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1

            # 防止除以零
            v_norm = np.linalg.norm(v, axis=1)[:, np.newaxis]
            v_norm[v_norm == 0] = 1e-6
            v = v / v_norm

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)

            # 確保序列長度不超過 seq_length
            seq = seq[-seq_length:]

            # 繪製關節點
            if time.time() - st > 0.1:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # 進行手勢識別
            if len(seq) == seq_length:
                input_data = torch.FloatTensor(np.array(seq).reshape(1, seq_length, -1)).to(device)
                with torch.no_grad():
                    y_pred = model(input_data)
                values, indices = torch.max(y_pred.data, dim=1)
                confidence = values.item()
                if (confidence >= 0.7):
                    label_idx = indices.item()
                    action = class_name.get(label_idx, '?')
                else:
                    label_idx = 99  # 未知手勢索引
                    action = '?'

                # 準備標記資料並加入緩衝區
                total_time = round(frame_count / fps, 2)
                labeled_data = np.hstack(
                    [np.array(seq), np.full((seq_length, 1), label_idx), np.full((seq_length, 1), total_time)]
                )

                data_buffer.append(labeled_data)

                # 更新動作序列
                action_seq.append(action)
                if len(action_seq) > 8:
                    action_seq = action_seq[-8:]

                # 判斷當前動作
                this_action = '?'
                if action_seq.count(action_seq[-1]) == len(action_seq):
                    this_action = action_seq[-1]

                # 在圖像上標註動作和置信度
                cv2.putText(img, this_action, org=(int(img.shape[1] / 2), int(img.shape[0] / 2)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
                            color=(0, 0, 255), thickness=3)
                cv2.putText(img, f'{confidence:.2f}', org=(int(img.shape[1] / 2), int(img.shape[0] / 2) + 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5,
                            color=(0, 0, 255), thickness=3)

        # 更新幀計數和總時間
        frame_count += 1
        st = time.time()
        total_time = round(frame_count / fps, 2)
        cv2.putText(img, f"Time: {total_time:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    else:
        # 未檢測到手部，不進行處理
        # 只更新幀計數和時間
        frame_count += 1
        st = time.time()
        total_time = round(frame_count / fps, 2)
        cv2.putText(img, f"Time: {total_time:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 寫入 CSV 檔案
    if data_buffer:
        file_path = 'static/test.csv'
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for data in data_buffer:
                writer.writerows(data)
        data_buffer.clear()  # 清空緩衝區

    # 寫入視頻幀
    if out is not None and is_recording:
        with video_write_lock:
            out.write(img)

    return img

def process_and_emit():
    while not stop_event.is_set():
        if frame_queue:
            data = frame_queue.popleft()
            try:
                frame = process_image(data)
            except Exception as e:
                print(f"Error processing frame: {e}")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 返回一個黑色影像

            _, buffer = cv2.imencode('.jpg', frame)
            response = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('response', response)
        else:
            socketio.sleep(0.01)

threading.Thread(target=process_and_emit).start()

@socketio.on('image')
def handle_image(data):
    global is_recording

    if stop_event.is_set():
        return

    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 80:
        if is_recording:
            is_recording = False  # 暫停錄影
            print("High memory usage detected. Pausing recording.")
    else:
        if not is_recording:
            is_recording = True  # 恢復錄影
            print("Memory usage normal. Resuming recording.")

    if len(frame_queue) < frame_queue.maxlen:
        frame_queue.append(data)
    else:
        print("Frame queue is full. Skipping frame.")

@app.route('/start')
def start_processing():
    global out, frame_count, total_time, recording_start_time, is_recording

    # 重置變數
    stop_event.clear()
    frame_count = 0
    total_time = 0
    recording_start_time = 0
    is_recording = True

    # 初始化 VideoWriter
    out = cv2.VideoWriter('static/output.webm', cv2.VideoWriter_fourcc(*'VP80'), 30, (640, 480))

    # 初始化 CSV 文件，寫入標題行
    file_path = 'static/test.csv'
    num_features = 99  # 特徵數量
    header = [f'feature{i}' for i in range(num_features)] + ['label', 'time']
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    return "Started video processing"

@app.route('/stop')
def stop_processing():
    global out, is_recording

    # 停止錄影
    is_recording = False
    executor.shutdown(wait=True)

    if out is not None:
        with video_write_lock:
            out.release()
            out = None

    return "Stopped video processing"

@app.route('/models', methods=['GET', 'POST'])
def models():
    models_folder = 'model'  # Replace with the path to your models directory
    models = os.listdir(models_folder)
    return jsonify({'models': models})

@app.route('/delete_model', methods=['POST'])
def delete_model():
    data = request.get_json(silent=False)  # This prevents throwing errors if the content is not JSON
    if data and 'model' in data:
        model_to_delete = data['model']
        model_path = os.path.join('model', model_to_delete)
        try:
            os.remove(model_path)
            return jsonify({'status': 'success', 'message': 'Model deleted.'})
        except FileNotFoundError:
            return jsonify({'status': 'error', 'message': 'Model not found.'}), 404
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        return jsonify({'status': 'error', 'message': 'Model key is missing in request.'}), 400

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

        if not updated:
            return jsonify({"error": "Time not found in CSV"}), 404

        # Write back to CSV
        with open(csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(csv_data)

        return jsonify({"message": "Success"}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process-csv', methods=['GET'])
def process_csv():
    """
    將 CSV 文件中的資料處理並插入到資料庫中。
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

        # 清理列名，确保它们是有效的 SQL 列名
        sanitized_headers = [header.strip().replace(' ', '_').lower() for header in headers]
        num_columns = len(sanitized_headers) + 1  # +1 為 username 列

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
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_values = []
                for header in headers:
                    value = row[header].strip()
                    if header == 'label':
                        try:
                            # 先将字符串转换为浮点数，再转换为整数
                            row_values.append(int(float(value)))
                        except ValueError:
                            return jsonify({'error': f"Invalid label value: {value}"}), 400
                    elif header == 'time':
                        row_values.append(float(value))
                    else:
                        row_values.append(float(value))
                row_values.append(username)  # 添加 username 到每行的最後

                placeholders = ', '.join(['%s'] * num_columns)
                insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
                cur.execute(insert_query, tuple(row_values))

        # 提交更改並關閉連接
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'success': 'CSV file processed and data inserted into database'}), 200
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
        if (username == 'default'):
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

# @app.route('/get_users', methods=['GET', 'POST'])
# def get_users():
#     """
#     返回 data 目錄下的所有使用者列表。
#     """
#     try:
#         data_dir = 'data'
#         if not os.path.exists(data_dir):
#             os.makedirs(data_dir)
        
#         users = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
#         return jsonify({'users': users}), 200
#     except Exception as e:
#         app.logger.error(f'Error in /get_users: {e}')
#         return jsonify({'error': 'Internal Server Error'}), 500

# 获取用户的手势列表
@app.route('/get_gestures/<username>', methods=['GET', 'POST'])
def get_gestures(username):
    gestures_file = f'./data/{username}/gestures.json'
    if os.path.exists(gestures_file):
        with open(gestures_file, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"message": "Gestures file not found"}), 404


@app.route('/get_models')
def get_models():
    models_dir = 'model'
    models = os.listdir(models_dir)
    return jsonify(models=models)


# 获取用户的模型列表
@app.route('/get_user_models/<username>/<neural_network>', methods=['GET', 'POST'])
def get_user_models(username, neural_network):
    """
    獲取指定用戶和神經網絡類型的模型列表
    """
    try:
        if username == 'default':
            # 對於默認用戶，檢查特定神經網絡的目錄
            models_dir = f'./model/{neural_network}'
        else:
            # 對於其他用戶，檢查其專屬目錄下的特定神經網絡目錄
            models_dir = f'./model/{username}/{neural_network}'
        
        # 確保目錄存在
        os.makedirs(models_dir, exist_ok=True)
        
        # 只獲取 .pth 文件
        models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        return jsonify(models=models)
        
    except Exception as e:
        app.logger.error(f'Error in get_user_models: {str(e)}')
        return jsonify(models=[])

@app.route('/get_model/<username>/<neural_network>/<model_name>', methods=['GET'])
def get_model(username, neural_network, model_name):
    """
    提供模型文件的下載。
    """
    try:
        models_dir = os.path.join('model', username, neural_network)
        if not os.path.exists(models_dir):
            return jsonify({'error': 'Model directory not found'}), 404

        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404

        # 使用 send_from_directory 來提供文件下載，使用位置參數
        return send_from_directory(models_dir, model_name, as_attachment=True)
    except Exception as e:
        app.logger.error(f'Error in /get_model: {e}')
        return jsonify({'error': 'Internal Server Error'}), 500


@app.route('/train', methods=['POST', 'GET'])
def start_training():
    """
    啟動訓練線程，使用當前設置的超參數。
    """
    global training_thread, stop_training_event

    if training_thread and training_thread.is_alive():
        app.logger.error('Training is already running')
        return jsonify({'message': 'Training is already running'}), 400

    with hyperparameters_lock:
        current_hyperparameters = hyperparameters.copy()

    if not current_hyperparameters:
        return jsonify({'message': 'Hyperparameters not set'}), 400

    username = current_hyperparameters.get('username')
    model_name_selected = current_hyperparameters.get('model_name')
    neural_network = current_hyperparameters.get('neuralNetwork')

    if not username or not neural_network:
        return jsonify({'message': 'Username and Neural Network must be set'}), 400

    # 獲取手勢文件
    gestures_file = f'./data/{username}/gestures.json'
    if os.path.exists(gestures_file):
        with open(gestures_file, 'r') as f:
            user_data = json.load(f)
        gestures = user_data.get('gestures', [])
        num_classes = 9 + len(gestures)  # 假設默認有9個手勢
    else:
        app.logger.error(f'Gestures file not found for user: {username}')
        return jsonify({'message': 'Gestures file not found for the user'}), 400

    # 初始化訓練進度和指標
    with progress_lock:
        training_progress = {"epoch": 0, "batch": 0, "loss": 0.0, "progress": 0, "accuracy": 0.0}
    with metrics_lock:
        training_metrics = {"epoch": 0, "avg_train_loss": 0.0, "Train_accuracy": 0.0, 
                            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                            "confusion": []}

    # 重置停止事件
    stop_training_event.clear()
    print(f"Training started for user: {username}, model: {model_name_selected}",num_classes)

    # 啟動訓練線程
    try:
        training_thread = threading.Thread(target=train_model, args=(username, model_name_selected, num_classes))
        training_thread.start()
        app.logger.info('Training thread started.')
    except Exception as e:
        app.logger.error(f'Error starting training thread: {e}')
        return jsonify({'message': 'Failed to start training'}), 500

    app.logger.info('Training started successfully')
    return jsonify({'message': 'Training started'}), 200


@app.route('/stop_train', methods=['POST', 'GET'])
def stop_training_route():
    global stop_training_event, training_thread
    if not training_thread or not training_thread.is_alive():
        return jsonify({'message': 'No training process is running'}), 400

    # 設置停止事件
    stop_training_event.set()

    # 等待訓練線程結束
    training_thread.join()
    return jsonify({'message': 'Training process stopped'}), 200

@app.route('/get_train_process_update', methods=['GET', 'POST'])
def train_process_update():
    try:
        with progress_lock:
            return jsonify(training_progress), 200
    except Exception as e:
        app.logger.error(f'Error in /get_train_process_update: {e}')
        return jsonify({'message': 'Internal Server Error'}), 500

@app.route('/metrics_update', methods=['GET', 'POST'])
def metrics_update():
    try:
        with metrics_lock:
            return jsonify(training_metrics), 200
    except Exception as e:
        app.logger.error(f'Error in /metrics_update: {e}')
        return jsonify({'message': 'Internal Server Error'}), 500

# 定义训练函数
# 定義訓練函數
def train_model(username, model_name_selected, num_classes):
    """
    執行模型訓練的函數。
    """
    global training_progress, training_metrics, hyperparameters

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import psycopg2
    import numpy as np
    from retrain_dataset import retrain_dataset
    from model_1 import CNN_LSTM, CBiLSTM, Transformer_rev_Complex, GRUClassifier, CRNN
    from torch.utils.data import Dataset, DataLoader, random_split
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import pandas as pd
    import threading, time
    import time  # 引入time模組

    # 獲取超參數
    # 獲取超參數
    with hyperparameters_lock:
        learning_rate = hyperparameters.get("learningRate", 0.001)
        num_epochs = hyperparameters.get("epoch", 10)
        batch_size = hyperparameters.get("batchSize", 30)
        loss_function_name = hyperparameters.get("lossFunction", "CrossEntropyLoss")
        optimizer_name = hyperparameters.get("optimizer", "adam")
        neural_network = hyperparameters.get("neuralNetwork", "Transformer")
        dropout = hyperparameters.get("dropout", 0.2)

    # 初始化模型
    try:
        if neural_network == "Transformer":
            model = Transformer_rev_Complex(99, num_classes, 32, 2,dropout).to(device)
        elif neural_network == "CNN_LSTM":
            model = CNN_LSTM(99, num_classes, 64, 32,dropout).to(device)
        elif neural_network == "CBiLSTM":
            model = CBiLSTM(99, 64, 32,dropout).to(device)
        elif neural_network == "GRUClassifier":
            model = GRUClassifier(99, 64, 32, num_classes,dropout).to(device)
        elif neural_network == "CRNN":
            model = CRNN(99, 64, 32,dropout).to(device)
        else:
            app.logger.error(f'Unknown neural network architecture: {neural_network}')
            return
    except Exception as e:
        app.logger.error(f'Error initializing model: {e}')
        return

    # 加載用戶選擇的模型或從頭開始訓練
    if model_name_selected and model_name_selected != 'train_scratch':
        model_path = f"./model/{username}/{model_name_selected}"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                app.logger.info(f'Loaded model from {model_path}')
            except Exception as e:
                app.logger.error(f'Error loading model from {model_path}: {e}')
                print("Error loading model. Training from scratch.")
        else:
            app.logger.warning(f'Selected model {model_name_selected} not found. Training from scratch.')

    # 定義損失函數和優化器
    try:
        if loss_function_name == "CrossEntropyLoss":
            loss_func = nn.CrossEntropyLoss()
        elif loss_function_name == "NLLLoss":
            loss_func = nn.NLLLoss()
        else:
            app.logger.error(f'Unknown loss function: {loss_function_name}')
            return

        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            app.logger.error(f'Unknown optimizer: {optimizer_name}')
            return
    except Exception as e:
        app.logger.error(f'Error setting loss function or optimizer: {e}')
        return

    # 連接到資料庫
    db_host = "100.68.188.72"
    db_name = "gesture"
    db_user = "t1204"
    db_password = "t1204"

    try:
        conn = psycopg2.connect(
            host=db_host,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cur = conn.cursor()
        table_name = "gesture_1_test"  # 假設表名

        # 根據模式獲取資料
        if username != 'default':
            # Custom 模式：獲取自定義使用者和 default 使用者的資料
            query = f"SELECT * FROM {table_name} WHERE username = %s OR username = 'default';"
            cur.execute(query, (username,))
        else:
            # Default 模式：僅獲取 default 使用者的資料
            query = f"SELECT * FROM {table_name} WHERE username = %s;"
            cur.execute(query, (username,))

        data = cur.fetchall()

        # 關閉連接
        cur.close()
        conn.close()

        if not data:
            app.logger.error("No data found for user. Exiting training.")
            return

        # 將資料轉換為 numpy 陣列，根據資料結構提取特徵和標籤
        # 假設資料格式：列 0-98 = 手勢資料，列 -3 = 標籤，列 -2 = 時間，列 -1 = 使用者名稱
        try:
            data_np = np.array([list(map(float, row[0:99])) for row in data])  # 手勢資料
            labels_np = np.array([int(row[-3]) for row in data])             # 標籤
        except Exception as e:
            app.logger.error(f'Error processing data from database: {e}')
            return

    except Exception as e:
        app.logger.error(f'Database error: {e}')
        return

    # 加載數據集
    try:
        train_dataset = retrain_dataset(window_size=30, data=data_np, labels=labels_np)
    except Exception as e:
        app.logger.error(f'Error initializing retrain_dataset: {e}')
        return

    # 數據加載器
    try:
        # 根據數據集大小動態調整批次大小
        total_samples = len(data_np)
        batch_size = min(batch_size, total_samples // 10)  # 確保批次大小不會太大
        batch_size = max(30, batch_size)  # 確保批次大小不會太小
        
        # 加入資料採樣
        max_samples = 50000  # 設置最大樣本數
        if len(data_np) > max_samples:
            indices = np.random.choice(len(data_np), max_samples, replace=False)
            data_np = data_np[indices]
            labels_np = labels_np[indices]

        train_dataset = retrain_dataset(window_size=30, data=data_np, labels=labels_np)
        
        # 創建 DataLoader 時使用 drop_last=True
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,  # 丟棄最後一個不完整的批次
            num_workers=4,
            pin_memory=True
        )
        
        if len(train_loader) == 0:
            raise ValueError("Dataset is too small for the current batch size")
            
    except Exception as e:
        app.logger.error(f'Error initializing dataset: {str(e)}')
        return

    # 設定每個 epoch 使用的最大樣本數
    max_samples_per_epoch = 10000  # 您可以根據需要調整此值

    # 在訓練循環中，每個 epoch 開始時進行數據抽樣
    for epoch in range(1, num_epochs + 1):
        if stop_training_event.is_set():
            app.logger.info(f"Training stopped by user at epoch {epoch}")
            break

        # 在每個 epoch 開始時，隨機抽樣數據
        try:
            if len(data_np) > max_samples_per_epoch:
                indices = np.random.choice(len(data_np), max_samples_per_epoch, replace=False)
                sampled_data_np = data_np[indices]
                sampled_labels_np = labels_np[indices]
            else:
                sampled_data_np = data_np
                sampled_labels_np = labels_np

            # 創建新的數據集和數據加載器
            train_dataset = retrain_dataset(window_size=30, data=sampled_data_np, labels=sampled_labels_np)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=4,
                pin_memory=True
            )
            if len(train_loader) == 0:
                raise ValueError("Dataset is too small for the current batch size")

        except Exception as e:
            app.logger.error(f'Error sampling data for epoch {epoch}: {str(e)}')
            continue

        # 開始本 epoch 的訓練
        model.train()
        Train_total_loss = 0.0
        Train_correct_predictions = 0
        all_labels = []
        all_preds = []

        # 使用新的 train_loader 進行訓練
        with tqdm(train_loader, unit="batch", disable=False) as tepoch:
            for batch_idx, (x_train, y_train) in enumerate(tepoch):
                if stop_training_event.is_set():
                    break

                try:
                    x_train = x_train.to(device)
                    y_train = y_train.to(device).long()
                    
                    # 檢查批次大小
                    if x_train.size(0) != y_train.size(0):
                        continue
                        
                    y_predict = model(x_train)

                    loss = loss_func(y_predict, y_train)
                    Train_total_loss += loss.item()

                    _, indices = torch.max(y_predict, dim=1)
                    Train_correct_predictions += (indices == y_train).sum().item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    all_labels.append(y_train.cpu().numpy())
                    all_preds.append(indices.cpu().numpy())
                    tepoch.set_description(f"Epoch: {epoch}")
                    tepoch.update(1)

                    progress_percentage = (tepoch.n / tepoch.total) * 100

                    # 更新訓練進度
                    with progress_lock:
                        training_progress.update({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss": loss.item(),
                            "progress": progress_percentage,
                            "accuracy": Train_correct_predictions / ((batch_idx + 1) * batch_size)
                        })

                    time.sleep(0.1)  # 添加0.1秒的延遲
                except Exception as e:
                    app.logger.error(f'Error during training at epoch {epoch}, batch {batch_idx}: {e}')
                    continue

        try:
            avg_train_loss = Train_total_loss / len(train_loader)
            Train_accuracy = 100. * Train_correct_predictions / len(train_loader.dataset)

            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            accuracies = accuracy_score(all_labels, all_preds)
            precisions = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            confusion = confusion_matrix(all_labels, all_preds)

            # 更新訓練指標
            with metrics_lock:
                training_metrics.update({
                    "epoch": epoch,
                    "avg_train_loss": avg_train_loss,
                    "Train_accuracy": Train_accuracy,
                    "accuracy": accuracies,
                    "precision": precisions,
                    "recall": recall,
                    "f1": f1,
                    "confusion": confusion.tolist()
                })

            # 儲存結果到 CSV
            os.makedirs('result', exist_ok=True)
            results = {
                'loss': [avg_train_loss],
                'accuracy': [Train_accuracy],
                'precision': [precisions],
                'recall': [recall],
                'f1_score': [f1]
            }
            df = pd.DataFrame(results)
            df.to_csv('result/training_results.csv', index=False)

            # Save confusion matrix to CSV
            if epoch == num_epochs:
                confusion_df = pd.DataFrame(confusion)
                confusion_df.to_csv('result/final_confusion_matrix.csv', index=False)
        except Exception as e:
            app.logger.error(f'Error updating training metrics: {e}')
            continue

    # 訓練結束後儲存模型
    try:
        # 保存模型
        model.eval()
        user_model_dir = f'./model/{username}/{neural_network}'
        os.makedirs(user_model_dir, exist_ok=True)

        # 生成新的模型名稱
        existing_models = os.listdir(user_model_dir)
        version_numbers = []
        for name in existing_models:
            if name.startswith(neural_network) and name.endswith('.pth'):
                parts = name.split('_')
                if len(parts) < 3:
                    continue  # 跳過不符合格式的文件
                version_part = parts[-1].split('.')[0]
                if version_part.isdigit():
                    version_numbers.append(int(version_part))

        if version_numbers:
            new_version = max(version_numbers) + 1
        else:
            new_version = 1

        model_filename = f"{neural_network}_{username}_{new_version}.pth"
        torch.save(model.state_dict(), os.path.join(user_model_dir, model_filename))
        app.logger.info(f'Model saved to {os.path.join(user_model_dir, model_filename)}')

        app.logger.info("Training process has ended.")
    except Exception as e:
        app.logger.error(f'Error saving model: {e}')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    # train_model('default', 'train_scratch', 9)