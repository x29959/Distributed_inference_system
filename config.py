"""
Configuration management for the Distributed Inference System.
Centralizes all configuration parameters for better maintainability.
"""
import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class MQTTConfig:
    """MQTT configuration for TEMI robot communication"""
    host: str = '128.204.223.100'
    port: int = 1883
    username: str = 'steve'
    password: str = '062028633'
    serial: str = "00120495065"


@dataclass
class CameraConfig:
    """Camera stream configuration"""
    yolo_source: str = 'rtmp://100.70.26.103:1935/live/stream'
    gesture_source: str = 'rtmp://100.100.87.86:1935/live/stream'
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class ModelConfig:
    """Model configuration"""
    yolo_model_path: str = 'best_m.pt'
    tracker_weights_path: str = 'osnet_x0_25_msmt17.pt'
    gesture_model_dir: str = './model'
    default_neural_network: str = 'Transformer'
    confidence_threshold: float = 0.6
    num_classes: int = 9
    input_features: int = 99
    seq_length: int = 30


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration"""
    host: str = 'localhost'
    name: str = 'gesture'
    user: str = 't1204'
    password: str = 't1204'
    table_name: str = 'gesture_1_test'


@dataclass
class ServerConfig:
    """Flask server configuration"""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = False
    secret_key: str = 'your-secret-key-change-this'
    target_device_ip: str = '100.72.105.61'
    target_device_port: str = '5000'


@dataclass
class ProcessingConfig:
    """Processing configuration"""
    shot_interval: float = 10.0
    fire_gesture_interval: float = 5.0
    fire_gesture_duration: float = 5.0
    gesture_interval: float = 3.0
    command_send_interval: float = 2.0
    rotation_interval: float = 1.0
    angle_threshold: float = 1.0
    fov: float = 55.0
    confidence_threshold_for_fire: float = 0.6
    angle_confidence_threshold: float = 0.6


@dataclass
class FilePathConfig:
    """File path configuration"""
    ground_truth_file: str = 'ground_truth.csv'
    tracking_results_file: str = 'tracking_results.csv'
    gesture_inference_data: str = './static/gesture_inference_data.csv'
    data_dir: str = './data'
    model_dir: str = './model'
    static_dir: str = './static'
    templates_dir: str = './templates'
    video_dir: str = './video'


class SystemConfig:
    """Main system configuration container"""

    def __init__(self, config_file: Optional[str] = None):
        self.mqtt = MQTTConfig()
        self.camera = CameraConfig()
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.server = ServerConfig()
        self.processing = ProcessingConfig()
        self.filepaths = FilePathConfig()

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        else:
            self.load_from_env()

    def load_from_env(self):
        """Load configuration from environment variables"""
        # MQTT
        self.mqtt.host = os.getenv('MQTT_HOST', self.mqtt.host)
        self.mqtt.port = int(os.getenv('MQTT_PORT', self.mqtt.port))
        self.mqtt.username = os.getenv('MQTT_USERNAME', self.mqtt.username)
        self.mqtt.password = os.getenv('MQTT_PASSWORD', self.mqtt.password)
        self.mqtt.serial = os.getenv('TEMI_SERIAL', self.mqtt.serial)

        # Camera
        self.camera.yolo_source = os.getenv('YOLO_CAMERA_SOURCE', self.camera.yolo_source)
        self.camera.gesture_source = os.getenv('GESTURE_CAMERA_SOURCE', self.camera.gesture_source)

        # Database
        self.database.host = os.getenv('DB_HOST', self.database.host)
        self.database.name = os.getenv('DB_NAME', self.database.name)
        self.database.user = os.getenv('DB_USER', self.database.user)
        self.database.password = os.getenv('DB_PASSWORD', self.database.password)

        # Server
        self.server.host = os.getenv('SERVER_HOST', self.server.host)
        self.server.port = int(os.getenv('SERVER_PORT', self.server.port))
        self.server.secret_key = os.getenv('SECRET_KEY', self.server.secret_key)

    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            if 'mqtt' in config_data:
                self.mqtt = MQTTConfig(**config_data['mqtt'])
            if 'camera' in config_data:
                self.camera = CameraConfig(**config_data['camera'])
            if 'model' in config_data:
                self.model = ModelConfig(**config_data['model'])
            if 'database' in config_data:
                self.database = DatabaseConfig(**config_data['database'])
            if 'server' in config_data:
                self.server = ServerConfig(**config_data['server'])
            if 'processing' in config_data:
                self.processing = ProcessingConfig(**config_data['processing'])
            if 'filepaths' in config_data:
                self.filepaths = FilePathConfig(**config_data['filepaths'])
        except Exception as e:
            print(f"Error loading config file: {e}")

    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file"""
        config_data = {
            'mqtt': asdict(self.mqtt),
            'camera': asdict(self.camera),
            'model': asdict(self.model),
            'database': asdict(self.database),
            'server': asdict(self.server),
            'processing': asdict(self.processing),
            'filepaths': asdict(self.filepaths)
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)

    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.filepaths.data_dir,
            self.filepaths.model_dir,
            self.filepaths.static_dir,
            self.filepaths.templates_dir,
            self.filepaths.video_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Global configuration instance
config = SystemConfig()
