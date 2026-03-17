"""
Utility functions for the Distributed Inference System.
Includes error handling, validation, and helper functions.
"""
import logging
import functools
import time
import traceback
from typing import Callable, Any, Optional, Dict
import cv2
import numpy as np
import torch


class ErrorHandler:
    """Centralized error handling and logging"""

    @staticmethod
    def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
        """
        Setup logging configuration for the application

        Args:
            log_file: Optional log file path
            level: Logging level
        """
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        handlers = [logging.StreamHandler()]

        if log_file:
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=handlers
        )

        # Suppress verbose logging from third-party libraries
        logging.getLogger('boxmot').setLevel(logging.WARNING)
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        logging.getLogger('socketio').setLevel(logging.WARNING)
        logging.getLogger('engineio').setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    @staticmethod
    def log_exception(logger: logging.Logger, error: Exception, context: str = ""):
        """
        Log exception with context and traceback

        Args:
            logger: Logger instance
            error: Exception to log
            context: Additional context information
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        logger.error(error_msg)
        logger.debug(traceback.format_exc())

    @staticmethod
    def safe_execute(func: Callable, default_return: Any = None,
                    logger: Optional[logging.Logger] = None) -> Any:
        """
        Safely execute a function with error handling

        Args:
            func: Function to execute
            default_return: Default return value on error
            logger: Optional logger for error logging

        Returns:
            Function result or default_return on error
        """
        try:
            return func()
        except Exception as e:
            if logger:
                ErrorHandler.log_exception(logger, e, f"Error executing {func.__name__}")
            return default_return


def retry_on_failure(max_retries: int = 3, delay: float = 1.0,
                    exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on failure

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}"
                    )
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def validate_image(image: np.ndarray) -> bool:
    """
    Validate if image is valid

    Args:
        image: Image array

    Returns:
        True if valid, False otherwise
    """
    if image is None:
        return False
    if not isinstance(image, np.ndarray):
        return False
    if image.size == 0:
        return False
    if len(image.shape) not in [2, 3]:
        return False
    return True


def validate_model_input(data: Any, expected_shape: tuple) -> bool:
    """
    Validate model input data

    Args:
        data: Input data
        expected_shape: Expected shape tuple

    Returns:
        True if valid, False otherwise
    """
    try:
        if isinstance(data, torch.Tensor):
            return data.shape == expected_shape
        elif isinstance(data, np.ndarray):
            return data.shape == expected_shape
        return False
    except Exception:
        return False


class PerformanceMonitor:
    """Monitor performance metrics"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.metrics: Dict[str, list] = {}

    def record(self, metric_name: str, value: float):
        """Record a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append(value)

        # Keep only recent values
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name].pop(0)

    def get_average(self, metric_name: str) -> float:
        """Get average value for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])

    def get_latest(self, metric_name: str) -> float:
        """Get latest value for a metric"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return self.metrics[metric_name][-1]

    def clear(self, metric_name: Optional[str] = None):
        """Clear metrics"""
        if metric_name:
            self.metrics[metric_name] = []
        else:
            self.metrics.clear()


class ResourceManager:
    """Manage system resources"""

    @staticmethod
    def release_camera(cap):
        """Safely release camera resource"""
        if cap is not None:
            try:
                cap.release()
                return True
            except Exception as e:
                logging.error(f"Error releasing camera: {e}")
                return False
        return True

    @staticmethod
    def release_video_writer(writer):
        """Safely release video writer"""
        if writer is not None:
            try:
                writer.release()
                return True
            except Exception as e:
                logging.error(f"Error releasing video writer: {e}")
                return False
        return True

    @staticmethod
    def check_gpu_available() -> bool:
        """Check if GPU is available"""
        return torch.cuda.is_available()

    @staticmethod
    def get_device() -> str:
        """Get best available device"""
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def free_gpu_memory():
        """Free GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_black_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a black frame

    Args:
        width: Frame width
        height: Frame height

    Returns:
        Black frame array
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def safe_frame_encode(frame: np.ndarray, format: str = '.jpg') -> Optional[bytes]:
    """
    Safely encode frame to bytes

    Args:
        frame: Image frame
        format: Encoding format

    Returns:
        Encoded bytes or None on error
    """
    try:
        if not validate_image(frame):
            return None
        ret, buffer = cv2.imencode(format, frame)
        if not ret:
            return None
        return buffer.tobytes()
    except Exception as e:
        logging.error(f"Error encoding frame: {e}")
        return None


class ThreadSafeCounter:
    """Thread-safe counter"""

    def __init__(self, initial: int = 0):
        self._value = initial
        import threading
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """Increment counter"""
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """Decrement counter"""
        with self._lock:
            self._value -= amount
            return self._value

    def get(self) -> int:
        """Get current value"""
        with self._lock:
            return self._value

    def reset(self):
        """Reset counter to 0"""
        with self._lock:
            self._value = 0


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks

    Args:
        filename: Input filename

    Returns:
        Sanitized filename
    """
    import os
    # Remove directory components
    filename = os.path.basename(filename)
    # Remove any potentially dangerous characters
    dangerous_chars = ['..', '/', '\\', '\0']
    for char in dangerous_chars:
        filename = filename.replace(char, '')
    return filename


def validate_json_input(data: dict, required_fields: list) -> tuple[bool, Optional[str]]:
    """
    Validate JSON input data

    Args:
        data: Input data dictionary
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not data:
        return False, "No data provided"

    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    return True, None


# ============================================================================
# Security Functions for PyTorch Model Loading
# ============================================================================

def safe_torch_load(model_path: str, map_location=None, trusted_source: bool = False):
    """
    Safely load PyTorch models with security checks

    WARNING: Current PyTorch version (2.2.2) has known vulnerabilities.
    See SECURITY_ADVISORY.md for details and mitigation steps.

    Args:
        model_path: Path to model file
        map_location: Device mapping
        trusted_source: Only set True for models from verified sources

    Returns:
        Loaded model state dict or None on error
    """
    logger = logging.getLogger(__name__)

    # Validate model file first
    if not validate_model_file(model_path):
        logger.error(f"Model file validation failed: {model_path}")
        return None

    if not trusted_source:
        logger.warning(
            f"Loading model from untrusted source: {model_path}. "
            "This may pose security risks. See SECURITY_ADVISORY.md"
        )

    try:
        # Load model
        # Note: weights_only parameter behavior varies by PyTorch version
        # In PyTorch < 2.6.0, weights_only=True has vulnerabilities
        state_dict = torch.load(
            model_path,
            map_location=map_location,
        )

        logger.info(f"Successfully loaded model from {model_path}")
        return state_dict

    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None


def validate_model_file(model_path: str, max_size_gb: float = 5.0) -> bool:
    """
    Validate model file before loading

    Args:
        model_path: Path to model file
        max_size_gb: Maximum allowed file size in GB

    Returns:
        True if file appears safe, False otherwise
    """
    import os

    if not os.path.exists(model_path):
        logging.error(f"Model file does not exist: {model_path}")
        return False

    if not os.path.isfile(model_path):
        logging.error(f"Model path is not a file: {model_path}")
        return False

    # Check file size (prevent extremely large files)
    max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
    file_size = os.path.getsize(model_path)

    if file_size > max_size_bytes:
        logging.warning(
            f"Model file {model_path} size ({file_size} bytes) "
            f"exceeds maximum ({max_size_bytes} bytes)"
        )
        return False

    if file_size == 0:
        logging.error(f"Model file {model_path} is empty")
        return False

    return True


def verify_model_checksum(model_path: str, expected_hash: str) -> bool:
    """
    Verify model file integrity using SHA256 checksum

    Args:
        model_path: Path to model file
        expected_hash: Expected SHA256 hash

    Returns:
        True if checksum matches, False otherwise
    """
    import hashlib

    try:
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        actual_hash = sha256_hash.hexdigest()
        matches = actual_hash == expected_hash

        if not matches:
            logging.error(
                f"Model checksum mismatch for {model_path}. "
                f"Expected: {expected_hash}, Got: {actual_hash}"
            )

        return matches

    except Exception as e:
        logging.error(f"Error computing checksum for {model_path}: {e}")
        return False


def is_trusted_model_source(model_path: str, trusted_sources: Optional[list] = None) -> bool:
    """
    Check if model is from a trusted source

    Args:
        model_path: Path to model file
        trusted_sources: List of trusted source directories

    Returns:
        True if source is trusted, False otherwise
    """
    import os

    if trusted_sources is None:
        # Default trusted sources
        trusted_sources = [
            './model/',
            './best_m.pt',
            './osnet_x0_25_msmt17.pt',
        ]

    abs_path = os.path.abspath(model_path)

    for source in trusted_sources:
        trusted_abs = os.path.abspath(source)
        # Check if file is the trusted source or within trusted directory
        if abs_path == trusted_abs or abs_path.startswith(trusted_abs + os.sep):
            return True

    return False
