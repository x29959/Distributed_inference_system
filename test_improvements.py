"""
Example tests for the improved system components.
These demonstrate how to use the new utilities.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SystemConfig
from utils import (
    ErrorHandler, PerformanceMonitor, ResourceManager,
    validate_image, validate_json_input, sanitize_filename,
    retry_on_failure, ThreadSafeCounter
)
from health_monitor import HealthMonitor
import numpy as np
import logging


def test_configuration():
    """Test configuration management"""
    print("Testing Configuration Management...")

    config = SystemConfig()

    # Test accessing configuration
    assert config.mqtt.port == 1883
    assert config.camera.width == 640
    assert config.model.num_classes == 9

    # Test creating directories
    config.create_directories()

    print("✓ Configuration tests passed")


def test_error_handler():
    """Test error handling utilities"""
    print("\nTesting Error Handler...")

    # Setup logging
    ErrorHandler.setup_logging(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test safe execute
    def risky_func():
        raise ValueError("Test error")

    result = ErrorHandler.safe_execute(
        risky_func,
        default_return="default",
        logger=logger
    )
    assert result == "default"

    # Test log exception
    try:
        raise Exception("Test exception")
    except Exception as e:
        ErrorHandler.log_exception(logger, e, "Test context")

    print("✓ Error handler tests passed")


def test_validators():
    """Test validation functions"""
    print("\nTesting Validators...")

    # Test image validation
    valid_image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert validate_image(valid_image) == True
    assert validate_image(None) == False
    assert validate_image([]) == False

    # Test JSON validation
    valid_data = {"username": "test", "model": "test.pth"}
    is_valid, error = validate_json_input(
        valid_data,
        required_fields=["username", "model"]
    )
    assert is_valid == True
    assert error is None

    invalid_data = {"username": "test"}
    is_valid, error = validate_json_input(
        invalid_data,
        required_fields=["username", "model"]
    )
    assert is_valid == False
    assert "model" in error

    # Test filename sanitization
    dangerous_name = "../../../etc/passwd"
    safe_name = sanitize_filename(dangerous_name)
    assert ".." not in safe_name
    assert "/" not in safe_name

    print("✓ Validator tests passed")


def test_performance_monitor():
    """Test performance monitoring"""
    print("\nTesting Performance Monitor...")

    monitor = PerformanceMonitor(window_size=5)

    # Record some metrics
    for i in range(10):
        monitor.record('test_metric', i * 0.1)

    # Should only keep last 5 values
    avg = monitor.get_average('test_metric')
    latest = monitor.get_latest('test_metric')

    assert latest == 0.9
    assert 0 < avg < 1

    # Clear metrics
    monitor.clear('test_metric')
    assert monitor.get_average('test_metric') == 0.0

    print("✓ Performance monitor tests passed")


def test_resource_manager():
    """Test resource management"""
    print("\nTesting Resource Manager...")

    # Test GPU check
    has_gpu = ResourceManager.check_gpu_available()
    device = ResourceManager.get_device()
    assert device in ['cuda', 'cpu']

    # Test camera release (with None - should handle gracefully)
    assert ResourceManager.release_camera(None) == True

    print("✓ Resource manager tests passed")


def test_health_monitor():
    """Test health monitoring"""
    print("\nTesting Health Monitor...")

    monitor = HealthMonitor()

    # Register services
    monitor.register_service('test_service_1')
    monitor.register_service('test_service_2')

    # Update statuses
    monitor.update_service_status('test_service_1', 'running')
    monitor.update_service_status('test_service_2', 'error', 'Test error')

    # Get status
    status = monitor.get_service_status('test_service_1')
    assert status['status'] == 'running'

    status = monitor.get_service_status('test_service_2')
    assert status['status'] == 'error'
    assert status['error_count'] == 1

    # Get health status
    health = monitor.get_health_status()
    assert 'status' in health
    assert 'metrics' in health
    assert 'services' in health

    # Get metrics
    metrics = monitor.get_current_metrics()
    assert 'cpu_percent' in metrics
    assert 'memory_percent' in metrics

    print("✓ Health monitor tests passed")


def test_retry_decorator():
    """Test retry decorator"""
    print("\nTesting Retry Decorator...")

    call_count = 0

    @retry_on_failure(max_retries=3, delay=0.1)
    def unstable_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Not yet")
        return "success"

    result = unstable_function()
    assert result == "success"
    assert call_count == 3

    print("✓ Retry decorator tests passed")


def test_thread_safe_counter():
    """Test thread-safe counter"""
    print("\nTesting Thread-Safe Counter...")

    counter = ThreadSafeCounter(initial=0)

    assert counter.get() == 0
    assert counter.increment() == 1
    assert counter.increment(5) == 6
    assert counter.decrement(2) == 4

    counter.reset()
    assert counter.get() == 0

    print("✓ Thread-safe counter tests passed")


def main():
    """Run all tests"""
    print("=" * 50)
    print("Running System Improvement Tests")
    print("=" * 50)

    try:
        test_configuration()
        test_error_handler()
        test_validators()
        test_performance_monitor()
        test_resource_manager()
        test_health_monitor()
        test_retry_decorator()
        test_thread_safe_counter()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
