# Quick Reference - System Improvements

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

### 2. Configure System
```bash
# Copy example config
cp .env.example .env

# Edit with your settings
nano .env
```

### 3. Use in Your Code
```python
from config import config
from utils import ErrorHandler
from health_monitor import health_monitor

# Setup logging
ErrorHandler.setup_logging()

# Start monitoring
health_monitor.start_monitoring()

# Use config instead of hardcoded values
mqtt_client = temi.connect(
    config.mqtt.host,
    config.mqtt.port,
    config.mqtt.username,
    config.mqtt.password
)
```

## 📋 Common Tasks

### Configuration
```python
# Load config
from config import config

# Access settings
host = config.mqtt.host
port = config.server.port
db_name = config.database.name

# Create directories
config.create_directories()
```

### Error Handling
```python
from utils import ErrorHandler, retry_on_failure

# Safe execution
result = ErrorHandler.safe_execute(
    lambda: risky_function(),
    default_return=None,
    logger=logger
)

# Auto-retry on failure
@retry_on_failure(max_retries=3, delay=1.0)
def unstable_operation():
    # Your code here
    pass
```

### Input Validation
```python
from utils import validate_image, validate_json_input, sanitize_filename

# Validate image
if validate_image(frame):
    process(frame)

# Validate JSON
is_valid, error = validate_json_input(
    data,
    required_fields=['username', 'model']
)

# Sanitize filename
safe_name = sanitize_filename(user_input)
```

### Health Monitoring
```python
from health_monitor import health_monitor

# Start monitoring
health_monitor.start_monitoring(interval=5.0)

# Register service
health_monitor.register_service('yolo_processing')

# Update status
health_monitor.update_service_status('yolo_processing', 'running')

# Record error
health_monitor.record_error('service_name', 'error message')

# Check health
if health_monitor.is_healthy():
    print("System OK")

# Get metrics
metrics = health_monitor.get_current_metrics()
print(f"CPU: {metrics['cpu_percent']}%")
print(f"Memory: {metrics['memory_percent']}%")
```

### Performance Monitoring
```python
from utils import PerformanceMonitor
import time

monitor = PerformanceMonitor()

# Record metric
start = time.time()
# ... do work ...
monitor.record('processing_time', time.time() - start)

# Get statistics
avg = monitor.get_average('processing_time')
latest = monitor.get_latest('processing_time')
```

### Resource Management
```python
from utils import ResourceManager

# Check GPU
if ResourceManager.check_gpu_available():
    device = 'cuda'
else:
    device = 'cpu'

# Free GPU memory
ResourceManager.free_gpu_memory()

# Release resources safely
ResourceManager.release_camera(cap)
ResourceManager.release_video_writer(writer)
```

## 🔧 Flask Integration

### Add Health Endpoints
```python
from flask import Flask, jsonify
from health_monitor import health_monitor

app = Flask(__name__)

@app.route('/health')
def health():
    health = health_monitor.get_health_status()
    status = 200 if health['status'] == 'healthy' else 503
    return jsonify(health), status

@app.route('/metrics')
def metrics():
    return jsonify(health_monitor.get_current_metrics())

@app.route('/services')
def services():
    return jsonify(health_monitor.get_service_status())
```

## ⚡ Best Practices

### DO ✅
```python
# Use configuration
from config import config
cap = cv2.VideoCapture(config.camera.yolo_source)

# Validate inputs
from utils import validate_image
if validate_image(frame):
    process(frame)

# Handle errors gracefully
try:
    result = process()
except Exception as e:
    ErrorHandler.log_exception(logger, e, "Processing")

# Use retry for unstable operations
@retry_on_failure(max_retries=3)
def connect():
    return mqtt.connect()
```

### DON'T ❌
```python
# Don't hardcode values
cap = cv2.VideoCapture('rtmp://100.70.26.103:1935/live/stream')

# Don't skip validation
process(frame)  # What if frame is None?

# Don't ignore errors
try:
    result = process()
except:
    pass  # Silent failure!

# Don't ignore unstable operations
mqtt.connect()  # What if connection fails?
```

## 📊 Monitoring

### Check System Health
```bash
# HTTP endpoint
curl http://localhost:5000/health

# Python
health = health_monitor.get_health_status()
print(health['status'])  # 'healthy', 'degraded', or 'unhealthy'
```

### View Metrics
```bash
# HTTP endpoint
curl http://localhost:5000/metrics

# Python
metrics = health_monitor.get_current_metrics()
print(f"CPU: {metrics['cpu_percent']}%")
print(f"Memory: {metrics['memory_percent']}%")
```

## 🐛 Debugging

### Enable Debug Logging
```python
from utils import ErrorHandler
import logging

ErrorHandler.setup_logging(
    log_file='debug.log',
    level=logging.DEBUG
)
```

### View Logs
```bash
# Tail log file
tail -f app.log

# Search for errors
grep ERROR app.log

# View last 100 lines
tail -n 100 app.log
```

## 🔐 Security

### Environment Variables
```bash
# .env file
MQTT_PASSWORD=secret123
DB_PASSWORD=dbsecret456
SECRET_KEY=randomkey789

# Never commit .env!
# It's in .gitignore
```

### Sanitize User Input
```python
from utils import sanitize_filename, validate_json_input

# Prevent path traversal
safe = sanitize_filename(user_filename)

# Validate JSON
valid, error = validate_json_input(data, ['username', 'model'])
if not valid:
    return error, 400
```

## 📦 Deployment

### Local Development
```bash
python inference.py
```

### Production (systemd)
```bash
sudo systemctl start inference
sudo systemctl status inference
```

### Docker
```bash
docker-compose up -d
docker-compose logs -f
```

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error | `pip install -r requirements.txt` |
| Config not found | Copy `.env.example` to `.env` |
| Camera fails | Check RTMP URL in config |
| Database error | Verify PostgreSQL is running |
| High memory | Call `ResourceManager.free_gpu_memory()` |

## 📚 Learn More

- **[SUMMARY.md](SUMMARY.md)** - Overview of all improvements
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Detailed guide with examples
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment instructions
- **[test_improvements.py](test_improvements.py)** - Usage examples

## 🎯 Key Takeaways

1. **Use `config`** instead of hardcoded values
2. **Validate inputs** before processing
3. **Handle errors** with ErrorHandler
4. **Monitor health** with health_monitor
5. **Track performance** with PerformanceMonitor
6. **Manage resources** with ResourceManager
7. **Read the docs** for detailed information

---

**Quick Test:**
```bash
python test_improvements.py
```

**Status Check:**
```bash
curl http://localhost:5000/health
```

**Ready to deploy!** 🚀
