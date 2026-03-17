# Deployment Guide - Distributed Inference System

## Quick Start

### 1. Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional but recommended)
- PostgreSQL database
- MQTT broker (for TEMI robot communication)
- RTMP streaming server

### 2. Installation

#### Option A: Using Conda (Recommended)

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate combine
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

#### Option A: Using Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

#### Option B: Using Configuration File

```bash
# Copy example config
cp config.example.json config.json

# Edit config.json with your settings
nano config.json
```

### 4. Running the System

#### Edge Server (with YOLO and Gesture Recognition)

```bash
# Run the main inference server
python inference.py
```

The server will start on `http://0.0.0.0:5000`

#### Cloud Retraining Server

```bash
# Navigate to cloud_retrain directory
cd cloud_retrain

# Run the cloud server
python cloud.py
```

The cloud server will start on `http://0.0.0.0:5000`

### 5. RTMP Streaming Server

#### For Android

```bash
cd Rtmp_server/Android_rtmp
npm install
node app.js
```

#### For Raspberry Pi

```bash
cd Rtmp_server/RaspberryPi_rtmp
python app3.py
```

## System Components

### Edge Servers

1. **Remote Edge Server (RTX3060)**
   - Runs YOLO object detection
   - Gesture recognition
   - Real-time tracking

2. **Local Edge Server (Raspberry Pi 5)**
   - Local processing
   - Reduced latency for robot control

### Cloud Server

- Model retraining
- Data aggregation
- Centralized model storage

### RTMP Servers

- Drone camera feed
- Raspberry Pi gun mechanism feed

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MQTT_HOST` | MQTT broker hostname | - |
| `MQTT_PORT` | MQTT broker port | 1883 |
| `MQTT_USERNAME` | MQTT username | - |
| `MQTT_PASSWORD` | MQTT password | - |
| `TEMI_SERIAL` | TEMI robot serial number | - |
| `YOLO_CAMERA_SOURCE` | YOLO camera RTMP URL | - |
| `GESTURE_CAMERA_SOURCE` | Gesture camera RTMP URL | - |
| `DB_HOST` | PostgreSQL host | localhost |
| `DB_NAME` | Database name | gesture |
| `DB_USER` | Database user | - |
| `DB_PASSWORD` | Database password | - |
| `SERVER_HOST` | Flask server host | 0.0.0.0 |
| `SERVER_PORT` | Flask server port | 5000 |
| `SECRET_KEY` | Flask secret key | - |

## Database Setup

### PostgreSQL

```sql
-- Create database
CREATE DATABASE gesture;

-- Create user (if needed)
CREATE USER t1204 WITH PASSWORD 'your_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE gesture TO t1204;
```

The table will be created automatically on first run.

## Health Monitoring

### Health Check Endpoints

Add these to your Flask app for monitoring:

```python
from health_monitor import health_monitor

@app.route('/health')
def health_check():
    health = health_monitor.get_health_status()
    return jsonify(health), 200 if health['status'] == 'healthy' else 503

@app.route('/metrics')
def metrics():
    return jsonify(health_monitor.get_current_metrics())
```

### Monitoring Metrics

- CPU usage
- Memory usage
- Disk usage
- Service status
- Error counts
- Uptime

## Production Deployment

### Using systemd (Linux)

Create `/etc/systemd/system/inference.service`:

```ini
[Unit]
Description=Distributed Inference System
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/Distributed_inference_system
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python inference.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable inference
sudo systemctl start inference
sudo systemctl status inference
```

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python3", "inference.py"]
```

Build and run:

```bash
docker build -t distributed-inference .
docker run -p 5000:5000 --gpus all distributed-inference
```

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  inference:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MQTT_HOST=${MQTT_HOST}
      - DB_HOST=postgres
    depends_on:
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=gesture
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Run:

```bash
docker-compose up -d
```

## Troubleshooting

### Common Issues

#### 1. Camera Connection Fails

```
Error: Failed to open camera stream
```

**Solution:**
- Verify RTMP server is running
- Check camera source URLs in config
- Test stream with VLC or ffplay

#### 2. MQTT Connection Fails

```
Error: Failed to connect to TEMI robot
```

**Solution:**
- Verify MQTT broker is accessible
- Check credentials in config
- Test with mosquitto_sub

#### 3. Database Connection Fails

```
Error: Could not connect to database
```

**Solution:**
- Verify PostgreSQL is running
- Check database credentials
- Ensure database exists

#### 4. GPU Not Detected

```
Warning: CUDA not available, using CPU
```

**Solution:**
- Install CUDA toolkit
- Verify nvidia-smi shows GPU
- Check PyTorch CUDA compatibility

#### 5. High Memory Usage

**Solution:**
- Reduce batch sizes
- Lower frame queue sizes
- Enable GPU memory cleanup:
  ```python
  from utils import ResourceManager
  ResourceManager.free_gpu_memory()
  ```

### Logging

View logs:

```bash
# If running directly
tail -f app.log

# If using systemd
journalctl -u inference -f

# If using Docker
docker logs -f inference
```

### Performance Tuning

1. **Reduce latency:**
   - Lower frame resolution
   - Reduce queue sizes
   - Use GPU if available

2. **Increase throughput:**
   - Increase worker threads
   - Optimize batch sizes
   - Enable frame skipping

3. **Save resources:**
   - Reduce logging verbosity
   - Lower monitoring frequency
   - Disable unused features

## Maintenance

### Regular Tasks

1. **Clean up old data:**
   ```bash
   # Remove old videos
   find ./video -name "*.avi" -mtime +7 -delete

   # Clean old CSVs
   find . -name "*.csv" -mtime +30 -delete
   ```

2. **Backup database:**
   ```bash
   pg_dump gesture > backup_$(date +%Y%m%d).sql
   ```

3. **Update models:**
   ```bash
   # Download new model
   # Update config to point to new model
   # Restart service
   ```

4. **Monitor health:**
   ```bash
   # Check health endpoint
   curl http://localhost:5000/health

   # Check metrics
   curl http://localhost:5000/metrics
   ```

### Updating

```bash
# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
sudo systemctl restart inference
```

## Security Best Practices

1. **Use environment variables** for sensitive data
2. **Never commit** `.env` or `config.json` with real credentials
3. **Use strong passwords** for database and MQTT
4. **Enable HTTPS** in production
5. **Regularly update** dependencies
6. **Limit network exposure** using firewalls
7. **Monitor access logs** for suspicious activity

## Support

For issues and questions:
- Check the IMPROVEMENTS.md documentation
- Review logs for error messages
- Test individual components
- Verify configuration settings

## Scaling

### Horizontal Scaling

Deploy multiple edge servers:

```
Load Balancer
    ├── Edge Server 1 (Camera Feed 1)
    ├── Edge Server 2 (Camera Feed 2)
    └── Edge Server 3 (Camera Feed 3)
            ↓
    Central Cloud Server
            ↓
    Shared Database
```

### Vertical Scaling

- Upgrade GPU (RTX 3060 → RTX 4090)
- Increase RAM
- Use faster storage (SSD)
- Optimize model size

## Next Steps

After deployment:

1. Configure monitoring and alerts
2. Set up automated backups
3. Implement log rotation
4. Create deployment pipelines
5. Document custom configurations
6. Train team on operations
7. Plan disaster recovery
