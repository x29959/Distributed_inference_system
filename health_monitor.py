"""
Health monitoring and metrics collection for the Distributed Inference System.
Provides endpoints and utilities for monitoring system health.
"""
import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging


@dataclass
class SystemMetrics:
    """System metrics data"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    timestamp: float


@dataclass
class ServiceStatus:
    """Service status information"""
    name: str
    status: str  # 'running', 'stopped', 'error'
    uptime: float
    error_count: int
    last_error: Optional[str]


class HealthMonitor:
    """Monitor system health and collect metrics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.metrics_history: list[SystemMetrics] = []
        self.service_status: Dict[str, ServiceStatus] = {}
        self.error_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval: float = 5.0):
        """
        Start monitoring system metrics

        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring:
            self.logger.warning("Monitoring already started")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")

    def _monitor_loop(self, interval: float):
        """Monitoring loop"""
        while self.monitoring:
            try:
                self._collect_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=psutil.virtual_memory().percent,
                memory_available_mb=psutil.virtual_memory().available / (1024 * 1024),
                disk_usage_percent=psutil.disk_usage('/').percent,
                timestamp=time.time()
            )

            with self._lock:
                self.metrics_history.append(metrics)
                # Keep only last 100 metrics
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_mb': round(psutil.virtual_memory().available / (1024 * 1024), 2),
                'memory_used_mb': round(psutil.virtual_memory().used / (1024 * 1024), 2),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'uptime_seconds': round(time.time() - self.start_time, 2)
            }
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return {}

    def get_metrics_history(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Get recent metrics history

        Args:
            limit: Number of recent metrics to return

        Returns:
            List of metrics dictionaries
        """
        with self._lock:
            recent_metrics = self.metrics_history[-limit:]
            return [asdict(m) for m in recent_metrics]

    def register_service(self, service_name: str):
        """Register a service for monitoring"""
        with self._lock:
            self.service_status[service_name] = ServiceStatus(
                name=service_name,
                status='running',
                uptime=0.0,
                error_count=0,
                last_error=None
            )
            self.error_counts[service_name] = 0

    def update_service_status(self, service_name: str, status: str,
                            error: Optional[str] = None):
        """
        Update service status

        Args:
            service_name: Name of the service
            status: Service status ('running', 'stopped', 'error')
            error: Optional error message
        """
        with self._lock:
            if service_name not in self.service_status:
                self.register_service(service_name)

            service = self.service_status[service_name]
            service.status = status

            if error:
                service.error_count += 1
                service.last_error = error
                self.error_counts[service_name] = service.error_count

            service.uptime = time.time() - self.start_time

    def record_error(self, service_name: str, error: str):
        """Record an error for a service"""
        self.update_service_status(service_name, 'error', error)

    def get_service_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get service status

        Args:
            service_name: Optional service name, returns all if None

        Returns:
            Service status dictionary
        """
        with self._lock:
            if service_name:
                if service_name in self.service_status:
                    return asdict(self.service_status[service_name])
                return {}
            else:
                return {name: asdict(status)
                       for name, status in self.service_status.items()}

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status

        Returns:
            Health status dictionary
        """
        metrics = self.get_current_metrics()
        services = self.get_service_status()

        # Determine overall health
        health = 'healthy'
        issues = []

        # Check system resources
        if metrics.get('cpu_percent', 0) > 90:
            health = 'degraded'
            issues.append('High CPU usage')

        if metrics.get('memory_percent', 0) > 90:
            health = 'degraded'
            issues.append('High memory usage')

        if metrics.get('disk_usage_percent', 0) > 90:
            health = 'degraded'
            issues.append('High disk usage')

        # Check service statuses
        for service_name, service_data in services.items():
            if service_data.get('status') == 'error':
                health = 'unhealthy'
                issues.append(f"Service {service_name} has errors")
            elif service_data.get('status') == 'stopped':
                health = 'degraded'
                issues.append(f"Service {service_name} is stopped")

        return {
            'status': health,
            'timestamp': time.time(),
            'uptime': round(time.time() - self.start_time, 2),
            'metrics': metrics,
            'services': services,
            'issues': issues
        }

    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        status = self.get_health_status()
        return status['status'] == 'healthy'

    def get_error_summary(self) -> Dict[str, int]:
        """Get error count summary for all services"""
        with self._lock:
            return self.error_counts.copy()


# Global health monitor instance
health_monitor = HealthMonitor()
