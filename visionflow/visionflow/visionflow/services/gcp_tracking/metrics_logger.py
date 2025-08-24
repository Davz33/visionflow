"""
GCP Cloud Monitoring metrics logger
Provides Prometheus-compatible metrics logging to Cloud Monitoring
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict

from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import TimeSeries, Point, TimeInterval, Metric
from google.cloud.monitoring_v3.types import MetricKind, ValueType
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, start_http_server

from ...shared.config import get_settings
from ...shared.monitoring import get_logger

logger = get_logger("cloud_metrics")
settings = get_settings()


class CloudMetrics:
    """GCP Cloud Monitoring integration for real-time metrics"""
    
    def __init__(self):
        self.settings = get_settings()
        self.project_id = (
            self.settings.monitoring.vertex_ai_project 
            if hasattr(self.settings.monitoring, 'vertex_ai_project') 
            else "visionflow-gcp-project"
        )
        
        # Initialize monitoring client
        self._initialize_monitoring()
        
        # Initialize Prometheus metrics for local monitoring
        self._initialize_prometheus()
        
        # Metrics cache for batching
        self.metrics_buffer = defaultdict(list)
        self.last_flush = time.time()
        self.flush_interval = 30  # seconds
    
    def _initialize_monitoring(self):
        """Initialize Cloud Monitoring client"""
        try:
            self.monitoring_client = monitoring_v3.MetricServiceClient()
            self.project_name = f"projects/{self.project_id}"
            
            logger.info("Cloud Monitoring client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cloud Monitoring: {e}")
            self.monitoring_client = None
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics for local monitoring"""
        try:
            # Create custom registry
            self.prometheus_registry = CollectorRegistry()
            
            # Core VisionFlow metrics
            self.video_generation_counter = Counter(
                'visionflow_video_generations_total',
                'Total number of video generations',
                ['status', 'model', 'orchestration_type'],
                registry=self.prometheus_registry
            )
            
            self.video_generation_duration = Histogram(
                'visionflow_video_generation_duration_seconds',
                'Video generation duration in seconds',
                ['model', 'orchestration_type'],
                registry=self.prometheus_registry
            )
            
            self.orchestration_quality_score = Histogram(
                'visionflow_orchestration_quality_score',
                'Quality score from evaluator-optimizer loops',
                ['orchestration_type'],
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                registry=self.prometheus_registry
            )
            
            self.agent_tool_calls = Counter(
                'visionflow_agent_tool_calls_total',
                'Total agent tool calls',
                ['tool_name', 'status'],
                registry=self.prometheus_registry
            )
            
            self.workflow_iterations = Histogram(
                'visionflow_workflow_iterations',
                'Number of iterations in evaluator-optimizer workflows',
                ['workflow_type'],
                buckets=[1, 2, 3, 4, 5, 10],
                registry=self.prometheus_registry
            )
            
            # System metrics
            self.active_jobs = Gauge(
                'visionflow_active_jobs',
                'Number of currently active video generation jobs',
                registry=self.prometheus_registry
            )
            
            self.gcp_service_health = Gauge(
                'visionflow_gcp_service_health',
                'Health status of GCP services (1=healthy, 0=unhealthy)',
                ['service'],
                registry=self.prometheus_registry
            )
            
            logger.info("Prometheus metrics initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Prometheus metrics: {e}")
    
    def start_prometheus_server(self, port: int = 9091):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(port, registry=self.prometheus_registry)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def log_video_generation_metrics(
        self,
        duration: float,
        status: str,
        model: str = "wan2-1-fast",
        orchestration_type: str = "workflow",
        quality_score: Optional[float] = None,
        iterations: Optional[int] = None
    ):
        """Log video generation metrics"""
        
        # Prometheus metrics
        self.video_generation_counter.labels(
            status=status,
            model=model,
            orchestration_type=orchestration_type
        ).inc()
        
        if status == "completed":
            self.video_generation_duration.labels(
                model=model,
                orchestration_type=orchestration_type
            ).observe(duration)
            
            if quality_score is not None:
                self.orchestration_quality_score.labels(
                    orchestration_type=orchestration_type
                ).observe(quality_score)
            
            if iterations is not None:
                self.workflow_iterations.labels(
                    workflow_type=orchestration_type
                ).observe(iterations)
        
        # Cloud Monitoring metrics
        if self.monitoring_client:
            metrics_data = {
                "video_generation_duration": duration,
                "video_generation_success": 1 if status == "completed" else 0,
            }
            
            if quality_score is not None:
                metrics_data["quality_score"] = quality_score
            
            if iterations is not None:
                metrics_data["workflow_iterations"] = iterations
            
            labels = {
                "model": model,
                "orchestration_type": orchestration_type,
                "status": status
            }
            
            self._send_to_cloud_monitoring(metrics_data, labels)
    
    def log_agent_tool_call(self, tool_name: str, status: str = "success"):
        """Log agent tool call metrics"""
        
        # Prometheus
        self.agent_tool_calls.labels(
            tool_name=tool_name,
            status=status
        ).inc()
        
        # Cloud Monitoring
        if self.monitoring_client:
            self._send_to_cloud_monitoring(
                {"agent_tool_calls": 1},
                {"tool_name": tool_name, "status": status}
            )
    
    def update_active_jobs(self, count: int):
        """Update active jobs count"""
        self.active_jobs.set(count)
        
        if self.monitoring_client:
            self._send_to_cloud_monitoring(
                {"active_jobs": count},
                {"component": "orchestrator"}
            )
    
    def update_gcp_service_health(self, service: str, healthy: bool):
        """Update GCP service health status"""
        health_value = 1 if healthy else 0
        
        self.gcp_service_health.labels(service=service).set(health_value)
        
        if self.monitoring_client:
            self._send_to_cloud_monitoring(
                {f"{service}_health": health_value},
                {"service": service}
            )
    
    def _send_to_cloud_monitoring(
        self, 
        metrics: Dict[str, float], 
        labels: Dict[str, str] = None
    ):
        """Send metrics to Cloud Monitoring (with batching)"""
        if not self.monitoring_client:
            return
        
        try:
            # Add to buffer
            timestamp = time.time()
            for metric_name, value in metrics.items():
                self.metrics_buffer[metric_name].append({
                    "value": value,
                    "labels": labels or {},
                    "timestamp": timestamp
                })
            
            # Flush if interval exceeded
            if timestamp - self.last_flush > self.flush_interval:
                self._flush_metrics_buffer()
                
        except Exception as e:
            logger.error(f"Failed to buffer metrics: {e}")
    
    def _flush_metrics_buffer(self):
        """Flush buffered metrics to Cloud Monitoring"""
        if not self.monitoring_client or not self.metrics_buffer:
            return
        
        try:
            time_series_list = []
            
            for metric_name, data_points in self.metrics_buffer.items():
                for data_point in data_points:
                    # Create time series
                    series = TimeSeries()
                    series.metric.type = f"custom.googleapis.com/visionflow/{metric_name}"
                    series.resource.type = "global"
                    
                    # Add labels
                    for label_key, label_value in data_point["labels"].items():
                        series.metric.labels[label_key] = str(label_value)
                    
                    # Create point
                    point = Point()
                    point.value.double_value = float(data_point["value"])
                    point.interval.end_time.FromDatetime(
                        datetime.fromtimestamp(data_point["timestamp"])
                    )
                    series.points = [point]
                    
                    time_series_list.append(series)
            
            # Send batch to Cloud Monitoring
            if time_series_list:
                self.monitoring_client.create_time_series(
                    name=self.project_name,
                    time_series=time_series_list
                )
                
                logger.debug(f"Flushed {len(time_series_list)} metrics to Cloud Monitoring")
            
            # Clear buffer
            self.metrics_buffer.clear()
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics to Cloud Monitoring: {e}")
            # Clear buffer even on error to prevent accumulation
            self.metrics_buffer.clear()
            self.last_flush = time.time()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics"""
        try:
            from prometheus_client import generate_latest
            return generate_latest(self.prometheus_registry).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return "# Metrics unavailable\n"
    
    def log_custom_metric(
        self, 
        metric_name: str, 
        value: float, 
        labels: Dict[str, str] = None
    ):
        """Log custom metric to both Prometheus and Cloud Monitoring"""
        
        # For Cloud Monitoring
        if self.monitoring_client:
            self._send_to_cloud_monitoring(
                {metric_name: value},
                labels or {}
            )
        
        logger.debug(f"Logged custom metric: {metric_name}={value}")
    
    def force_flush(self):
        """Force flush all buffered metrics"""
        self._flush_metrics_buffer()
    
    def health_check(self) -> Dict[str, Any]:
        """Check metrics logging health"""
        health = {
            "cloud_monitoring_available": False,
            "prometheus_available": False,
            "metrics_buffer_size": len(self.metrics_buffer)
        }
        
        try:
            # Check Cloud Monitoring
            if self.monitoring_client:
                # Try to list metric descriptors (minimal call)
                list(self.monitoring_client.list_metric_descriptors(
                    name=self.project_name,
                    page_size=1
                ))
                health["cloud_monitoring_available"] = True
            
            # Check Prometheus
            if hasattr(self, 'prometheus_registry'):
                health["prometheus_available"] = True
            
        except Exception as e:
            logger.error(f"Metrics health check failed: {e}")
        
        return health


# Global instance
_cloud_metrics_instance = None

def get_metrics_logger() -> CloudMetrics:
    """Get or create metrics logger instance"""
    global _cloud_metrics_instance
    if _cloud_metrics_instance is None:
        _cloud_metrics_instance = CloudMetrics()
    return _cloud_metrics_instance
