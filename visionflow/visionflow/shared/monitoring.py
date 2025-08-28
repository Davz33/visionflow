"""Monitoring and observability utilities."""

import asyncio
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

import structlog
from prometheus_client import Counter, Histogram, Info, start_http_server, Gauge
from prometheus_client.core import CollectorRegistry

from .config import get_settings


# Prometheus metrics - use default registry for simplicity
# REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    "visionflow_requests_total",
    "Total number of requests",
    ["service", "method", "status"]
)

REQUEST_DURATION = Histogram(
    "visionflow_request_duration_seconds",
    "Request duration in seconds",
    ["service", "method"]
)

# Video generation metrics
VIDEO_GENERATION_COUNT = Counter(
    "visionflow_videos_generated_total",
    "Total number of videos generated",
    ["quality", "status"]
)

VIDEO_GENERATION_DURATION = Histogram(
    "visionflow_video_generation_duration_seconds",
    "Video generation duration in seconds",
    ["quality"]
)

# Enhanced job tracking metrics for monitoring dashboard
VIDEO_GENERATION_JOBS_ACTIVE = Gauge(
    "visionflow_video_generation_jobs_active",
    "Number of currently active video generation jobs",
    ["status", "quality"]
)

VIDEO_GENERATION_QUEUE_LENGTH = Gauge(
    "visionflow_video_generation_queue_length",
    "Number of jobs waiting in queue"
)

VIDEO_GENERATION_JOB_PROGRESS = Gauge(
    "visionflow_video_generation_job_progress",
    "Progress percentage of video generation jobs",
    ["job_id", "status"]
)

VIDEO_GENERATION_AVERAGE_WAIT_TIME = Histogram(
    "visionflow_video_generation_wait_time_seconds",
    "Time jobs spend waiting in queue",
    ["quality"]
)

VIDEO_GENERATION_SUCCESS_RATE = Gauge(
    "visionflow_video_generation_success_rate",
    "Success rate of video generation jobs (0-1)",
    ["quality"]
)

# Error metrics
ERROR_COUNT = Counter(
    "visionflow_errors_total",
    "Total number of errors",
    ["service", "error_type"]
)

# Cache metrics
CACHE_HITS = Counter(
    "visionflow_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"]
)

CACHE_MISSES = Counter(
    "visionflow_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"]
)

# Service info
SERVICE_INFO = Info(
    "visionflow_service",
    "Service information"
)


def setup_logging() -> None:
    """Setup structured logging."""
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.monitoring.log_format == "json"
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.api.log_level.upper()),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger."""
    return structlog.get_logger(name)


def start_metrics_server() -> None:
    """Start Prometheus metrics server."""
    settings = get_settings()
    if settings.monitoring.enable_metrics:
        start_http_server(settings.monitoring.prometheus_port)


def track_request_metrics(service: str) -> Callable:
    """Decorator to track request metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            method = func.__name__
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                ERROR_COUNT.labels(service=service, error_type=type(e).__name__).inc()
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(service=service, method=method, status=status).inc()
                REQUEST_DURATION.labels(service=service, method=method).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            method = func.__name__
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                ERROR_COUNT.labels(service=service, error_type=type(e).__name__).inc()
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_COUNT.labels(service=service, method=method, status=status).inc()
                REQUEST_DURATION.labels(service=service, method=method).observe(duration)
        
        # Return appropriate wrapper based on function type
        if hasattr(func, "__call__"):
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
        return sync_wrapper
    
    return decorator


def track_video_generation(quality: str) -> Callable:
    """Decorator to track video generation metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                # Update active jobs count
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=quality).inc()
                
                result = await func(*args, **kwargs)
                
                # Record success
                duration = time.time() - start_time
                VIDEO_GENERATION_COUNT.labels(quality=quality, status="success").inc()
                VIDEO_GENERATION_DURATION.labels(quality=quality).observe(duration)
                
                # Update active jobs count
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=quality).dec()
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="completed", quality=quality).inc()
                
                return result
                
            except Exception as e:
                # Record failure
                duration = time.time() - start_time
                VIDEO_GENERATION_COUNT.labels(quality=quality, status="failed").inc()
                VIDEO_GENERATION_DURATION.labels(quality=quality).observe(duration)
                
                # Update active jobs count
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=quality).dec()
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="failed", quality=quality).inc()
                
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                # Update active jobs count
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=quality).inc()
                
                result = func(*args, **kwargs)
                
                # Record success
                duration = time.time() - start_time
                VIDEO_GENERATION_COUNT.labels(quality=quality, status="success").inc()
                VIDEO_GENERATION_DURATION.labels(quality=quality).observe(duration)
                
                # Update active jobs count
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=quality).dec()
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="completed", quality=quality).inc()
                
                return result
                
            except Exception as e:
                # Record failure
                duration = time.time() - start_time
                VIDEO_GENERATION_COUNT.labels(quality=quality, status="failed").inc()
                VIDEO_GENERATION_DURATION.labels(quality=quality).observe(duration)
                
                # Update active jobs count
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="processing", quality=quality).dec()
                VIDEO_GENERATION_JOBS_ACTIVE.labels(status="failed", quality=quality).inc()
                
                raise
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Helper functions for real-time job tracking
def update_job_queue_length(length: int) -> None:
    """Update the current queue length metric."""
    VIDEO_GENERATION_QUEUE_LENGTH.set(length)

def update_job_progress(job_id: str, status: str, progress: float) -> None:
    """Update job progress metric."""
    VIDEO_GENERATION_JOB_PROGRESS.labels(job_id=job_id, status=status).set(progress)

def update_job_wait_time(quality: str, wait_time: float) -> None:
    """Update job wait time metric."""
    VIDEO_GENERATION_AVERAGE_WAIT_TIME.labels(quality=quality).observe(wait_time)

def update_success_rate(quality: str, success_rate: float) -> None:
    """Update success rate metric."""
    VIDEO_GENERATION_SUCCESS_RATE.labels(quality=quality).set(success_rate)

def reset_job_metrics() -> None:
    """Reset job-related metrics (useful for testing)."""
    VIDEO_GENERATION_JOBS_ACTIVE.clear()
    VIDEO_GENERATION_QUEUE_LENGTH.set(0)
    VIDEO_GENERATION_JOB_PROGRESS.clear()


@contextmanager
def track_cache_usage(cache_type: str):
    """Context manager to track cache usage."""
    hit = False
    try:
        yield lambda: setattr(track_cache_usage, 'hit', True)
    finally:
        if getattr(track_cache_usage, 'hit', False):
            CACHE_HITS.labels(cache_type=cache_type).inc()
            setattr(track_cache_usage, 'hit', False)
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()


class PerformanceTracker:
    """Performance tracking utility."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.logger = get_logger(f"performance.{operation_name}")
    
    def start(self) -> None:
        """Start tracking."""
        self.start_time = time.time()
        self.logger.info("Operation started", operation=self.operation_name)
    
    def end(self, additional_data: Optional[Dict[str, Any]] = None) -> float:
        """End tracking and return duration."""
        if self.start_time is None:
            raise RuntimeError("Must call start() before end()")
        
        duration = time.time() - self.start_time
        log_data = {
            "operation": self.operation_name,
            "duration": duration,
        }
        
        if additional_data:
            log_data.update(additional_data)
        
        self.logger.info("Operation completed", **log_data)
        return duration
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end({"success": exc_type is None})


def set_service_info(name: str, version: str, description: str) -> None:
    """Set service information metrics."""
    SERVICE_INFO.info({
        'name': name,
        'version': version,
        'description': description
    })
