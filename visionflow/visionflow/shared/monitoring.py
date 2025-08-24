"""Monitoring and observability utilities."""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional

import structlog
from prometheus_client import Counter, Histogram, Info, start_http_server
from prometheus_client.core import CollectorRegistry

from .config import get_settings


# Prometheus metrics
REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    "visionflow_requests_total",
    "Total number of requests",
    ["service", "method", "status"],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    "visionflow_request_duration_seconds",
    "Request duration in seconds",
    ["service", "method"],
    registry=REGISTRY
)

# Video generation metrics
VIDEO_GENERATION_COUNT = Counter(
    "visionflow_videos_generated_total",
    "Total number of videos generated",
    ["quality", "status"],
    registry=REGISTRY
)

VIDEO_GENERATION_DURATION = Histogram(
    "visionflow_video_generation_duration_seconds",
    "Video generation duration in seconds",
    ["quality"],
    registry=REGISTRY
)

# Error metrics
ERROR_COUNT = Counter(
    "visionflow_errors_total",
    "Total number of errors",
    ["service", "error_type"],
    registry=REGISTRY
)

# Cache metrics
CACHE_HITS = Counter(
    "visionflow_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
    registry=REGISTRY
)

CACHE_MISSES = Counter(
    "visionflow_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
    registry=REGISTRY
)

# Service info
SERVICE_INFO = Info(
    "visionflow_service",
    "Service information",
    registry=REGISTRY
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
        start_http_server(settings.monitoring.prometheus_port, registry=REGISTRY)


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
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                VIDEO_GENERATION_COUNT.labels(quality=quality, status=status).inc()
                if status == "success":
                    VIDEO_GENERATION_DURATION.labels(quality=quality).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                VIDEO_GENERATION_COUNT.labels(quality=quality, status=status).inc()
                if status == "success":
                    VIDEO_GENERATION_DURATION.labels(quality=quality).observe(duration)
        
        # Return appropriate wrapper based on function type
        if hasattr(func, "__call__"):
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
        return sync_wrapper
    
    return decorator


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
