"""Telemetry and metrics collection."""

from typing import Callable
from functools import wraps
import time

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ml_api.core.config import settings

# Create registry
registry = CollectorRegistry()

# HTTP Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    registry=registry,
)

# Job Metrics
jobs_started_total = Counter(
    "jobs_started_total",
    "Total jobs started",
    ["job_type"],
    registry=registry,
)

jobs_completed_total = Counter(
    "jobs_completed_total",
    "Total jobs completed",
    ["job_type", "status"],
    registry=registry,
)

jobs_duration_seconds = Histogram(
    "jobs_duration_seconds",
    "Job duration in seconds",
    ["job_type"],
    registry=registry,
)

jobs_active = Gauge(
    "jobs_active",
    "Number of active jobs",
    ["job_type"],
    registry=registry,
)

# Model Metrics
predictions_total = Counter(
    "predictions_total",
    "Total predictions made",
    ["model_type", "task_type"],
    registry=registry,
)

prediction_duration_seconds = Histogram(
    "prediction_duration_seconds",
    "Prediction duration in seconds",
    ["model_type"],
    registry=registry,
)

# Training Metrics
training_trials_total = Counter(
    "training_trials_total",
    "Total training trials",
    ["model_type", "task_type"],
    registry=registry,
)

training_duration_seconds = Histogram(
    "training_duration_seconds",
    "Training duration in seconds",
    ["model_type"],
    registry=registry,
)

# Data Metrics
data_splits_created_total = Counter(
    "data_splits_created_total",
    "Total data splits created",
    ["strategy"],
    registry=registry,
)

data_rows_processed = Histogram(
    "data_rows_processed",
    "Number of rows processed",
    ["operation"],
    registry=registry,
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics."""

    async def dispatch(self, request: Request, call_next):
        """Collect metrics for HTTP requests."""
        if not settings.enable_metrics:
            return await call_next(request)

        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time

        # Record metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        return response


def track_prediction(model_type: str, task_type: str):
    """Decorator to track prediction metrics."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not settings.enable_metrics:
                return await func(*args, **kwargs)

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                predictions_total.labels(
                    model_type=model_type,
                    task_type=task_type,
                ).inc()
                return result
            finally:
                duration = time.time() - start_time
                prediction_duration_seconds.labels(model_type=model_type).observe(duration)

        return wrapper

    return decorator


def track_training_trial(model_type: str, task_type: str):
    """Context manager to track training trial metrics."""

    class TrainingTrialTracker:
        def __enter__(self):
            if settings.enable_metrics:
                self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if settings.enable_metrics:
                duration = time.time() - self.start_time
                training_trials_total.labels(
                    model_type=model_type,
                    task_type=task_type,
                ).inc()
                training_duration_seconds.labels(model_type=model_type).observe(duration)

    return TrainingTrialTracker()


def get_metrics() -> bytes:
    """Get current metrics in Prometheus format."""
    return generate_latest(registry)
