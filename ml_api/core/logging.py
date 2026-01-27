"""Structured logging with request_id correlation and traceability."""

import sys
import uuid
import logging
from contextvars import ContextVar
from typing import Any

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ml_api.core.config import settings

# Context variable for request ID
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_ctx.get()


def set_request_id(request_id: str) -> None:
    """Set request ID in context."""
    request_id_ctx.set(request_id)


def add_request_id(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Add request_id to all log entries."""
    request_id = get_request_id()
    if request_id:
        event_dict["request_id"] = request_id
    return event_dict


def add_service_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Add service metadata to all log entries."""
    event_dict["service"] = settings.app_name
    event_dict["environment"] = settings.environment
    event_dict["version"] = settings.app_version
    return event_dict


def configure_logging() -> None:
    """Configure structured logging for the application."""
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Configure structlog
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_service_context,
        add_request_id,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_development:
        # Pretty console output for development
        processors = shared_processors + [structlog.dev.ConsoleRenderer()]
    else:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request_id to all requests and responses."""

    async def dispatch(self, request: Request, call_next):
        """Process request and add request_id."""
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        set_request_id(request_id)

        # Add to request state for easy access
        request.state.request_id = request_id

        # Log request start
        logger = get_logger(__name__)
        logger.info(
            "request_started",
            method=request.method,
            url=str(request.url),
            client_host=request.client.host if request.client else None,
        )

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Log request completion
        logger.info(
            "request_completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
        )

        return response


def log_function_call(logger: structlog.stdlib.BoundLogger, function_name: str, **kwargs) -> None:
    """Log function call with parameters (redacting sensitive data)."""
    safe_kwargs = {k: v for k, v in kwargs.items() if not _is_sensitive_field(k)}
    logger.info(f"{function_name}_started", **safe_kwargs)


def log_function_result(
    logger: structlog.stdlib.BoundLogger, function_name: str, duration_ms: float, **kwargs
) -> None:
    """Log function result with duration."""
    safe_kwargs = {k: v for k, v in kwargs.items() if not _is_sensitive_field(k)}
    logger.info(f"{function_name}_completed", duration_ms=duration_ms, **safe_kwargs)


def log_exception(
    logger: structlog.stdlib.BoundLogger, exc: Exception, context: str = "", **kwargs
) -> None:
    """Log exception with context and stack trace."""
    safe_kwargs = {k: v for k, v in kwargs.items() if not _is_sensitive_field(k)}
    logger.error(
        "exception_occurred",
        exc_type=type(exc).__name__,
        exc_message=str(exc),
        context=context,
        **safe_kwargs,
        exc_info=True,
    )


def _is_sensitive_field(field_name: str) -> bool:
    """Check if field name indicates sensitive data."""
    sensitive_patterns = [
        "password",
        "secret",
        "token",
        "key",
        "credential",
        "auth",
        "api_key",
        "private",
    ]
    field_lower = field_name.lower()
    return any(pattern in field_lower for pattern in sensitive_patterns)


def log_dataframe_info(
    logger: structlog.stdlib.BoundLogger, df_name: str, df: Any, context: str = ""
) -> None:
    """Log dataframe metadata without logging raw data."""
    import polars as pl

    if isinstance(df, pl.DataFrame):
        logger.info(
            "dataframe_info",
            df_name=df_name,
            context=context,
            rows=df.shape[0],
            cols=df.shape[1],
            columns=df.columns,
            dtypes={col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            memory_estimate_mb=round(df.estimated_size() / (1024 * 1024), 2),
        )
    else:
        # Fallback for pandas or other types
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            logger.info(
                "dataframe_info",
                df_name=df_name,
                context=context,
                rows=len(df),
                cols=len(df.columns),
                columns=list(df.columns),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                memory_estimate_mb=round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            )


def log_conversion(
    logger: structlog.stdlib.BoundLogger,
    from_type: str,
    to_type: str,
    reason: str,
    rows: int,
    cols: int,
    memory_before_mb: float,
    memory_after_mb: float,
) -> None:
    """Log explicit dataframe type conversions."""
    logger.warning(
        "dataframe_conversion",
        from_type=from_type,
        to_type=to_type,
        reason=reason,
        rows=rows,
        cols=cols,
        memory_before_mb=memory_before_mb,
        memory_after_mb=memory_after_mb,
    )
