"""Custom exceptions and error handlers."""

from typing import Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml_api.core.logging import get_logger, log_exception

logger = get_logger(__name__)


class MLAPIException(Exception):
    """Base exception for ML API errors."""

    def __init__(
        self,
        message: str,
        code: str = "internal_error",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ResourceNotFoundError(MLAPIException):
    """Resource not found error."""

    def __init__(self, resource_type: str, resource_id: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=f"{resource_type} not found: {resource_id}",
            code="resource_not_found",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details or {"resource_type": resource_type, "resource_id": resource_id},
        )


class ResourceConflictError(MLAPIException):
    """Resource conflict error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="resource_conflict",
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


class ValidationError(MLAPIException):
    """Business validation error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="validation_error",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class ExternalServiceError(MLAPIException):
    """External service error (GCS, Redis, etc.)."""

    def __init__(self, service: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=f"{service} error: {message}",
            code="external_service_error",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details=details or {"service": service},
        )


class JobError(MLAPIException):
    """Background job error."""

    def __init__(
        self, message: str, job_id: str | None = None, details: dict[str, Any] | None = None
    ):
        super().__init__(
            message=message,
            code="job_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details or {"job_id": job_id},
        )


class ModelTrainingError(MLAPIException):
    """Model training error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="model_training_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class DataProcessingError(MLAPIException):
    """Data processing error."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(
            message=message,
            code="data_processing_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


def create_error_response(
    request_id: str,
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create standardized error response."""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }


async def mlapi_exception_handler(request: Request, exc: MLAPIException) -> JSONResponse:
    """Handle MLAPIException errors."""
    request_id = getattr(request.state, "request_id", "unknown")

    log_exception(
        logger,
        exc,
        context="mlapi_exception",
        code=exc.code,
        status_code=exc.status_code,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            request_id=request_id,
            code=exc.code,
            message=exc.message,
            status_code=exc.status_code,
            details=exc.details,
        ),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        "validation_error",
        errors=exc.errors(),
        body=str(exc.body) if hasattr(exc, "body") else None,
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            request_id=request_id,
            code="validation_error",
            message="Request validation failed",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"errors": exc.errors()},
        ),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        detail=exc.detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            request_id=request_id,
            code="http_error",
            message=exc.detail,
            status_code=exc.status_code,
        ),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")

    log_exception(logger, exc, context="unexpected_exception")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            request_id=request_id,
            code="internal_error",
            message="An unexpected error occurred",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
    )
