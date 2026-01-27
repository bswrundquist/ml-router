"""Main FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml_api.core.config import settings
from ml_api.core.logging import configure_logging, get_logger, RequestIDMiddleware
from ml_api.core.exceptions import (
    MLAPIException,
    mlapi_exception_handler,
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
)
from ml_api.core.telemetry import MetricsMiddleware, get_metrics
from ml_api.db.session import init_db, close_db

# Import routes
from ml_api.api.routes import splits, health

# TODO: Create these route modules
# from ml_api.api.routes import experiments, predict, importance, models

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("application_starting", environment=settings.environment)
    configure_logging()

    if not settings.is_development:
        await init_db()

    logger.info("application_started")

    yield

    # Shutdown
    logger.info("application_shutting_down")
    await close_db()
    logger.info("application_stopped")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade ML API with Polars, FastAPI, and GCS",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RequestIDMiddleware)

if settings.enable_metrics:
    app.add_middleware(MetricsMiddleware)

if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add exception handlers
app.add_exception_handler(MLAPIException, mlapi_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(splits.router, prefix="/v1", tags=["Data Splits"])
# TODO: Create these route modules and uncomment
# app.include_router(experiments.router, prefix="/v1", tags=["Experiments"])
# app.include_router(models.router, prefix="/v1", tags=["Models"])
# app.include_router(predict.router, prefix="/v1", tags=["Predictions"])
# app.include_router(importance.router, prefix="/v1", tags=["Feature Importance"])


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not settings.enable_metrics:
        return {"error": "Metrics disabled"}
    return get_metrics()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
