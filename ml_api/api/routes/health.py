"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ml_api.core.config import settings
from ml_api.core.logging import get_logger
from ml_api.db.session import get_db
from ml_api.clients import get_gcs_client

logger = get_logger(__name__)
router = APIRouter()


@router.get("/healthz")
async def healthz():
    """Basic health check."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/readyz")
async def readyz(db: AsyncSession = Depends(get_db)):
    """Readiness check with dependencies."""
    checks = {
        "database": False,
        "gcs": False,
    }

    # Check database
    try:
        await db.execute("SELECT 1")
        checks["database"] = True
    except Exception as e:
        logger.error("readyz_db_check_failed", error=str(e))

    # Check GCS
    try:
        gcs_client = get_gcs_client()
        checks["gcs"] = gcs_client.verify_connectivity()
    except Exception as e:
        logger.error("readyz_gcs_check_failed", error=str(e))

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }


@router.get("/version")
async def version():
    """Version information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }
