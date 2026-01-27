"""Data split API routes."""

from uuid import UUID
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ml_api.core.logging import get_logger
from ml_api.core.exceptions import ResourceNotFoundError
from ml_api.db.session import get_db
from ml_api.db.models.split import DataSplit, SplitStatus
from ml_api.schemas.split import (
    DataSplitCreate,
    DataSplitUpdate,
    DataSplitResponse,
    DataSplitListResponse,
)
from ml_api.services.split_service import SplitService
from ml_api.clients import get_gcs_client

logger = get_logger(__name__)
router = APIRouter()


def get_split_service(
    db: AsyncSession = Depends(get_db),
) -> SplitService:
    """Get split service dependency."""
    from ml_api.services.split_service import SplitService

    return SplitService(db, get_gcs_client())


@router.post("/splits", response_model=DataSplitResponse, status_code=201)
async def create_split(
    request: DataSplitCreate,
    service: SplitService = Depends(get_split_service),
):
    """
    Create a new data split.

    Accepts either inline data or dataset URI, splits into train/val/test,
    stores artifacts in GCS, and persists metadata to database.
    """
    logger.info(
        "create_split_request",
        entity_id=request.entity_id,
        split_strategy=request.split_strategy,
    )

    split = await service.create_split(request)

    logger.info("create_split_completed", split_id=str(split.id))
    return split


@router.get("/splits/{split_id}", response_model=DataSplitResponse)
async def get_split(
    split_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get data split by ID."""
    logger.info("get_split_request", split_id=str(split_id))

    result = await db.execute(select(DataSplit).where(DataSplit.id == split_id))
    split = result.scalar_one_or_none()

    if not split:
        raise ResourceNotFoundError("DataSplit", str(split_id))

    return split


@router.get("/splits", response_model=DataSplitListResponse)
async def list_splits(
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    status: Optional[SplitStatus] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db),
):
    """List data splits with filtering and pagination."""
    logger.info(
        "list_splits_request",
        entity_id=entity_id,
        status=status,
        page=page,
        page_size=page_size,
    )

    # Build query
    query = select(DataSplit)

    if entity_id:
        query = query.where(DataSplit.entity_id == entity_id)
    if status:
        query = query.where(DataSplit.status == status)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Paginate
    query = query.order_by(DataSplit.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    # Execute
    result = await db.execute(query)
    splits = result.scalars().all()

    logger.info("list_splits_completed", count=len(splits), total=total)

    return DataSplitListResponse(
        items=splits,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.patch("/splits/{split_id}", response_model=DataSplitResponse)
async def update_split(
    split_id: UUID,
    request: DataSplitUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update data split metadata (safe operations only)."""
    logger.info("update_split_request", split_id=str(split_id))

    result = await db.execute(select(DataSplit).where(DataSplit.id == split_id))
    split = result.scalar_one_or_none()

    if not split:
        raise ResourceNotFoundError("DataSplit", str(split_id))

    # Only allow updating split_params for now
    if request.split_params is not None:
        split.split_params = request.split_params

    await db.commit()
    await db.refresh(split)

    logger.info("update_split_completed", split_id=str(split_id))
    return split


@router.delete("/splits/{split_id}", status_code=204)
async def delete_split(
    split_id: UUID,
    delete_artifacts: bool = Query(False, description="Also delete GCS artifacts"),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete data split (soft delete by default).

    Set delete_artifacts=true to also remove GCS artifacts.
    """
    logger.info(
        "delete_split_request",
        split_id=str(split_id),
        delete_artifacts=delete_artifacts,
    )

    result = await db.execute(select(DataSplit).where(DataSplit.id == split_id))
    split = result.scalar_one_or_none()

    if not split:
        raise ResourceNotFoundError("DataSplit", str(split_id))

    # Soft delete - mark as failed/deleted
    split.status = SplitStatus.FAILED

    if delete_artifacts:
        # Delete GCS artifacts (implement in service)
        gcs_client = get_gcs_client()
        for uri in [split.train_uri, split.val_uri, split.test_uri]:
            if uri and uri.startswith("gs://"):
                blob_path = uri.replace(f"gs://{gcs_client.bucket_name}/", "")
                try:
                    gcs_client.delete(blob_path)
                except Exception as e:
                    logger.warning("failed_to_delete_artifact", uri=uri, error=str(e))

    await db.commit()

    logger.info("delete_split_completed", split_id=str(split_id))
