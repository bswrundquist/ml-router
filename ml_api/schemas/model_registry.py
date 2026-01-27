"""Model registry schemas."""

from typing import Optional
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ml_api.db.models.model_registry import ModelStage


class ModelRegistryResponse(BaseModel):
    """Schema for model registry response."""

    id: UUID
    experiment_id: UUID
    version: int
    stage: ModelStage
    artifact_uri: str
    signature: dict
    preprocess_config: Optional[dict]
    postprocess_config: Optional[dict]
    metrics: Optional[dict]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class ModelRegistryListResponse(BaseModel):
    """Schema for model registry list response."""

    items: list[ModelRegistryResponse]
    total: int
    page: int
    page_size: int

    model_config = ConfigDict(from_attributes=True)


class ModelPromoteRequest(BaseModel):
    """Schema for promoting a model."""

    target_stage: ModelStage = Field(..., description="Target stage")

    model_config = ConfigDict(use_enum_values=True)
