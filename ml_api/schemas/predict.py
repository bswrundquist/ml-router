"""Prediction schemas."""

from typing import Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ml_api.db.models.model_registry import ModelStage


class PredictRequest(BaseModel):
    """Schema for prediction request."""

    # Model selector (one of these must be provided)
    model_id: Optional[UUID] = Field(None, description="Model ID")
    experiment_id: Optional[UUID] = Field(None, description="Experiment ID (uses best model)")
    stage: Optional[ModelStage] = Field(None, description="Model stage (production/staging)")

    # Input data
    instances: list[dict] = Field(..., description="Input instances", min_length=1)

    # Optional overrides
    preprocess_config: Optional[dict] = Field(None, description="Override preprocessing config")
    postprocess_config: Optional[dict] = Field(None, description="Override postprocessing config")

    model_config = ConfigDict(use_enum_values=True)


class PredictResponse(BaseModel):
    """Schema for prediction response."""

    predictions: list[Any]
    probabilities: Optional[list[list[float]]] = Field(
        None, description="Class probabilities (classification only)"
    )
    model_id: UUID
    model_version: int
    model_type: str
    task_type: str
    request_id: str
    duration_ms: float

    model_config = ConfigDict(from_attributes=True)
