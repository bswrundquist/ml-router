"""Feature importance schemas."""

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ml_api.db.models.model_registry import ModelStage


class ImportanceRequest(BaseModel):
    """Schema for feature importance request."""

    # Model selector (one of these must be provided)
    model_id: Optional[UUID] = Field(None, description="Model ID")
    experiment_id: Optional[UUID] = Field(None, description="Experiment ID (uses best model)")
    stage: Optional[ModelStage] = Field(None, description="Model stage (production/staging)")

    # Optional data for SHAP
    instances: Optional[list[dict]] = Field(None, description="Instances for SHAP explanation")
    dataset_uri: Optional[str] = Field(None, description="Dataset URI for SHAP background")

    # Configuration
    max_samples: int = Field(1000, description="Max samples for SHAP", ge=1, le=10000)
    method_preference: list[str] = Field(
        default_factory=lambda: ["shap", "native", "permutation"],
        description="Preference order for importance methods",
    )

    model_config = ConfigDict(use_enum_values=True)


class FeatureImportance(BaseModel):
    """Feature importance entry."""

    feature: str
    importance: float
    rank: int

    model_config = ConfigDict(from_attributes=True)


class ImportanceResponse(BaseModel):
    """Schema for feature importance response."""

    global_importance: list[FeatureImportance]
    method_used: str
    model_id: UUID
    model_version: int
    request_id: str
    shap_values: Optional[list[dict]] = Field(None, description="Per-instance SHAP values")
    base_value: Optional[float] = Field(None, description="SHAP base value")

    model_config = ConfigDict(from_attributes=True)
