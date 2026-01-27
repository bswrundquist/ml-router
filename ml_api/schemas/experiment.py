"""Experiment schemas."""

from typing import Optional
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ml_api.db.models.experiment import TaskType, ModelType, ExperimentStatus
from ml_api.db.models.trial import TrialStatus


class OptunaConfig(BaseModel):
    """Optuna hyperparameter tuning configuration."""

    n_trials: int = Field(50, description="Number of trials", ge=1, le=1000)
    timeout_seconds: Optional[int] = Field(None, description="Timeout in seconds", ge=60)
    sampler: str = Field("TPE", description="Sampler type (TPE, Random, Grid)")
    pruner: str = Field("MedianPruner", description="Pruner type (MedianPruner, HyperbandPruner)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    model_config = ConfigDict(use_enum_values=True)


class ExperimentCreate(BaseModel):
    """Schema for creating an experiment."""

    split_id: UUID = Field(..., description="Data split ID")
    name: str = Field(..., description="Experiment name", min_length=1, max_length=255)
    target_column: str = Field(..., description="Target column name")
    feature_columns: list[str] = Field(..., description="Feature column names", min_length=1)
    task_type: TaskType = Field(..., description="Task type (classification/regression)")
    model_type: ModelType = Field(ModelType.CATBOOST, description="Model type")
    metric_name: Optional[str] = Field(
        None, description="Metric to optimize (auto-selected if not provided)"
    )
    optuna_config: OptunaConfig = Field(
        default_factory=OptunaConfig, description="Optuna configuration"
    )
    preprocess_config: Optional[dict] = Field(None, description="Preprocessing configuration")
    postprocess_config: Optional[dict] = Field(None, description="Postprocessing configuration")

    model_config = ConfigDict(use_enum_values=True)


class ExperimentUpdate(BaseModel):
    """Schema for updating an experiment."""

    name: Optional[str] = Field(None, description="Updated experiment name")

    model_config = ConfigDict(use_enum_values=True)


class TrialResponse(BaseModel):
    """Schema for trial response."""

    id: UUID
    experiment_id: UUID
    number: int
    params: dict
    metrics: Optional[dict]
    artifact_uri: Optional[str]
    status: TrialStatus
    error_message: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class ExperimentResponse(BaseModel):
    """Schema for experiment response."""

    id: UUID
    split_id: UUID
    name: str
    target_column: str
    feature_columns: list[str]
    task_type: TaskType
    model_type: ModelType
    optuna_config: dict
    search_space: Optional[dict]
    metric_name: str
    status: ExperimentStatus
    best_trial_id: Optional[UUID]
    artifact_uri: Optional[str]
    error_message: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class ExperimentListResponse(BaseModel):
    """Schema for experiment list response."""

    items: list[ExperimentResponse]
    total: int
    page: int
    page_size: int

    model_config = ConfigDict(from_attributes=True)
