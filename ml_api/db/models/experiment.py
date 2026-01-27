"""Experiment database model."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Text, Enum, ForeignKey, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ml_api.db.base import Base

if TYPE_CHECKING:
    from ml_api.db.models.split import DataSplit
    from ml_api.db.models.trial import Trial
    from ml_api.db.models.model_registry import ModelRegistry


class TaskType(str, enum.Enum):
    """ML task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelType(str, enum.Enum):
    """Supported model types."""

    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class ExperimentStatus(str, enum.Enum):
    """Experiment execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class Experiment(Base):
    """Experiment table for hyperparameter tuning runs."""

    __tablename__ = "experiments"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # Foreign keys
    split_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("data_splits.id"),
        nullable=False,
        index=True,
    )
    best_trial_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("trials.id", use_alter=True),
        nullable=True,
    )

    # Configuration
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_columns: Mapped[list] = mapped_column(JSON, nullable=False)

    # Model configuration
    task_type: Mapped[TaskType] = mapped_column(
        Enum(TaskType, name="task_type_enum"),
        nullable=False,
    )
    model_type: Mapped[ModelType] = mapped_column(
        Enum(ModelType, name="model_type_enum"),
        nullable=False,
        default=ModelType.CATBOOST,
    )

    # Optuna configuration
    optuna_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    search_space: Mapped[dict] = mapped_column(JSON, nullable=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Results
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus, name="experiment_status_enum"),
        nullable=False,
        default=ExperimentStatus.PENDING,
        index=True,
    )
    artifact_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    split: Mapped["DataSplit"] = relationship("DataSplit", back_populates="experiments")
    trials: Mapped[list["Trial"]] = relationship(
        "Trial",
        back_populates="experiment",
        foreign_keys="Trial.experiment_id",
        cascade="all, delete-orphan",
    )
    best_trial: Mapped[Optional["Trial"]] = relationship(
        "Trial",
        foreign_keys=[best_trial_id],
        post_update=True,
    )
    model_versions: Mapped[list["ModelRegistry"]] = relationship(
        "ModelRegistry",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("ix_experiments_split_id_created_at", "split_id", "created_at"),
        Index("ix_experiments_model_type_task_type", "model_type", "task_type"),
    )

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name={self.name}, status={self.status})>"
