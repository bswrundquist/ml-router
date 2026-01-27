"""Model Registry database model."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Integer, Text, Enum, ForeignKey, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ml_api.db.base import Base

if TYPE_CHECKING:
    from ml_api.db.models.experiment import Experiment


class ModelStage(str, enum.Enum):
    """Model deployment stage."""

    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistry(Base):
    """Model registry table for versioned model artifacts."""

    __tablename__ = "model_registry"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # Foreign keys
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id"),
        nullable=False,
        index=True,
    )

    # Version info
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    stage: Mapped[ModelStage] = mapped_column(
        Enum(ModelStage, name="model_stage_enum"),
        nullable=False,
        default=ModelStage.STAGING,
        index=True,
    )

    # Artifacts
    artifact_uri: Mapped[str] = mapped_column(Text, nullable=False)

    # Model metadata
    signature: Mapped[dict] = mapped_column(JSON, nullable=False)
    preprocess_config: Mapped[dict] = mapped_column(JSON, nullable=True)
    postprocess_config: Mapped[dict] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="model_versions",
    )

    # Indexes
    __table_args__ = (
        Index("ix_model_registry_experiment_id_version", "experiment_id", "version", unique=True),
        Index("ix_model_registry_stage", "stage"),
    )

    def __repr__(self) -> str:
        return f"<ModelRegistry(id={self.id}, version={self.version}, stage={self.stage})>"
