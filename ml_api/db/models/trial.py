"""Trial database model."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Integer, Text, Enum, ForeignKey, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ml_api.db.base import Base

if TYPE_CHECKING:
    from ml_api.db.models.experiment import Experiment


class TrialStatus(str, enum.Enum):
    """Trial execution status."""

    RUNNING = "running"
    COMPLETED = "completed"
    PRUNED = "pruned"
    FAILED = "failed"


class Trial(Base):
    """Trial table for individual Optuna trial runs."""

    __tablename__ = "trials"

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

    # Trial info
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Artifacts
    artifact_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status
    status: Mapped[TrialStatus] = mapped_column(
        Enum(TrialStatus, name="trial_status_enum"),
        nullable=False,
        default=TrialStatus.RUNNING,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Relationships
    experiment: Mapped["Experiment"] = relationship(
        "Experiment",
        back_populates="trials",
        foreign_keys=[experiment_id],
    )

    # Indexes
    __table_args__ = (
        Index("ix_trials_experiment_id_number", "experiment_id", "number", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Trial(id={self.id}, experiment_id={self.experiment_id}, number={self.number})>"
