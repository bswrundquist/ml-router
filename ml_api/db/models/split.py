"""DataSplit database model."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, Text, Enum, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ml_api.db.base import Base

if TYPE_CHECKING:
    from ml_api.db.models.experiment import Experiment


class SplitStrategy(str, enum.Enum):
    """Split strategy types."""

    RANDOM = "random"
    TIME_BASED = "time_based"
    ENTITY_BASED = "entity_based"


class SplitStatus(str, enum.Enum):
    """Split processing status."""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class DataSplit(Base):
    """Data split table for storing train/val/test splits."""

    __tablename__ = "data_splits"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # Identification
    entity_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Data source
    dataset_uri: Mapped[str] = mapped_column(Text, nullable=False)

    # Split configuration
    split_strategy: Mapped[SplitStrategy] = mapped_column(
        Enum(SplitStrategy, name="split_strategy_enum"),
        nullable=False,
    )
    split_params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Split artifacts (GCS URIs)
    train_uri: Mapped[str] = mapped_column(Text, nullable=True)
    val_uri: Mapped[str] = mapped_column(Text, nullable=True)
    test_uri: Mapped[str] = mapped_column(Text, nullable=True)

    # Metadata
    row_count_train: Mapped[int] = mapped_column(Integer, nullable=True)
    row_count_val: Mapped[int] = mapped_column(Integer, nullable=True)
    row_count_test: Mapped[int] = mapped_column(Integer, nullable=True)
    schema_json: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Status
    status: Mapped[SplitStatus] = mapped_column(
        Enum(SplitStatus, name="split_status_enum"),
        nullable=False,
        default=SplitStatus.PENDING,
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment",
        back_populates="split",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("ix_data_splits_entity_id_created_at", "entity_id", "created_at"),
        Index("ix_data_splits_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<DataSplit(id={self.id}, entity_id={self.entity_id}, status={self.status})>"
