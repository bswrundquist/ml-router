"""Data split schemas."""

from typing import Optional
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from ml_api.db.models.split import SplitStrategy, SplitStatus


class DataSplitCreate(BaseModel):
    """Schema for creating a data split."""

    entity_id: str = Field(..., description="Entity identifier")
    dataset_uri: Optional[str] = Field(
        None, description="URI to dataset (if not providing inline data)"
    )
    inline_data: Optional[list[dict]] = Field(None, description="Inline data rows")
    split_strategy: SplitStrategy = Field(SplitStrategy.RANDOM, description="Split strategy")
    split_params: dict = Field(
        default_factory=dict,
        description="Split parameters (e.g., {'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15, 'seed': 42})",
    )

    model_config = ConfigDict(use_enum_values=True)


class DataSplitUpdate(BaseModel):
    """Schema for updating a data split."""

    split_params: Optional[dict] = Field(None, description="Updated split parameters")

    model_config = ConfigDict(use_enum_values=True)


class DataSplitResponse(BaseModel):
    """Schema for data split response."""

    id: UUID
    entity_id: str
    dataset_uri: str
    split_strategy: SplitStrategy
    split_params: dict
    train_uri: Optional[str]
    val_uri: Optional[str]
    test_uri: Optional[str]
    row_count_train: Optional[int]
    row_count_val: Optional[int]
    row_count_test: Optional[int]
    schema_json: Optional[dict]
    status: SplitStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)


class DataSplitListResponse(BaseModel):
    """Schema for data split list response."""

    items: list[DataSplitResponse]
    total: int
    page: int
    page_size: int

    model_config = ConfigDict(from_attributes=True)
