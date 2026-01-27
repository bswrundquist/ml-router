"""Data split service implementation."""

import time
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from ml_api.core.logging import get_logger, log_function_call, log_function_result
from ml_api.core.exceptions import ValidationError, DataProcessingError
from ml_api.db.models.split import DataSplit, SplitStatus
from ml_api.schemas.split import DataSplitCreate
from ml_api.clients import GCSClient
from ml_api.services.training.dataset_io import (
    load_dataset_from_uri,
    load_dataset_from_records,
    save_dataset_to_gcs,
    split_dataset,
)

logger = get_logger(__name__)


class SplitService:
    """Service for data split operations."""

    def __init__(self, db: AsyncSession, gcs_client: GCSClient):
        self.db = db
        self.gcs_client = gcs_client

    async def create_split(self, request: DataSplitCreate) -> DataSplit:
        """Create a new data split."""
        start_time = time.time()
        split_id = uuid4()

        log_function_call(
            logger,
            "create_split",
            split_id=str(split_id),
            entity_id=request.entity_id,
            strategy=request.split_strategy,
        )

        try:
            # Create database record
            split = DataSplit(
                id=split_id,
                entity_id=request.entity_id,
                dataset_uri=request.dataset_uri or f"inline_{split_id}",
                split_strategy=request.split_strategy,
                split_params=request.split_params,
                status=SplitStatus.PENDING,
            )

            self.db.add(split)
            await self.db.commit()
            await self.db.refresh(split)

            # Load data
            if request.inline_data:
                df = load_dataset_from_records(request.inline_data)
            elif request.dataset_uri:
                df = load_dataset_from_uri(request.dataset_uri, self.gcs_client)
            else:
                raise ValidationError("Either inline_data or dataset_uri must be provided")

            # Perform split
            train_df, val_df, test_df = split_dataset(
                df,
                request.split_strategy.value,
                request.split_params,
            )

            # Save splits to GCS
            train_uri = save_dataset_to_gcs(
                train_df,
                f"splits/{split_id}/train.parquet",
                self.gcs_client,
            )
            val_uri = save_dataset_to_gcs(
                val_df,
                f"splits/{split_id}/val.parquet",
                self.gcs_client,
            )
            test_uri = save_dataset_to_gcs(
                test_df,
                f"splits/{split_id}/test.parquet",
                self.gcs_client,
            )

            # Update database record
            split.train_uri = train_uri
            split.val_uri = val_uri
            split.test_uri = test_uri
            split.row_count_train = len(train_df)
            split.row_count_val = len(val_df)
            split.row_count_test = len(test_df)
            split.schema_json = {
                "columns": df.columns,
                "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            }
            split.status = SplitStatus.READY

            await self.db.commit()
            await self.db.refresh(split)

            duration_ms = (time.time() - start_time) * 1000
            log_function_result(
                logger,
                "create_split",
                duration_ms=duration_ms,
                split_id=str(split_id),
                train_rows=split.row_count_train,
                val_rows=split.row_count_val,
                test_rows=split.row_count_test,
            )

            return split

        except Exception as e:
            # Mark as failed
            split.status = SplitStatus.FAILED
            await self.db.commit()

            logger.error("create_split_failed", split_id=str(split_id), error=str(e))
            raise DataProcessingError(f"Failed to create split: {str(e)}")
