"""Dataset I/O operations using Polars (primary) with explicit pandas conversion when needed."""

from typing import Tuple
import tempfile
from pathlib import Path

import polars as pl
import pandas as pd

from ml_api.core.logging import get_logger, log_dataframe_info, log_conversion
from ml_api.core.exceptions import DataProcessingError
from ml_api.clients import GCSClient

logger = get_logger(__name__)


def load_dataset_from_uri(uri: str, gcs_client: GCSClient) -> pl.DataFrame:
    """Load dataset from URI (GCS or local) using Polars."""
    logger.info("load_dataset_started", uri=uri)

    try:
        if uri.startswith("gs://"):
            # Download from GCS to temp file
            blob_path = uri.replace(f"gs://{gcs_client.bucket_name}/", "")

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                gcs_client.download_to_file(blob_path, tmp.name)
                df = pl.read_parquet(tmp.name)
                Path(tmp.name).unlink()
        else:
            # Local file
            if uri.endswith(".parquet"):
                df = pl.read_parquet(uri)
            elif uri.endswith(".csv"):
                df = pl.read_csv(uri)
            else:
                raise DataProcessingError(f"Unsupported file format: {uri}")

        log_dataframe_info(logger, "loaded_dataset", df, context=f"uri={uri}")
        logger.info("load_dataset_completed", uri=uri, rows=df.shape[0], cols=df.shape[1])
        return df

    except Exception as e:
        logger.error("load_dataset_failed", uri=uri, error=str(e))
        raise DataProcessingError(f"Failed to load dataset from {uri}: {str(e)}")


def load_dataset_from_records(records: list[dict]) -> pl.DataFrame:
    """Load dataset from inline records using Polars."""
    logger.info("load_dataset_from_records_started", num_records=len(records))

    try:
        df = pl.DataFrame(records)
        log_dataframe_info(logger, "loaded_from_records", df)
        logger.info("load_dataset_from_records_completed", rows=df.shape[0], cols=df.shape[1])
        return df

    except Exception as e:
        logger.error("load_dataset_from_records_failed", error=str(e))
        raise DataProcessingError(f"Failed to create dataset from records: {str(e)}")


def save_dataset_to_gcs(
    df: pl.DataFrame,
    blob_path: str,
    gcs_client: GCSClient,
) -> str:
    """Save Polars DataFrame to GCS as Parquet."""
    logger.info("save_dataset_started", blob_path=blob_path, rows=df.shape[0], cols=df.shape[1])

    try:
        # Write to temp file then upload
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.write_parquet(tmp.name)
            uri = gcs_client.upload_file(
                blob_path, tmp.name, content_type="application/octet-stream"
            )
            Path(tmp.name).unlink()

        logger.info("save_dataset_completed", uri=uri)
        return uri

    except Exception as e:
        logger.error("save_dataset_failed", blob_path=blob_path, error=str(e))
        raise DataProcessingError(f"Failed to save dataset to {blob_path}: {str(e)}")


def split_dataset(
    df: pl.DataFrame,
    strategy: str,
    params: dict,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split dataset into train/val/test using Polars."""
    logger.info(
        "split_dataset_started",
        strategy=strategy,
        total_rows=df.shape[0],
        params=params,
    )

    try:
        if strategy == "random":
            return _split_random(df, params)
        elif strategy == "time_based":
            return _split_time_based(df, params)
        elif strategy == "entity_based":
            return _split_entity_based(df, params)
        else:
            raise DataProcessingError(f"Unknown split strategy: {strategy}")

    except Exception as e:
        logger.error("split_dataset_failed", strategy=strategy, error=str(e))
        raise DataProcessingError(f"Failed to split dataset: {str(e)}")


def _split_random(
    df: pl.DataFrame, params: dict
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Random split using Polars."""
    train_ratio = params.get("train_ratio", 0.7)
    val_ratio = params.get("val_ratio", 0.15)
    test_ratio = params.get("test_ratio", 0.15)
    seed = params.get("seed", 42)

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise DataProcessingError(f"Split ratios must sum to 1.0, got {total_ratio}")

    # Shuffle and split
    df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)

    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]

    logger.info(
        "random_split_completed",
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
    )

    return train_df, val_df, test_df


def _split_time_based(
    df: pl.DataFrame, params: dict
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Time-based split using Polars."""
    time_column = params.get("time_column")
    if not time_column:
        raise DataProcessingError("time_column required for time_based split")

    if time_column not in df.columns:
        raise DataProcessingError(f"Time column '{time_column}' not found in dataset")

    train_ratio = params.get("train_ratio", 0.7)
    val_ratio = params.get("val_ratio", 0.15)

    # Sort by time
    df_sorted = df.sort(time_column)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df_sorted[:train_end]
    val_df = df_sorted[train_end:val_end]
    test_df = df_sorted[val_end:]

    logger.info(
        "time_based_split_completed",
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
    )

    return train_df, val_df, test_df


def _split_entity_based(
    df: pl.DataFrame, params: dict
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Entity-based (group) split using Polars."""
    entity_column = params.get("entity_column")
    if not entity_column:
        raise DataProcessingError("entity_column required for entity_based split")

    if entity_column not in df.columns:
        raise DataProcessingError(f"Entity column '{entity_column}' not found in dataset")

    train_ratio = params.get("train_ratio", 0.7)
    val_ratio = params.get("val_ratio", 0.15)
    seed = params.get("seed", 42)

    # Get unique entities and shuffle them
    unique_entities = df.select(pl.col(entity_column).unique()).to_series()
    entities_df = pl.DataFrame({entity_column: unique_entities})
    entities_shuffled = entities_df.sample(fraction=1.0, shuffle=True, seed=seed)

    # Split entities
    n_entities = len(entities_shuffled)
    train_end = int(n_entities * train_ratio)
    val_end = train_end + int(n_entities * val_ratio)

    train_entities = entities_shuffled[:train_end]
    val_entities = entities_shuffled[train_end:val_end]
    test_entities = entities_shuffled[val_end:]

    # Filter original data by entity sets
    train_df = df.join(train_entities, on=entity_column, how="inner")
    val_df = df.join(val_entities, on=entity_column, how="inner")
    test_df = df.join(test_entities, on=entity_column, how="inner")

    logger.info(
        "entity_based_split_completed",
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
    )

    return train_df, val_df, test_df


def polars_to_pandas(df: pl.DataFrame, reason: str) -> pd.DataFrame:
    """
    Convert Polars DataFrame to Pandas with explicit logging.

    This should ONLY be called when absolutely necessary (e.g., ML library requires pandas).
    """
    memory_before = df.estimated_size() / (1024 * 1024)

    # Convert to pandas
    pdf = df.to_pandas()

    memory_after = pdf.memory_usage(deep=True).sum() / (1024 * 1024)

    log_conversion(
        logger,
        from_type="polars.DataFrame",
        to_type="pandas.DataFrame",
        reason=reason,
        rows=len(df),
        cols=len(df.columns),
        memory_before_mb=round(memory_before, 2),
        memory_after_mb=round(memory_after, 2),
    )

    return pdf


def pandas_to_polars(df: pd.DataFrame, reason: str) -> pl.DataFrame:
    """Convert Pandas DataFrame back to Polars with explicit logging."""
    memory_before = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Convert to polars
    pldf = pl.from_pandas(df)

    memory_after = pldf.estimated_size() / (1024 * 1024)

    log_conversion(
        logger,
        from_type="pandas.DataFrame",
        to_type="polars.DataFrame",
        reason=reason,
        rows=len(df),
        cols=len(df.columns),
        memory_before_mb=round(memory_before, 2),
        memory_after_mb=round(memory_after, 2),
    )

    return pldf
