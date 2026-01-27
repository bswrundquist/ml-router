"""Preprocessing operations using Polars."""

from typing import Tuple, Optional
import polars as pl

from ml_api.core.logging import get_logger, log_dataframe_info
from ml_api.core.exceptions import DataProcessingError

logger = get_logger(__name__)


def preprocess_features(
    df: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: Optional[dict] = None,
) -> Tuple[pl.DataFrame, pl.Series, dict]:
    """
    Preprocess features using Polars.

    Returns:
        - X: Features DataFrame (Polars)
        - y: Target Series (Polars)
        - artifacts: Preprocessing artifacts (category maps, etc.)
    """
    logger.info(
        "preprocess_started",
        total_cols=len(df.columns),
        feature_cols=len(feature_columns),
        target_column=target_column,
    )

    config = config or {}
    artifacts = {}

    try:
        # Validate columns exist
        missing_cols = set(feature_columns + [target_column]) - set(df.columns)
        if missing_cols:
            raise DataProcessingError(f"Missing columns: {missing_cols}")

        # Extract features and target
        X = df.select(feature_columns)
        y = df.select(target_column).to_series()

        log_dataframe_info(logger, "X_before_preprocessing", X)

        # Handle missing values
        missing_strategy = config.get("missing_strategy", "drop")
        if missing_strategy == "drop":
            null_count_before = X.null_count().sum(axis=1)[0]
            if null_count_before > 0:
                logger.info("dropping_rows_with_nulls", null_count=null_count_before)
                # Get mask where any feature is null
                null_mask = X.select(pl.any_horizontal(pl.all().is_null())).to_series()
                X = X.filter(~null_mask)
                y = y.filter(~null_mask)

        elif missing_strategy == "fill_mean":
            for col in X.columns:
                if X[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                    mean_val = X[col].mean()
                    X = X.with_columns(pl.col(col).fill_null(mean_val))

        # Encode categorical features
        categorical_cols = [
            col for col in X.columns if X[col].dtype == pl.Utf8 or X[col].dtype == pl.Categorical
        ]

        if categorical_cols:
            logger.info("encoding_categoricals", cols=categorical_cols)
            category_maps = {}

            for col in categorical_cols:
                # Create category mapping
                unique_vals = X[col].unique().drop_nulls().to_list()
                category_map = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                category_maps[col] = category_map

                # Apply mapping
                X = X.with_columns(pl.col(col).map_dict(category_map, default=-1).alias(col))

            artifacts["category_maps"] = category_maps

        # Cast to appropriate types
        for col in X.columns:
            if X[col].dtype == pl.Utf8:
                # Remaining string columns -> cast to int if possible
                try:
                    X = X.with_columns(pl.col(col).cast(pl.Float64))
                except Exception:
                    logger.warning("could_not_cast_to_numeric", col=col)

        log_dataframe_info(logger, "X_after_preprocessing", X)
        logger.info("preprocess_completed", rows=len(X), cols=len(X.columns))

        return X, y, artifacts

    except Exception as e:
        logger.error("preprocess_failed", error=str(e))
        raise DataProcessingError(f"Preprocessing failed: {str(e)}")


def apply_preprocessing(
    df: pl.DataFrame,
    feature_columns: list[str],
    artifacts: dict,
) -> pl.DataFrame:
    """Apply saved preprocessing artifacts to new data (for prediction)."""
    logger.info("apply_preprocessing_started", rows=len(df), cols=len(df.columns))

    try:
        X = df.select(feature_columns)

        # Apply category mappings
        category_maps = artifacts.get("category_maps", {})
        for col, category_map in category_maps.items():
            if col in X.columns:
                X = X.with_columns(pl.col(col).map_dict(category_map, default=-1).alias(col))

        # Cast types
        for col in X.columns:
            if X[col].dtype == pl.Utf8:
                try:
                    X = X.with_columns(pl.col(col).cast(pl.Float64))
                except Exception:
                    pass

        logger.info("apply_preprocessing_completed", rows=len(X), cols=len(X.columns))
        return X

    except Exception as e:
        logger.error("apply_preprocessing_failed", error=str(e))
        raise DataProcessingError(f"Failed to apply preprocessing: {str(e)}")
