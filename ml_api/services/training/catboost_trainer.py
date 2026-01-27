"""CatBoost model trainer (default)."""

from typing import Any, Tuple

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score

from ml_api.core.logging import get_logger
from ml_api.core.exceptions import ModelTrainingError
from ml_api.db.models.experiment import TaskType
from ml_api.services.training.dataset_io import polars_to_pandas

logger = get_logger(__name__)


class CatBoostTrainer:
    """CatBoost trainer for classification and regression."""

    def build_search_space(self, task_type: str) -> dict:
        """Build Optuna search space for CatBoost."""
        # Return search space structure (trial will sample from these)
        return {
            "iterations": {"type": "int", "low": 50, "high": 500},
            "depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
            "l2_leaf_reg": {"type": "float", "low": 1, "high": 10},
            "border_count": {"type": "int", "low": 32, "high": 255},
            "bagging_temperature": {"type": "float", "low": 0, "high": 1},
        }

    def train(
        self,
        params: dict,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        task_type: str,
    ) -> Tuple[Any, dict]:
        """Train CatBoost model."""
        logger.info("catboost_train_started", task_type=task_type, params=params)

        try:
            # Convert Polars to Pandas (CatBoost supports pandas natively)
            X_train_pd = polars_to_pandas(X_train, "CatBoost training requires pandas")
            y_train_pd = y_train.to_pandas()
            X_val_pd = polars_to_pandas(X_val, "CatBoost validation requires pandas")
            y_val_pd = y_val.to_pandas()

            # Create model based on task type
            if task_type == TaskType.CLASSIFICATION:
                model = CatBoostClassifier(
                    iterations=params.get("iterations", 100),
                    depth=params.get("depth", 6),
                    learning_rate=params.get("learning_rate", 0.03),
                    l2_leaf_reg=params.get("l2_leaf_reg", 3),
                    border_count=params.get("border_count", 128),
                    bagging_temperature=params.get("bagging_temperature", 1),
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                )
            else:  # regression
                model = CatBoostRegressor(
                    iterations=params.get("iterations", 100),
                    depth=params.get("depth", 6),
                    learning_rate=params.get("learning_rate", 0.03),
                    l2_leaf_reg=params.get("l2_leaf_reg", 3),
                    border_count=params.get("border_count", 128),
                    bagging_temperature=params.get("bagging_temperature", 1),
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                )

            # Train
            model.fit(
                X_train_pd,
                y_train_pd,
                eval_set=(X_val_pd, y_val_pd),
                verbose=False,
            )

            # Compute metrics
            metrics = self._compute_metrics(model, X_val_pd, y_val_pd, task_type)

            logger.info("catboost_train_completed", metrics=metrics)
            return model, metrics

        except Exception as e:
            logger.error("catboost_train_failed", error=str(e))
            raise ModelTrainingError(f"CatBoost training failed: {str(e)}")

    def predict(self, model: Any, X: Any) -> np.ndarray:
        """Make predictions."""
        X_pd = polars_to_pandas(X, "CatBoost prediction requires pandas")
        return model.predict(X_pd)

    def predict_proba(self, model: Any, X: Any) -> np.ndarray:
        """Make probability predictions (classification only)."""
        X_pd = polars_to_pandas(X, "CatBoost prediction requires pandas")
        return model.predict_proba(X_pd)

    def feature_importance(self, model: Any) -> dict:
        """Get feature importance from trained model."""
        importance_values = model.get_feature_importance()
        feature_names = model.feature_names_

        importance_dict = {
            name: float(importance) for name, importance in zip(feature_names, importance_values)
        }

        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance

    def save_model(self, model: Any, path: str) -> None:
        """Save model to file."""
        model.save_model(path)
        logger.info("catboost_model_saved", path=path)

    def load_model(self, path: str, task_type: str) -> Any:
        """Load model from file."""
        if task_type == TaskType.CLASSIFICATION:
            model = CatBoostClassifier()
        else:
            model = CatBoostRegressor()

        model.load_model(path)
        logger.info("catboost_model_loaded", path=path)
        return model

    def _compute_metrics(self, model: Any, X_val: Any, y_val: Any, task_type: str) -> dict:
        """Compute evaluation metrics."""
        y_pred = model.predict(X_val)

        if task_type == TaskType.CLASSIFICATION:
            y_proba = model.predict_proba(X_val)

            # Handle binary vs multiclass
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")

            accuracy = accuracy_score(y_val, y_pred)

            return {
                "auc": float(auc),
                "accuracy": float(accuracy),
            }
        else:  # regression
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)

            return {
                "rmse": float(rmse),
                "mae": float(mae),
            }
