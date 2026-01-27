"""LightGBM model trainer."""

from typing import Any, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score

from ml_api.core.logging import get_logger
from ml_api.core.exceptions import ModelTrainingError
from ml_api.db.models.experiment import TaskType
from ml_api.services.training.dataset_io import polars_to_pandas

logger = get_logger(__name__)


class LightGBMTrainer:
    """LightGBM trainer for classification and regression."""

    def build_search_space(self, task_type: str) -> dict:
        """Build Optuna search space for LightGBM."""
        return {
            "n_estimators": {"type": "int", "low": 50, "high": 500},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
            "num_leaves": {"type": "int", "low": 20, "high": 150},
            "subsample": {"type": "float", "low": 0.6, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 0, "high": 1},
            "reg_lambda": {"type": "float", "low": 0, "high": 1},
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
        """Train LightGBM model."""
        logger.info("lightgbm_train_started", task_type=task_type)

        try:
            X_train_pd = polars_to_pandas(X_train, "LightGBM training")
            y_train_pd = y_train.to_pandas()
            X_val_pd = polars_to_pandas(X_val, "LightGBM validation")
            y_val_pd = y_val.to_pandas()

            if task_type == TaskType.CLASSIFICATION:
                model = lgb.LGBMClassifier(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", -1),
                    learning_rate=params.get("learning_rate", 0.1),
                    num_leaves=params.get("num_leaves", 31),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    reg_alpha=params.get("reg_alpha", 0),
                    reg_lambda=params.get("reg_lambda", 0),
                    random_state=42,
                    verbose=-1,
                )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", -1),
                    learning_rate=params.get("learning_rate", 0.1),
                    num_leaves=params.get("num_leaves", 31),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    reg_alpha=params.get("reg_alpha", 0),
                    reg_lambda=params.get("reg_lambda", 0),
                    random_state=42,
                    verbose=-1,
                )

            model.fit(X_train_pd, y_train_pd, eval_set=[(X_val_pd, y_val_pd)])

            metrics = self._compute_metrics(model, X_val_pd, y_val_pd, task_type)
            logger.info("lightgbm_train_completed", metrics=metrics)
            return model, metrics

        except Exception as e:
            logger.error("lightgbm_train_failed", error=str(e))
            raise ModelTrainingError(f"LightGBM training failed: {str(e)}")

    def predict(self, model: Any, X: Any) -> np.ndarray:
        """Make predictions."""
        X_pd = polars_to_pandas(X, "LightGBM prediction")
        return model.predict(X_pd)

    def predict_proba(self, model: Any, X: Any) -> np.ndarray:
        """Make probability predictions."""
        X_pd = polars_to_pandas(X, "LightGBM prediction")
        return model.predict_proba(X_pd)

    def feature_importance(self, model: Any) -> dict:
        """Get feature importance."""
        importance = model.feature_importances_
        feature_names = model.feature_name_

        return {
            name: float(imp)
            for name, imp in sorted(
                zip(feature_names, importance), key=lambda x: x[1], reverse=True
            )
        }

    def save_model(self, model: Any, path: str) -> None:
        """Save model."""
        model.booster_.save_model(path)

    def load_model(self, path: str, task_type: str) -> Any:
        """Load model."""
        if task_type == TaskType.CLASSIFICATION:
            model = lgb.LGBMClassifier()
        else:
            model = lgb.LGBMRegressor()
        model.booster_ = lgb.Booster(model_file=path)
        return model

    def _compute_metrics(self, model: Any, X_val: Any, y_val: Any, task_type: str) -> dict:
        """Compute metrics."""
        y_pred = model.predict(X_val)

        if task_type == TaskType.CLASSIFICATION:
            y_proba = model.predict_proba(X_val)
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")

            return {"auc": float(auc), "accuracy": float(accuracy_score(y_val, y_pred))}
        else:
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
                "mae": float(mean_absolute_error(y_val, y_pred)),
            }
