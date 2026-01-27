"""Training dispatcher with match/case for task_type and model_type."""

from typing import Any, Tuple, Protocol

from ml_api.core.logging import get_logger
from ml_api.core.exceptions import ValidationError
from ml_api.db.models.experiment import TaskType, ModelType

logger = get_logger(__name__)


class ModelTrainer(Protocol):
    """Protocol for model trainers."""

    def build_search_space(self, task_type: str) -> dict:
        """Build Optuna search space for hyperparameters."""
        ...

    def train(
        self,
        params: dict,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        task_type: str,
    ) -> Tuple[Any, dict]:
        """Train model and return (model, metrics)."""
        ...

    def predict(self, model: Any, X: Any) -> Any:
        """Make predictions."""
        ...

    def predict_proba(self, model: Any, X: Any) -> Any:
        """Make probability predictions (classification only)."""
        ...

    def feature_importance(self, model: Any) -> dict:
        """Get feature importance."""
        ...


def get_trainer(model_type: str, task_type: str) -> "ModelTrainer":
    """
    Dispatch to appropriate trainer based on model_type and task_type.

    Uses match/case for explicit dispatching.
    """
    logger.info("dispatching_trainer", model_type=model_type, task_type=task_type)

    match model_type:
        case ModelType.CATBOOST:
            from ml_api.services.training.catboost_trainer import CatBoostTrainer

            trainer = CatBoostTrainer()

        case ModelType.XGBOOST:
            from ml_api.services.training.xgboost_trainer import XGBoostTrainer

            trainer = XGBoostTrainer()

        case ModelType.LIGHTGBM:
            from ml_api.services.training.lightgbm_trainer import LightGBMTrainer

            trainer = LightGBMTrainer()

        case _:
            raise ValidationError(
                f"Unsupported model type: {model_type}", {"model_type": model_type}
            )

    # Validate task type
    match task_type:
        case TaskType.CLASSIFICATION | TaskType.REGRESSION:
            pass
        case _:
            raise ValidationError(f"Unsupported task type: {task_type}", {"task_type": task_type})

    logger.info("trainer_dispatched", trainer=type(trainer).__name__)
    return trainer


def get_default_metric(task_type: str) -> str:
    """Get default metric based on task type (match/case)."""
    match task_type:
        case TaskType.CLASSIFICATION:
            return "auc"
        case TaskType.REGRESSION:
            return "rmse"
        case _:
            raise ValidationError(f"Unknown task type: {task_type}")
