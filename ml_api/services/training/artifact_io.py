"""Model artifact I/O to/from GCS."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import UUID

from ml_api.core.logging import get_logger
from ml_api.core.exceptions import ExternalServiceError
from ml_api.clients import GCSClient

logger = get_logger(__name__)


def save_model_artifacts(
    model: Any,
    experiment_id: UUID,
    version: int,
    trainer: Any,
    task_type: str,
    model_type: str,
    preprocess_artifacts: Dict,
    postprocess_config: Dict,
    metrics: Dict,
    feature_columns: list[str],
    gcs_client: GCSClient,
) -> str:
    """
    Save complete model artifacts to GCS.

    Returns the base URI for the model artifacts.
    """
    logger.info(
        "save_model_artifacts_started",
        experiment_id=str(experiment_id),
        version=version,
        model_type=model_type,
    )

    try:
        base_path = f"models/{experiment_id}/v{version}"

        # Save model binary
        with tempfile.NamedTemporaryFile(
            suffix=get_model_extension(model_type), delete=False
        ) as tmp:
            trainer.save_model(model, tmp.name)
            gcs_client.upload_file(
                f"{base_path}/model{get_model_extension(model_type)}",
                tmp.name,
            )
            Path(tmp.name).unlink()

        # Save preprocessing artifacts
        gcs_client.upload_json(
            f"{base_path}/preprocess.json",
            preprocess_artifacts,
        )

        # Save postprocessing config
        gcs_client.upload_json(
            f"{base_path}/postprocess.json",
            postprocess_config,
        )

        # Save metrics
        gcs_client.upload_json(
            f"{base_path}/metrics.json",
            metrics,
        )

        # Save model signature (expected input schema)
        signature = {
            "feature_columns": feature_columns,
            "task_type": task_type,
            "model_type": model_type,
        }
        gcs_client.upload_json(
            f"{base_path}/signature.json",
            signature,
        )

        logger.info(
            "save_model_artifacts_completed",
            experiment_id=str(experiment_id),
            version=version,
            base_uri=f"gs://{gcs_client.bucket_name}/{base_path}",
        )

        return f"gs://{gcs_client.bucket_name}/{base_path}"

    except Exception as e:
        logger.error(
            "save_model_artifacts_failed",
            experiment_id=str(experiment_id),
            error=str(e),
        )
        raise ExternalServiceError("GCS", f"Failed to save model artifacts: {str(e)}")


def load_model_artifacts(
    artifact_uri: str,
    trainer: Any,
    task_type: str,
    model_type: str,
    gcs_client: GCSClient,
) -> Tuple[Any, Dict, Dict, Dict]:
    """
    Load complete model artifacts from GCS.

    Returns:
        - model: Loaded model
        - preprocess_artifacts: Preprocessing artifacts
        - postprocess_config: Postprocessing config
        - signature: Model signature
    """
    logger.info("load_model_artifacts_started", artifact_uri=artifact_uri)

    try:
        # Extract base path from URI
        base_path = artifact_uri.replace(f"gs://{gcs_client.bucket_name}/", "")

        # Load model binary
        model_path = f"{base_path}/model{get_model_extension(model_type)}"
        with tempfile.NamedTemporaryFile(
            suffix=get_model_extension(model_type), delete=False
        ) as tmp:
            gcs_client.download_to_file(model_path, tmp.name)
            model = trainer.load_model(tmp.name, task_type)
            Path(tmp.name).unlink()

        # Load preprocessing artifacts
        preprocess_artifacts = gcs_client.download_json(f"{base_path}/preprocess.json")

        # Load postprocessing config
        postprocess_config = gcs_client.download_json(f"{base_path}/postprocess.json")

        # Load signature
        signature = gcs_client.download_json(f"{base_path}/signature.json")

        logger.info("load_model_artifacts_completed", artifact_uri=artifact_uri)

        return model, preprocess_artifacts, postprocess_config, signature

    except Exception as e:
        logger.error("load_model_artifacts_failed", artifact_uri=artifact_uri, error=str(e))
        raise ExternalServiceError("GCS", f"Failed to load model artifacts: {str(e)}")


def save_trial_artifacts(
    model: Any,
    experiment_id: UUID,
    trial_number: int,
    trainer: Any,
    model_type: str,
    params: Dict,
    metrics: Dict,
    gcs_client: GCSClient,
) -> str:
    """
    Save trial artifacts to GCS (for individual Optuna trials).

    Returns the URI for the trial artifacts.
    """
    logger.info(
        "save_trial_artifacts_started",
        experiment_id=str(experiment_id),
        trial_number=trial_number,
    )

    try:
        base_path = f"experiments/{experiment_id}/trials/{trial_number}"

        # Save model binary
        with tempfile.NamedTemporaryFile(
            suffix=get_model_extension(model_type), delete=False
        ) as tmp:
            trainer.save_model(model, tmp.name)
            gcs_client.upload_file(
                f"{base_path}/model{get_model_extension(model_type)}",
                tmp.name,
            )
            Path(tmp.name).unlink()

        # Save trial metadata
        metadata = {
            "trial_number": trial_number,
            "params": params,
            "metrics": metrics,
        }
        gcs_client.upload_json(f"{base_path}/metadata.json", metadata)

        logger.info(
            "save_trial_artifacts_completed",
            experiment_id=str(experiment_id),
            trial_number=trial_number,
        )

        return f"gs://{gcs_client.bucket_name}/{base_path}"

    except Exception as e:
        logger.error(
            "save_trial_artifacts_failed",
            experiment_id=str(experiment_id),
            trial_number=trial_number,
            error=str(e),
        )
        # Don't raise - trials can fail to save without breaking the study
        return ""


def get_model_extension(model_type: str) -> str:
    """Get file extension for model type."""
    extensions = {
        "catboost": ".cbm",
        "xgboost": ".json",
        "lightgbm": ".txt",
    }
    return extensions.get(model_type, ".pkl")


def delete_model_artifacts(artifact_uri: str, gcs_client: GCSClient) -> None:
    """Delete all artifacts for a model from GCS."""
    logger.info("delete_model_artifacts_started", artifact_uri=artifact_uri)

    try:
        base_path = artifact_uri.replace(f"gs://{gcs_client.bucket_name}/", "")

        # List all blobs with this prefix
        blobs = gcs_client.list_blobs(base_path)

        # Delete each blob
        for blob_path in blobs:
            gcs_client.delete(blob_path)

        logger.info(
            "delete_model_artifacts_completed",
            artifact_uri=artifact_uri,
            deleted_count=len(blobs),
        )

    except Exception as e:
        logger.error("delete_model_artifacts_failed", artifact_uri=artifact_uri, error=str(e))
        raise ExternalServiceError("GCS", f"Failed to delete model artifacts: {str(e)}")
