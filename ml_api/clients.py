"""External clients for GCS and other services."""

from typing import Optional
import json

from google.cloud import storage
from google.cloud.exceptions import NotFound

from ml_api.core.config import settings
from ml_api.core.logging import get_logger
from ml_api.core.exceptions import ExternalServiceError

logger = get_logger(__name__)


class GCSClient:
    """Google Cloud Storage client for artifact management."""

    def __init__(self):
        """Initialize GCS client."""
        self.client = storage.Client(project=settings.gcs_project_id)
        self.bucket_name = settings.gcs_bucket
        self._bucket: Optional[storage.Bucket] = None

    @property
    def bucket(self) -> storage.Bucket:
        """Get or create bucket reference."""
        if self._bucket is None:
            try:
                self._bucket = self.client.bucket(self.bucket_name)
                # Check if bucket exists
                if not self._bucket.exists():
                    logger.warning(
                        "gcs_bucket_not_found",
                        bucket_name=self.bucket_name,
                    )
                    raise ExternalServiceError(
                        "GCS",
                        f"Bucket not found: {self.bucket_name}",
                        {"bucket_name": self.bucket_name},
                    )
            except Exception as e:
                logger.error("gcs_client_init_failed", error=str(e))
                raise ExternalServiceError("GCS", f"Failed to initialize client: {str(e)}")
        return self._bucket

    def upload_bytes(
        self, blob_path: str, data: bytes, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload bytes to GCS."""
        try:
            logger.info(
                "gcs_upload_started",
                blob_path=blob_path,
                size_bytes=len(data),
            )

            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(data, content_type=content_type)

            uri = f"gs://{self.bucket_name}/{blob_path}"
            logger.info("gcs_upload_completed", uri=uri)
            return uri

        except Exception as e:
            logger.error("gcs_upload_failed", blob_path=blob_path, error=str(e))
            raise ExternalServiceError("GCS", f"Upload failed: {str(e)}", {"blob_path": blob_path})

    def upload_file(
        self, blob_path: str, file_path: str, content_type: str = "application/octet-stream"
    ) -> str:
        """Upload file to GCS."""
        try:
            logger.info("gcs_upload_file_started", blob_path=blob_path, file_path=file_path)

            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(file_path, content_type=content_type)

            uri = f"gs://{self.bucket_name}/{blob_path}"
            logger.info("gcs_upload_file_completed", uri=uri)
            return uri

        except Exception as e:
            logger.error("gcs_upload_file_failed", blob_path=blob_path, error=str(e))
            raise ExternalServiceError("GCS", f"Upload failed: {str(e)}", {"blob_path": blob_path})

    def download_bytes(self, blob_path: str) -> bytes:
        """Download bytes from GCS."""
        try:
            logger.info("gcs_download_started", blob_path=blob_path)

            blob = self.bucket.blob(blob_path)
            data = blob.download_as_bytes()

            logger.info("gcs_download_completed", blob_path=blob_path, size_bytes=len(data))
            return data

        except NotFound:
            logger.error("gcs_blob_not_found", blob_path=blob_path)
            raise ExternalServiceError(
                "GCS", f"Blob not found: {blob_path}", {"blob_path": blob_path}
            )
        except Exception as e:
            logger.error("gcs_download_failed", blob_path=blob_path, error=str(e))
            raise ExternalServiceError(
                "GCS", f"Download failed: {str(e)}", {"blob_path": blob_path}
            )

    def download_to_file(self, blob_path: str, file_path: str) -> None:
        """Download blob to file."""
        try:
            logger.info("gcs_download_to_file_started", blob_path=blob_path, file_path=file_path)

            blob = self.bucket.blob(blob_path)
            blob.download_to_filename(file_path)

            logger.info("gcs_download_to_file_completed", blob_path=blob_path, file_path=file_path)

        except NotFound:
            logger.error("gcs_blob_not_found", blob_path=blob_path)
            raise ExternalServiceError(
                "GCS", f"Blob not found: {blob_path}", {"blob_path": blob_path}
            )
        except Exception as e:
            logger.error("gcs_download_to_file_failed", blob_path=blob_path, error=str(e))
            raise ExternalServiceError(
                "GCS", f"Download failed: {str(e)}", {"blob_path": blob_path}
            )

    def upload_json(self, blob_path: str, data: dict) -> str:
        """Upload JSON data to GCS."""
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        return self.upload_bytes(blob_path, json_bytes, content_type="application/json")

    def download_json(self, blob_path: str) -> dict:
        """Download JSON data from GCS."""
        data = self.download_bytes(blob_path)
        return json.loads(data.decode("utf-8"))

    def exists(self, blob_path: str) -> bool:
        """Check if blob exists."""
        try:
            blob = self.bucket.blob(blob_path)
            return blob.exists()
        except Exception as e:
            logger.error("gcs_exists_check_failed", blob_path=blob_path, error=str(e))
            return False

    def delete(self, blob_path: str) -> None:
        """Delete blob from GCS."""
        try:
            logger.info("gcs_delete_started", blob_path=blob_path)

            blob = self.bucket.blob(blob_path)
            blob.delete()

            logger.info("gcs_delete_completed", blob_path=blob_path)

        except NotFound:
            logger.warning("gcs_delete_blob_not_found", blob_path=blob_path)
        except Exception as e:
            logger.error("gcs_delete_failed", blob_path=blob_path, error=str(e))
            raise ExternalServiceError("GCS", f"Delete failed: {str(e)}", {"blob_path": blob_path})

    def list_blobs(self, prefix: str) -> list[str]:
        """List blobs with given prefix."""
        try:
            logger.info("gcs_list_blobs_started", prefix=prefix)

            blobs = self.bucket.list_blobs(prefix=prefix)
            blob_names = [blob.name for blob in blobs]

            logger.info("gcs_list_blobs_completed", prefix=prefix, count=len(blob_names))
            return blob_names

        except Exception as e:
            logger.error("gcs_list_blobs_failed", prefix=prefix, error=str(e))
            raise ExternalServiceError("GCS", f"List failed: {str(e)}", {"prefix": prefix})

    def verify_connectivity(self) -> bool:
        """Verify GCS connectivity."""
        try:
            # Try to access the bucket
            _ = self.bucket
            logger.info("gcs_connectivity_verified", bucket_name=self.bucket_name)
            return True
        except Exception as e:
            logger.error("gcs_connectivity_failed", bucket_name=self.bucket_name, error=str(e))
            return False


# Global GCS client instance
_gcs_client: Optional[GCSClient] = None


def get_gcs_client() -> GCSClient:
    """Get or create GCS client instance."""
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = GCSClient()
    return _gcs_client
