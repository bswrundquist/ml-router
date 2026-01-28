"""Application configuration using Pydantic Settings."""

from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application
    app_name: str = "ml-api"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://localhost:5432/mlapi",
        description="PostgreSQL connection string",
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Google Cloud Storage
    gcs_bucket: str = Field(default="", description="GCS bucket for artifacts")
    gcs_project_id: str = Field(default="", description="GCP project ID")
    google_application_credentials: str = Field(
        default="", description="Path to GCP service account key"
    )

    # Redis (for Arq background jobs)
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 10

    # Job Queue
    job_queue_enabled: bool = True
    job_timeout_seconds: int = 7200
    max_concurrent_jobs: int = 3

    # Model defaults
    default_model_type: Literal["catboost", "xgboost", "lightgbm"] = "catboost"
    default_classification_metric: str = "auc"
    default_regression_metric: str = "rmse"

    # Optuna defaults
    default_optuna_n_trials: int = 50
    default_optuna_timeout_seconds: int = 3600
    default_optuna_sampler: str = "TPE"
    default_optuna_pruner: str = "MedianPruner"

    # Data
    max_upload_size_mb: int = 500
    default_random_seed: int = 42

    # Observability
    enable_metrics: bool = True
    metrics_port: int = 9090
    sentry_dsn: str = ""

    # Security
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"
    cors_enabled: bool = True

    @field_validator("allowed_origins")
    @classmethod
    def parse_origins(cls, v: str) -> list[str]:
        """Parse comma-separated origins into list."""
        return [origin.strip() for origin in v.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"


# Global settings instance
settings = Settings()
