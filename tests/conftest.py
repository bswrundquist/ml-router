"""Pytest configuration - minimal setup."""

import os

# Set minimal environment variables for tests
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost:5432/mlapi_test")
os.environ.setdefault("GCS_BUCKET", "test-bucket")
os.environ.setdefault("GCS_PROJECT_ID", "test-project")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/fake-gcp-creds.json")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("ENVIRONMENT", "development")

# Create fake GCP credentials file
import json

fake_creds = {
    "type": "service_account",
    "project_id": "test-project",
}
with open("/tmp/fake-gcp-creds.json", "w") as f:
    json.dump(fake_creds, f)
