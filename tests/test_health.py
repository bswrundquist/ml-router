"""Tests for health check endpoints."""

from fastapi.testclient import TestClient
from ml_api.main import app

# Create a simple test client without database dependencies
client = TestClient(app)


def test_healthz():
    """Test basic health check."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "ml-api"


def test_app_version():
    """Test version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "environment" in data
