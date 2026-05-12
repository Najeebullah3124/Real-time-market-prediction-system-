"""Fast, offline-safe checks for CI (no live API, no Docker)."""
from fastapi.testclient import TestClient

from app import app


def test_health_returns_json():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "model_loaded" in body
    assert "scaler_loaded" in body
