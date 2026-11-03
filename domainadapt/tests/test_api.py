"""Unit tests for FastAPI serving endpoints and middleware."""

from fastapi.testclient import TestClient
from domainadapt.api.main import app, state

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "device" in data
    assert "X-Correlation-ID" in response.headers


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "total_parameters" in data
    assert data["total_parameters"] > 0


def test_predict_validation_error():
    # Empty instruction should trigger 422 Unprocessable Entity
    response = client.post("/predict", json={"instruction": ""})
    assert response.status_code == 422


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0.0


def test_correlation_id_propagation():
    custom_id = "test-corr-id-12345"
    response = client.get("/health", headers={"X-Correlation-ID": custom_id})
    assert response.headers.get("X-Correlation-ID") == custom_id
