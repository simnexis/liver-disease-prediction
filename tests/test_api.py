"""Integration tests for FastAPI Liver Disease Risk Prediction service."""

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized model artifacts."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test that the root endpoint returns 200 and points to docs."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "docs_url" in data


def test_health_check(client):
    """Test /health endpoint reports healthy and model_loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "screening_threshold" in data


def test_model_info_endpoint(client):
    """Test /api/v1/model-info endpoint returns metadata and validation stats."""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "chosen_model" in data
    assert "held_out_test_metrics" in data


def test_predict_single_healthy(client):
    """Test /api/v1/predict with a typical healthy profile."""
    payload = {
        "age": 30,
        "gender": "Female",
        "total_bilirubin": 0.8,
        "direct_bilirubin": 0.2,
        "alkaline_phosphatase": 130,
        "alamine_aminotransferase": 20,
        "aspartate_aminotransferase": 22,
        "total_proteins": 7.2,
        "albumin": 4.1,
        "albumin_globulin_ratio": 1.3,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["screening_prediction"] in [0, 1]
    assert data["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]


def test_predict_single_high_risk(client):
    """Test /api/v1/predict with severe liver biomarker elevations."""
    payload = {
        "age": 58,
        "gender": "Male",
        "total_bilirubin": 6.8,
        "direct_bilirubin": 3.4,
        "alkaline_phosphatase": 420,
        "alamine_aminotransferase": 180,
        "aspartate_aminotransferase": 210,
        "total_proteins": 5.4,
        "albumin": 2.3,
        "albumin_globulin_ratio": 0.7,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_probability"] > 0.60
    assert data["screening_prediction"] == 1
    assert data["is_screen_positive"] is True


def test_predict_single_missing_ratio(client):
    """Test /api/v1/predict handles omitted albumin_globulin_ratio via median imputation."""
    payload = {
        "age": 42,
        "gender": "Male",
        "total_bilirubin": 1.0,
        "direct_bilirubin": 0.3,
        "alkaline_phosphatase": 160,
        "alamine_aminotransferase": 35,
        "aspartate_aminotransferase": 40,
        "total_proteins": 6.5,
        "albumin": 3.4,
        "albumin_globulin_ratio": None,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data


def test_predict_validation_error(client):
    """Test that out-of-range inputs trigger Pydantic 422 Unprocessable Entity."""
    # Age < 1
    payload = {
        "age": -5,
        "gender": "Male",
        "total_bilirubin": 1.0,
        "direct_bilirubin": 0.3,
        "alkaline_phosphatase": 160,
        "alamine_aminotransferase": 35,
        "aspartate_aminotransferase": 40,
        "total_proteins": 6.5,
        "albumin": 3.4,
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch(client):
    """Test /api/v1/predict/batch endpoint with multiple patient cases."""
    patients = [
        {
            "age": 28,
            "gender": "Female",
            "total_bilirubin": 0.7,
            "direct_bilirubin": 0.1,
            "alkaline_phosphatase": 110,
            "alamine_aminotransferase": 18,
            "aspartate_aminotransferase": 20,
            "total_proteins": 7.0,
            "albumin": 4.0,
            "albumin_globulin_ratio": 1.33,
        },
        {
            "age": 62,
            "gender": "Male",
            "total_bilirubin": 4.5,
            "direct_bilirubin": 2.2,
            "alkaline_phosphatase": 310,
            "alamine_aminotransferase": 95,
            "aspartate_aminotransferase": 115,
            "total_proteins": 5.8,
            "albumin": 2.8,
            "albumin_globulin_ratio": 0.93,
        },
    ]
    response = client.post("/api/v1/predict/batch", json={"patients": patients})
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 2
    assert len(data["predictions"]) == 2
    assert data["positive_screens"] + data["negative_screens"] == 2
