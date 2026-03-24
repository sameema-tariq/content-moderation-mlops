"""Tests for the FastAPI prediction endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Start the app with lifespan (loads model) and yield a test client."""
    with TestClient(app) as c:
        yield c


def test_health(client):
    """GET /health should return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_spam(client):
    """POST /predict with spam text should return label spam and valid confidence."""
    response = client.post("/predict", json={"text": "Congratulations! You won a free prize. Call now!"})
    assert response.status_code == 200
    assert response.json()["label"] == "spam"
    assert 0 < response.json()["confidence"] <= 1


def test_predict_ham(client):
    """POST /predict with ham text should return label ham."""
    response = client.post("/predict", json={"text": "Hey, are we still meeting for lunch today?"})
    assert response.status_code == 200
    assert response.json()["label"] == "ham"


def test_predict_empty_text(client):
    """POST /predict with empty string should return 422 validation error."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
