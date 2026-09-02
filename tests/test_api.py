"""
Unit tests for the FastAPI backend.
"""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import sys
sys.modules['cv2'] = MagicMock()
sys.modules['shap'] = MagicMock()

from src.api.main import app
from config.settings import SETTINGS

client = TestClient(app)


def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "AegisAI API is running"}


@patch("src.api.main.predict_url")
def test_predict_url_valid(mock_predict_url):
    """Test /predict/url with valid input."""
    mock_predict_url.return_value = {
        "url": "https://example.com",
        "verdict": "legitimate",
        "label": 0,
        "phishing_probability": 0.05,
        "features": {},
        "model_name": "random_forest"
    }

    response = client.post("/predict/url", json={"url": "https://example.com"})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "legitimate"
    assert data["phishing_probability"] == 0.05
    mock_predict_url.assert_called_once_with("https://example.com")


@patch("src.api.main.predict_url")
def test_predict_url_invalid(mock_predict_url):
    """Test /predict/url with empty string handled by logic."""
    mock_predict_url.return_value = {
        "verdict": "invalid",
        "error": "Empty or invalid URL provided."
    }

    # "  " is technically length >= 1 but is empty after strip
    response = client.post("/predict/url", json={"url": "   "})
    assert response.status_code == 400
    assert "Empty or invalid URL" in response.json()["detail"]


def test_predict_url_validation_error():
    """Test /predict/url with Pydantic validation error."""
    # Min length is 1, so empty string should fail validation immediately
    response = client.post("/predict/url", json={"url": ""})
    assert response.status_code == 422


@patch("src.api.main.predict_text")
def test_predict_text_valid(mock_predict_text):
    """Test /predict/text with valid input."""
    mock_predict_text.return_value = {
        "text": "Win a free iPhone!",
        "verdict": "phishing",
        "label": 1,
        "phishing_probability": 0.99,
        "cleaned_text": "win free iphone",
        "model_name": "linear_svm"
    }

    response = client.post("/predict/text", json={"text": "Win a free iPhone!"})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "phishing"
    assert data["phishing_probability"] == 0.99


@patch("src.api.main.process_qr")
def test_predict_qr_valid_image(mock_process_qr):
    """Test /predict/qr with a valid image file mock."""
    mock_process_qr.return_value = {
        "qr_detected": True,
        "payload": "https://phishing.com",
        "payload_type": "url",
        "is_url": True,
        "verdict": "phishing",
        "score": 0.95,
        "risk_indicators": ["URL classified as highly suspicious/phishing"]
    }

    # Simulate an image upload
    files = {"file": ("test.png", b"dummy_image_bytes", "image/png")}
    response = client.post("/predict/qr", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["qr_detected"] is True
    assert data["verdict"] == "phishing"
    
    # Verify process_qr was called with bytes
    mock_process_qr.assert_called_once_with(b"dummy_image_bytes")


def test_predict_qr_invalid_content_type():
    """Test /predict/qr rejects non-image files."""
    files = {"file": ("test.txt", b"hello world", "text/plain")}
    response = client.post("/predict/qr", files=files)
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]


@patch("src.api.main.explain_url")
def test_explain_url_valid(mock_explain_url):
    """Test /explain/url with valid input."""
    mock_explain_url.return_value = {
        "prediction": "PHISHING",
        "confidence": 0.94,
        "top_features": ["has_ip_address"],
        "feature_contributions": {"has_ip_address": 0.3},
        "reasons": ["IP address detected instead of domain name"]
    }

    response = client.post("/explain/url", json={"url": "http://192.168.1.1"})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "PHISHING"
    assert "IP address detected" in data["reasons"][0]


@patch("src.api.main.explain_url")
def test_explain_url_missing_model(mock_explain_url):
    """Test /explain/url handles missing model error gracefully."""
    mock_explain_url.return_value = {
        "prediction": "error",
        "error": "Model failed to load."
    }

    response = client.post("/explain/url", json={"url": "http://192.168.1.1"})
    assert response.status_code == 503
    assert "Model failed to load" in response.json()["detail"]
