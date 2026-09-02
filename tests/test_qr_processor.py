"""
Unit tests for the QR code processing module.
"""

from unittest.mock import patch, MagicMock
import sys

# Mock cv2 before importing our modules, in case it's not installed
sys.modules['cv2'] = MagicMock()

import numpy as np

from src.qr.detector import decode_qr
from src.qr.classifier import process_qr, is_valid_url


def test_is_valid_url():
    """Test the robust URL regex validator."""
    assert is_valid_url("https://example.com")
    assert is_valid_url("http://192.168.1.1/login")
    assert is_valid_url("https://sub.domain.co.uk/path?q=1")
    
    assert not is_valid_url("Hello World")
    assert not is_valid_url("example.com") # Missing scheme
    assert not is_valid_url("ftp://example.com") # Not http/https


@patch("src.qr.detector.os.path.exists", return_value=True)
@patch("src.qr.detector.cv2.imread")
@patch("src.qr.detector.cv2.QRCodeDetector")
def test_decode_qr_valid(mock_detector_cls, mock_imread, mock_exists):
    """Test valid QR decoding using OpenCV."""
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    
    mock_detector = MagicMock()
    # detectAndDecode returns (data, bbox, straight_qrcode)
    mock_detector.detectAndDecode.return_value = ("https://phishing.com", None, None)
    mock_detector_cls.return_value = mock_detector
    
    payload = decode_qr("fake_path.png")
    assert payload == "https://phishing.com"


@patch("src.qr.detector.os.path.exists", return_value=True)
@patch("src.qr.detector.cv2.imread")
@patch("src.qr.detector.cv2.QRCodeDetector")
def test_decode_qr_fallback(mock_detector_cls, mock_imread, mock_exists):
    """Test fallback to pyzbar if OpenCV fails."""
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    
    mock_detector = MagicMock()
    mock_detector.detectAndDecode.return_value = ("", None, None)
    mock_detector_cls.return_value = mock_detector
    
    # Mock pyzbar
    # Ensure pyzbar_decode exists in the module namespace for patching
    import src.qr.detector as detector
    detector._PYZBAR_AVAILABLE = True
    detector.pyzbar_decode = MagicMock()
    
    mock_obj = MagicMock()
    mock_obj.type = "QRCODE"
    mock_obj.data = b"fallback payload"
    detector.pyzbar_decode.return_value = [mock_obj]
    
    payload = decode_qr("fake_path.png")
    assert payload == "fallback payload"


def test_decode_qr_invalid_input():
    """Test decoding with invalid file paths or empty input."""
    with patch("os.path.exists", return_value=False):
        assert decode_qr("nonexistent.png") is None
    
    assert decode_qr(b"") is None


@patch("src.qr.classifier.decode_qr")
def test_process_qr_no_code(mock_decode):
    """Test processing when no QR code is found."""
    mock_decode.return_value = None
    
    result = process_qr("dummy_path")
    assert result["qr_detected"] is False
    assert result["verdict"] == "qr_not_detected"


@patch("src.qr.classifier.decode_qr")
def test_process_qr_plain_text(mock_decode):
    """Test processing a plain text payload."""
    mock_decode.return_value = "Hello World"
    
    result = process_qr("dummy_path")
    assert result["qr_detected"] is True
    assert result["is_url"] is False
    assert result["payload_type"] == "text"
    assert result["verdict"] == "non_url_payload"
    assert "Payload is not a URL" in result["risk_indicators"][0]


@patch("src.qr.classifier.predict_url")
@patch("src.qr.classifier.decode_qr")
def test_process_qr_url(mock_decode, mock_predict_url):
    """Test processing a URL payload routes to the existing URL model."""
    mock_decode.return_value = "https://phishing.com"
    
    # Mock the existing Phase 3 URL model response
    mock_predict_url.return_value = {
        "verdict": "phishing",
        "phishing_probability": 0.95,
        "features": {
            "has_ip_address": 0,
            "has_https": 1,
            "url_length": 30
        }
    }
    
    result = process_qr("dummy_path")
    
    # Verify predict_url was called with the exact payload
    mock_predict_url.assert_called_once_with("https://phishing.com", model_path=None)
    
    assert result["qr_detected"] is True
    assert result["is_url"] is True
    assert result["payload_type"] == "url"
    assert result["verdict"] == "phishing"
    assert result["score"] == 0.95
    assert "URL classified as highly suspicious/phishing" in result["risk_indicators"]


@patch("src.qr.classifier.predict_url")
@patch("src.qr.classifier.decode_qr")
def test_process_qr_url_features(mock_decode, mock_predict_url):
    """Test that risk indicators are generated correctly from URL features."""
    mock_decode.return_value = "http://192.168.1.1/very/long/path/with/many/subdomains"
    
    mock_predict_url.return_value = {
        "verdict": "legitimate",  # Even if legitimate, features should generate indicators
        "phishing_probability": 0.2,
        "features": {
            "has_ip_address": 1,
            "has_https": 0,
            "url_length": 80,
            "num_subdomains": 3
        }
    }
    
    result = process_qr("dummy_path")
    
    indicators = result["risk_indicators"]
    assert "Uses IP address instead of domain name" in indicators
    assert "Missing HTTPS (HTTP only)" in indicators
    assert "Suspiciously long URL" in indicators
    assert "Excessive subdomains" in indicators
