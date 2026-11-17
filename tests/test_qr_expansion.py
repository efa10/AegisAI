"""
Tests for QR code processing with shortened URL payloads and safe handling.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.qr.classifier import process_qr


class TestQRExpansionPipeline:
    """Verifies that QR codes containing shortened URLs use the safe pipeline."""

    @patch("src.qr.classifier.decode_qr")
    @patch("src.qr.classifier.predict_url")
    def test_qr_with_shortened_url(self, mock_predict_url, mock_decode_qr):
        """QR payload with shortened URL routes through safe URL pipeline."""
        mock_decode_qr.return_value = "https://tinyurl.com/securelink"

        mock_predict_url.return_value = {
            "original_url": "https://tinyurl.com/securelink",
            "is_shortened": True,
            "shortening_service": "tinyurl.com",
            "expanded_url": "https://destination.org/home",
            "redirect_chain": ["https://tinyurl.com/securelink", "https://destination.org/home"],
            "redirect_count": 1,
            "expansion_success": True,
            "expansion_error": None,
            "ml_verdict": "legitimate",
            "verdict": "legitimate",
            "phishing_probability": 0.12,
            "confidence": 0.88,
            "risk_score": 12.0,
            "risk_indicators": ["Shortened URL detected via tinyurl.com"],
            "threat_intelligence": {"overall_verdict": "clean"},
            "url": "https://tinyurl.com/securelink",
            "label": 0,
            "features": {"url_length": 30},
            "model_name": "xgboost",
        }

        result = process_qr(b"fake_image_bytes")

        assert result["qr_detected"] is True
        assert result["is_url"] is True
        assert result["payload"] == "https://tinyurl.com/securelink"
        assert result["is_shortened"] is True
        assert result["shortening_service"] == "tinyurl.com"
        assert result["expanded_url"] == "https://destination.org/home"
        assert result["redirect_count"] == 1
        assert result["expansion_success"] is True
        assert result["verdict"] == "legitimate"
        assert result["score"] == 0.12
        assert result["confidence"] == 0.88

    @patch("src.qr.classifier.decode_qr")
    def test_qr_plain_text_payload_not_routed_to_network(self, mock_decode_qr):
        """Plain text payloads in QR codes must not trigger network/URL calls."""
        mock_decode_qr.return_value = "WIFI:T:WPA;S:OfficeNet;P:SecretPass;;"

        result = process_qr(b"fake_image_bytes")

        assert result["qr_detected"] is True
        assert result["is_url"] is False
        assert result["payload_type"] == "text"
        assert result["verdict"] == "non_url_payload"
        assert result["score"] is None
