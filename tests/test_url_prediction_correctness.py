"""
Regression tests for URL prediction correctness, class mapping, and score semantics.
Verifies that legitimate search engine URLs and benign domains are not misclassified
as phishing with 100% confidence, and ensures score semantics are mathematically truthful.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.ml.url_model import predict_url
from config.settings import SETTINGS


class TestPredictionCorrectness:
    """Verifies predictions on legitimate and suspicious inputs."""

    @patch("src.preprocessing.url_expansion.check_host_ssrf")
    def test_bing_search_prediction_semantics(self, mock_ssrf):
        """Regression test: https://www.bing.com/search?q=google must NOT be 100% phishing."""
        mock_ssrf.return_value = (True, "")

        url = "https://www.bing.com/search?q=google"
        res = predict_url(url)

        assert res["ml_verdict"] == "legitimate"
        assert res["verdict"] == "legitimate"
        assert res["label"] == 0
        assert res["phishing_probability"] < 0.50

        # Confidence semantics: for legitimate, confidence = 1.0 - phishing_prob
        expected_conf = round(1.0 - res["phishing_probability"], 5)
        assert abs(res["confidence"] - expected_conf) < 1e-4
        assert res["confidence"] > 0.50

    @pytest.mark.parametrize(
        "legit_url",
        [
            "https://www.google.com/search?q=weather",
            "https://en.wikipedia.org/wiki/Phishing",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.nytimes.com/section/technology",
            "https://www.bbc.com/news",
        ],
    )
    @patch("src.preprocessing.url_expansion.check_host_ssrf")
    def test_multiple_legitimate_urls_classified_correctly(self, mock_ssrf, legit_url):
        """Verifies multiple common benign URLs without hardcoded domain whitelisting."""
        mock_ssrf.return_value = (True, "")

        res = predict_url(legit_url)
        assert res["ml_verdict"] == "legitimate"
        assert res["phishing_probability"] < 0.50
        assert res["confidence"] >= 0.50

    @patch("src.preprocessing.url_expansion.check_host_ssrf")
    def test_synthetic_suspicious_url(self, mock_ssrf):
        """Suspicious synthetic probe with known attack patterns (IP host + target keywords)."""
        mock_ssrf.return_value = (True, "")

        url = "http://paypal-security-update.account-verification-alert.com/login.php"
        res = predict_url(url)

        assert res["ml_verdict"] == "phishing"
        assert res["phishing_probability"] >= 0.50
        assert res["confidence"] == res["phishing_probability"]
        assert res["risk_score"] >= 50.0

    def test_score_and_confidence_semantics(self):
        """Confidence must equal chosen class probability."""
        url = "https://www.bing.com/search?q=google"
        res = predict_url(url)

        p = res["phishing_probability"]
        v = res["ml_verdict"]

        if v == "legitimate":
            assert res["confidence"] == round(1.0 - p, 5)
        else:
            assert res["confidence"] == round(p, 5)

    def test_backward_compatibility_fields_present(self):
        """All legacy fields must remain present for backward compatibility."""
        res = predict_url("https://example.com")

        # Legacy fields
        legacy_keys = ["url", "verdict", "label", "phishing_probability", "features", "model_name"]
        for k in legacy_keys:
            assert k in res, f"Missing legacy backward-compatibility field: {k}"

        # New structured fields
        new_keys = [
            "original_url",
            "is_shortened",
            "shortening_service",
            "expanded_url",
            "redirect_chain",
            "redirect_count",
            "expansion_success",
            "expansion_error",
            "ml_verdict",
            "confidence",
            "risk_score",
            "risk_indicators",
            "threat_intelligence",
        ]
        for k in new_keys:
            assert k in res, f"Missing structured field: {k}"

    def test_feature_names_match_schema(self):
        """Features extracted during inference must strictly match configured schema."""
        res = predict_url("https://example.com/test")
        extracted_keys = list(res["features"].keys())
        assert set(extracted_keys) == set(SETTINGS.URL_FEATURE_NAMES)
