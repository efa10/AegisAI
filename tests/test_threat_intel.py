"""
Tests for Threat Intelligence integrations (VirusTotal, URLhaus, and PhishTank).
All network requests are mocked to ensure deterministic, isolated execution.
"""

import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import Timeout

from src.threat_intel.threat_intel import (
    check_virustotal,
    check_urlhaus,
    check_phishtank,
    check_threat_intelligence,
)


class TestVirusTotal:
    """Tests for VirusTotal threat intelligence feed."""

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("VIRUSTOTAL_API_KEY", raising=False)
        result = check_virustotal("https://example.com")
        assert result["status"] == "not_configured"
        assert result["verdict"] == "unknown"

    @patch("requests.get")
    def test_virustotal_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = check_virustotal("https://fresh-domain.com", api_key="dummy_key")
        assert result["status"] == "available"
        assert result["verdict"] == "unknown"
        assert "not previously scanned" in result["detail"]

    @patch("requests.get")
    def test_virustotal_rate_limit(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        result = check_virustotal("https://example.com", api_key="dummy_key")
        assert result["status"] == "unavailable"
        assert "rate limit exceeded" in result["detail"]

    @patch("requests.get")
    def test_virustotal_flagged_malicious(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "attributes": {
                    "last_analysis_stats": {
                        "malicious": 14,
                        "suspicious": 3,
                        "harmless": 50,
                    }
                }
            }
        }
        mock_get.return_value = mock_resp

        result = check_virustotal("http://bad-site.com", api_key="dummy_key")
        assert result["status"] == "available"
        assert result["verdict"] == "malicious"
        assert result["positives"] == 17

    @patch("requests.get")
    def test_virustotal_timeout(self, mock_get):
        mock_get.side_effect = Timeout("VT timeout")
        result = check_virustotal("https://example.com", api_key="dummy_key")
        assert result["status"] == "unavailable"
        assert "timed out" in result["detail"]


class TestURLhaus:
    """Tests for URLhaus threat intelligence feed."""

    @patch("requests.post")
    def test_urlhaus_clean_result(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"query_status": "no_results"}
        mock_post.return_value = mock_resp

        result = check_urlhaus("https://safe-domain.org")
        assert result["status"] == "available"
        assert result["verdict"] == "clean"

    @patch("requests.post")
    def test_urlhaus_malicious_result(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "query_status": "ok",
            "threat": "malware_download",
            "url_status": "online",
        }
        mock_post.return_value = mock_resp

        result = check_urlhaus("http://payload-distribution.com/malware.exe")
        assert result["status"] == "available"
        assert result["verdict"] == "malicious"

    @patch("requests.post")
    def test_urlhaus_timeout(self, mock_post):
        mock_post.side_effect = Timeout("URLhaus timeout")
        result = check_urlhaus("https://example.com")
        assert result["status"] == "unavailable"
        assert "timed out" in result["detail"]


class TestThreatIntelligenceAggregator:
    """Tests for the multi-source aggregator."""

    @patch("src.threat_intel.threat_intel.check_virustotal")
    @patch("src.threat_intel.threat_intel.check_urlhaus")
    @patch("src.threat_intel.threat_intel.check_phishtank")
    def test_overall_verdict_malicious_if_any_feed_flags(
        self, mock_pt, mock_uh, mock_vt
    ):
        mock_vt.return_value = {"status": "available", "verdict": "malicious"}
        mock_uh.return_value = {"status": "available", "verdict": "clean"}
        mock_pt.return_value = {"status": "available", "verdict": "not_listed"}

        report = check_threat_intelligence("http://flagged-site.com")
        assert report["overall_verdict"] == "malicious"

    @patch("src.threat_intel.threat_intel.check_virustotal")
    @patch("src.threat_intel.threat_intel.check_urlhaus")
    @patch("src.threat_intel.threat_intel.check_phishtank")
    def test_overall_verdict_clean(self, mock_pt, mock_uh, mock_vt):
        mock_vt.return_value = {"status": "available", "verdict": "clean"}
        mock_uh.return_value = {"status": "available", "verdict": "clean"}
        mock_pt.return_value = {"status": "available", "verdict": "not_listed"}

        report = check_threat_intelligence("https://clean-site.com")
        assert report["overall_verdict"] == "clean"
