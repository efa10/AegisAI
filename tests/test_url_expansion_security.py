"""
Tests for URL expansion security, SSRF protection, and bounded redirect resolution.
All network and DNS interactions are strictly mocked to avoid external calls.
"""

import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import Timeout, RequestException
import socket

from src.preprocessing.url_expansion import (
    expand_url,
    is_shortened_url,
    is_safe_scheme,
    validate_ip_address,
    check_host_ssrf,
    normalize_url,
)


class TestShortenerDetection:
    """Test shortener service identification."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://tinyurl.com/abc1234", True),
            ("https://bit.ly/xyz987", True),
            ("https://t.co/random123", True),
            ("https://is.gd/short", True),
            ("https://ow.ly/test", True),
            ("https://cutt.ly/link", True),
            ("https://www.google.com/search?q=test", False),
            ("https://www.bing.com/search?q=google", False),
            ("https://github.com", False),
        ],
    )
    def test_is_shortened_url(self, url, expected):
        assert is_shortened_url(url) == expected


class TestSSRFProtection:
    """Test blocking of loopback, private, link-local, and cloud metadata addresses."""

    @pytest.mark.parametrize(
        "ip,is_safe",
        [
            ("127.0.0.1", False),
            ("127.0.0.2", False),
            ("::1", False),
            ("10.0.0.1", False),
            ("10.255.255.255", False),
            ("172.16.0.1", False),
            ("172.31.255.255", False),
            ("192.168.1.1", False),
            ("192.168.0.254", False),
            ("169.254.169.254", False),  # AWS / Cloud metadata
            ("169.254.1.1", False),      # Link-local
            ("0.0.0.0", False),
            ("224.0.0.1", False),        # Multicast
            ("8.8.8.8", True),           # Public DNS
            ("93.184.216.34", True),     # Public IP
            ("1.1.1.1", True),           # Public Cloudflare
        ],
    )
    def test_validate_ip_address(self, ip, is_safe):
        safe, reason = validate_ip_address(ip)
        assert safe == is_safe

    def test_blocked_cloud_metadata_hostname(self):
        safe, reason = check_host_ssrf("metadata.google.internal")
        assert not safe
        assert "Blocked sensitive hostname" in reason

    def test_blocked_localhost_hostname(self):
        safe, reason = check_host_ssrf("localhost")
        assert not safe
        assert "Blocked sensitive hostname" in reason

    @patch("socket.getaddrinfo")
    def test_hostname_resolving_to_private_ip_is_blocked(self, mock_getaddrinfo):
        """DNS rebinding / SSRF probe resolving to private IP must be rejected."""
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.50", 80))
        ]
        safe, reason = check_host_ssrf("evil-internal.attack.com")
        assert not safe
        assert "SSRF blocked" in reason

    @patch("socket.getaddrinfo")
    def test_hostname_resolving_to_public_ip_is_allowed(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        safe, reason = check_host_ssrf("example.com")
        assert safe
        assert reason == ""


class TestSchemeValidation:
    """Test scheme constraints."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("http://example.com", True),
            ("https://example.com/login", True),
            ("file:///etc/passwd", False),
            ("javascript:alert(1)", False),
            ("data:text/html;base64,PHNjcmlwdD4=", False),
            ("ftp://ftp.example.com", False),
            ("gopher://gopher.example.com", False),
        ],
    )
    def test_is_safe_scheme(self, url, expected):
        assert is_safe_scheme(url) == expected

    def test_expand_url_rejects_unsupported_scheme(self):
        res = expand_url("file:///etc/passwd")
        assert not res["expansion_success"]
        assert "Unsupported or unsafe URL scheme" in res["expansion_error"]


class TestBoundedRedirects:
    """Test redirect limits, loop detection, and timeouts."""

    @patch("src.preprocessing.url_expansion.check_host_ssrf")
    @patch("requests.Session.head")
    def test_successful_expansion(self, mock_head, mock_ssrf):
        mock_ssrf.return_value = (True, "")

        # Step 1: tinyurl -> intermediate redirect
        resp1 = MagicMock()
        resp1.status_code = 301
        resp1.headers = {"Location": "https://landing-page.com/target"}

        # Step 2: intermediate -> final
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.headers = {}

        mock_head.side_effect = [resp1, resp2]

        res = expand_url("https://tinyurl.com/xyz123")
        assert res["expansion_success"]
        assert res["expanded_url"] == "https://landing-page.com/target"
        assert res["redirect_count"] == 1
        assert len(res["redirect_chain"]) == 2

    @patch("src.preprocessing.url_expansion.check_host_ssrf")
    @patch("requests.Session.head")
    def test_redirect_loop_detection(self, mock_head, mock_ssrf):
        mock_ssrf.return_value = (True, "")

        resp1 = MagicMock()
        resp1.status_code = 302
        resp1.headers = {"Location": "https://hop-b.com"}

        resp2 = MagicMock()
        resp2.status_code = 302
        resp2.headers = {"Location": "https://hop-a.com"}

        resp3 = MagicMock()
        resp3.status_code = 302
        resp3.headers = {"Location": "https://hop-b.com"}  # Loop!

        mock_head.side_effect = [resp1, resp2, resp3]

        res = expand_url("https://hop-a.com")
        assert not res["expansion_success"]
        assert "Redirect loop detected" in res["expansion_error"]

    @patch("src.preprocessing.url_expansion.check_host_ssrf")
    @patch("requests.Session.head")
    def test_timeout_handling(self, mock_head, mock_ssrf):
        mock_ssrf.return_value = (True, "")
        mock_head.side_effect = Timeout("Connection timed out")

        res = expand_url("https://tinyurl.com/slow")
        assert not res["expansion_success"]
        assert "Connection timed out" in res["expansion_error"]

    @patch("src.preprocessing.url_expansion.check_host_ssrf")
    def test_redirect_to_ssrf_blocked_target(self, mock_ssrf):
        # Initial host is safe, but redirect target is private IP
        mock_ssrf.side_effect = [
            (True, ""),
            (False, "SSRF blocked: internal host resolves to 10.0.0.1"),
        ]

        with patch("requests.Session.head") as mock_head:
            resp1 = MagicMock()
            resp1.status_code = 301
            resp1.headers = {"Location": "http://10.0.0.1/admin"}
            mock_head.return_value = resp1

            res = expand_url("https://tinyurl.com/exploit")
            assert not res["expansion_success"]
            assert "SSRF blocked" in res["expansion_error"]
