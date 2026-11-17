"""
AegisAI — Secure URL Expansion Module

Resolves shortened URLs securely with strict SSRF protections:
- Validates URL schemes (HTTP/HTTPS only).
- Protects against SSRF and DNS rebinding by resolving hostnames and blocking
  private, loopback, link-local, multicast, and cloud metadata addresses.
- Follows bounded redirects (max depth 5) using short timeouts (3.0s).
- Enforces response size limits and closes streams immediately without downloading page content.
- Never executes scripts, submits forms, or opens URLs in a browser.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse, urljoin
import requests
from requests.exceptions import RequestException, Timeout, TooManyRedirects

SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "ow.ly",
    "buff.ly",
    "rebrand.ly",
    "shorturl.at",
    "cutt.ly",
    "tiny.cc",
    "qr.net",
    "goo.gl",
    "tr.im",
    "v.gd",
    "clck.ru",
    "bl.ink",
}

MAX_REDIRECTS = 5
REQUEST_TIMEOUT = 3.0  # seconds
MAX_RESPONSE_BYTES = 10 * 1024  # 10 KB limit for response headers/metadata
USER_AGENT = "AegisAI-Security-Scanner/1.0 (Phishing Detection Analyzer)"

# Explicit list of dangerous cloud metadata hostnames
BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "instance-data",
    "169.254.169.254",
    "localhost",
}


def normalize_url(url: str) -> str:
    """Normalize URL by stripping whitespace and adding default scheme if missing."""
    if not isinstance(url, str):
        url = str(url)
    url = url.strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://", "ftp://", "file://", "javascript:", "data:")):
        url = "https://" + url
    return url


def get_domain(url: str) -> str:
    """Extract normalized lowercase domain without 'www.'."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        hostname = hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


def is_shortened_url(url: str) -> bool:
    """Check if the URL belongs to a known URL shortener service."""
    domain = get_domain(url)
    return domain in SHORTENER_DOMAINS


def is_safe_scheme(url: str) -> bool:
    """Check if the scheme is strictly HTTP or HTTPS."""
    try:
        parsed = urlparse(url)
        return parsed.scheme.lower() in ("http", "https")
    except Exception:
        return False


def validate_ip_address(ip_str: str) -> tuple[bool, str]:
    """Validate that an IP address is public and not private/loopback/cloud metadata.

    Returns (is_safe, reason).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False, f"Invalid IP address format: {ip_str}"

    if ip.is_loopback:
        return False, f"Blocked loopback IP ({ip_str})"
    if ip.is_private:
        return False, f"Blocked private IP address ({ip_str})"
    if ip.is_link_local:
        return False, f"Blocked link-local address ({ip_str})"
    if ip.is_multicast:
        return False, f"Blocked multicast address ({ip_str})"
    if ip.is_reserved:
        return False, f"Blocked reserved IP address ({ip_str})"
    if ip.is_unspecified:
        return False, f"Blocked unspecified address ({ip_str})"

    # Explicit cloud metadata check (169.254.169.254)
    if str(ip) == "169.254.169.254":
        return False, "Blocked cloud metadata endpoint (169.254.169.254)"

    return True, ""


def check_host_ssrf(hostname: str) -> tuple[bool, str]:
    """Resolve hostname via DNS and ensure all resolved IP addresses are public.

    Protects against DNS rebinding and internal network reconnaissance.
    """
    if not hostname:
        return False, "Empty hostname"

    hostname_lower = hostname.lower().strip("[]")
    if hostname_lower in BLOCKED_HOSTNAMES:
        return False, f"Blocked sensitive hostname: {hostname}"

    # Try direct IP check first if hostname is an IP literal
    try:
        ip = ipaddress.ip_address(hostname_lower)
        return validate_ip_address(str(ip))
    except ValueError:
        pass

    # Resolve via DNS
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        if not addr_info:
            return False, f"Unable to resolve hostname: {hostname}"

        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            is_safe, reason = validate_ip_address(ip_str)
            if not is_safe:
                return False, f"SSRF blocked: {hostname} resolved to forbidden IP {ip_str} ({reason})"

        return True, ""
    except socket.gaierror as e:
        return False, f"DNS resolution failed for {hostname}: {e}"
    except Exception as e:
        return False, f"DNS lookup error for {hostname}: {str(e)}"


def expand_url(
    url: str,
    max_redirects: int = MAX_REDIRECTS,
    timeout: float = REQUEST_TIMEOUT,
    verify_ssrf: bool = True,
) -> dict[str, Any]:
    """Securely resolve shortened URLs by following HTTP redirects.

    Enforces SSRF prevention, scheme validation, loop detection, and depth limits.

    Parameters
    ----------
    url : str
        The original input URL.
    max_redirects : int
        Maximum number of redirects to follow (default 5).
    timeout : float
        Timeout in seconds for network requests (default 3.0s).
    verify_ssrf : bool
        Whether to perform strict IP/DNS SSRF validation.

    Returns
    -------
    dict[str, Any]
        Structured expansion result containing:
        - original_url: submitted URL
        - is_shortened: whether original URL matches shortener pattern
        - shortening_service: detected shortening service name or None
        - expanded_url: resolved destination URL
        - redirect_chain: list of traversed URLs
        - redirect_count: number of redirect hops
        - expansion_success: True if expansion completed safely
        - expansion_error: error description or None
        - shortener_service: alias for backward compatibility
        - error: alias for backward compatibility
    """
    raw_input = url
    normalized = normalize_url(url)

    shortened = is_shortened_url(normalized)
    service = get_domain(normalized) if shortened else None

    result: dict[str, Any] = {
        "original_url": raw_input,
        "is_shortened": shortened,
        "shortening_service": service,
        "expanded_url": normalized,
        "redirect_chain": [normalized],
        "redirect_count": 0,
        "expansion_success": False,
        "expansion_error": None,
        # Backward compatibility fields
        "shortener_service": service,
        "error": None,
    }

    if not normalized:
        result["expansion_error"] = "Empty or invalid URL"
        result["error"] = result["expansion_error"]
        return result

    if not is_safe_scheme(normalized):
        result["expansion_error"] = f"Unsupported or unsafe URL scheme: {urlparse(normalized).scheme}"
        result["error"] = result["expansion_error"]
        return result

    current_url = normalized
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT}

    for step in range(max_redirects):
        parsed = urlparse(current_url)
        hostname = parsed.hostname or ""

        if verify_ssrf:
            safe_host, ssrf_reason = check_host_ssrf(hostname)
            if not safe_host:
                result["expansion_error"] = ssrf_reason
                result["error"] = ssrf_reason
                result["expansion_success"] = False
                return result

        try:
            # First attempt HEAD request to avoid body downloads
            response = session.head(
                current_url,
                headers=headers,
                allow_redirects=False,
                timeout=timeout,
            )

            # Fallback to GET with stream=True if server rejects HEAD
            if response.status_code in (405, 403, 501):
                response = session.get(
                    current_url,
                    headers=headers,
                    allow_redirects=False,
                    timeout=timeout,
                    stream=True,
                )
                # Read at most MAX_RESPONSE_BYTES and close immediately
                _ = response.raw.read(MAX_RESPONSE_BYTES)
                response.close()

            # Check for redirect (3xx)
            if 300 <= response.status_code < 400 and "Location" in response.headers:
                location = response.headers["Location"].strip()
                next_url = urljoin(current_url, location)

                if not is_safe_scheme(next_url):
                    result["expansion_error"] = f"Redirected to unsupported scheme: {next_url}"
                    result["error"] = result["expansion_error"]
                    break

                if next_url in result["redirect_chain"]:
                    result["expansion_error"] = "Redirect loop detected"
                    result["error"] = result["expansion_error"]
                    break

                result["redirect_chain"].append(next_url)
                current_url = next_url
                result["expanded_url"] = current_url
                result["redirect_count"] = len(result["redirect_chain"]) - 1
            else:
                # Terminal destination reached
                result["expansion_success"] = True
                break

        except Timeout:
            result["expansion_error"] = f"Connection timed out after {timeout}s"
            result["error"] = result["expansion_error"]
            break
        except TooManyRedirects:
            result["expansion_error"] = "Exceeded maximum redirects limit"
            result["error"] = result["expansion_error"]
            break
        except RequestException as e:
            result["expansion_error"] = f"HTTP request error: {str(e)}"
            result["error"] = result["expansion_error"]
            break
        except Exception as e:
            result["expansion_error"] = f"Unexpected expansion error: {str(e)}"
            result["error"] = result["expansion_error"]
            break
    else:
        # Loop finished without breaking
        if result["redirect_count"] >= max_redirects:
            result["expansion_error"] = f"Exceeded maximum redirect depth of {max_redirects}"
            result["error"] = result["expansion_error"]

    if not result["expansion_error"]:
        result["expansion_success"] = True

    return result
