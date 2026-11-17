"""
AegisAI — Threat Intelligence Aggregator

Integrates external threat intelligence feeds (VirusTotal, URLhaus, and local PhishTank).
- API keys are retrieved server-side from environment variables.
- Timeouts, rate limits, network failures, or missing keys NEVER crash local ML prediction.
- Fails safely by reporting 'unavailable' or 'not_configured', never as an untrue clean bill of health.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any
import requests
from config.settings import SETTINGS

VIRUSTOTAL_TIMEOUT = 2.5
URLHAUS_TIMEOUT = 2.5

# In-memory cache for local PhishTank database lookup
_phishtank_url_set: set[str] | None = None


def get_phishtank_set() -> set[str]:
    """Lazily load local verified PhishTank database into memory set for sub-millisecond lookup."""
    global _phishtank_url_set
    if _phishtank_url_set is not None:
        return _phishtank_url_set

    _phishtank_url_set = set()
    phish_path = SETTINGS.DATA_RAW_DIR / "verified_online.csv"
    if phish_path.exists():
        try:
            import pandas as pd
            # Only read the 'url' column to minimize memory
            df = pd.read_csv(phish_path, usecols=["url"], nrows=100000)
            _phishtank_url_set = set(df["url"].dropna().astype(str).str.strip().str.lower())
        except Exception:
            _phishtank_url_set = set()
    return _phishtank_url_set


def check_virustotal(url: str, api_key: str | None = None) -> dict[str, Any]:
    """Query VirusTotal v3 API for URL analysis.

    Requires VIRUSTOTAL_API_KEY environment variable if api_key not passed.
    """
    key = api_key or os.environ.get("VIRUSTOTAL_API_KEY")
    if not key:
        return {
            "status": "not_configured",
            "verdict": "unknown",
            "detail": "VIRUSTOTAL_API_KEY environment variable not configured",
        }

    try:
        # Base64 URL-safe encode without padding as per VirusTotal API v3 spec
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        headers = {"x-apikey": key, "User-Agent": "AegisAI-Scanner/1.0"}

        response = requests.get(endpoint, headers=headers, timeout=VIRUSTOTAL_TIMEOUT)

        if response.status_code == 404:
            return {
                "status": "available",
                "verdict": "unknown",
                "positives": 0,
                "total": 0,
                "detail": "URL not previously scanned in VirusTotal dataset",
            }
        elif response.status_code == 401 or response.status_code == 403:
            return {
                "status": "unavailable",
                "verdict": "unknown",
                "detail": "Invalid or unauthorized VirusTotal API key",
            }
        elif response.status_code == 429:
            return {
                "status": "unavailable",
                "verdict": "unknown",
                "detail": "VirusTotal API rate limit exceeded",
            }
        elif response.status_code == 200:
            data = response.json()
            stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            malicious = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            harmless = stats.get("harmless", 0)
            total = sum(stats.values()) if stats else 0

            positives = malicious + suspicious
            verdict = "malicious" if positives > 0 else ("clean" if harmless > 0 else "unknown")

            return {
                "status": "available",
                "verdict": verdict,
                "positives": positives,
                "total": total,
                "malicious_count": malicious,
                "suspicious_count": suspicious,
                "detail": f"{positives}/{total} security vendors flagged this URL",
            }
        else:
            return {
                "status": "unavailable",
                "verdict": "unknown",
                "detail": f"VirusTotal returned HTTP {response.status_code}",
            }
    except requests.Timeout:
        return {
            "status": "unavailable",
            "verdict": "unknown",
            "detail": "VirusTotal API request timed out",
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "verdict": "unknown",
            "detail": f"VirusTotal lookup error: {str(e)}",
        }


def check_urlhaus(url: str, api_key: str | None = None) -> dict[str, Any]:
    """Query URLhaus (abuse.ch) API for malware/phishing URL intelligence."""
    # URLhaus public API does not strictly require an auth key for URL lookups,
    # but checks URLHAUS_API_KEY if provided for authenticated rate tiers.
    key = api_key or os.environ.get("URLHAUS_API_KEY")

    # If URLHAUS_ENABLED is set to false or not configured and user wants offline:
    # URLhaus query:
    try:
        endpoint = "https://urlhaus-api.abuse.ch/v1/url/"
        data = {"url": url}
        headers = {"User-Agent": "AegisAI-Scanner/1.0"}
        if key:
            headers["Auth-Key"] = key

        response = requests.post(endpoint, data=data, headers=headers, timeout=URLHAUS_TIMEOUT)

        if response.status_code == 200:
            res_json = response.json()
            status = res_json.get("query_status")
            if status == "ok":
                url_status = res_json.get("url_status", "unknown")
                threat = res_json.get("threat", "malware_url")
                return {
                    "status": "available",
                    "verdict": "malicious",
                    "threat": threat,
                    "url_status": url_status,
                    "detail": f"Active threat listed on URLhaus ({threat})",
                }
            elif status == "no_results":
                return {
                    "status": "available",
                    "verdict": "clean",
                    "detail": "No malicious reports on URLhaus",
                }
            else:
                return {
                    "status": "available",
                    "verdict": "unknown",
                    "detail": f"URLhaus query status: {status}",
                }
        else:
            return {
                "status": "unavailable",
                "verdict": "unknown",
                "detail": f"URLhaus returned HTTP {response.status_code}",
            }
    except requests.Timeout:
        return {
            "status": "unavailable",
            "verdict": "unknown",
            "detail": "URLhaus API request timed out",
        }
    except Exception as e:
        return {
            "status": "unavailable",
            "verdict": "unknown",
            "detail": f"URLhaus lookup error: {str(e)}",
        }


def check_phishtank(url: str) -> dict[str, Any]:
    """Check if URL exists in the local verified PhishTank database."""
    try:
        phishtank_set = get_phishtank_set()
        if not phishtank_set:
            return {
                "status": "unavailable",
                "verdict": "unknown",
                "detail": "Local PhishTank database not loaded or empty",
            }

        url_norm = url.strip().lower()
        if url_norm in phishtank_set:
            return {
                "status": "available",
                "verdict": "phishing",
                "in_database": True,
                "detail": "Verified phishing URL in PhishTank feed",
            }
        else:
            return {
                "status": "available",
                "verdict": "not_listed",
                "in_database": False,
                "detail": "Not listed in local PhishTank verified database",
            }
    except Exception as e:
        return {
            "status": "unavailable",
            "verdict": "unknown",
            "detail": f"PhishTank lookup error: {str(e)}",
        }


def check_threat_intelligence(url: str) -> dict[str, Any]:
    """Aggregate all threat intelligence sources safely.

    Never blocks ML classification. Returns unified report with individual
    source results.
    """
    vt_result = check_virustotal(url)
    uh_result = check_urlhaus(url)
    pt_result = check_phishtank(url)

    # Determine overall threat intel verdict
    if (
        vt_result.get("verdict") == "malicious"
        or uh_result.get("verdict") == "malicious"
        or pt_result.get("verdict") == "phishing"
    ):
        overall = "malicious"
    elif (
        vt_result.get("verdict") == "clean"
        or uh_result.get("verdict") == "clean"
    ):
        overall = "clean"
    else:
        overall = "unknown"

    return {
        "overall_verdict": overall,
        "virustotal": vt_result,
        "urlhaus": uh_result,
        "phishtank": pt_result,
    }
