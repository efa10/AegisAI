"""
AegisAI — QR/Quishing Classifier

Orchestrates QR code processing. It extracts the payload from an image,
classifies the payload type, and safely routes URLs to the existing
Phase 3 URL classifier without ever visiting or requesting the URL.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import numpy as np

from src.qr.detector import decode_qr
from src.ml.url_model import predict_url


# A robust regex for identifying if a string is a URL.
# This checks for a scheme (http/https) and a valid-looking host/path structure.
URL_REGEX = re.compile(
    r"^(?:http|https)://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain
    r"localhost|"  # localhost
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # or ipv4
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE
)

def is_valid_url(payload: str) -> bool:
    """Check if the payload is a valid HTTP/HTTPS URL."""
    payload = payload.strip()
    return bool(URL_REGEX.match(payload))


def process_qr(
    image_source: str | Path | bytes | np.ndarray,
    url_model_path: Path | None = None
) -> dict[str, Any]:
    """Process a QR code image and classify its payload safely.

    Parameters
    ----------
    image_source : str | Path | bytes | np.ndarray
        The input image containing a QR code.
    url_model_path : Path | None
        Path to the saved URL model. Defaults to standard location.

    Returns
    -------
    dict[str, Any]
        A structured result describing the QR code payload, its type, and risk.
    """
    # 1. Decode the payload
    payload = decode_qr(image_source)

    if not payload:
        return {
            "qr_detected": False,
            "payload": None,
            "payload_type": None,
            "is_url": False,
            "verdict": "qr_not_detected",
            "score": None,
            "risk_indicators": []
        }

    # 2. Determine payload type
    if is_valid_url(payload):
        # 3. Route to existing URL classifier
        # IMPORTANT: This safely extracts features as strings without making HTTP requests.
        url_prediction = predict_url(payload, model_path=url_model_path)
        
        if url_prediction.get("verdict") == "invalid":
            return {
                "qr_detected": True,
                "payload": payload,
                "payload_type": "url",
                "is_url": True,
                "verdict": "error",
                "score": None,
                "risk_indicators": ["Invalid URL format processing error."]
            }
        
        # Build safe risk indicators based on existing URL features
        features = url_prediction.get("features", {})
        indicators = []
        if features.get("url_length", 0) > 75:
            indicators.append("Suspiciously long URL")
        if features.get("has_ip_address", 0) == 1:
            indicators.append("Uses IP address instead of domain name")
        if features.get("num_subdomains", 0) > 2:
            indicators.append("Excessive subdomains")
        if features.get("has_https", 0) == 0:
            indicators.append("Missing HTTPS (HTTP only)")

        if url_prediction["verdict"] == "phishing":
            indicators.insert(0, "URL classified as highly suspicious/phishing")

        return {
            "qr_detected": True,
            "payload": payload,
            "payload_type": "url",
            "is_url": True,
            "verdict": url_prediction["verdict"],
            "score": url_prediction["phishing_probability"],
            "confidence": url_prediction.get("confidence"),
            "risk_score": url_prediction.get("risk_score"),
            "is_shortened": url_prediction.get("is_shortened", False),
            "shortening_service": url_prediction.get("shortening_service"),
            "expanded_url": url_prediction.get("expanded_url", payload),
            "redirect_chain": url_prediction.get("redirect_chain", [payload]),
            "redirect_count": url_prediction.get("redirect_count", 0),
            "expansion_success": url_prediction.get("expansion_success", False),
            "expansion_error": url_prediction.get("expansion_error"),
            "risk_indicators": url_prediction.get("risk_indicators", indicators),
            "threat_intelligence": url_prediction.get("threat_intelligence"),
        }
    else:
        # 4. Handle non-URL text payloads
        indicators = ["Payload is not a URL (plain text)"]
        if len(payload) < 5:
            indicators.append("Very short text payload")

        return {
            "qr_detected": True,
            "payload": payload,
            "payload_type": "text",
            "is_url": False,
            "verdict": "non_url_payload",
            "score": None,
            "risk_indicators": indicators
        }
