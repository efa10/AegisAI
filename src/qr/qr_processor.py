"""
AegisAI — QR Code Processor

Decodes QR codes from images using OpenCV and pyzbar, then delegates the
extracted URL to the URL phishing classifier.

Usage (after implementation):
    from src.qr.qr_processor import decode_qr_image
    urls = decode_qr_image(image_bytes)
"""

from __future__ import annotations


def decode_qr_image(image_bytes: bytes) -> list[str]:
    """Decode QR codes from an image and return extracted data strings.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file content (PNG, JPEG, etc.).

    Returns
    -------
    list[str]
        List of decoded data strings (typically URLs). Empty if no QR found.

    TODO
    ----
    - Decode image bytes with OpenCV (cv2.imdecode)
    - Use pyzbar.decode() to extract QR data
    - Validate extracted strings as URLs
    - Return list of decoded URL strings
    """
    raise NotImplementedError("QR decoding not yet implemented.")


def classify_qr_urls(urls: list[str]) -> list[dict]:
    """Classify extracted QR URLs using the URL phishing classifier.

    Parameters
    ----------
    urls : list[str]
        URLs decoded from QR codes.

    Returns
    -------
    list[dict]
        Classification results per URL, each containing:
        - url: str
        - is_phishing: bool
        - confidence: float
        - explanation: dict (SHAP values)

    TODO
    ----
    - Load URL classifier from src.ml.url_model
    - Extract features via src.preprocessing.url_features
    - Run prediction and return structured results
    """
    raise NotImplementedError("QR URL classification not yet implemented.")
