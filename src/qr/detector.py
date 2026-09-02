"""
AegisAI — QR Code Detector

Provides robust extraction of QR code payloads from images using OpenCV
with an optional fallback to pyzbar if available.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    _PYZBAR_AVAILABLE = True
except ImportError:
    _PYZBAR_AVAILABLE = False


def decode_qr(image_source: str | Path | bytes | np.ndarray) -> str | None:
    """Detect and decode a QR code from an image.

    Parameters
    ----------
    image_source : str, Path, bytes, or np.ndarray
        Path to the image, raw image bytes, or an OpenCV numpy array.

    Returns
    -------
    str | None
        The decoded payload string, or None if no QR code was found
        or the image was invalid.
    """
    img = None

    if isinstance(image_source, (str, Path)):
        path_str = str(image_source)
        if not os.path.exists(path_str):
            return None
        img = cv2.imread(path_str)
    elif isinstance(image_source, bytes):
        if not image_source:
            return None
        np_arr = np.frombuffer(image_source, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif isinstance(image_source, np.ndarray):
        img = image_source

    if img is None or img.size == 0:
        return None

    # Primary detection using OpenCV
    try:
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img)
        if data:
            return data
    except Exception:
        pass  # Fallback to pyzbar if OpenCV fails unexpectedly

    # Fallback detection using pyzbar (handles some cases OpenCV misses)
    if _PYZBAR_AVAILABLE:
        try:
            decoded_objects = pyzbar_decode(img)
            for obj in decoded_objects:
                if obj.type == 'QRCODE':
                    return obj.data.decode('utf-8', errors='ignore')
        except Exception:
            pass

    return None
