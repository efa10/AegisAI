"""
AegisAI — SHAP Explainer for URL Models

Calculates SHAP values for a given URL using the existing Phase 3 Random Forest model.
Provides human-readable reasons mapping raw feature values to phishing indicators.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Suppress noisy SHAP warnings
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*numpy.ufunc size changed.*")

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from config.settings import SETTINGS
from src.ml import url_model
from src.ml.url_model import predict_url, get_url_classifier

# Module-level references kept for test mock patching compatibility
_cached_classifier = None
_cached_model_path = None


# ---------------------------------------------------------------------------
# Feature mapping dictionary
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "url_length": ("Suspicious URL length", "Normal URL length"),
    "hostname_length": ("Unusual hostname length", "Normal hostname length"),
    "path_length": ("Suspicious path length", "Normal path length"),
    "query_length": ("Suspicious query string length", "Normal query string"),
    "num_dots": ("Excessive number of dots", "Normal dot count"),
    "num_hyphens": ("Multiple hyphens used", "Normal hyphen usage"),
    "num_underscores": ("Multiple underscores used", "Normal underscore usage"),
    "num_special_chars": ("High special-character count", "Normal character set"),
    "num_digits": ("Suspicious number of digits", "Normal digit count"),
    "num_subdomains": ("High subdomain count", "Normal subdomain count"),
    "has_https": ("Missing HTTPS", "Uses secure HTTPS"),
    "has_ip_address": ("IP address detected instead of domain name", "Uses standard domain name"),
    "has_at_symbol": ("Contains '@' symbol (often used to obscure URLs)", "No '@' symbol present"),
    "url_entropy": ("High URL entropy/randomness", "Normal URL predictability"),
    "domain_length": ("Suspicious domain length", "Normal domain length"),
    "path_depth": ("Deep directory path", "Shallow directory path"),
    "query_param_count": ("Numerous query parameters", "Normal query parameter count"),
}


# Module-level cache for SHAP explainer to avoid recalculating base values
_cached_explainer: Any = None
_cached_explainer_model_path: Path | None = None
_cached_model_id: int | None = None


def explain_url(url: str, model_path: Path | None = None) -> dict[str, Any]:
    """Generate a SHAP explanation for a URL prediction.

    Parameters
    ----------
    url : str
        The URL to explain.
    model_path : Path | None
        Path to the saved URL model.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the prediction, confidence, top features,
        their SHAP contributions, and human-readable reasons.
    """
    global _cached_explainer, _cached_explainer_model_path, _cached_model_id

    if not _SHAP_AVAILABLE:
        return {
            "prediction": "error",
            "error": "SHAP library is not installed."
        }

    # 1. Run the existing URL prediction pipeline
    prediction_result = predict_url(url, model_path=model_path)

    if prediction_result.get("verdict") == "invalid":
        return {
            "prediction": "error",
            "error": prediction_result.get("error", "Invalid URL processing error.")
        }
        
    verdict = prediction_result["verdict"]
    score = prediction_result["phishing_probability"]
    features_dict = prediction_result["features"]
    
    # Format features in the exact order the model expects
    feature_names = SETTINGS.URL_FEATURE_NAMES
    X_single = pd.DataFrame([features_dict], columns=feature_names)
    
    # 2. Get the trained model instance from the url_model module cache
    # Check if _cached_classifier was explicitly patched / set (e.g. in unit tests)
    clf = _cached_classifier
    if clf is None:
        clf = getattr(url_model, "_cached_classifier", None)
    if clf is None:
        try:
            clf = get_url_classifier(model_path)
        except Exception:
            clf = None

    model = getattr(clf, "model", None) if clf is not None else None
    if model is None:
        return {
            "prediction": "error",
            "error": "Model failed to load."
        }

    # 3. Initialize SHAP Explainer (cached for performance)
    current_model_path = model_path or SETTINGS.MODELS_DIR / "best_url_model.joblib"
    if (
        _cached_explainer is None
        or _cached_explainer_model_path != current_model_path
        or _cached_model_id != id(model)
    ):
        # Use TreeExplainer for Random Forest / XGBoost, which are our primary models.
        _cached_explainer = shap.TreeExplainer(model)
        _cached_explainer_model_path = current_model_path
        _cached_model_id = id(model)

    # 4. Calculate SHAP values
    shap_values = _cached_explainer.shap_values(X_single)
    
    # shap_values format varies slightly depending on sklearn version and model type.
    # For a binary Random Forest, shap_values is typically a list of length 2 (one array per class),
    # where shap_values[1] contains the values for the positive class (phishing).
    if isinstance(shap_values, list):
        if verdict == "phishing":
            # Contributions towards the phishing class (class 1)
            contributions = shap_values[1][0]
        else:
            # Contributions towards the legitimate class (class 0)
            contributions = shap_values[0][0]
    else:
        # For single-output models (like XGBoost sometimes), shap_values is just an array
        contributions = shap_values[0]
        if verdict == "legitimate":
            # If predicting legitimate, reverse the sign so positive values mean "pushed towards legitimate"
            contributions = -contributions

    # 5. Extract top features
    # Pair feature names with their absolute contribution magnitude
    feature_contributions = list(zip(feature_names, contributions))
    
    # We want features that positively pushed the model towards the chosen verdict.
    # So we sort by the actual contribution value (descending), keeping only those > 0.
    positive_contributors = [(feat, val) for feat, val in feature_contributions if val > 0]
    positive_contributors.sort(key=lambda x: x[1], reverse=True)
    
    if positive_contributors:
        top_features = positive_contributors[:SETTINGS.SHAP_MAX_DISPLAY_FEATURES]
    else:
        sorted_by_mag = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_by_mag[:SETTINGS.SHAP_MAX_DISPLAY_FEATURES]
    
    # 6. Generate human-readable reasons
    reasons = []
    for feat, val in top_features:
        if feat in FEATURE_DESCRIPTIONS:
            # If phishing, use the first description (suspicious indicator)
            # If legitimate, use the second description (safe indicator)
            idx = 0 if verdict == "phishing" else 1
            reasons.append(FEATURE_DESCRIPTIONS[feat][idx])
        else:
            reasons.append(feat) # Fallback

    if not reasons:
        reasons.append("Model relied on a combination of subtle baseline features.")

    return {
        "prediction": verdict.upper(),
        "confidence": prediction_result.get("confidence", score if score is not None else 1.0),
        "phishing_probability": score,
        "top_features": [f[0] for f in top_features],
        "feature_contributions": {f[0]: round(float(f[1]), 5) for f in top_features},
        "reasons": reasons
    }
