"""
AegisAI — URL Phishing Classifier

Trains and serves machine learning models for URL-based phishing detection.
Uses engineered URL features from src.preprocessing.url_features.

Supports: Random Forest, Naive Bayes, SVM, XGBoost.

Usage:
    from src.ml.url_model import URLClassifier, predict_url
    clf = URLClassifier("random_forest")
    clf.train(X_train, y_train)
    result = predict_url("https://suspicious-site.com/login")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config.settings import SETTINGS
from src.preprocessing.url_features import extract_url_features

# Try importing xgboost — optional dependency
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _create_model(
    model_name: str,
    random_state: int = SETTINGS.RANDOM_STATE,
    class_weight: dict | str | None = None,
) -> Any:
    """Create an unfitted sklearn-compatible estimator by name.

    Parameters
    ----------
    model_name : str
        One of 'random_forest', 'naive_bayes', 'svm', 'xgboost'.
    random_state : int
        Reproducibility seed.
    class_weight : dict | str | None
        Class weight strategy. Supported by RF and SVM.
    """
    name = model_name.lower().replace(" ", "_")

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight=class_weight or "balanced",
            n_jobs=-1,
        )
    elif name == "naive_bayes":
        # GaussianNB doesn't support class_weight natively.
        # Weighting is handled via sample_weight during fit.
        return GaussianNB()
    elif name == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight=class_weight or "balanced",
            max_iter=5000,
        )
    elif name == "xgboost":
        if not _XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: random_forest, naive_bayes, svm, xgboost")


# ---------------------------------------------------------------------------
# URLClassifier
# ---------------------------------------------------------------------------

class URLClassifier:
    """URL phishing classifier.

    Wraps a scikit-learn compatible estimator with train / predict / save / load.

    Attributes
    ----------
    model_name : str
        Name of the underlying algorithm.
    model : object or None
        The trained estimator. None until train() is called.
    """

    def __init__(self, model_name: str = "random_forest") -> None:
        self.model_name = model_name.lower().replace(" ", "_")
        self.model: Any | None = None

    def train(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        class_weight: dict | str | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Train the URL classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Binary labels (0 = legitimate, 1 = phishing).
        class_weight : dict | str | None
            Class weight strategy for models that support it.
        sample_weight : np.ndarray | None
            Per-sample weights (used for Naive Bayes imbalance handling).

        Returns
        -------
        dict
            Training metadata including elapsed time.
        """
        self.model = _create_model(self.model_name, class_weight=class_weight)

        start = time.time()

        if sample_weight is not None and hasattr(self.model, "fit"):
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

        elapsed = time.time() - start
        return {"model_name": self.model_name, "training_time_seconds": round(elapsed, 3)}

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict phishing labels.

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | None:
        """Predict phishing probabilities.

        Returns None if the model does not support probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> dict[str, Any]:
        """Evaluate the classifier.

        Returns
        -------
        dict
            accuracy, precision, recall, f1, confusion_matrix, inference_latency_ms
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        start = time.time()
        y_pred = self.model.predict(X)
        total_time = time.time() - start
        n_samples = len(y)

        return {
            "model_name": self.model_name,
            "accuracy": round(accuracy_score(y, y_pred), 5),
            "precision": round(precision_score(y, y_pred, zero_division=0), 5),
            "recall": round(recall_score(y, y_pred, zero_division=0), 5),
            "f1": round(f1_score(y, y_pred, zero_division=0), 5),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "inference_latency_ms_per_sample": round(
                (total_time / n_samples) * 1000, 5
            ),
            "total_inference_time_seconds": round(total_time, 4),
        }

    def save_model(self, path: Path) -> None:
        """Save trained model to disk using joblib."""
        if self.model is None:
            raise RuntimeError("No trained model to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model_name": self.model_name, "model": self.model},
            path,
        )

    def load_model(self, path: Path) -> None:
        """Load a trained model from disk."""
        data = joblib.load(path)
        self.model_name = data["model_name"]
        self.model = data["model"]


# ---------------------------------------------------------------------------
# High-level prediction function
# ---------------------------------------------------------------------------

# Module-level cache for the loaded model
_cached_classifier: URLClassifier | None = None
_cached_model_path: Path | None = None


def get_url_classifier(model_path: Path | None = None) -> URLClassifier:
    """Retrieve or load the cached URLClassifier instance.

    Parameters
    ----------
    model_path : Path | None
        Path to a saved model. Defaults to models/best_url_model.joblib.

    Returns
    -------
    URLClassifier
        The cached and loaded URLClassifier instance.
    """
    global _cached_classifier, _cached_model_path

    if model_path is None:
        model_path = SETTINGS.MODELS_DIR / "best_url_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Train a model first."
        )

    if _cached_classifier is None or _cached_model_path != model_path:
        _cached_classifier = URLClassifier()
        _cached_classifier.load_model(model_path)
        _cached_model_path = model_path

    return _cached_classifier


def predict_url(url: str, model_path: Path | None = None) -> dict[str, Any]:
    """Predict whether a URL is phishing.

    Parameters
    ----------
    url : str
        The raw URL to classify.
    model_path : Path | None
        Path to a saved model. Defaults to models/best_url_model.joblib.

    Returns
    -------
    dict
        verdict, probability, features, model_name
    """
    clf = get_url_classifier(model_path)

    # Validate input
    if not isinstance(url, str) or not url.strip():
        return {
            "original_url": url,
            "url": url,
            "is_shortened": False,
            "shortening_service": None,
            "shortener_service": None,
            "expanded_url": url,
            "redirect_chain": [url] if url else [],
            "redirect_count": 0,
            "expansion_success": False,
            "expansion_error": "Empty or invalid URL provided.",
            "error": "Empty or invalid URL provided.",
            "ml_verdict": "invalid",
            "verdict": "invalid",
            "label": -1,
            "phishing_probability": None,
            "confidence": None,
            "risk_score": None,
            "risk_indicators": ["Empty or invalid URL format."],
            "threat_intelligence": {
                "overall_verdict": "unknown",
                "virustotal": {"status": "not_configured", "verdict": "unknown"},
                "urlhaus": {"status": "not_configured", "verdict": "unknown"},
                "phishtank": {"status": "not_configured", "verdict": "unknown"},
            },
            "features": {},
            "model_name": clf.model_name,
        }

    from src.preprocessing.url_expansion import expand_url, normalize_url
    from src.threat_intel.threat_intel import check_threat_intelligence

    # Expand shortened URLs safely (with SSRF protection)
    expansion_result = expand_url(url)
    target_url = expansion_result.get("expanded_url") or normalize_url(url)

    # Extract features
    features = extract_url_features(target_url)
    feature_df = pd.DataFrame([features])

    # Predict
    prediction = clf.predict(feature_df)[0]
    proba = None
    proba_result = clf.predict_proba(feature_df)
    if proba_result is not None:
        proba = round(float(proba_result[0][1]), 5)  # probability of phishing

    verdict = "phishing" if prediction == SETTINGS.LABEL_PHISHING else "legitimate"

    # Correct confidence semantics:
    # If verdict is phishing: confidence that it is phishing = proba
    # If verdict is legitimate: confidence that it is legitimate = 1.0 - proba
    if proba is not None:
        confidence = round(proba if verdict == "phishing" else (1.0 - proba), 5)
        risk_score = round(proba * 100, 1)
    else:
        confidence = 1.0
        risk_score = 100.0 if verdict == "phishing" else 0.0

    # Risk indicators
    indicators: list[str] = []
    if expansion_result.get("is_shortened"):
        svc = expansion_result.get("shortening_service", "Unknown")
        if expansion_result.get("expansion_success"):
            indicators.append(f"Shortened URL detected via {svc} (resolved to target destination)")
        else:
            indicators.append(f"Shortened URL via {svc} (expansion failed: {expansion_result.get('expansion_error')})")

    if features.get("has_ip_address") == 1:
        indicators.append("Host uses direct IP address instead of domain name")
    if features.get("has_at_symbol") == 1:
        indicators.append("Contains '@' symbol (often used to obscure target hostname)")
    if features.get("num_subdomains", 0) > 2:
        indicators.append(f"High subdomain count ({features.get('num_subdomains')} subdomains)")
    if features.get("url_length", 0) > 85:
        indicators.append(f"Abnormally long URL ({features.get('url_length')} characters)")
    if features.get("url_entropy", 0) > 4.8:
        indicators.append("Elevated character entropy/randomness in URL structure")
    if features.get("num_digits", 0) > 8:
        indicators.append(f"High numeric character concentration ({features.get('num_digits')} digits)")

    # Query threat intelligence feeds non-blockingly
    threat_intel = check_threat_intelligence(target_url)
    if threat_intel.get("overall_verdict") == "malicious":
        indicators.insert(0, "Flagged as malicious by external threat intelligence database")
        # Elevate risk score if threat intel confirmed active threat
        risk_score = max(risk_score, 95.0)

    return {
        "original_url": url,
        "is_shortened": expansion_result["is_shortened"],
        "shortening_service": expansion_result["shortening_service"],
        "expanded_url": target_url,
        "redirect_chain": expansion_result["redirect_chain"],
        "redirect_count": expansion_result["redirect_count"],
        "expansion_success": expansion_result["expansion_success"],
        "expansion_error": expansion_result["expansion_error"],
        "ml_verdict": verdict,
        "phishing_probability": proba,
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_indicators": indicators,
        "threat_intelligence": threat_intel,
        # Backward-compatible fields
        "url": url,
        "shortener_service": expansion_result["shortening_service"],
        "error": expansion_result["expansion_error"],
        "verdict": verdict,
        "label": int(prediction),
        "features": features,
        "model_name": clf.model_name,
    }
