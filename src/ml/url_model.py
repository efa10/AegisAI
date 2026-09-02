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
            max_depth=20,
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
    global _cached_classifier, _cached_model_path

    if model_path is None:
        model_path = SETTINGS.MODELS_DIR / "best_url_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Train a model first."
        )

    # Load model (with caching)
    if _cached_classifier is None or _cached_model_path != model_path:
        _cached_classifier = URLClassifier()
        _cached_classifier.load_model(model_path)
        _cached_model_path = model_path

    clf = _cached_classifier

    # Validate input
    if not isinstance(url, str) or not url.strip():
        return {
            "url": url,
            "verdict": "invalid",
            "probability": None,
            "features": {},
            "model_name": clf.model_name,
            "error": "Empty or invalid URL provided.",
        }

    # Extract features
    features = extract_url_features(url)
    feature_df = pd.DataFrame([features])

    # Predict
    prediction = clf.predict(feature_df)[0]
    proba = None
    proba_result = clf.predict_proba(feature_df)
    if proba_result is not None:
        proba = round(float(proba_result[0][1]), 5)  # probability of phishing

    verdict = "phishing" if prediction == SETTINGS.LABEL_PHISHING else "legitimate"

    return {
        "url": url,
        "verdict": verdict,
        "label": int(prediction),
        "phishing_probability": proba,
        "features": features,
        "model_name": clf.model_name,
    }
