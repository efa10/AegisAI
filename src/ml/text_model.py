"""
AegisAI — Text/Email/SMS Phishing Classifier

Trains and serves machine learning models for text-based phishing detection.
Uses TF-IDF features from src.preprocessing.text_features.

Supports: Multinomial Naive Bayes, Logistic Regression, Linear SVM.

Usage:
    from src.ml.text_model import TextClassifier, predict_text
    clf = TextClassifier("logistic_regression")
    clf.train(X_train, y_train)
    result = predict_text("Urgent! Click here to verify your account")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config.settings import SETTINGS
from src.preprocessing.text_features import clean_text, build_tfidf_pipeline


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _create_text_model(
    model_name: str,
    random_state: int = SETTINGS.RANDOM_STATE,
    class_weight: dict | str | None = None,
) -> Any:
    """Create an unfitted sklearn-compatible estimator for text."""
    name = model_name.lower().replace(" ", "_")

    if name == "naive_bayes":
        # MultinomialNB is standard for text/TF-IDF. It does not support class_weight.
        return MultinomialNB()
    elif name == "logistic_regression":
        return LogisticRegression(
            random_state=random_state,
            class_weight=class_weight or "balanced",
            max_iter=1000,
        )
    elif name == "linear_svm":
        return LinearSVC(
            random_state=random_state,
            class_weight=class_weight or "balanced",
            max_iter=2000,
            dual="auto",
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: naive_bayes, logistic_regression, linear_svm")


# ---------------------------------------------------------------------------
# TextClassifier
# ---------------------------------------------------------------------------

class TextClassifier:
    """Text phishing classifier.

    Wraps a TF-IDF vectorizer + sklearn estimator with train / predict / save / load.

    Attributes
    ----------
    model_name : str
        Name of the underlying algorithm.
    model : object or None
        The trained estimator.
    vectorizer : object or None
        The fitted TF-IDF vectorizer.
    """

    def __init__(self, model_name: str = "logistic_regression") -> None:
        self.model_name = model_name.lower().replace(" ", "_")
        self.model: Any | None = None
        self.vectorizer: Any | None = None

    def train(
        self,
        X_text: pd.Series | list[str],
        y: np.ndarray | pd.Series,
        class_weight: dict | str | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Train the text classifier.

        Parameters
        ----------
        X_text : array-like of strings
            Raw text strings. (Already cleaned strings are fine too).
        y : array-like of shape (n_samples,)
            Binary labels (0 = legitimate, 1 = phishing).
        class_weight : dict | str | None
            Class weight strategy.
        sample_weight : np.ndarray | None
            Per-sample weights.

        Returns
        -------
        dict
            Training metadata including elapsed time.
        """
        # Fit vectorizer
        start = time.time()
        self.vectorizer = build_tfidf_pipeline()
        # Ensure it's a list of strings
        X_list = list(X_text)
        X_tfidf = self.vectorizer.fit_transform(X_list)
        
        # Fit model
        self.model = _create_text_model(self.model_name, class_weight=class_weight)

        if sample_weight is not None and hasattr(self.model, "fit"):
            self.model.fit(X_tfidf, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_tfidf, y)

        elapsed = time.time() - start
        return {"model_name": self.model_name, "training_time_seconds": round(elapsed, 3)}

    def predict(self, X_text: pd.Series | list[str]) -> np.ndarray:
        """Predict phishing labels."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X_tfidf = self.vectorizer.transform(list(X_text))
        return self.model.predict(X_tfidf)

    def predict_proba(self, X_text: pd.Series | list[str]) -> np.ndarray | None:
        """Predict phishing probabilities. None if unsupported."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if hasattr(self.model, "predict_proba"):
            X_tfidf = self.vectorizer.transform(list(X_text))
            return self.model.predict_proba(X_tfidf)
        elif hasattr(self.model, "decision_function"):
            # For LinearSVC, we can get confidence scores
            X_tfidf = self.vectorizer.transform(list(X_text))
            scores = self.model.decision_function(X_tfidf)
            # Normalise scores to 0-1 range roughly using sigmoid for API compat
            probs = 1 / (1 + np.exp(-scores))
            # Return [prob_class_0, prob_class_1]
            return np.vstack((1 - probs, probs)).T
        return None

    def evaluate(
        self,
        X_text: pd.Series | list[str],
        y: np.ndarray | pd.Series,
    ) -> dict[str, Any]:
        """Evaluate the classifier."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not trained. Call train() first.")

        start = time.time()
        y_pred = self.predict(X_text)
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
        """Save trained model and vectorizer to disk using joblib."""
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("No trained model to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": self.model_name,
                "model": self.model,
                "vectorizer": self.vectorizer
            },
            path,
        )

    def load_model(self, path: Path) -> None:
        """Load a trained model and vectorizer from disk."""
        data = joblib.load(path)
        self.model_name = data["model_name"]
        self.model = data["model"]
        self.vectorizer = data["vectorizer"]


# ---------------------------------------------------------------------------
# High-level prediction function
# ---------------------------------------------------------------------------

_cached_text_classifier: TextClassifier | None = None
_cached_text_model_path: Path | None = None


def predict_text(text: str, model_path: Path | None = None) -> dict[str, Any]:
    """Predict whether a text message is phishing/spam.

    Parameters
    ----------
    text : str
        The raw text string to classify.
    model_path : Path | None
        Path to a saved model. Defaults to models/best_text_model.joblib.

    Returns
    -------
    dict
        verdict, probability, cleaned_text, model_name
    """
    global _cached_text_classifier, _cached_text_model_path

    if model_path is None:
        model_path = SETTINGS.MODELS_DIR / "best_text_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Train a model first."
        )

    # Load model (with caching)
    if _cached_text_classifier is None or _cached_text_model_path != model_path:
        _cached_text_classifier = TextClassifier()
        _cached_text_classifier.load_model(model_path)
        _cached_text_model_path = model_path

    clf = _cached_text_classifier

    # Validate input
    if not isinstance(text, str) or not text.strip():
        return {
            "text": text,
            "verdict": "invalid",
            "probability": None,
            "cleaned_text": "",
            "model_name": clf.model_name,
            "error": "Empty or invalid text provided.",
        }

    # Preprocess
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "text": text,
            "verdict": "invalid",
            "probability": None,
            "cleaned_text": "",
            "model_name": clf.model_name,
            "error": "Text was empty after cleaning.",
        }

    # Predict
    prediction = clf.predict([cleaned])[0]
    proba = None
    proba_result = clf.predict_proba([cleaned])
    if proba_result is not None:
        proba = round(float(proba_result[0][1]), 5)

    verdict = "phishing" if prediction == SETTINGS.LABEL_PHISHING else "legitimate"

    return {
        "text": text,
        "verdict": verdict,
        "label": int(prediction),
        "phishing_probability": proba,
        "cleaned_text": cleaned,
        "model_name": clf.model_name,
    }
