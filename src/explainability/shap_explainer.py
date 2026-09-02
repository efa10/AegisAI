"""
AegisAI — SHAP Explainer

Generates SHAP-based explanations for individual predictions, enabling
users to understand *why* a URL or text was flagged as phishing.

Usage (after implementation):
    from src.explainability.shap_explainer import explain_prediction
    explanation = explain_prediction(model, features, feature_names)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def explain_prediction(
    model: Any,
    features: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Generate SHAP values for a single prediction.

    Parameters
    ----------
    model : Any
        A trained scikit-learn compatible model.
    features : np.ndarray
        Feature vector for a single sample (1, n_features).
    feature_names : list[str]
        Human-readable names for each feature.

    Returns
    -------
    dict[str, float]
        Mapping of feature name → SHAP value.

    TODO
    ----
    - Create a SHAP TreeExplainer (for tree models) or KernelExplainer
    - Compute SHAP values for the input
    - Return sorted feature importance dict
    """
    raise NotImplementedError("SHAP explanation not yet implemented.")


def explain_text_prediction(
    model: Any,
    text: str,
) -> dict[str, Any]:
    """Generate SHAP explanation for a text classification prediction.

    Parameters
    ----------
    model : Any
        Trained text classification pipeline (TF-IDF + classifier).
    text : str
        Raw text input.

    Returns
    -------
    dict[str, Any]
        Explanation data including top contributing tokens.

    TODO
    ----
    - Use SHAP with the text pipeline
    - Extract top positive/negative contributing words
    """
    raise NotImplementedError("Text SHAP explanation not yet implemented.")
