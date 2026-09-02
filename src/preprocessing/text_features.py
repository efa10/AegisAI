"""
AegisAI — Text Feature Extraction

Preprocesses and vectorises text data (email bodies, SMS messages) for the
phishing text classifier. Uses TF-IDF with NLP preprocessing.

Usage:
    from src.preprocessing.text_features import build_tfidf_pipeline, clean_text
    vectorizer = build_tfidf_pipeline()
    vectorizer.fit(X_train)
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from config.settings import SETTINGS


def clean_text(text: str) -> str:
    """Normalise and clean a single text string.

    Parameters
    ----------
    text : str
        Raw email body or SMS message.

    Returns
    -------
    str
        Cleaned text ready for vectorisation.
    """
    if pd.isna(text) or not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    # Lowercase
    text = text.lower()
    
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove excessive punctuation but KEEP some indicators (like £, $, !)
    # We want to keep words and common currency/exclamation symbols
    # So we replace anything that is NOT a letter, number, space, !, $, £, ? with a space.
    text = re.sub(r'[^a-z0-9\s!\$\£\?]', ' ', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def build_tfidf_pipeline(**kwargs: Any) -> TfidfVectorizer:
    """Build and return a scikit-learn TF-IDF vectorisation pipeline.

    Parameters
    ----------
    **kwargs
        Override default TF-IDF parameters from config.

    Returns
    -------
    TfidfVectorizer
        Unfitted vectoriser.
    """
    max_features = kwargs.get("max_features", SETTINGS.TFIDF_MAX_FEATURES)
    ngram_range = kwargs.get("ngram_range", SETTINGS.TFIDF_NGRAM_RANGE)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        lowercase=False  # We do this in clean_text
    )
    
    return vectorizer


def preprocess_text_batch(texts: pd.Series) -> pd.Series:
    """Clean a batch of text strings.

    Parameters
    ----------
    texts : pd.Series
        Raw text data.

    Returns
    -------
    pd.Series
        Cleaned text data.
    """
    return texts.apply(clean_text)
