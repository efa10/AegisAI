"""
Tests for src.preprocessing.text_features
"""

import pytest

from src.preprocessing.text_features import clean_text, build_tfidf_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class TestCleanText:
    """Unit tests for text cleaning."""

    def test_lowercases_text(self):
        """clean_text should lowercase the input."""
        cleaned = clean_text("HELLO WORLD")
        assert cleaned == "hello world"

    def test_removes_html_tags(self):
        """clean_text should strip HTML tags."""
        cleaned = clean_text("<p>Hello <b>World</b></p>")
        assert cleaned == "hello world"

    def test_handles_empty_string(self):
        """Should handle empty string input gracefully."""
        cleaned = clean_text("")
        assert cleaned == ""

    def test_removes_special_characters(self):
        """Should remove most punctuation but keep indicators like ! or $."""
        cleaned = clean_text("Hello, world! 123 @#$%")
        assert cleaned == "hello world! 123 $"

class TestBuildTfidfPipeline:
    """Unit tests for TF-IDF pipeline construction."""

    def test_returns_pipeline(self):
        """build_tfidf_pipeline should return a valid pipeline object."""
        pipeline = build_tfidf_pipeline()
        assert isinstance(pipeline, TfidfVectorizer)

    def test_respects_max_features(self):
        """Pipeline should respect the max_features parameter."""
        pipeline = build_tfidf_pipeline(max_features=100)
        assert pipeline.max_features == 100
