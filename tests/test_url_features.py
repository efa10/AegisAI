"""
Tests for src.preprocessing.url_features
"""

import pytest
import pandas as pd

from src.preprocessing.url_features import extract_url_features, extract_url_features_batch


class TestExtractURLFeatures:
    """Unit tests for single-URL feature extraction."""

    def test_returns_dict(self):
        """extract_url_features should return a dict of features."""
        features = extract_url_features("https://example.com/login")
        assert isinstance(features, dict)
        assert "url_length" in features

    def test_handles_empty_url(self):
        """Should handle empty string input gracefully."""
        features = extract_url_features("")
        assert features["url_length"] == 0
        assert features["hostname_length"] == 0

    def test_handles_url_with_special_chars(self):
        """Should correctly process URLs with special characters."""
        features = extract_url_features("https://example.com/test?param=1&foo=bar@_")
        assert features["num_special_chars"] > 0
        assert features["has_at_symbol"] == 1
        assert features["query_param_count"] == 2

    def test_feature_values_are_numeric(self):
        """All returned feature values should be int or float."""
        features = extract_url_features("https://example.com")
        for val in features.values():
            assert isinstance(val, (int, float))

    def test_ip_address_detection(self):
        features_ip = extract_url_features("http://192.168.1.1/login")
        features_no_ip = extract_url_features("http://example.com/login")
        assert features_ip["has_ip_address"] == 1
        assert features_no_ip["has_ip_address"] == 0


class TestExtractURLFeaturesBatch:
    """Unit tests for batch URL feature extraction."""

    def test_returns_dataframe(self):
        """extract_url_features_batch should return a pandas DataFrame."""
        urls = pd.Series(["https://example.com", "http://192.168.1.1"])
        df = extract_url_features_batch(urls)
        assert isinstance(df, pd.DataFrame)
        assert "url_length" in df.columns

    def test_correct_row_count(self):
        """Output should have the same number of rows as input."""
        urls = pd.Series(["https://example.com", "http://192.168.1.1", ""])
        df = extract_url_features_batch(urls)
        assert len(df) == 3
