"""
Tests for src.ml.url_model
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.ml.url_model import URLClassifier, predict_url
from src.preprocessing.url_features import extract_url_features
from config.settings import SETTINGS


class TestURLClassifier:
    """Unit tests for the URLClassifier class."""

    def _make_dummy_data(self, n=100):
        """Create small dummy feature data for testing."""
        rng = np.random.RandomState(42)
        X = rng.rand(n, 17)
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return X, y

    def test_train_random_forest(self):
        """Should train a Random Forest model without error."""
        X, y = self._make_dummy_data()
        clf = URLClassifier("random_forest")
        result = clf.train(X, y)
        assert clf.model is not None
        assert "training_time_seconds" in result

    def test_predict_returns_array(self):
        """Predict should return an array of labels."""
        X, y = self._make_dummy_data()
        clf = URLClassifier("random_forest")
        clf.train(X, y)
        preds = clf.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_returns_array(self):
        """predict_proba should return probability arrays."""
        X, y = self._make_dummy_data()
        clf = URLClassifier("random_forest")
        clf.train(X, y)
        proba = clf.predict_proba(X)
        assert proba is not None
        assert proba.shape == (len(X), 2)

    def test_evaluate_returns_metrics(self):
        """evaluate() should return a dict with accuracy, precision, recall, f1."""
        X, y = self._make_dummy_data()
        clf = URLClassifier("random_forest")
        clf.train(X, y)
        metrics = clf.evaluate(X, y)
        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
            assert key in metrics

    def test_save_and_load_model(self):
        """Should save and reload a model correctly."""
        X, y = self._make_dummy_data()
        clf = URLClassifier("random_forest")
        clf.train(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.joblib"
            clf.save_model(path)
            assert path.exists()

            clf2 = URLClassifier()
            clf2.load_model(path)
            assert clf2.model is not None
            assert clf2.model_name == "random_forest"

            # Predictions should be identical
            preds1 = clf.predict(X)
            preds2 = clf2.predict(X)
            np.testing.assert_array_equal(preds1, preds2)

    def test_predict_without_training_raises(self):
        """Should raise RuntimeError if predicting without training."""
        clf = URLClassifier("random_forest")
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict(np.array([[1, 2, 3]]))

    def test_unknown_model_raises(self):
        """Should raise ValueError for unknown model name."""
        clf = URLClassifier("unknown_model")
        X, y = self._make_dummy_data()
        with pytest.raises(ValueError, match="Unknown model"):
            clf.train(X, y)


class TestPredictURL:
    """Tests for the predict_url() high-level function."""

    @pytest.fixture(autouse=True)
    def _setup_model(self, tmp_path):
        """Train and save a small model for testing."""
        # Create a tiny trained model
        rng = np.random.RandomState(42)
        X = rng.rand(50, 17)
        y = np.array([0] * 25 + [1] * 25)
        clf = URLClassifier("random_forest")
        clf.train(X, y)
        self.model_path = tmp_path / "test_model.joblib"
        clf.save_model(self.model_path)

    def test_predict_url_returns_dict(self):
        """predict_url should return a dict with required keys."""
        result = predict_url("https://example.com", model_path=self.model_path)
        assert isinstance(result, dict)
        assert "verdict" in result
        assert "phishing_probability" in result
        assert "features" in result
        assert "model_name" in result
        assert result["verdict"] in ("phishing", "legitimate")

    def test_predict_url_invalid_input(self):
        """Should handle empty URL gracefully."""
        result = predict_url("", model_path=self.model_path)
        assert result["verdict"] == "invalid"

    def test_predict_url_features_match(self):
        """Features in prediction should match extract_url_features output."""
        url = "https://example.com/test?q=1"
        result = predict_url(url, model_path=self.model_path)
        expected_features = extract_url_features(url)
        assert set(result["features"].keys()) == set(expected_features.keys())

    def test_predict_url_missing_model(self):
        """Should raise FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError):
            predict_url("https://example.com", model_path=Path("/nonexistent/model.joblib"))


class TestFeatureConsistency:
    """Tests to verify feature extraction produces correct dimensions."""

    def test_feature_count_matches_config(self):
        """Number of features should match SETTINGS.URL_FEATURE_NAMES."""
        features = extract_url_features("https://example.com")
        assert len(features) == len(SETTINGS.URL_FEATURE_NAMES)

    def test_feature_names_match_config(self):
        """Feature names should match SETTINGS.URL_FEATURE_NAMES."""
        features = extract_url_features("https://example.com")
        assert set(features.keys()) == set(SETTINGS.URL_FEATURE_NAMES)
