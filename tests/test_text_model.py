import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.ml.text_model import TextClassifier, predict_text
from config.settings import SETTINGS

@pytest.fixture
def sample_text_data():
    texts = [
        "Hey, are we still meeting for lunch?",
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
        "Can you send me the report by 5?",
        "Please update your account info by clicking this link http://spam.com/login",
        "See you tomorrow!",
        "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!"
    ]
    labels = np.array([0, 1, 0, 1, 0, 1])
    return texts, labels


def test_text_classifier_init():
    clf = TextClassifier("logistic_regression")
    assert clf.model_name == "logistic_regression"
    assert clf.model is None
    assert clf.vectorizer is None


def test_text_classifier_invalid_model():
    with pytest.raises(ValueError):
        clf = TextClassifier("non_existent_model")
        clf.train(["test"], np.array([0]))


def test_text_classifier_train_predict(sample_text_data):
    texts, labels = sample_text_data
    clf = TextClassifier("logistic_regression")
    
    # Train
    result = clf.train(texts, labels)
    assert result["model_name"] == "logistic_regression"
    assert "training_time_seconds" in result
    
    # Predict
    preds = clf.predict(texts)
    assert len(preds) == len(labels)
    assert set(preds).issubset({0, 1})
    
    # Predict proba
    probs = clf.predict_proba(texts)
    assert probs is not None
    assert probs.shape == (len(texts), 2)
    assert np.all((probs >= 0) & (probs <= 1))


def test_text_classifier_evaluate(sample_text_data):
    texts, labels = sample_text_data
    clf = TextClassifier("naive_bayes")
    clf.train(texts, labels)
    
    metrics = clf.evaluate(texts, labels)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert "inference_latency_ms_per_sample" in metrics


def test_text_classifier_save_load(sample_text_data, tmp_path):
    texts, labels = sample_text_data
    clf = TextClassifier("linear_svm")
    clf.train(texts, labels)
    
    model_path = tmp_path / "test_model.joblib"
    clf.save_model(model_path)
    assert model_path.exists()
    
    # Load into new instance
    clf_new = TextClassifier()
    clf_new.load_model(model_path)
    assert clf_new.model_name == "linear_svm"
    assert clf_new.model is not None
    assert clf_new.vectorizer is not None
    
    # Verify predictions match
    preds_orig = clf.predict(texts)
    preds_new = clf_new.predict(texts)
    np.testing.assert_array_equal(preds_orig, preds_new)


def test_predict_text_valid(sample_text_data, tmp_path):
    texts, labels = sample_text_data
    clf = TextClassifier("logistic_regression")
    clf.train(texts, labels)
    
    model_path = tmp_path / "best_text_model.joblib"
    clf.save_model(model_path)
    
    res = predict_text("URGENT! You won a prize", model_path=model_path)
    assert res["verdict"] in ["legitimate", "phishing"]
    assert res["model_name"] == "logistic_regression"
    assert res["phishing_probability"] is not None


def test_predict_text_empty(sample_text_data, tmp_path):
    texts, labels = sample_text_data
    clf = TextClassifier("logistic_regression")
    clf.train(texts, labels)
    
    model_path = tmp_path / "best_text_model.joblib"
    clf.save_model(model_path)
    
    # Test empty
    res = predict_text("   ", model_path=model_path)
    assert res["verdict"] == "invalid"
    assert "error" in res


def test_predict_text_no_model():
    with pytest.raises(FileNotFoundError):
        predict_text("Hello", model_path=Path("/does/not/exist.joblib"))
