"""
Unit tests for the SHAP explainability module.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Mock shap before importing our modules, in case it's not installed
sys.modules['shap'] = MagicMock()

from src.explainability.url_explainer import explain_url


@patch("src.explainability.url_explainer.predict_url")
@patch("src.explainability.url_explainer._cached_classifier")
@patch("src.explainability.url_explainer.shap")
def test_explain_url_phishing(mock_shap, mock_classifier, mock_predict_url):
    """Test generating an explanation for a phishing URL."""
    
    # Mock the output of predict_url
    mock_predict_url.return_value = {
        "verdict": "phishing",
        "phishing_probability": 0.95,
        "features": {
            "url_length": 85,
            "has_ip_address": 1,
            "num_subdomains": 4,
            "has_https": 0
        }
    }
    
    # Mock the trained model
    mock_model = MagicMock()
    mock_classifier.model = mock_model
    
    # Mock SHAP Explainer and shap_values
    mock_explainer_inst = MagicMock()
    
    # Simulate binary classification SHAP values (list of two arrays)
    # shap_values[1][0] corresponds to the phishing class features
    # Let's say url_length (+0.2), has_ip (+0.3), subdomains (+0.1), https (-0.05)
    # The order maps to SETTINGS.URL_FEATURE_NAMES, but we'll just mock an array 
    # of the correct length (17 features).
    phishing_shap_array = [0.0] * 17
    # url_length is index 0, num_subdomains is 9, has_https is 10, has_ip_address is 11
    phishing_shap_array[0] = 0.2
    phishing_shap_array[9] = 0.1
    phishing_shap_array[10] = -0.05 # Pushed toward legitimate, so it shouldn't be in the top positive contributors
    phishing_shap_array[11] = 0.3
    
    mock_explainer_inst.shap_values.return_value = [[], [phishing_shap_array]]
    mock_shap.TreeExplainer.return_value = mock_explainer_inst
    
    result = explain_url("http://192.168.1.1/fake_login")
    
    assert result["prediction"] == "PHISHING"
    assert result["confidence"] == 0.95
    assert len(result["top_features"]) == 3
    
    # has_ip_address has highest shap value (0.3)
    assert result["top_features"][0] == "has_ip_address"
    assert "IP address detected instead of domain name" in result["reasons"][0]
    
    assert result["top_features"][1] == "url_length"
    assert "Suspicious URL length" in result["reasons"][1]


@patch("src.explainability.url_explainer.predict_url")
@patch("src.explainability.url_explainer._cached_classifier")
@patch("src.explainability.url_explainer.shap")
def test_explain_url_legitimate(mock_shap, mock_classifier, mock_predict_url):
    """Test generating an explanation for a legitimate URL."""
    
    mock_predict_url.return_value = {
        "verdict": "legitimate",
        "phishing_probability": 0.05,
        "features": {
            "url_length": 25,
            "has_https": 1
        }
    }
    mock_classifier.model = MagicMock()
    
    mock_explainer_inst = MagicMock()
    
    # Legitimate class is class 0. So we look at shap_values[0][0]
    legit_shap_array = [0.0] * 17
    # url_length (0) = 0.1, has_https (10) = 0.4
    legit_shap_array[0] = 0.1
    legit_shap_array[10] = 0.4
    
    mock_explainer_inst.shap_values.return_value = [[legit_shap_array], [legit_shap_array]]
    mock_shap.TreeExplainer.return_value = mock_explainer_inst
    
    result = explain_url("https://google.com", model_path=Path("dummy2.joblib"))
    
    assert result["prediction"] == "LEGITIMATE"
    assert result["confidence"] == 0.05
    assert len(result["top_features"]) == 2
    assert result["top_features"][0] == "has_https"
    assert "Uses secure HTTPS" in result["reasons"][0]


@patch("src.explainability.url_explainer.predict_url")
def test_explain_url_invalid(mock_predict_url):
    """Test handling of invalid/empty URLs."""
    
    mock_predict_url.return_value = {
        "verdict": "invalid",
        "error": "Empty or invalid URL provided."
    }
    
    result = explain_url("")
    
    assert result["prediction"] == "error"
    assert "Empty or invalid URL" in result["error"]


@patch("src.explainability.url_explainer.predict_url")
@patch("src.explainability.url_explainer._cached_classifier")
def test_explain_url_missing_model(mock_classifier, mock_predict_url):
    """Test handling when model fails to load."""
    
    mock_predict_url.return_value = {
        "verdict": "phishing",
        "phishing_probability": 0.99,
        "features": {"url_length": 100}
    }
    
    # Simulate missing model
    mock_classifier.model = None
    
    result = explain_url("http://bad-site.com")
    
    assert result["prediction"] == "error"
    assert "Model failed to load" in result["error"]


@patch("src.explainability.url_explainer.predict_url")
@patch("src.explainability.url_explainer.get_url_classifier")
@patch("src.explainability.url_explainer.shap")
def test_explain_url_fallback_when_cached_classifier_is_none(mock_shap, mock_get_clf, mock_predict_url):
    """Test that explain_url dynamically retrieves classifier when _cached_classifier is None."""
    import src.explainability.url_explainer as exp_mod
    exp_mod._cached_classifier = None
    
    mock_predict_url.return_value = {
        "verdict": "phishing",
        "phishing_probability": 0.95,
        "features": {feat: 0 for feat in ["url_length", "has_ip_address"]}
    }
    
    mock_clf = MagicMock()
    mock_clf.model = MagicMock()
    mock_get_clf.return_value = mock_clf
    
    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = [[], [[0.1] * 17]]
    mock_shap.TreeExplainer.return_value = mock_explainer
    
    res = explain_url("http://phish.com")
    assert res["prediction"] == "PHISHING"
    mock_get_clf.assert_called_once()
