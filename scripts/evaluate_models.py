"""
AegisAI — Final Model Evaluation Script

Loads the existing processed datasets and trained models to compute
final evaluation metrics for Phase 9 reporting.
"""

import sys
import pandas as pd
from pathlib import Path

from config.settings import SETTINGS
from src.ml.url_model import URLClassifier
from src.ml.text_model import TextClassifier
from src.preprocessing.utils import split_data

def evaluate_url_model():
    print("=" * 50)
    print("URL MODEL EVALUATION")
    print("=" * 50)
    
    data_path = SETTINGS.DATA_PROCESSED_DIR / "urls_clean.csv"
    if not data_path.exists():
        print("URL dataset not found.")
        return
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=["url", "label", "domain"], errors="ignore")
    y = df["label"]
    
    from src.preprocessing.url_features import extract_url_features_batch
    import time
    import numpy as np
    
    # Use the same split as training to get the test set
    _, X_test_urls, _, y_test = split_data(df, "label", test_size=SETTINGS.TEST_SIZE, random_state=SETTINGS.RANDOM_STATE)
    
    print("Extracting URL features (this may take a few seconds)...")
    X_test = extract_url_features_batch(X_test_urls["url"])
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    
    model_path = SETTINGS.MODELS_DIR / "best_url_model.joblib"
    if not model_path.exists():
        print("URL model not found.")
        return
        
    clf = URLClassifier()
    clf.load_model(model_path)
    
    results = clf.evaluate(X_test, y_test)
    
    print(f"Algorithm: {results['model_name']}")
    print(f"Test Set Size: {len(y_test)} samples")
    print(f"Class Distribution: {y_test.value_counts().to_dict()}")
    print("-" * 30)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1']:.4f}")
    print("-" * 30)
    
    cm = results['confusion_matrix']
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]}  | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}  | TP: {cm[1][1]}")
    print()


def evaluate_text_model():
    print("=" * 50)
    print("TEXT MODEL EVALUATION")
    print("=" * 50)
    
    data_path = SETTINGS.DATA_PROCESSED_DIR / "text_clean.csv"
    if not data_path.exists():
        print("Text dataset not found.")
        return
        
    df = pd.read_csv(data_path)
    # Filter out empty/NaN text
    df = df.dropna(subset=["text"])
    
    _, X_test, _, y_test = split_data(df, "label", test_size=SETTINGS.TEST_SIZE, random_state=SETTINGS.RANDOM_STATE)
    X_test = X_test["text"]
    
    model_path = SETTINGS.MODELS_DIR / "best_text_model.joblib"
    if not model_path.exists():
        print("Text model not found.")
        return
        
    clf = TextClassifier()
    clf.load_model(model_path)
    
    results = clf.evaluate(X_test, y_test)
    
    print(f"Algorithm: {results['model_name']}")
    print(f"Test Set Size: {len(y_test)} samples")
    print(f"Class Distribution: {y_test.value_counts().to_dict()}")
    print("-" * 30)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1']:.4f}")
    print("-" * 30)
    
    cm = results['confusion_matrix']
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]}  | FP: {cm[0][1]}")
    print(f"FN: {cm[1][0]}  | TP: {cm[1][1]}")
    print()


if __name__ == "__main__":
    evaluate_url_model()
    evaluate_text_model()
