"""
AegisAI — Training Script

CLI entry point for training the URL and text phishing classifiers.
Loads the processed dataset, extracts features, trains multiple models,
evaluates them, and saves the best model.

Usage:
    python scripts/train.py --pipeline url
    python scripts/train.py --pipeline text
    python scripts/train.py --pipeline all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from config.settings import SETTINGS

# URL Pipeline imports
from src.preprocessing.url_features import extract_url_features_batch
from src.ml.url_model import URLClassifier

# Text Pipeline imports
from src.ml.text_model import TextClassifier


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AegisAI — Train phishing detection models",
    )
    parser.add_argument(
        "--pipeline",
        choices=["url", "text", "all"],
        default="all",
        help="Which pipeline to train (default: all)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Common Utilities
# ---------------------------------------------------------------------------

def generate_confusion_matrix_plot(cm, model_name: str, output_dir: Path, prefix: str = ""):
    """Save a confusion matrix plot as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)

        classes = ["Legitimate (0)", "Phishing (1)"]
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=classes,
            yticklabels=classes,
            ylabel="True Label",
            xlabel="Predicted Label",
            title=f"Confusion Matrix — {model_name}",
        )

        # Write cell values
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i][j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i][j] > cm.max() / 2 else "black",
                        fontsize=14)

        fig.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        filename = f"confusion_matrix_{prefix}{safe_name}.png"
        path = output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"       Saved: {path}")
    except ImportError:
        print(f"[WARN] matplotlib not available — skipping confusion matrix plot for {model_name}")


def create_splits(X, y, random_state=SETTINGS.RANDOM_STATE):
    """Create 70/15/15 stratified train/val/test split."""
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    # Second split: 50/50 of the 30% → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    print(f"\n[INFO] Data splits (stratified, seed={random_state}):")
    print(f"       Train: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
    print(f"       Val:   {len(y_val)} ({len(y_val)/len(y)*100:.1f}%)")
    print(f"       Test:  {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")

    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = ys.value_counts().to_dict()
        print(f"       {name} class dist: {counts}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# URL Pipeline
# ---------------------------------------------------------------------------

def load_url_dataset() -> pd.DataFrame:
    data_path = SETTINGS.DATA_PROCESSED_DIR / "urls_clean.csv"
    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        sys.exit(1)
    df = pd.read_csv(data_path)
    return df

def extract_url_features_with_log(urls: pd.Series, label: str) -> pd.DataFrame:
    print(f"[INFO] Extracting features for {label} set ({len(urls)} URLs)...")
    start = time.time()
    features = extract_url_features_batch(urls)
    elapsed = time.time() - start
    print(f"       Done in {elapsed:.1f}s — shape: {features.shape}")
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    return features


def train_url_pipeline():
    """Full URL model training pipeline."""
    print("=" * 60)
    print("PHASE 3: URL PHISHING ML TRAINING")
    print("=" * 60)

    df = load_url_dataset()
    X_train_urls, X_val_urls, X_test_urls, y_train, y_val, y_test = create_splits(df["url"], df["label"])

    X_train = extract_url_features_with_log(X_train_urls, "train")
    X_val = extract_url_features_with_log(X_val_urls, "val")
    X_test = extract_url_features_with_log(X_test_urls, "test")

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_configs = [
        {"name": "random_forest", "display_name": "Random Forest", "imbalance_strategy": "class_weight='balanced'"},
        {"name": "naive_bayes", "display_name": "Naive Bayes", "imbalance_strategy": "sample_weight"},
        {"name": "svm", "display_name": "SVM", "imbalance_strategy": "class_weight='balanced'"},
        {"name": "xgboost", "display_name": "XGBoost", "imbalance_strategy": "scale_pos_weight"},
    ]

    all_results = []
    
    n_legit = (y_train == 0).sum()
    n_phish = (y_train == 1).sum()
    scale_pos_weight = n_legit / n_phish if n_phish > 0 else 1.0
    sw = compute_sample_weight("balanced", y_train)

    for config in model_configs:
        name = config["name"]
        display = config["display_name"]
        print(f"\n{'—' * 60}")
        print(f"Training: {display}")

        clf = URLClassifier(name)
        try:
            if name == "xgboost":
                clf.model = None
                from xgboost import XGBClassifier
                clf.model_name = name
                clf.model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=SETTINGS.RANDOM_STATE, use_label_encoder=False, eval_metric="logloss", scale_pos_weight=scale_pos_weight, n_jobs=-1)
                s = time.time()
                clf.model.fit(X_train, y_train)
                train_time = round(time.time() - s, 3)
            else:
                train_meta = clf.train(X_train, y_train, sample_weight=sw if name == "naive_bayes" else None)
                train_time = train_meta["training_time_seconds"]

            val_metrics = clf.evaluate(X_val, y_val)
            val_metrics["training_time_seconds"] = train_time
            val_metrics["display_name"] = display
            val_metrics["imbalance_strategy"] = config["imbalance_strategy"]
            
            print(f"       F1: {val_metrics['f1']}")
            
            generate_confusion_matrix_plot(np.array(val_metrics["confusion_matrix"]), display, reports_dir, prefix="url_")
            all_results.append({"classifier": clf, "metrics": val_metrics})
        except Exception as e:
            print(f"[ERROR] Failed to train {display}: {e}")

    # Select best model (by F1 score)
    best = max(all_results, key=lambda r: r["metrics"]["f1"])
    best_name = best["metrics"]["display_name"]
    best_clf = best["classifier"]

    print(f"\n[INFO] BEST URL MODEL: {best_name} (F1: {best['metrics']['f1']})")
    
    # Save the best model
    model_path = SETTINGS.MODELS_DIR / "best_url_model.joblib"
    best_clf.save_model(model_path)
    print(f"[INFO] Best model saved to: {model_path}")


# ---------------------------------------------------------------------------
# Text Pipeline
# ---------------------------------------------------------------------------

def load_text_dataset() -> pd.DataFrame:
    data_path = SETTINGS.DATA_PROCESSED_DIR / "text_clean.csv"
    if not data_path.exists():
        print(f"[ERROR] Dataset not found: {data_path}")
        print("[ERROR] Run scripts/prepare_text_data.py first.")
        sys.exit(1)
    df = pd.read_csv(data_path)
    df.dropna(subset=['text'], inplace=True)
    return df


def train_text_pipeline():
    """Full Text model training pipeline."""
    print("=" * 60)
    print("PHASE 4: TEXT PHISHING ML TRAINING")
    print("=" * 60)

    df = load_text_dataset()
    X_train_text, X_val_text, X_test_text, y_train, y_val, y_test = create_splits(df["text"], df["label"])

    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_configs = [
        {"name": "naive_bayes", "display_name": "Multinomial Naive Bayes", "imbalance_strategy": "sample_weight"},
        {"name": "logistic_regression", "display_name": "Logistic Regression", "imbalance_strategy": "class_weight='balanced'"},
        {"name": "linear_svm", "display_name": "Linear SVM", "imbalance_strategy": "class_weight='balanced'"},
    ]

    all_results = []
    sw = compute_sample_weight("balanced", y_train)

    for config in model_configs:
        name = config["name"]
        display = config["display_name"]
        print(f"\n{'—' * 60}")
        print(f"Training: {display}")

        clf = TextClassifier(name)
        try:
            train_meta = clf.train(
                X_train_text, y_train, 
                sample_weight=sw if name == "naive_bayes" else None,
                class_weight="balanced" if name != "naive_bayes" else None
            )
            train_time = train_meta["training_time_seconds"]

            val_metrics = clf.evaluate(X_val_text, y_val)
            val_metrics["training_time_seconds"] = train_time
            val_metrics["display_name"] = display
            val_metrics["imbalance_strategy"] = config["imbalance_strategy"]
            
            print(f"       F1: {val_metrics['f1']}  Precision: {val_metrics['precision']}  Recall: {val_metrics['recall']}")
            
            generate_confusion_matrix_plot(np.array(val_metrics["confusion_matrix"]), display, reports_dir, prefix="text_")
            all_results.append({"classifier": clf, "metrics": val_metrics})
        except Exception as e:
            print(f"[ERROR] Failed to train {display}: {e}")

    if not all_results:
        print("[ERROR] No models were successfully trained. Aborting.")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON (Validation Set)")
    print(f"{'=' * 80}")
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Train(s)':>10} {'Latency(ms)':>12}"
    print(header)
    print("-" * len(header))

    comparison_rows = []
    for r in all_results:
        m = r["metrics"]
        row = {
            "model": m["display_name"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "training_time_s": m["training_time_seconds"],
            "latency_ms_per_sample": m["inference_latency_ms_per_sample"],
            "confusion_matrix": m["confusion_matrix"],
            "imbalance_strategy": m["imbalance_strategy"],
        }
        comparison_rows.append(row)
        print(f"{row['model']:<25} {row['accuracy']:>10.5f} {row['precision']:>10.5f} "
              f"{row['recall']:>8.5f} {row['f1']:>8.5f} {row['training_time_s']:>10.3f} "
              f"{row['latency_ms_per_sample']:>12.5f}")

    # Save comparison
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(reports_dir / "text_model_results.csv", index=False)
    with open(reports_dir / "text_model_results.json", "w") as f:
        json.dump(comparison_rows, f, indent=2)

    # Select best model (by F1 score)
    best = max(all_results, key=lambda r: r["metrics"]["f1"])
    best_name = best["metrics"]["display_name"]
    best_clf = best["classifier"]

    print(f"\n{'=' * 60}")
    print(f"BEST TEXT MODEL: {best_name}")
    print(f"{'=' * 60}")
    print(f"  F1:        {best['metrics']['f1']}")

    # Final evaluation on TEST set
    test_metrics = best_clf.evaluate(X_test_text, y_test)
    print(f"\nFINAL TEST SET EVALUATION")
    print(f"  Accuracy:  {test_metrics['accuracy']}")
    print(f"  Precision: {test_metrics['precision']}")
    print(f"  Recall:    {test_metrics['recall']}")
    print(f"  F1:        {test_metrics['f1']}")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']}")

    generate_confusion_matrix_plot(np.array(test_metrics["confusion_matrix"]), f"{best_name} (Test)", reports_dir, prefix="text_")

    test_results = {"model": best_name, "dataset": "test", **test_metrics}
    with open(reports_dir / "best_text_model_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Save the best model
    model_path = SETTINGS.MODELS_DIR / "best_text_model.joblib"
    best_clf.save_model(model_path)
    print(f"\n[INFO] Best text model saved to: {model_path}")

    # Quick smoke test
    from src.ml.text_model import predict_text
    test_texts = [
        "Hey, are we still meeting for lunch today?",
        "URGENT: Your account has been suspended. Click here to verify: http://bad.com"
    ]
    print("\nSMOKE TEST: predict_text()")
    for t in test_texts:
        res = predict_text(t, model_path=model_path)
        print(f"  Text: {t[:40]}...")
        print(f"    Verdict: {res['verdict']} (prob={res['phishing_probability']})")

    print("\nPHASE 4 COMPLETE")


def main() -> None:
    args = parse_args()

    if args.pipeline in ("url", "all"):
        train_url_pipeline()

    if args.pipeline in ("text", "all"):
        train_text_pipeline()


if __name__ == "__main__":
    main()
