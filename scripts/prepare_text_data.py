"""
AegisAI — Prepare Text Data

Loads raw text datasets (SMS/Email) from data/raw/ using config-driven mappings,
cleans them, removes duplicates, normalizes labels, and saves the result to
data/processed/text_clean.csv.

Usage:
    python scripts/prepare_text_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.settings import SETTINGS
from src.preprocessing.text_features import clean_text


def load_dataset(filepath: Path, config: dict) -> pd.DataFrame:
    """Load a dataset if it exists, otherwise return an empty DataFrame."""
    if not filepath.exists():
        print(f"[WARN] Dataset not present: {filepath}")
        return pd.DataFrame()

    print(f"[INFO] Loading {filepath.name}...")
    try:
        kwargs = {}
        if config.get("separator"):
            kwargs["sep"] = config["separator"]
        if config.get("has_header") is False:
            kwargs["header"] = None
            
        df = pd.read_csv(filepath, on_bad_lines="skip", **kwargs)
        print(f"       Loaded {len(df)} rows, columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath.name}: {e}")
        return pd.DataFrame()


def extract_text_label(df: pd.DataFrame, config: dict, filename: str) -> pd.DataFrame:
    """Extract text and label columns according to config."""
    result = pd.DataFrame()
    
    text_col = config["text_col"]
    label_col = config["label_col"]

    if text_col not in df.columns:
        print(f"[WARN] Skipping {filename}: expected text column '{text_col}' not found.")
        return pd.DataFrame()
    
    result["text"] = df[text_col]

    if config.get("fixed_label") is not None:
        result["label"] = config["fixed_label"]
    elif label_col is not None:
        if label_col not in df.columns:
            print(f"[WARN] Skipping {filename}: expected label column '{label_col}' not found.")
            return pd.DataFrame()
            
        raw_labels = df[label_col]
        label_map = config.get("label_map")
        
        if label_map:
            # Map labels
            mapped = raw_labels.map(label_map)
            unmapped = mapped.isna() & raw_labels.notna()
            if unmapped.any():
                print(f"[WARN] {filename}: {unmapped.sum()} rows had unmapped labels. Dropping them.")
                mapped = mapped.dropna()
            result["label"] = mapped.astype(int)
        else:
            result["label"] = raw_labels.astype(int)
            
        print(f"       Label distribution: {result['label'].value_counts().to_dict()}")
    else:
        print(f"[ERROR] Skipping {filename}: no label_col or fixed_label configured.")
        return pd.DataFrame()

    result["source"] = filename
    return result


def prepare_text_datasets() -> None:
    """Main pipeline for preparing Text datasets."""
    SETTINGS.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    
    for config in SETTINGS.RAW_TEXT_DATASET_CONFIGS:
        filename = config["filename"]
        filepath = SETTINGS.DATA_RAW_DIR / filename
        df = load_dataset(filepath, config)
        
        if df.empty:
            continue
            
        extracted = extract_text_label(df, config, filename)
        if not extracted.empty:
            dfs.append(extracted)

    if not dfs:
        print("[INFO] Dataset not present — preprocessing pipeline is ready but waiting for the dataset.")
        return

    # Combine all datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    initial_rows = len(combined_df)
    print(f"\n[INFO] Combined {len(dfs)} source(s): {initial_rows} total rows")
    
    # 1. Remove missing/empty text
    combined_df.dropna(subset=['text'], inplace=True)
    combined_df['text'] = combined_df['text'].astype(str).str.strip()
    combined_df = combined_df[combined_df['text'] != '']
    valid_text_rows = len(combined_df)
    empty_removed = initial_rows - valid_text_rows
    
    # 2. Normalize text and remove duplicates
    # Note: We clean first, then deduplicate based on cleaned text to catch functional duplicates.
    print("[INFO] Cleaning text...")
    combined_df['text'] = combined_df['text'].apply(clean_text)
    
    # Drop empty strings again after cleaning (e.g. if a text was just HTML tags)
    combined_df = combined_df[combined_df['text'].str.strip() != '']
    
    pre_dedup = len(combined_df)
    combined_df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    unique_rows = len(combined_df)
    duplicates_removed = pre_dedup - unique_rows
    
    # 3. Validate labels
    valid_labels = combined_df["label"].isin([SETTINGS.LABEL_LEGITIMATE, SETTINGS.LABEL_PHISHING])
    invalid_label_rows = (~valid_labels).sum()
    if invalid_label_rows > 0:
        print(f"[WARN] Dropping {invalid_label_rows} rows with invalid labels")
        combined_df = combined_df[valid_labels]
        
    final_rows = len(combined_df)
    legitimate_count = (combined_df['label'] == SETTINGS.LABEL_LEGITIMATE).sum()
    phishing_count = (combined_df['label'] == SETTINGS.LABEL_PHISHING).sum()
    
    # Generate summary
    print("\n" + "="*50)
    print("TEXT DATASET SUMMARY")
    print("="*50)
    print(f"Total initial rows (all sources):  {initial_rows}")
    print(f"Empty/missing text removed:        {empty_removed}")
    print(f"Duplicate messages removed:        {duplicates_removed}")
    print(f"Invalid label rows removed:        {invalid_label_rows}")
    print(f"Final valid unique rows:           {final_rows}")
    print("-" * 50)
    print(f"Legitimate/Ham (label=0):  {legitimate_count} ({legitimate_count / final_rows * 100:.2f}%)")
    print(f"Phishing/Spam  (label=1):  {phishing_count} ({phishing_count / final_rows * 100:.2f}%)")
    print("-" * 50)
    
    print("Source breakdown:")
    for src, count in combined_df["source"].value_counts().items():
        print(f"  {src}: {count}")
    print("="*50 + "\n")
    
    # Save cleaned dataset
    output_path = SETTINGS.DATA_PROCESSED_DIR / "text_clean.csv"
    combined_df[['text', 'label']].to_csv(output_path, index=False)
    print(f"[INFO] Cleaned text dataset saved to: {output_path}")


if __name__ == "__main__":
    prepare_text_datasets()
