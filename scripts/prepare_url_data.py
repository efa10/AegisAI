"""
AegisAI — Prepare URL Data

Loads raw URL datasets from data/raw/ using config-driven column mappings,
cleans them, removes duplicates, normalizes labels, and saves the result to
data/processed/urls_clean.csv.

Supports combining multiple sources (PhishTank, Kaggle, etc.) without
duplicate URLs or conflicting labels.

Usage:
    python scripts/prepare_url_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config.settings import SETTINGS


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load a CSV dataset if it exists, otherwise return an empty DataFrame."""
    if not filepath.exists():
        print(f"[WARN] Dataset not present: {filepath}")
        return pd.DataFrame()

    print(f"[INFO] Loading {filepath.name}...")
    try:
        df = pd.read_csv(filepath)
        print(f"       Loaded {len(df)} rows, columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath.name}: {e}")
        return pd.DataFrame()


def extract_url_label(df: pd.DataFrame, config: dict, filename: str) -> pd.DataFrame:
    """Extract url and label columns from a dataset according to its config.

    Parameters
    ----------
    df : pd.DataFrame
        The raw loaded DataFrame.
    config : dict
        Dataset configuration with url_col, label_col, fixed_label, label_map.
    filename : str
        The filename (for log messages).

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly two columns: 'url' and 'label'.
    """
    url_col = config["url_col"]

    if url_col not in df.columns:
        print(f"[WARN] Skipping {filename}: expected URL column '{url_col}' not found. "
              f"Columns: {list(df.columns)}")
        return pd.DataFrame()

    result = pd.DataFrame()
    result["url"] = df[url_col]

    # Determine labels
    if config.get("fixed_label") is not None:
        # All rows get the same fixed label (e.g. PhishTank = all phishing)
        result["label"] = config["fixed_label"]
        print(f"       Assigned fixed label={config['fixed_label']} to all {len(result)} rows")
    elif config.get("label_col") is not None:
        label_col = config["label_col"]
        if label_col not in df.columns:
            print(f"[WARN] Skipping {filename}: expected label column '{label_col}' not found. "
                  f"Columns: {list(df.columns)}")
            return pd.DataFrame()

        raw_labels = df[label_col]

        # Apply label map if provided
        label_map = config.get("label_map")
        if label_map:
            mapped = raw_labels.map(label_map)
            unmapped = mapped.isna() & raw_labels.notna()
            if unmapped.any():
                unmapped_vals = raw_labels[unmapped].unique()[:10]
                print(f"[WARN] {filename}: {unmapped.sum()} rows had unmapped label values: {unmapped_vals}")
                print(f"       Dropping rows with unmapped labels.")
                mapped = mapped.dropna()
            result["label"] = mapped.astype(int)
        else:
            # Assume numeric 0/1
            result["label"] = raw_labels.astype(int)

        print(f"       Label distribution: {result['label'].value_counts().to_dict()}")
    else:
        print(f"[ERROR] Skipping {filename}: no label_col or fixed_label configured.")
        return pd.DataFrame()

    result["source"] = filename
    return result


def prepare_url_datasets() -> None:
    """Main pipeline for preparing URL datasets."""
    SETTINGS.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []

    for config in SETTINGS.RAW_URL_DATASET_CONFIGS:
        filename = config["filename"]
        filepath = SETTINGS.DATA_RAW_DIR / filename
        df = load_dataset(filepath)

        if df.empty:
            continue

        extracted = extract_url_label(df, config, filename)
        if not extracted.empty:
            dfs.append(extracted)

    if not dfs:
        print("[INFO] Dataset not present — preprocessing pipeline is ready but waiting for the dataset.")
        return

    # Combine all datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    initial_rows = len(combined_df)
    print(f"\n[INFO] Combined {len(dfs)} source(s): {initial_rows} total rows")

    # 1. Remove missing/empty URLs
    combined_df.dropna(subset=["url"], inplace=True)
    combined_df["url"] = combined_df["url"].astype(str).str.strip()
    combined_df = combined_df[combined_df["url"] != ""]
    valid_urls_rows = len(combined_df)
    empty_removed = initial_rows - valid_urls_rows

    # 2. Normalize URLs (strip whitespace, lowercase)
    combined_df["url"] = combined_df["url"].str.lower()

    # 3. Remove exact duplicate URLs
    # When duplicates exist from different sources, keep the first occurrence.
    # This avoids conflicting labels — first-seen wins.
    pre_dedup = len(combined_df)
    combined_df.drop_duplicates(subset=["url"], keep="first", inplace=True)
    unique_rows = len(combined_df)
    duplicates_removed = pre_dedup - unique_rows

    # 4. Validate labels — must be 0 or 1
    valid_labels = combined_df["label"].isin([SETTINGS.LABEL_LEGITIMATE, SETTINGS.LABEL_PHISHING])
    invalid_label_rows = (~valid_labels).sum()
    if invalid_label_rows > 0:
        print(f"[WARN] Dropping {invalid_label_rows} rows with invalid labels")
        combined_df = combined_df[valid_labels]

    final_rows = len(combined_df)
    legitimate_count = (combined_df["label"] == SETTINGS.LABEL_LEGITIMATE).sum()
    phishing_count = (combined_df["label"] == SETTINGS.LABEL_PHISHING).sum()

    # 5. Balance classes to prevent severe artificial skew and feature correlation inversion
    if legitimate_count > 0 and phishing_count > legitimate_count:
        print(f"[INFO] Balancing classes to avoid artificial skew: matching phishing to legitimate ({legitimate_count} each)...")
        legit_df = combined_df[combined_df["label"] == SETTINGS.LABEL_LEGITIMATE]
        phish_df = combined_df[combined_df["label"] == SETTINGS.LABEL_PHISHING]
        # Prioritize matching phishing from same source as legitimate if available
        same_source_phish = phish_df[phish_df["source"].isin(legit_df["source"].unique())]
        if len(same_source_phish) >= len(legit_df):
            phish_sampled = same_source_phish.sample(n=len(legit_df), random_state=SETTINGS.RANDOM_STATE)
        else:
            remaining = len(legit_df) - len(same_source_phish)
            other_phish = phish_df[~phish_df.index.isin(same_source_phish.index)]
            other_sampled = other_phish.sample(n=min(remaining, len(other_phish)), random_state=SETTINGS.RANDOM_STATE)
            phish_sampled = pd.concat([same_source_phish, other_sampled])
        
        combined_df = pd.concat([legit_df, phish_sampled], ignore_index=True).sample(
            frac=1.0, random_state=SETTINGS.RANDOM_STATE
        ).reset_index(drop=True)
        final_rows = len(combined_df)
        legitimate_count = (combined_df["label"] == SETTINGS.LABEL_LEGITIMATE).sum()
        phishing_count = (combined_df["label"] == SETTINGS.LABEL_PHISHING).sum()

    # Print summary
    print("\n" + "=" * 50)
    print("URL DATASET SUMMARY")
    print("=" * 50)
    print(f"Total initial rows (all sources):  {initial_rows}")
    print(f"Empty/missing URLs removed:        {empty_removed}")
    print(f"Duplicate URLs removed:            {duplicates_removed}")
    print(f"Invalid label rows removed:        {invalid_label_rows}")
    print(f"Final valid unique rows:           {final_rows}")
    print("-" * 50)
    print(f"Legitimate URLs (label=0):  {legitimate_count} ({legitimate_count / final_rows * 100:.2f}%)")
    print(f"Phishing URLs   (label=1):  {phishing_count} ({phishing_count / final_rows * 100:.2f}%)")
    print("-" * 50)

    # Source breakdown
    print("Source breakdown:")
    for src, count in combined_df["source"].value_counts().items():
        print(f"  {src}: {count}")
    print("=" * 50 + "\n")

    # Save cleaned dataset (drop the 'source' helper column)
    output_path = SETTINGS.DATA_PROCESSED_DIR / "urls_clean.csv"
    combined_df[["url", "label"]].to_csv(output_path, index=False)
    print(f"[INFO] Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    prepare_url_datasets()
