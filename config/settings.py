"""
AegisAI — Centralized Configuration

All project-wide settings, file paths, model parameters, and feature
definitions live here. Import this module wherever configuration is needed
instead of hard-coding values.

Usage:
    from config.settings import SETTINGS
    print(SETTINGS.DATA_RAW_DIR)
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project root (two levels up from this file: config/settings.py → AegisAI/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _Settings:
    """Immutable project-wide settings."""

    # --- Paths ---
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # --- Raw Dataset Expected Filenames and Column Mappings ---
    # Each entry: (filename, url_column, label_column_or_None, label_value_map_or_None)
    # If label_column is None, a fixed_label must be provided.
    RAW_URL_DATASET_CONFIGS: list[dict] = [
        {
            "filename": "verified_online.csv",       # PhishTank
            "url_col": "url",
            "label_col": None,                       # No label column — all rows are phishing
            "fixed_label": 1,                        # 1 = phishing
            "label_map": None,
        },
        {
            "filename": "phishtank_online_valid.csv", # PhishTank (downloaded via script)
            "url_col": "url",
            "label_col": None,
            "fixed_label": 1,
            "label_map": None,
        },
        {
            "filename": "dataset_phishing.csv",       # Kaggle — both phishing and legitimate
            "url_col": "url",
            "label_col": "status",
            "fixed_label": None,
            "label_map": {"legitimate": 0, "phishing": 1},
        },
        {
            "filename": "kaggle_phishing_urls.csv",   # Alternative Kaggle filename
            "url_col": "url",
            "label_col": "label",
            "fixed_label": None,
            "label_map": None,                        # Assumes numeric 0/1
        },
    ]
    # --- Raw Text Dataset Expected Filenames and Column Mappings ---
    RAW_TEXT_DATASET_CONFIGS: list[dict] = [
        {
            "filename": "SMSSpamCollection",          # UCI SMS Spam
            "separator": "\t",                        # TSV
            "has_header": False,
            "url_col": None,
            "text_col": 1,                            # column index 1 (second column)
            "label_col": 0,                           # column index 0 (first column)
            "fixed_label": None,
            "label_map": {"ham": 0, "spam": 1},
        },
        {
            "filename": "spamassassin_emails.csv",    # Placeholder if added later
            "separator": ",",
            "has_header": True,
            "url_col": None,
            "text_col": "text",
            "label_col": "label",
            "fixed_label": None,
            "label_map": {"ham": 0, "spam": 1},
        }
    ]

    # --- Preprocessing ---
    LABEL_LEGITIMATE: int = 0
    LABEL_PHISHING: int = 1
    
    URL_TARGET_COL: str = "label"
    TEXT_TARGET_COL: str = "label"
    TEXT_FEATURE_COL: str = "text"
    URL_FEATURE_COL: str = "url"

    # --- URL Feature Extraction ---
    URL_FEATURE_NAMES: list[str] = [
        "url_length",
        "hostname_length",
        "path_length",
        "query_length",
        "num_dots",
        "num_hyphens",
        "num_underscores",
        "num_special_chars",
        "num_digits",
        "num_subdomains",
        "has_https",
        "has_ip_address",
        "has_at_symbol",
        "url_entropy",
        "domain_length",
        "path_depth",
        "query_param_count"
    ]

    # --- Text/NLP ---
    TFIDF_MAX_FEATURES: int = 5000
    TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)

    # --- Model Defaults ---
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2

    # --- API ---
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # --- QR ---
    MAX_QR_IMAGE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # --- SHAP ---
    SHAP_MAX_DISPLAY_FEATURES: int = 20


SETTINGS = _Settings()
