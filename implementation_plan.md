# AegisAI — Project Structure Scaffolding

Create a clean, modular Python project structure for an AI-powered phishing detection MVP.

## Proposed Folder Structure

```
AegisAI/
├── README.md                    # Project overview, setup, usage
├── requirements.txt             # Pinned Python dependencies
├── .gitignore                   # Python/ML/IDE ignores
├── config/
│   └── settings.py              # Centralized configuration (paths, thresholds, model params)
├── data/
│   ├── raw/                     # Original datasets (gitignored)
│   ├── processed/               # Cleaned/split datasets (gitignored)
│   └── README.md                # Documents expected datasets & download instructions
├── notebooks/
│   └── README.md                # Explains notebook conventions
├── models/
│   └── README.md                # Saved model artifacts go here (gitignored)
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── url_features.py      # URL feature extraction (length, entropy, special chars, etc.)
│   │   └── text_features.py     # TF-IDF text preprocessing pipeline
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── url_model.py         # URL phishing classifier (train + predict)
│   │   └── text_model.py        # Text/email/SMS classifier (train + predict)
│   ├── qr/
│   │   ├── __init__.py
│   │   └── qr_processor.py      # QR decode via OpenCV → feed URL to URL classifier
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_explainer.py    # SHAP explanations for model predictions
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI app with /check-url, /check-text, /check-qr
│   └── app/
│       ├── __init__.py
│       └── streamlit_app.py     # Streamlit frontend
├── tests/
│   ├── __init__.py
│   ├── test_url_features.py     # Unit tests for URL feature extraction
│   ├── test_text_features.py    # Unit tests for text preprocessing
│   ├── test_qr_processor.py     # Unit tests for QR processing
│   └── test_api.py              # Integration tests for FastAPI endpoints
└── scripts/
    └── train.py                 # CLI entry point for training pipelines (placeholder)
```

## Design Rationale

| Decision | Why |
|---|---|
| `src/` as top-level package | Keeps source separate from data, notebooks, configs — standard Python layout |
| `preprocessing/` separate from `ml/` | Feature engineering is distinct from model training; enables independent iteration |
| `qr/` as its own module | QR decoding is a pure utility; it delegates to `url_model` for classification |
| `config/settings.py` | Single source of truth for paths, thresholds, model hyperparameters |
| `data/raw/` + `data/processed/` | Clear separation of immutable source data vs. derived artifacts |
| `models/` gitignored | Model binaries are large; only code and config go into Git |
| `scripts/train.py` | CLI entry point keeps training separate from library code |

## What Each File Will Contain (This Task)

Every Python file will contain:
- A module docstring explaining its purpose
- Placeholder class/function stubs with `TODO` comments
- Correct imports referencing sibling modules (to verify import structure)

**No fake metrics, no pretrained models, no dataset downloads.** Just clean scaffolding.

## Dependencies

```
# Core Data Science
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
joblib>=1.2.0
tqdm>=4.65.0

# ML Models
xgboost>=1.7.0
imbalanced-learn>=0.10.0

# NLP
nltk>=3.8.0
spacy>=3.5.0

# Explainability
shap>=0.42.0

# QR Code Processing
opencv-python>=4.7.0
pyzbar>=0.1.9

# API
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
python-multipart>=0.0.6

# Frontend
streamlit>=1.25.0

# Testing
pytest>=7.4.0
httpx>=0.24.0
```

All versions use compatible minimum bounds (`>=`), not hard pins. Python 3.10+ required.

## Files to Create

### [NEW] README.md
Project overview, feature list, setup instructions, usage guide.

### [NEW] requirements.txt
Dependencies with compatible minimum versions (see above).

### [NEW] .gitignore
Python, ML, IDE, and OS-specific ignores.

### [NEW] config/settings.py
Centralized paths, model config, feature lists.

### [NEW] All `src/` modules
Stub implementations with docstrings and TODOs.

### [NEW] All `tests/` modules
Stub test cases with meaningful names.

### [NEW] data/README.md, notebooks/README.md, models/README.md
Documentation placeholders.

### [NEW] scripts/train.py
CLI placeholder for training pipeline.

## Verification Plan

### Automated Tests
```bash
python -m py_compile src/__init__.py
python -c "import src"
pytest tests/ --collect-only
```
We will verify that all files are syntactically valid Python and that pytest can discover the test stubs.
