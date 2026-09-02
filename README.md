# AegisAI — AI-Powered Phishing Detection System

AegisAI is a machine-learning-based phishing detection system that identifies
phishing threats across **URLs**, **text messages (email/SMS)**, and **QR codes**.

> **Status:** Phase 8 complete — Streamlit Frontend and Full System Integration.

---

## Features (MVP Scope)

| Capability | Method | Verified Test Accuracy |
|---|---|---|
| **URL phishing detection** | Engineered URL features + classical ML (Random Forest) | 92.36% (F1: 0.9580) |
| **Text/email/SMS detection** | TF-IDF vectorization + classical ML (Linear SVM) | 97.86% (F1: 0.9127) |
| **QR code detection** | OpenCV/pyzbar decode → URL classifier pipeline | — |
| **Explainability** | SHAP values for per-prediction explanations | — |
| **FastAPI Backend** | Production-ready REST endpoints | 58/58 Tests Passed |
| **Streamlit Frontend** | Interactive cybersecurity dashboard | — |

> 📄 **See [docs/final_evaluation.md](file:///c:/Users/fathima%20efa/OneDrive/Desktop/AegisAI/docs/final_evaluation.md) for the complete end-to-end evaluation report, confusion matrices, and security validations.**

## Project Structure

```
AegisAI/
├── config/           # Centralized configuration
├── data/             # Raw and processed datasets (gitignored)
├── models/           # Saved model artifacts (gitignored)
├── notebooks/        # Exploratory analysis and experiments
├── reports/          # Evaluation results, confusion matrices
├── scripts/          # CLI entry points (training, evaluation)
├── src/              # Main source package
│   ├── preprocessing/  # Feature engineering (URL and Text)
│   ├── ml/             # Model training and prediction (URL and Text)
│   ├── qr/             # QR code processing
│   ├── explainability/ # SHAP explanations
│   ├── api/            # FastAPI backend
│   └── app/            # Streamlit frontend
└── tests/            # Unit and integration tests
```

## Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AegisAI

# Create a virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Setup

```bash
# Download NLTK data (required for text preprocessing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model (if using spaCy-based features)
python -m spacy download en_core_web_sm
```

## Usage

### Phase 2: Data Preparation

Before training, you must collect and prepare the datasets. See `data/README.md` for download instructions. Once downloaded into `data/raw/`, run:

```bash
python scripts/prepare_url_data.py
python scripts/prepare_text_data.py
```

### Phase 3: URL Model Training

The URL model training pipeline trains and compares Random Forest, Naive Bayes, SVM, and XGBoost classifiers using 17 engineered URL features.

**Features used:** URL length, hostname length, path length, query length, number of dots, hyphens, underscores, special characters, digits, subdomains, HTTPS presence, IP-address detection, @ symbol detection, URL entropy, domain length, path depth, query parameter count.

**Class imbalance handling:** The dataset is heavily imbalanced (~93% phishing / ~7% legitimate). Random Forest and SVM use `class_weight='balanced'`; XGBoost uses `scale_pos_weight`; Naive Bayes uses computed sample weights.

**Train command:**

```bash
python scripts/train.py --pipeline url
```

This will output `models/best_url_model.joblib`.

### Phase 4: Text Model Training

The Text model training pipeline trains and compares Multinomial Naive Bayes, Logistic Regression, and Linear SVM classifiers using TF-IDF vectorization.

**Dataset used:** UCI SMS Spam Collection.
**Features used:** TF-IDF n-grams (1,2) with a maximum of 5,000 features. Punctuation is stripped except for key indicators (like `!`, `$`, `£`, `?`).
**Class imbalance handling:** `class_weight='balanced'` for SVM/LR, and `sample_weight` for Naive Bayes.

**Train command:**

```bash
python scripts/train.py --pipeline text
```

This will output `models/best_text_model.joblib`.

### Phase 5: QR / Quishing Detection

The QR code processor safely decodes payloads and routes them for classification without making any HTTP requests.

**Capabilities:** OpenCV and pyzbar (fallback) decoding, payload classification, safe URL extraction, and risk indicator generation.
**Integration:** Automatically routes URL payloads to the trained Phase 3 URL model.

**Usage (Python API):**

```python
from src.qr.classifier import process_qr
result = process_qr("path/to/qr_image.png")
print(result["verdict"])
```

### Phase 6: SHAP Explainability

The URL explainability module provides human-readable reasons for predictions by applying SHAP to the Random Forest model.

**Usage (Python API):**

```python
from src.explainability.url_explainer import explain_url
result = explain_url("http://192.168.1.1/secure/login")

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
print("Reasons:")
for reason in result['reasons']:
    print(f"- {reason}")
```

### Starting the API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```
*API documentation is available at `http://localhost:8000/docs` once the server is running.*

### API Endpoints

**1. Predict URL**
```bash
curl -X POST "http://localhost:8000/predict/url" \
     -H "Content-Type: application/json" \
     -d '{"url": "http://example-phishing-site.com"}'
```

**2. Predict Text (SMS/Email)**
```bash
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "URGENT: Click here to claim your prize!"}'
```

**3. Predict QR Code**
```bash
curl -X POST "http://localhost:8000/predict/qr" \
     -F "file=@path/to/qr_image.png"
```

**4. Explain URL Prediction (SHAP)**
```bash
curl -X POST "http://localhost:8000/explain/url" \
     -H "Content-Type: application/json" \
     -d '{"url": "http://192.168.1.1/login"}'
```

## Security Guarantees
- The system **never** automatically visits or fetches content from submitted URLs.
- The system **never** executes decoded QR payloads.
- All URL inputs are treated strictly as static strings for feature extraction.

### Streamlit Frontend

```bash
streamlit run src/app/streamlit_app.py
```

### Running Tests

```bash
pytest tests/ -v
```

## Future Scope

The following are planned for future development phases and are **not** part
of the current MVP:

- Siamese Vision Transformer for visual phishing detection
- GAN-based adversarial red-team engine
- Enterprise monitoring dashboard
- Mobile SDK
- Vector database integration (Pinecone)
- Large-scale production deployment

## License

This project is developed as a college project.

## Authors

AegisAI Team
