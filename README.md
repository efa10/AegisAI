# AegisAI — AI-Powered Phishing Detection & Threat Intelligence Platform

AegisAI is a multi-modal machine learning cybersecurity system engineered to detect phishing threats across **URLs**, **text messages (SMS/Email)**, and **QR codes (Quishing)** with integrated threat intelligence and zero-execution inspection.

> **Status:** Production-Ready MVP — Balanced Multi-Vector ML, Secure Shortened URL Expansion with SSRF Defense, and Operations Dashboard.

---

## Capabilities & Architecture

| Defense Vector | Engine / Methodology | Performance Metrics |
|---|---|---|
| **URL Phishing Detection** | 17 Lexical Features + Balanced Gradient Boosting (XGBoost) | **85.52% Accuracy / 85.37% F1** (2,286 balanced test samples) |
| **Shortened URL Expander** | Bounded HTTP redirect tracing with strict SSRF & DNS rebinding guard | Max 5 hops, 3.0s timeout, RFC 1918 & Cloud Metadata Blocking |
| **SMS / Email NLP** | TF-IDF n-grams (1,2) + Linear Support Vector Machine | **98.25% Accuracy / 92.80% F1** (1,026 test samples) |
| **QR / Quishing Defense** | OpenCV / PyZbar Matrix Decoder ➔ Zero-Execution URL Analysis | Safe payload parsing without browser launching |
| **Threat Intelligence** | Multi-source aggregator (VirusTotal v3, URLhaus, local PhishTank feed) | Resilient non-blocking fallback (API keys in env vars) |
| **Explainability (SHAP)** | TreeExplainer feature attributions & human-readable risk factors | Per-feature impact scores & safety explanations |
| **FastAPI Backend** | Asynchronous REST API with Pydantic validation | 121/121 Passing Automated Tests |
| **Cybersecurity Dashboard**| Streamlit Operations Dashboard (Charcoal/Navy, Cyan, Electric Violet) | Desktop & Mobile Responsive, High-Contrast WCAG AA |

---

## Pipeline & Prediction Semantics

### End-to-End Pipeline
```
[User Input] ──> [Normalization] ──> [Shortener Check] ──> [SSRF Validation] ──> [Bounded Expansion]
                                                                                        │
[Destination URL] <─────────────────────────────────────────────────────────────────────┘
       │
       ├──> 17 Lexical Engineered Features (Entropy, Subdomains, Hostname, Special Chars)
       ├──> Classical ML Inference (Trained on Balanced 50/50 Ground Truth)
       ├──> Non-Blocking External Threat Intel (VirusTotal, URLhaus, Local PhishTank)
       └──> Structured Security Verdict & Semantically Correct Confidence Scoring
```

### Truthful Score & Confidence Semantics
- **`phishing_probability`** ($P \in [0.0, 1.0]$): The raw model posterior probability that the input is malicious.
- **`confidence`** ($C \in [0.5, 1.0]$): The mathematical confidence in the **chosen verdict**:
  - If verdict is `phishing`: $\text{Confidence} = \text{phishing\_probability}$
  - If verdict is `legitimate`: $\text{Confidence} = 1.0 - \text{phishing\_probability}$
  *(Fixes previous UI inversion where safe URLs with 5% phishing probability were misleadingly displayed as having "5% confidence".)*
- **`risk_score`** ($R \in [0.0, 100.0]$): Operational risk score scaled from phishing probability and elevated ($\ge 95.0$) if threat intelligence confirms active threats.

---

## Shortened URL Support & SSRF Protection (Phase 3)

Attackers frequently use URL shortening services (such as TinyURL, Bitly, and t.co) to obfuscate malicious destinations. AegisAI supports transparent redirect tracing with strict cybersecurity controls:

### Supported Shorteners
Supports `tinyurl.com`, `bit.ly`, `t.co`, `is.gd`, `ow.ly`, `cutt.ly`, `buff.ly`, `rebrand.ly`, `shorturl.at`, `tiny.cc`, and generic 3xx redirection chains without assuming that shortened URLs are automatically malicious.

### Security Guarantees & SSRF Mitigations
1. **Zero Browser Execution:** Never launches a browser, never executes JavaScript, never renders HTML DOM, and never submits form actions.
2. **Strict Scheme Whitelisting:** Enforces `http://` and `https://` only. Rejects `file:`, `javascript:`, `data:`, `ftp:`, and `gopher:`.
3. **SSRF & Cloud Metadata Protection:**
   - Hostnames are resolved via DNS before connection.
   - All resolved IP addresses are checked against `ipaddress` rules.
   - **Blocks loopback addresses:** `127.0.0.0/8`, `::1`.
   - **Blocks RFC 1918 private subnets:** `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`.
   - **Blocks link-local addresses:** `169.254.0.0/16`, `fe80::/10`.
   - **Blocks multicast and reserved ranges:** `224.0.0.0/4`, `0.0.0.0/8`.
   - **Blocks cloud metadata endpoints:** Explicitly rejects `169.254.169.254`, `metadata.google.internal`, and AWS instance metadata probes.
   - Validates **every intermediate redirect hop** against the same SSRF guard to prevent open-redirect exploitation into internal clouds.
4. **Bounded Execution:** Enforces a maximum redirect depth of 5 hops, a short 3.0s network timeout, and a strict response-header size limit (10 KB) with immediate connection termination.
5. **Graceful Fallback:** If expansion fails (network timeout, DNS failure, or blocked SSRF), AegisAI reports `expansion_success: false`, provides the exact error, and analyzes the original URL without falsely claiming destination visibility.

---

## Threat Intelligence (Phase 4)

AegisAI aggregates external intelligence feeds to complement local lexical ML:
- **VirusTotal v3:** Queries multi-engine antivirus/security vendor verdicts using base64 URL identifiers. Configured via `VIRUSTOTAL_API_KEY`.
- **URLhaus (abuse.ch):** Checks active malware distribution databases. Configured via optional `URLHAUS_API_KEY`.
- **PhishTank:** Local sub-millisecond in-memory lookup against verified phishing feeds (`data/raw/verified_online.csv`).
- **Resilient Fallback:** Missing API keys, timeouts, or provider rate limits (HTTP 429) **never** interrupt or crash local ML inference. Unreachable feeds report status `"unavailable"` or `"not_configured"`, **never** a false clean bill of health.

---

## Project Structure

```
AegisAI/
├── config/
│   └── settings.py              # Centralized paths, schemas, and hyperparameters
├── data/
│   ├── raw/                     # Original datasets (dataset_phishing.csv, SMSSpamCollection)
│   └── processed/               # Cleaned & balanced datasets (urls_clean.csv, text_clean.csv)
├── models/
│   ├── best_url_model.joblib    # Trained balanced XGBoost classifier
│   └── best_text_model.joblib   # Trained Linear SVM classifier
├── reports/                     # Confusion matrices, CSV evaluations, and JSON metrics
├── scripts/
│   ├── prepare_url_data.py      # Balanced data preparation with duplicate removal
│   ├── prepare_text_data.py     # Text cleaning and preprocessing
│   ├── train.py                 # Multi-model training and evaluation CLI
│   └── evaluate_models.py       # Final evaluation verification script
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI REST backend with structured endpoints
│   ├── frontend/
│   │   └── app.py               # Futuristic Streamlit Operations Dashboard
│   ├── ml/
│   │   ├── url_model.py         # URL classifier serving & confidence semantics
│   │   └── text_model.py        # Text classification serving
│   ├── preprocessing/
│   │   ├── url_features.py      # 17 lexical feature extraction algorithms
│   │   ├── url_expansion.py     # Secure URL expander & SSRF firewall
│   │   └── text_features.py     # TF-IDF n-gram vectorization pipeline
│   ├── qr/
│   │   ├── detector.py          # OpenCV / PyZbar QR matrix decoding
│   │   └── classifier.py        # Safe zero-execution QR payload routing
│   ├── threat_intel/
│   │   └── threat_intel.py      # Multi-feed threat intelligence aggregator
│   └── explainability/
│       └── url_explainer.py     # SHAP tree explainer & human-readable reasons
└── tests/                       # 121 Comprehensive unit & regression tests
```

---

## Quickstart & Installation

### 1. Environment Setup
```bash
git clone <repository-url>
cd AegisAI

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (Optional Threat Intelligence)
```bash
export VIRUSTOTAL_API_KEY="your-virustotal-api-key"
export URLHAUS_API_KEY="your-urlhaus-api-key"  # Optional
```

### 3. Data Preparation & Training
```bash
# Prepare balanced datasets (5,715 legitimate / 5,715 phishing)
python scripts/prepare_url_data.py
python scripts/prepare_text_data.py

# Train models
python scripts/train.py --pipeline url
python scripts/train.py --pipeline text
```

### 4. Running the Dashboard (Frontend)
```bash
streamlit run src/frontend/app.py
```
*Access the operations console at `http://localhost:8501`.*

### 5. Running the API Backend
```bash
uvicorn src.api.main:app --reload --port 8000
```
*Interactive Swagger documentation available at `http://localhost:8000/docs`.*

### 6. Executing Tests
```bash
pytest -v
```
*(Runs all 121 tests verifying SSRF protection, redirect loops, prediction semantics, and model integrity.)*

---

## Known Limitations & Defensive Realism

1. **No Absolute Phishing Detection:** Machine learning phishing detection is probabilistic. No model can guarantee 100% precision or recall against novel zero-day threats.
2. **Lexical Domain Ambiguity:** Deep subpaths on legitimate developer domains (e.g., `github.com/org/repo/blob/master/path/file.ext`) exhibit elevated Shannon entropy and path depth, which purely lexical models may score with higher risk without domain reputation feeds.
3. **Network Constraints:** URL expansion requires network connectivity to resolve public DNS and follow redirects. Air-gapped environments will rely strictly on lexical features of the submitted URL.

---

## License

Developed as part of the AegisAI Academic Security Project.
