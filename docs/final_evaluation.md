# AegisAI — Final Project Evaluation & Validation Report

This report summarizes the final testing, metrics, and validation of the AegisAI Phishing Detection System. All metrics reported here are derived from programmatically executing the trained models against the held-out test datasets.

## 1. Project Overview
AegisAI is a multi-modal machine learning system designed to detect phishing threats across three vectors:
- **URLs**: Analyzed via 17 extracted lexical and structural features.
- **Text (SMS/Email)**: Analyzed via TF-IDF vectorization.
- **QR Codes (Quishing)**: Decoded and analyzed through the URL pipeline.

The system incorporates **SHAP** for human-readable explainability, a **FastAPI** backend for RESTful inference, and a **Streamlit** dashboard for end-user interaction.

## 2. Dataset Summary
- **URL Dataset**: Sourced from PhishTank (verified phishing) and Kaggle.
  - Total URLs evaluated in test set: `17,110`
  - Class Distribution (Test Set): `15,967` Phishing (1), `1,143` Legitimate (0)
- **Text Dataset**: Sourced from the UCI SMS Spam Collection.
  - Total Texts evaluated in test set: `1,026`
  - Class Distribution (Test Set): `126` Spam/Phishing (1), `900` Legitimate (0)

## 3. URL Model Results (Random Forest)
During Phase 3, four models were compared (Random Forest, Naive Bayes, SVM, XGBoost). **Random Forest** was selected for its superior F1 score (`0.958`) and accuracy over XGBoost (`0.939`) and Naive Bayes (`0.351`).

**Final Test Set Evaluation (Random Forest):**
- **Accuracy**: 0.9236
- **Precision**: 0.9841
- **Recall**: 0.9332
- **F1-score**: 0.9580

**Confusion Matrix:**
- **True Negatives (TN)**: 902
- **False Positives (FP)**: 241
- **False Negatives (FN)**: 1,066
- **True Positives (TP)**: 14,901

## 4. Text Model Results (Linear SVM)
During Phase 4, a Linear SVM trained on TF-IDF vectors was selected as the optimal model for text classification.

**Final Test Set Evaluation (Linear SVM):**
- **Accuracy**: 0.9786
- **Precision**: 0.9127
- **Recall**: 0.9127
- **F1-score**: 0.9127

**Confusion Matrix:**
- **True Negatives (TN)**: 889
- **False Positives (FP)**: 11
- **False Negatives (FN)**: 11
- **True Positives (TP)**: 115

## 5. QR / Quishing Validation
The Phase 5 QR detection component (powered by OpenCV/pyzbar) was tested against:
- QR codes containing URLs (correctly routed to URL classifier).
- QR codes containing plain text (correctly identified, bypassing URL classification).
- Invalid/No-QR images (safely failed with appropriate UI warnings).
- **Validation**: Decoded URLs are NEVER automatically opened or visited.

## 6. SHAP Validation
The Phase 6 explainability module successfully hooks into the Random Forest URL classifier.
- **Validation**: Generates correct feature contributions corresponding to the 17 extracted URL features.
- Maps raw feature impact to human-readable risk indicators (e.g., "High number of subdomains detected").

## 7. API Validation
The Phase 7 FastAPI backend passes all automated endpoint checks.
- `GET /health`: Returns 200 OK.
- `POST /predict/url`: Successfully handles valid URLs, empty strings (returns 400), and missing payloads.
- `POST /predict/qr`: Validates `multipart/form-data` and strictly enforces image content types.

## 8. Streamlit Validation
The Phase 8 Streamlit frontend connects seamlessly to the FastAPI backend.
- UI elements (Tabs, sidebars) render correctly.
- Background health checks verify the API status without blocking the main UI thread.
- **Validation**: The frontend strictly treats input as static text. It does not attempt to render HTML, fetch URL thumbnails, or trigger external HTTP requests.

## 9. Full Test Results
The automated test suite (`pytest tests/`) was executed on the final project state.
- **Total Tests**: 58
- **Passed**: 58
- **Failed**: 0
- **Skipped**: 0

## 10. Limitations
- **Synchronous UI**: The Streamlit frontend uses synchronous HTTP requests, which may block the UI if the backend experiences high latency.
- **Explainability Constraints**: SHAP explanations are currently only implemented for the URL Random Forest model; text classification does not yet support granular explainability.
- **Static Analysis**: The system relies entirely on static lexical analysis. It does not perform dynamic analysis (e.g., executing JS, following redirects, or analyzing DOM structure).

## 11. Security Considerations
- **No Active Probing**: AegisAI **never** makes external HTTP requests to submitted URLs or extracted QR payloads. This prevents the system from triggering malicious trackers or inadvertently downloading malware.
- **Payload Safety**: QR codes are decoded into text strings and treated purely as data.

## 12. Future Scope
- Implement dynamic analysis (DOM parsing via headless browser) in a secure sandbox.
- Expand text classification explainability using LLMs or LIME.
- Persist predictions and user feedback to a database for continuous model retraining.
