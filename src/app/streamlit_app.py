"""
AegisAI — Streamlit Frontend

Provides a simple web interface for:
    - Checking URLs for phishing
    - Checking text messages for phishing
    - Uploading QR code images for analysis

Usage:
    streamlit run src/app/streamlit_app.py

TODO
----
- Build URL input tab with prediction display
- Build text input tab with prediction display
- Build QR upload tab with image preview and prediction
- Display SHAP explanations as bar charts or waterfall plots
- Add confidence gauges and visual indicators
"""


def main() -> None:
    """Entry point for the Streamlit application.

    TODO
    ----
    - Set page config (title, icon, layout)
    - Create sidebar navigation or tabs for URL / Text / QR
    - Call API endpoints or model directly
    - Display results with SHAP visualisations
    """
    raise NotImplementedError("Streamlit app not yet implemented.")


if __name__ == "__main__":
    main()
