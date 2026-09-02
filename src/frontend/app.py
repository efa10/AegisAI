"""
AegisAI — Streamlit Frontend

Provides a professional cybersecurity-themed dashboard to interact
with the AegisAI FastAPI backend.

Usage:
    streamlit run src/frontend/app.py
"""

import os
import requests
import streamlit as st
import pandas as pd

# Configuration
API_URL = os.environ.get("AEGISAI_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AEGISAI - Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cybersecurity theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0b1121;
        color: #e2e8f0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Metrics and cards */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 5px;
        padding: 10px;
    }
    
    /* Verdict colors */
    .verdict-phishing {
        color: #ef4444;
        font-weight: bold;
        font-size: 1.5em;
    }
    .verdict-legitimate {
        color: #10b981;
        font-weight: bold;
        font-size: 1.5em;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #94a3b8;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def display_verdict(verdict: str, confidence: float = None):
    """Render the verdict with appropriate colors."""
    if verdict.lower() == "phishing":
        st.markdown('<p class="verdict-phishing">⚠️ VERDICT: PHISHING</p>', unsafe_allow_html=True)
        st.error("This input exhibits strong indicators of a phishing attack.")
    elif verdict.lower() == "legitimate":
        st.markdown('<p class="verdict-legitimate">✅ VERDICT: LEGITIMATE</p>', unsafe_allow_html=True)
        st.success("This input appears safe based on our models.")
    else:
        st.warning(f"Verdict: {verdict}")
        
    if confidence is not None:
        st.metric("Model Confidence", f"{confidence * 100:.2f}%")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ AEGISAI")
    st.markdown("### AI-Powered Phishing Detection")
    st.markdown("---")
    
    st.markdown("""
    **Detection Modes:**
    - 🔗 URL Classification
    - ✉️ SMS / Email Analysis
    - 📱 QR / Quishing Detection
    - 🧠 URL Explainability
    """)
    
    st.markdown("---")
    api_online = check_api_health()
    if api_online:
        st.success("🟢 API Backend: ONLINE")
    else:
        st.error("🔴 API Backend: OFFLINE")
        st.caption(f"Could not connect to {API_URL}")

# ---------------------------------------------------------------------------
# Main Interface
# ---------------------------------------------------------------------------

st.title("AEGISAI Dashboard")
st.markdown("Select an analysis mode below to evaluate suspicious content.")

if not api_online:
    st.warning("⚠️ The FastAPI backend is currently offline. Predictions will fail until the server is started.")

tab1, tab2, tab3, tab4 = st.tabs(["🔗 URL Detection", "✉️ SMS/Email Detection", "📱 QR Detection", "🧠 URL Explainability"])

# --- TAB 1: URL Detection ---
with tab1:
    st.subheader("URL Phishing Detection")
    url_input = st.text_input("Enter URL to analyze:", placeholder="https://example.com")
    
    if st.button("Analyze URL", type="primary"):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Analyzing URL..."):
                try:
                    response = requests.post(f"{API_URL}/predict/url", json={"url": url_input}, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        display_verdict(data["verdict"], data.get("phishing_probability"))
                        
                        with st.expander("View Raw Features"):
                            st.json(data.get("features", {}))
                    else:
                        st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
                except requests.RequestException as e:
                    st.error(f"Failed to connect to API: {e}")

# --- TAB 2: SMS/Email Detection ---
with tab2:
    st.subheader("Text Phishing Detection")
    text_input = st.text_area("Enter SMS or Email content:", placeholder="Urgent! Click here to claim your prize...", height=150)
    
    if st.button("Analyze Text", type="primary"):
        if not text_input.strip():
            st.warning("Please enter text content.")
        else:
            with st.spinner("Analyzing Text..."):
                try:
                    response = requests.post(f"{API_URL}/predict/text", json={"text": text_input}, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        display_verdict(data["verdict"], data.get("phishing_probability"))
                        
                        with st.expander("Preprocessed Text"):
                            st.write(data.get("cleaned_text", ""))
                    else:
                        st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
                except requests.RequestException as e:
                    st.error(f"Failed to connect to API: {e}")

# --- TAB 3: QR / Quishing Detection ---
with tab3:
    st.subheader("QR Code / Quishing Detection")
    st.markdown("Upload a screenshot or photo containing a QR code.")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        
        if st.button("Decode & Analyze QR", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/predict/qr", files=files, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if not data.get("qr_detected"):
                            st.warning("No QR code detected in the image.")
                        else:
                            st.info(f"**Decoded Payload ({data.get('payload_type', 'unknown')}):**\n{data.get('payload', '')}")
                            
                            if data.get("is_url"):
                                st.markdown("### URL Analysis")
                                display_verdict(data.get("verdict"), data.get("score"))
                                
                                if data.get("risk_indicators"):
                                    st.markdown("**Risk Indicators:**")
                                    for indicator in data["risk_indicators"]:
                                        st.write(f"- {indicator}")
                            else:
                                st.success("Payload is plain text (Not a URL). No phishing analysis required.")
                                
                    else:
                        st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
                except requests.RequestException as e:
                    st.error(f"Failed to connect to API: {e}")

# --- TAB 4: URL Explainability ---
with tab4:
    st.subheader("Model Explainability (SHAP)")
    st.markdown("Understand *why* the model made a specific prediction for a URL.")
    
    explain_input = st.text_input("Enter URL to explain:", placeholder="https://example.com", key="explain_input")
    
    if st.button("Generate Explanation", type="primary"):
        if not explain_input.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Generating SHAP explanation (this may take a moment)..."):
                try:
                    response = requests.post(f"{API_URL}/explain/url", json={"url": explain_input}, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            display_verdict(data["prediction"], data.get("confidence"))
                        
                        with col2:
                            st.markdown("### Top Risk/Safety Factors")
                            for reason in data.get("reasons", []):
                                st.markdown(f"- {reason}")
                                
                        st.markdown("### Raw Feature Contributions")
                        contribs = data.get("feature_contributions", {})
                        if contribs:
                            df = pd.DataFrame({
                                "Feature": list(contribs.keys()),
                                "Impact Score": list(contribs.values())
                            })
                            st.bar_chart(df.set_index("Feature"))
                        
                    else:
                        st.error(f"Error {response.status_code}: {response.json().get('detail', 'Unknown error')}")
                except requests.RequestException as e:
                    st.error(f"Failed to connect to API: {e}")
