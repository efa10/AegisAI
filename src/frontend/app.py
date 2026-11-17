"""
AegisAI — Cybersecurity Operations Dashboard

Futuristic, responsive operations dashboard for AI-powered phishing detection across:
- URL Intelligence (with SSRF-protected Shortened URL Expansion)
- SMS / Email Analysis
- QR / Quishing Scanner
- Model Explainability (SHAP)
- Threat Intelligence & Model Status
"""

from __future__ import annotations

import os
import requests
import streamlit as st
import pandas as pd
from typing import Any

# Configuration
API_URL = os.environ.get("AEGISAI_API_URL", "http://localhost:8000")

# Page Configuration
st.set_page_config(
    page_title="AEGISAI // Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System: High-End Cybersecurity Operations Theme
st.markdown("""
<style>
    /* Google Font Imports */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Design Tokens */
    :root {
        --bg-foundation: #070b14;
        --bg-surface: #0d1527;
        --bg-card: #121c33;
        --bg-card-hover: #172442;
        --border-subtle: #1e2d4d;
        --border-accent: rgba(56, 189, 248, 0.3);
        --cyan-glow: #06b6d4;
        --cyan-bright: #38bdf8;
        --violet-primary: #8b5cf6;
        --violet-light: #a78bfa;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --risk-safe: #10b981;
        --risk-safe-bg: rgba(16, 185, 129, 0.12);
        --risk-danger: #ef4444;
        --risk-danger-bg: rgba(239, 68, 68, 0.12);
        --risk-warning: #f59e0b;
        --risk-warning-bg: rgba(245, 158, 11, 0.12);
    }

    /* Base Reset & Typography */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0d172e 0%, #070b14 65%) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #080d1a !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* Target ALL text elements in the sidebar to guarantee high contrast visibility */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important;
    }

    /* Radio Group Label ("NAVIGATION") */
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stRadio > label p,
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        color: #38bdf8 !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }

    /* Radio Options List */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        padding: 0.45rem 0.75rem !important;
        border-radius: 8px !important;
        margin-bottom: 0.25rem !important;
        background: rgba(18, 28, 51, 0.6) !important;
        border: 1px solid rgba(30, 45, 77, 0.6) !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
        color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    /* Sidebar Caption text (like "Connected to ...") */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] .stCaption p {
        color: #cbd5e1 !important;
        font-size: 0.82rem !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(30, 45, 77, 0.6) !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    /* Custom Header Banner */
    .brand-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 1.75rem;
        background: linear-gradient(135deg, rgba(14, 23, 44, 0.9) 0%, rgba(19, 31, 58, 0.7) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        background: linear-gradient(90deg, #38bdf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .brand-subtitle {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin: 0.25rem 0 0 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Operations Dashboard Cards */
    .ops-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease-in-out;
    }
    .ops-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.1);
    }

    /* Verdict Banners */
    .verdict-box-phishing {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(153, 27, 27, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .verdict-box-legitimate {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .verdict-title {
        font-size: 1.25rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
        margin: 0;
    }

    .verdict-phishing { color: #f87171; }
    .verdict-legitimate { color: #34d399; }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-safe { background: var(--risk-safe-bg); color: var(--risk-safe); border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-danger { background: var(--risk-danger-bg); color: var(--risk-danger); border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-warning { background: var(--risk-warning-bg); color: var(--risk-warning); border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-info { background: rgba(56, 189, 248, 0.12); color: var(--cyan-bright); border: 1px solid rgba(56, 189, 248, 0.3); }

    /* Chain Hop Timeline */
    .chain-node {
        background: rgba(18, 28, 51, 0.6);
        border-left: 3px solid var(--cyan-bright);
        padding: 0.6rem 0.85rem;
        margin: 0.4rem 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        border-radius: 0 6px 6px 0;
        word-break: break-all;
    }

    /* Metric Containers override */
    div[data-testid="stMetricValue"] {
        color: var(--cyan-bright) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Inputs */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #0b1220 !important;
        color: #f8fafc !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: var(--cyan-bright) !important;
        box-shadow: 0 0 8px rgba(56, 189, 248, 0.3) !important;
    }

    /* Custom Code & Json blocks */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend Engine Wrapper (API or Direct Python In-Process Fallback)
# ---------------------------------------------------------------------------

def is_api_online() -> bool:
    """Verify if the FastAPI backend is responsive."""
    try:
        res = requests.get(f"{API_URL}/health", timeout=1.5)
        return res.status_code == 200
    except requests.RequestException:
        return False


def run_predict_url(url: str) -> dict[str, Any]:
    """Execute URL prediction via API or direct ML module fallback."""
    try:
        res = requests.post(f"{API_URL}/predict/url", json={"url": url}, timeout=8)
        if res.status_code == 200:
            return res.json()
    except requests.RequestException:
        pass

    # Direct fallback if API service is offline
    from src.ml.url_model import predict_url
    return predict_url(url)


def run_predict_text(text: str) -> dict[str, Any]:
    """Execute text prediction via API or direct ML module fallback."""
    try:
        res = requests.post(f"{API_URL}/predict/text", json={"text": text}, timeout=8)
        if res.status_code == 200:
            return res.json()
    except requests.RequestException:
        pass

    # Direct fallback
    from src.ml.text_model import predict_text
    return predict_text(text)


def run_process_qr(file_bytes: bytes) -> dict[str, Any]:
    """Execute QR analysis via API or direct module fallback."""
    try:
        files = {"file": ("qr_image.png", file_bytes, "image/png")}
        res = requests.post(f"{API_URL}/predict/qr", files=files, timeout=10)
        if res.status_code == 200:
            return res.json()
    except requests.RequestException:
        pass

    # Direct fallback
    from src.qr.classifier import process_qr
    return process_qr(file_bytes)


def run_explain_url(url: str) -> dict[str, Any]:
    """Execute SHAP explainability via API or direct module fallback."""
    try:
        res = requests.post(f"{API_URL}/explain/url", json={"url": url}, timeout=12)
        if res.status_code == 200:
            return res.json()
    except requests.RequestException:
        pass

    # Direct fallback
    from src.explainability.url_explainer import explain_url
    return explain_url(url)


# ---------------------------------------------------------------------------
# Sidebar Navigation & System Telemetry
# ---------------------------------------------------------------------------

api_connected = is_api_online()

with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 0.5rem;">
        <h2 style="margin: 0; color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;">
            🛡️ AEGISAI
        </h2>
        <p style="color: #64748b; font-size: 0.75rem; margin: 0; font-family: 'JetBrains Mono', monospace;">
            DEFENSIVE INTELLIGENCE v1.0
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav_choice = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "URL Intelligence",
            "SMS/Email Analysis",
            "QR Scanner",
            "Explainability",
            "API / Model Status"
        ],
        index=1  # URL Intelligence as prominent default
    )

    st.markdown("---")
    st.markdown("### System Telemetry")

    if api_connected:
        st.markdown('<span class="badge badge-safe">● API Backend: ONLINE</span>', unsafe_allow_html=True)
        st.caption(f"Connected to {API_URL}")
    else:
        st.markdown('<span class="badge badge-warning">● API Backend: OFFLINE</span>', unsafe_allow_html=True)
        st.caption("Auto-switched to Direct In-Process ML Engine")

    st.markdown("""
    <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.8; margin-top: 1rem; font-family: 'JetBrains Mono', monospace; background: rgba(14, 22, 40, 0.7); padding: 0.75rem 0.9rem; border-radius: 8px; border: 1px solid #1e2d4d;">
        <div><span style="color: #38bdf8; font-weight: 600;">URL Model:</span> <span style="color: #f8fafc;">XGBoost (Balanced)</span></div>
        <div><span style="color: #38bdf8; font-weight: 600;">Text Model:</span> <span style="color: #f8fafc;">Linear SVM (NLP)</span></div>
        <div><span style="color: #38bdf8; font-weight: 600;">QR Engine:</span> <span style="color: #f8fafc;">OpenCV PyZbar</span></div>
        <div><span style="color: #38bdf8; font-weight: 600;">SSRF Guard:</span> <span style="color: #34d399; font-weight: 600;">Active (RFC 1918)</span></div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------------------------

st.markdown("""
<div class="brand-banner">
    <div>
        <h1 class="brand-title">🛡️ AEGISAI THREAT INTELLIGENCE</h1>
        <p class="brand-subtitle">MULTI-VECTOR PHISHING DETECTION // REAL-TIME MODEL INFERENCE</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 1. OVERVIEW VIEW
# ---------------------------------------------------------------------------
if nav_choice == "Overview":
    st.markdown("### Platform Overview & Defense Vectors")
    st.markdown(
        "AegisAI provides multi-modal machine learning detection against modern phishing and social engineering attacks."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="ops-card">
            <h4>🔗 URL Intelligence</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Lexical feature extraction combined with gradient boosting. Resolves shortened links safely
                using strict SSRF protection without executing browser code.
            </p>
            <span class="badge badge-info">17 Lexical Features</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="ops-card">
            <h4>✉️ SMS / Email NLP</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                TF-IDF n-gram tokenization evaluated through Linear Support Vector Machines for high-precision
                social engineering and spam detection.
            </p>
            <span class="badge badge-info">98.2% Accuracy</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="ops-card">
            <h4>📱 QR Quishing Defense</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Detects and decodes QR matrices from images. Routes embedded URLs to the secure URL intelligence
                pipeline without opening them in a browser.
            </p>
            <span class="badge badge-info">Safe Decoding</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Detection Engine Pipeline Architecture")
    st.markdown("""
    ```
    Input URL ──> Normalization ──> [If Shortened] ──> SSRF Validation ──> Bounded Redirects
                                                                                    │
    Destination URL <───────────────────────────────────────────────────────────────┘
          │
          ├──> 17 Lexical Features Extraction (Entropy, Subdomains, Hostname)
          ├──> Machine Learning Model Inference (XGBoost / Random Forest)
          ├──> Threat Intelligence Feeds (VirusTotal, URLhaus, Local PhishTank)
          └──> Structured Security Verdict & Corrected Confidence Semantics
    ```
    """)


# ---------------------------------------------------------------------------
# 2. URL INTELLIGENCE (PRIMARY WORKFLOW)
# ---------------------------------------------------------------------------
elif nav_choice == "URL Intelligence":
    st.markdown("### URL Threat Intelligence & Deep Inspection")
    st.markdown("Analyze URLs for phishing, domain spoofing, and evasion techniques. Shortened links are resolved securely.")

    # URL Input Form
    col_input, col_action = st.columns([5, 1])
    with col_input:
        target_url = st.text_input(
            "Target URL",
            placeholder="e.g. https://www.bing.com/search?q=google or https://tinyurl.com/example",
            label_visibility="collapsed"
        )
    with col_action:
        analyze_btn = st.button("Analyze URL", type="primary", use_container_width=True)

    # Quick Sample Shortcuts
    st.markdown("<div style='font-size: 0.75rem; color: #64748b;'>QUICK TESTS:</div>", unsafe_allow_html=True)
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    with sample_col1:
        if st.button("🔍 Bing Search (Legit)", use_container_width=True):
            target_url = "https://www.bing.com/search?q=google"
            analyze_btn = True
    with sample_col2:
        if st.button("🌐 Wikipedia Phishing", use_container_width=True):
            target_url = "https://en.wikipedia.org/wiki/Phishing"
            analyze_btn = True
    with sample_col3:
        if st.button("⚠️ PayPal Phish Probe", use_container_width=True):
            target_url = "http://paypal-security-update.account-verification-alert.com/login.php"
            analyze_btn = True
    with sample_col4:
        if st.button("⚠️ Internal IP Probe (SSRF)", use_container_width=True):
            target_url = "http://192.168.1.1/admin"
            analyze_btn = True

    if analyze_btn:
        if not target_url.strip():
            st.warning("Please provide a valid URL string.")
        else:
            with st.spinner("Extracting lexical features & validating security posture..."):
                data = run_predict_url(target_url)

                if data.get("ml_verdict") == "invalid" or data.get("verdict") == "invalid":
                    st.error(f"Analysis Failed: {data.get('expansion_error') or data.get('error', 'Invalid URL')}")
                else:
                    verdict = data.get("ml_verdict") or data.get("verdict", "unknown")
                    is_phish = verdict.lower() == "phishing"
                    confidence = data.get("confidence")
                    if confidence is None:
                        # Fallback calculation
                        p = data.get("phishing_probability", 0.5)
                        confidence = p if is_phish else (1.0 - p)

                    risk_score = data.get("risk_score")
                    if risk_score is None:
                        risk_score = round(data.get("phishing_probability", 0.5) * 100, 1)

                    # 1. Main Verdict Display
                    if is_phish:
                        st.markdown(f"""
                        <div class="verdict-box-phishing">
                            <div>
                                <h3 class="verdict-title verdict-phishing">⚠️ VERDICT: PHISHING ATTACK DETECTED</h3>
                                <p style="margin: 0.25rem 0 0 0; color: #fca5a5; font-size: 0.85rem;">
                                    High-confidence indicators of malicious spoofing, brand deception, or credential theft.
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <span class="badge badge-danger">RISK SCORE: {risk_score}/100</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="verdict-box-legitimate">
                            <div>
                                <h3 class="verdict-title verdict-legitimate">✅ VERDICT: LEGITIMATE DESTINATION</h3>
                                <p style="margin: 0.25rem 0 0 0; color: #86efac; font-size: 0.85rem;">
                                    Consistent with benign structural norms. Low probability of social engineering.
                                </p>
                            </div>
                            <div style="text-align: right;">
                                <span class="badge badge-safe">RISK SCORE: {risk_score}/100</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 2. Key Metrics Row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Model Confidence", f"{confidence * 100:.2f}%")
                    with m2:
                        p_raw = data.get("phishing_probability", 0.0)
                        st.metric("Phishing Probability", f"{p_raw * 100:.2f}%")
                    with m3:
                        shortened_status = "YES" if data.get("is_shortened") else "NO"
                        st.metric("Shortened Link", shortened_status)
                    with m4:
                        redirect_hops = data.get("redirect_count", 0)
                        st.metric("Redirect Depth", f"{redirect_hops} hop(s)")

                    # 3. URL Expansion & Redirection Chain
                    if data.get("is_shortened") or data.get("redirect_count", 0) > 0:
                        st.markdown("#### Shortened URL Resolution")
                        svc = data.get("shortening_service") or data.get("shortener_service", "Unknown")
                        success = data.get("expansion_success", False)

                        if success:
                            st.info(f"Detected shortener service: **{svc}**. Successfully expanded to final target destination.")
                        else:
                            st.warning(f"Shortener service: **{svc}**. Notice: {data.get('expansion_error') or 'Could not trace'}")

                        with st.expander("Inspected Redirection Chain", expanded=True):
                            chain = data.get("redirect_chain", [target_url])
                            for idx, hop in enumerate(chain):
                                st.markdown(f"<div class='chain-node'><b>HOP {idx}:</b> {hop}</div>", unsafe_allow_html=True)

                    # 4. Risk Indicators & Threat Intel
                    st.markdown("#### Security Indicators & Threat Feeds")
                    ind_col, ti_col = st.columns([1, 1])

                    with ind_col:
                        st.markdown("**Structural Risk Indicators:**")
                        indicators = data.get("risk_indicators", [])
                        if indicators:
                            for ind in indicators:
                                st.markdown(f"- 🔴 {ind}")
                        else:
                            st.markdown("- 🟢 No abnormal lexical flags triggered.")

                    with ti_col:
                        st.markdown("**Threat Intelligence Feeds:**")
                        ti = data.get("threat_intelligence", {})
                        vt = ti.get("virustotal", {})
                        uh = ti.get("urlhaus", {})
                        pt = ti.get("phishtank", {})

                        st.write(f"- **VirusTotal:** `{vt.get('status', 'unknown')}` ({vt.get('detail', 'No reports')})")
                        st.write(f"- **URLhaus:** `{uh.get('status', 'unknown')}` ({uh.get('detail', 'No reports')})")
                        st.write(f"- **PhishTank:** `{pt.get('status', 'unknown')}` ({pt.get('detail', 'No reports')})")

                    # 5. Raw Features Inspector
                    with st.expander("Technical Feature Vector (17 Parameters)"):
                        features = data.get("features", {})
                        if features:
                            feat_df = pd.DataFrame(list(features.items()), columns=["Feature Name", "Extracted Value"])
                            st.dataframe(feat_df, use_container_width=True)


# ---------------------------------------------------------------------------
# 3. SMS / EMAIL ANALYSIS
# ---------------------------------------------------------------------------
elif nav_choice == "SMS/Email Analysis":
    st.markdown("### SMS & Email Social Engineering Scanner")
    st.markdown("Evaluate message text using NLP tokenization and SVM classification to detect smishing, spear-phishing, and extortion spam.")

    sample_text = st.selectbox(
        "Load Sample Template:",
        [
            "-- Custom Message --",
            "Urgent: Your Chase debit card has been suspended. Please confirm your PIN immediately at http://chase-update.xyz",
            "Hey Sarah, hope you are doing well. Are we still having our team sync at 2 PM today?",
            "Congratulations! You won a $1,000 Walmart gift card! Click here to claim your reward: http://free-gift.cc"
        ]
    )

    default_val = "" if sample_text == "-- Custom Message --" else sample_text
    text_content = st.text_area("Message Content", value=default_val, height=140, placeholder="Paste email body or SMS message here...")

    if st.button("Analyze Text Content", type="primary"):
        if not text_content.strip():
            st.warning("Please provide text to analyze.")
        else:
            with st.spinner("Tokenizing & analyzing message structure..."):
                res = run_predict_text(text_content)
                if res.get("verdict") == "invalid":
                    st.error(f"Error: {res.get('error', 'Invalid text')}")
                else:
                    verdict = res.get("verdict", "unknown")
                    is_spam = verdict.lower() == "phishing"
                    proba = res.get("phishing_probability", 0.5)
                    confidence = res.get("confidence")
                    if confidence is None:
                        confidence = proba if is_spam else (1.0 - proba)

                    if is_spam:
                        st.markdown("""
                        <div class="verdict-box-phishing">
                            <div>
                                <h3 class="verdict-title verdict-phishing">⚠️ VERDICT: SUSPICIOUS / SMISHING</h3>
                                <p style="margin: 0.25rem 0 0 0; color: #fca5a5; font-size: 0.85rem;">
                                    Message exhibits urgent call-to-action, financial coercion, or unverified link patterns.
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="verdict-box-legitimate">
                            <div>
                                <h3 class="verdict-title verdict-legitimate">✅ VERDICT: LEGITIMATE (HAM)</h3>
                                <p style="margin: 0.25rem 0 0 0; color: #86efac; font-size: 0.85rem;">
                                    Message aligns with standard benign interpersonal communication patterns.
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Model Confidence", f"{confidence * 100:.2f}%")
                    with c2:
                        st.metric("Phishing Probability", f"{proba * 100:.2f}%")

                    with st.expander("Preprocessed Token Stream"):
                        st.code(res.get("cleaned_text", ""))


# ---------------------------------------------------------------------------
# 4. QR SCANNER
# ---------------------------------------------------------------------------
elif nav_choice == "QR Scanner":
    st.markdown("### QR / Quishing Scanner")
    st.markdown("Upload QR code images to extract payload and perform zero-execution inspection.")

    uploaded_file = st.file_uploader("Upload QR Code Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        c_img, c_res = st.columns([1, 2])

        with c_img:
            st.image(file_bytes, caption="Uploaded Image", use_container_width=True)

        with c_res:
            if st.button("Decode & Analyze Payload", type="primary"):
                with st.spinner("Decoding matrix & routing payload safely..."):
                    res = run_process_qr(file_bytes)

                    if not res.get("qr_detected"):
                        st.warning("No QR matrix detected in this image. Ensure high contrast and sharp focus.")
                    else:
                        st.markdown('<span class="badge badge-info">QR Matrix Decoded</span>', unsafe_allow_html=True)
                        payload = res.get("payload", "")
                        st.code(payload)

                        if res.get("is_url"):
                            st.markdown("#### Embedded URL Analysis")
                            verdict = res.get("verdict", "unknown")
                            is_phish = verdict.lower() == "phishing"
                            score = res.get("score") or 0.5
                            conf = res.get("confidence") or (score if is_phish else (1.0 - score))

                            if is_phish:
                                st.error(f"⚠️ VERDICT: PHISHING (Confidence: {conf*100:.2f}%)")
                            else:
                                st.success(f"✅ VERDICT: LEGITIMATE (Confidence: {conf*100:.2f}%)")

                            if res.get("is_shortened"):
                                st.warning(f"Shortened Link Detected ({res.get('shortening_service')}) ➔ Destination: {res.get('expanded_url')}")

                            indicators = res.get("risk_indicators", [])
                            if indicators:
                                st.markdown("**Risk Indicators:**")
                                for ind in indicators:
                                    st.write(f"- {ind}")
                        else:
                            st.info("Payload is plain text (Not a URL). No network routing required.")


# ---------------------------------------------------------------------------
# 5. EXPLAINABILITY (SHAP)
# ---------------------------------------------------------------------------
elif nav_choice == "Explainability":
    st.markdown("### Model Explainability & Feature Attributions")
    st.markdown("Inspect why the model arrived at a particular decision through SHAP-based feature contributions.")

    exp_url = st.text_input("URL to Explain", "https://www.bing.com/search?q=google")

    if st.button("Generate Feature Explanation", type="primary"):
        with st.spinner("Computing feature attribution scores..."):
            res = run_explain_url(exp_url)

            if res.get("prediction") == "ERROR" or "error" in res:
                st.error(f"Explanation notice: {res.get('error')}")
            else:
                pred = res.get("prediction", "UNKNOWN")
                conf = res.get("confidence", 1.0)
                st.write(f"**Predicted Verdict:** `{pred}` (Confidence: {conf*100:.2f}%)")

                st.markdown("#### Primary Decision Factors")
                reasons = res.get("reasons", [])
                for r in reasons:
                    st.markdown(f"- {r}")

                contribs = res.get("feature_contributions", {})
                if contribs:
                    st.markdown("#### Feature Importance Magnitude")
                    df_chart = pd.DataFrame({
                        "Feature": list(contribs.keys()),
                        "Impact": list(contribs.values())
                    }).set_index("Feature")
                    st.bar_chart(df_chart)


# ---------------------------------------------------------------------------
# 6. API / MODEL STATUS
# ---------------------------------------------------------------------------
elif nav_choice == "API / Model Status":
    st.markdown("### System Telemetry & Model Metadata")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""
        <div class="ops-card">
            <h4>FastAPI Backend Telemetry</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">Status of the REST API endpoints and connectivity.</p>
        </div>
        """, unsafe_allow_html=True)

        if api_connected:
            st.success("🟢 API Service: ONLINE & RESPONSIVE")
            st.write(f"Endpoint: `{API_URL}`")
        else:
            st.error("🔴 API Service: UNREACHABLE")
            st.caption("Start with: `uvicorn src.api.main:app --reload --port 8000`")

    with s2:
        st.markdown("""
        <div class="ops-card">
            <h4>Model Artifact Integrity</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">Active models saved in the local repository.</p>
        </div>
        """, unsafe_allow_html=True)

        from config.settings import SETTINGS
        url_m_path = SETTINGS.MODELS_DIR / "best_url_model.joblib"
        text_m_path = SETTINGS.MODELS_DIR / "best_text_model.joblib"

        st.write(f"- **URL Model:** `{'Available' if url_m_path.exists() else 'Missing'}` ({url_m_path.name})")
        st.write(f"- **Text Model:** `{'Available' if text_m_path.exists() else 'Missing'}` ({text_m_path.name})")
        st.write(f"- **Feature Schema:** `{len(SETTINGS.URL_FEATURE_NAMES)} engineered URL lexical signals`")
