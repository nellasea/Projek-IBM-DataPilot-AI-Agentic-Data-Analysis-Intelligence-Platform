"""
ui/app.py
=========
Streamlit application entry point.

Run with:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running from ui/
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from core.config import get_settings
from core.logging import setup_logging

settings = get_settings()
setup_logging(log_level=settings.log_level.value, json_logs=False)

provider = settings.llm_provider.value

st.set_page_config(
    page_title="DataPilot AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "quality" not in st.session_state:
    st.session_state.quality = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ── Landing page ──────────────────────────────────────────────────────────────
st.title("🧭 DataPilot AI")
st.caption("Agentic Data Analysis & Intelligence Platform")

# ── Mode indicator ────────────────────────────────────────────────────────────
if provider == "fallback":
    st.info(
        "⚙️ **Rule-Based Agent Mode** — DataPilot is running without an AI language model. "
        "Analyses are performed using deterministic statistical tools (Pandas, DuckDB, SciPy, "
        "Scikit-learn). No external API credentials are required.",
        icon=None,
    )
else:
    st.success(
        f"🤖 **AI Mode** — LLM provider: `{provider.upper()}`. "
        "If your API key is missing or exhausted, the rule-based fallback will activate automatically.",
        icon=None,
    )

st.markdown("""
Welcome to **DataPilot AI** — upload a dataset and ask questions about it in plain English.

### Getting started
1. **Upload** a CSV or XLSX file using the **Upload** page in the sidebar.
2. **Profile** your data to understand its structure.
3. **Query** it using natural language on the **Ask DataPilot** page.
4. **Visualise** results and explore insights.
5. **Review** your analysis history.

> Use the navigation in the sidebar (☰) to move between pages.
""")

col1, col2, col3 = st.columns(3)
col1.metric("Supported formats", "CSV · XLSX")
col2.metric("Max file size", f"{settings.max_upload_mb} MB")
col3.metric(
    "Agent Mode",
    "Rule-Based" if provider == "fallback" else provider.upper(),
)

if st.session_state.dataframe is not None:
    st.success(
        f"✅ Dataset loaded: **{st.session_state.source_name}** "
        f"({len(st.session_state.dataframe):,} rows × "
        f"{len(st.session_state.dataframe.columns)} columns)"
    )
