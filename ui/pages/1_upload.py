"""
ui/pages/1_upload.py
====================
Upload page: file ingestion + quality overview.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uuid
import streamlit as st

from core.exceptions import IngestionError
from data.ingestion import ingest_file
from data.profiling import profile_dataframe
from data.quality import inspect_quality

st.set_page_config(page_title="Upload — DataPilot AI", page_icon="📤", layout="wide")
st.title("📤 Upload Dataset")


def _do_ingest(file) -> None:
    raw = file.read()
    try:
        with st.spinner("Loading and validating dataset…"):
            result = ingest_file(raw, source_name=file.name)
    except IngestionError as exc:
        st.error(f"Ingestion failed: {exc}")
        return

    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    st.session_state.dataframe = result.dataframe
    st.session_state.source_name = result.source_name
    st.session_state.profile = None   # invalidate cached profile
    st.session_state.quality = None

    st.success(
        f"✅ Loaded **{result.source_name}**: "
        f"{result.row_count:,} rows × {result.column_count} columns "
        f"({result.file_size_bytes / 1024:.1f} KB)"
    )

    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    # Auto-profile and quality-check on upload
    with st.spinner("Running automated profiling…"):
        st.session_state.profile = profile_dataframe(result.dataframe, result.source_name)
        st.session_state.quality = inspect_quality(result.dataframe, result.source_name)

    # Show schema table
    st.subheader("Column Schema")
    schema_data = [
        {
            "Column": c.name,
            "Type": c.dtype_category,
            "Raw dtype": c.dtype_raw,
            "Nullable": "Yes" if c.nullable else "No",
        }
        for c in result.columns
    ]
    st.dataframe(schema_data, use_container_width=True)

    # Quality score badge
    q = st.session_state.quality
    score_colour = "green" if q.score >= 80 else "orange" if q.score >= 50 else "red"
    st.subheader("Data Quality Score")
    st.markdown(
        f"<h2 style='color:{score_colour}'>{q.score:.0f}/100</h2>",
        unsafe_allow_html=True,
    )
    if q.issues:
        with st.expander(f"View {len(q.issues)} issue(s)"):
            for issue in q.issues:
                icon = "🔴" if issue.severity.value == "error" else "🟡" if issue.severity.value == "warning" else "🔵"
                st.markdown(f"{icon} **{issue.check}** — {issue.message}")

    st.subheader("Data Preview")
    st.dataframe(result.dataframe.head(50), use_container_width=True)


uploaded = st.file_uploader(
    "Choose a CSV or XLSX file",
    type=["csv", "xlsx", "xls"],
    help="Maximum file size is configured via MAX_UPLOAD_MB in your .env file.",
)

if uploaded is not None:
    _do_ingest(uploaded)
elif st.session_state.get("dataframe") is not None:
    st.info(
        f"ℹ️ Dataset already loaded: **{st.session_state.source_name}**. "
        "Upload a new file to replace it."
    )
    st.dataframe(st.session_state.dataframe.head(50), use_container_width=True)
