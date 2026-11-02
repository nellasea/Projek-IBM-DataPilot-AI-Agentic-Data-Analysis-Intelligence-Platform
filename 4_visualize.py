"""
ui/pages/4_visualize.py
=======================
Manual chart builder page.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Visualise — DataPilot AI", page_icon="📊", layout="wide")
st.title("📊 Visualise")

if st.session_state.get("dataframe") is None:
    st.warning("No dataset loaded. Please upload a file first.")
    st.stop()

df = st.session_state.dataframe
from tools.plotly_tool import plot_chart
from core.exceptions import ToolError

st.caption(f"Dataset: **{st.session_state.source_name}**")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
chart_type = col1.selectbox("Chart type", ["bar", "line", "scatter", "histogram", "box", "heatmap", "pie"])
title = col2.text_input("Chart title", value="My Chart")

cols = list(df.columns)
x_col = st.selectbox("X axis", options=[None] + cols, index=0)
y_col = st.selectbox("Y axis", options=[None] + cols, index=0)
color_col = st.selectbox("Color by (optional)", options=[None] + cols, index=0)

if st.button("Generate Chart", type="primary"):
    try:
        fig = plot_chart(
            df,
            chart_type=chart_type,
            x=x_col if x_col else None,
            y=y_col if y_col else None,
            color=color_col if color_col else None,
            title=title,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ToolError as exc:
        st.error(f"Chart error: {exc}")
