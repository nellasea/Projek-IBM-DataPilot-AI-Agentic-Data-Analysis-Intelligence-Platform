"""
ui/pages/2_profile.py
=====================
Data profiling page.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Profile — DataPilot AI", page_icon="🔍", layout="wide")
st.title("🔍 Data Profile")

if st.session_state.get("dataframe") is None:
    st.warning("No dataset loaded. Please upload a file first.")
    st.stop()

profile = st.session_state.get("profile")
if profile is None:
    from data.profiling import profile_dataframe
    with st.spinner("Profiling…"):
        profile = profile_dataframe(
            st.session_state.dataframe, st.session_state.source_name
        )
        st.session_state.profile = profile

# ── Dataset overview ──────────────────────────────────────────────────────────
st.subheader("Dataset Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{profile.row_count:,}")
c2.metric("Columns", profile.column_count)
c3.metric("Total Cells", f"{profile.total_cells:,}")
c4.metric("Missing", f"{profile.null_pct:.1f}%")
c5.metric("Duplicate Rows", f"{profile.duplicate_row_pct:.1f}%")

# ── Null heatmap ──────────────────────────────────────────────────────────────
null_data = {c.name: c.null_pct for c in profile.columns if c.null_pct > 0}
if null_data:
    st.subheader("Missing Values (% per column)")
    null_df = pd.DataFrame({"Column": list(null_data.keys()), "Missing %": list(null_data.values())})
    fig = px.bar(null_df, x="Column", y="Missing %", color="Missing %",
                 color_continuous_scale="Reds", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── Column-level detail ───────────────────────────────────────────────────────
st.subheader("Column Details")

dtype_counts = {}
for c in profile.columns:
    dtype_counts[c.dtype_category] = dtype_counts.get(c.dtype_category, 0) + 1

col_a, col_b = st.columns([1, 2])
with col_a:
    pie_fig = px.pie(
        values=list(dtype_counts.values()),
        names=list(dtype_counts.keys()),
        title="Column Types",
        template="plotly_white",
    )
    st.plotly_chart(pie_fig, use_container_width=True)

with col_b:
    for col_profile in profile.columns:
        with st.expander(f"**{col_profile.name}** ({col_profile.dtype_category})"):
            st.markdown(
                f"- **Dtype:** `{col_profile.dtype}`  \n"
                f"- **Nulls:** {col_profile.null_count} ({col_profile.null_pct:.1f}%)  \n"
                f"- **Distinct:** {col_profile.distinct_count} ({col_profile.distinct_pct:.1f}%)  \n"
                f"- **Unique:** {'Yes' if col_profile.is_unique else 'No'}"
            )
            if col_profile.numeric:
                n = col_profile.numeric
                st.dataframe(
                    {
                        "mean": [n.mean], "median": [n.median], "std": [n.std],
                        "min": [n.min], "max": [n.max], "q1": [n.q1], "q3": [n.q3],
                        "skewness": [n.skewness], "kurtosis": [n.kurtosis],
                    },
                    use_container_width=True,
                )
            elif col_profile.text:
                t = col_profile.text
                st.markdown(
                    f"- Unique: {t.unique_count} ({t.unique_pct:.1f}%)  \n"
                    f"- Avg length: {t.avg_length:.1f} chars"
                )
                if t.top_values:
                    top_df = pd.DataFrame(t.top_values)
                    st.dataframe(top_df, use_container_width=True)
            elif col_profile.datetime:
                d = col_profile.datetime
                st.markdown(
                    f"- Min: {d.min}  \n"
                    f"- Max: {d.max}  \n"
                    f"- Range: {d.range_days:.0f} days"
                )
