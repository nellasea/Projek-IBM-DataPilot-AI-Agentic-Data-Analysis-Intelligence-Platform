"""
ui/pages/5_history.py
=====================
Analysis history page.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="History — DataPilot AI", page_icon="📜", layout="wide")
st.title("📜 Analysis History")

history = st.session_state.get("analysis_history", [])

if not history:
    st.info("No analyses have been run in this session yet. Go to the Query page to get started.")
    st.stop()

st.caption(f"{len(history)} analysis record(s) this session.")

for i, record in enumerate(reversed(history), 1):
    with st.expander(f"**#{len(history) - i + 1}** — {record['question'][:80]}", expanded=(i == 1)):
        st.markdown(f"**Question:** {record['question']}")
        st.markdown(f"**Insight:** {record['insight']}")

        if record.get("eval_scores"):
            ev = record["eval_scores"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Faithfulness", f"{ev.get('faithfulness', '-')}/10")
            c2.metric("Relevance", f"{ev.get('relevance', '-')}/10")
            c3.metric("Groundedness", f"{ev.get('groundedness', '-')}/10")
            c4.metric("Completeness", f"{ev.get('completeness', '-')}/10")

        if record.get("plan"):
            with st.container():
                st.markdown("**Plan:**")
                for step in record["plan"]:
                    st.markdown(f"- `{step['tool']}`")
