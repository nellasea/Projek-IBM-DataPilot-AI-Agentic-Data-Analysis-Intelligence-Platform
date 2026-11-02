"""
ui/pages/3_query.py
===================
Natural language query page — Rule-Based Agent Mode.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="Ask DataPilot — DataPilot AI", page_icon="💬", layout="wide")
st.title("💬 Ask Your Data")

if st.session_state.get("dataframe") is None:
    st.warning("Belum ada dataset yang dimuat. Silakan unggah file terlebih dahulu.")
    st.stop()

df = st.session_state.dataframe
source = st.session_state.source_name

st.caption(f"Dataset: **{source}** · {len(df):,} baris × {len(df.columns)} kolom")

# ── Mode indicator (static — never shows LLM error in rule-based mode) ────────
from core.config import LLMProvider
_provider = LLMProvider.fallback  # This submission build is offline-only.
st.info(
    "⚙️ **Rule-Based Agent Mode** — Tidak memerlukan API key atau LLM eksternal. "
    "Analisis dilakukan menggunakan Pandas, DuckDB, SciPy, dan Scikit-learn.",
    icon=None,
)

# ── Question input ────────────────────────────────────────────────────────────
question = st.text_area(
    "Ketik pertanyaan Anda",
    height=80,
    placeholder=(
        "Contoh: Produk apa yang memiliki total penjualan tertinggi?\n"
        "        Berapa rata-rata penjualan per wilayah?\n"
        "        Apakah ada data yang hilang?"
    ),
)

run_btn = st.button("▶ Analisis", type="primary", use_container_width=True)

# ── Show column info to help users frame questions ────────────────────────────
with st.expander("ℹ️ Kolom dataset yang tersedia", expanded=False):
    col_types = {col: str(df[col].dtype) for col in df.columns}
    st.dataframe(
        {"Kolom": list(col_types.keys()), "Tipe": list(col_types.values())},
        use_container_width=True,
    )

if run_btn and question.strip():
    from agent.graph import run_analysis

    with st.spinner("⚙️ Menganalisis data…"):
        state = run_analysis(
            question=question.strip(),
            df=df,
            session_id=st.session_state.get("session_id", ""),
            evaluate=False,
        )

    # ── Error ─────────────────────────────────────────────────────────────────
    if state.get("error"):
        st.error(f"Analisis gagal: {state['error']}")

    else:
        fallback_mode: bool = state.get("fallback_mode", True)
        # In rule-based mode, NEVER show an LLM error banner
        # Only show a provider error when the user explicitly set a non-fallback provider

        # ── Plan ──────────────────────────────────────────────────────────────
        with st.expander("📋 Rencana Analisis", expanded=True):
            for i, step in enumerate(state.get("plan", []), 1):
                st.markdown(
                    f"**Langkah {i}:** `{step['tool']}` — parameter: `{step['params']}`"
                )

        # ── Tool results ──────────────────────────────────────────────────────
        with st.expander("🔧 Hasil Alat Analisis", expanded=False):
            for r in state.get("tool_results", []):
                status = "✅" if r.get("success") else "❌"
                st.markdown(
                    f"{status} **{r['tool']}** ({r.get('latency_ms', 0):.0f} ms)"
                )
                if r.get("error"):
                    st.error(r["error"])
                elif r.get("result") and r["result"] != "chart_generated":
                    # Render duckdb results as a table when possible
                    result = r["result"]
                    if (
                        isinstance(result, dict)
                        and "columns" in result
                        and "rows" in result
                    ):
                        import pandas as _pd
                        try:
                            tbl = _pd.DataFrame(result["rows"], columns=result["columns"])
                            st.dataframe(tbl, use_container_width=True)
                        except Exception:
                            st.json(result)
                    else:
                        st.json(result)

        # ── Charts ────────────────────────────────────────────────────────────
        if state.get("figures"):
            st.subheader("📊 Visualisasi")
            for fig in state["figures"]:
                st.plotly_chart(fig, use_container_width=True)

        # ── Insight ───────────────────────────────────────────────────────────
        st.subheader("💡 Hasil Analisis")
        st.markdown(state.get("insight", "_Tidak ada hasil analisis._"))

        # ── Optional evaluation output (not produced in offline mode) ────────
        if state.get("eval_scores"):
            ev = state["eval_scores"]
            st.subheader("🎯 Skor Evaluasi")
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Faithfulness", f"{ev.get('faithfulness', 0)}/10")
            ec2.metric("Relevance", f"{ev.get('relevance', 0)}/10")
            ec3.metric("Groundedness", f"{ev.get('groundedness', 0)}/10")
            ec4.metric("Completeness", f"{ev.get('completeness', 0)}/10")
            if ev.get("notes"):
                st.caption(f"Catatan: {ev['notes']}")

        # ── Save to history ───────────────────────────────────────────────────
        if "analysis_history" not in st.session_state:
            st.session_state.analysis_history = []
        st.session_state.analysis_history.append({
            "question": question,
            "insight": state.get("insight", ""),
            "plan": state.get("plan", []),
            "eval_scores": state.get("eval_scores"),
            "fallback_mode": fallback_mode,
        })

elif run_btn:
    st.warning("Silakan masukkan pertanyaan sebelum menjalankan analisis.")
