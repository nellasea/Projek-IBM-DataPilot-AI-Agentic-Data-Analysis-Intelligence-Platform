"""
agent/state.py
==============
LangGraph agent state definition.

``AgentState`` is the single typed dict that flows through every node
in the graph. All nodes read from and write to this dict.
"""
from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd
import plotly.graph_objects as go


class AgentState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    question: str                   # the user's natural language question
    dataframe: pd.DataFrame         # the active dataset
    schema_summary: str             # pre-built text summary of the schema
    session_id: str                 # analysis session identifier

    # ── Planning ──────────────────────────────────────────────────────────────
    plan: list[dict[str, Any]]      # [{"tool": ..., "params": {...}}, ...]

    # ── Execution ─────────────────────────────────────────────────────────────
    tool_results: list[dict[str, Any]]   # one entry per executed step
    figures: list[go.Figure]             # collected Plotly figures

    # ── Output ────────────────────────────────────────────────────────────────
    insight: str                    # final natural language insight
    error: str | None               # set if any node encounters an error

    # ── Evaluation ────────────────────────────────────────────────────────────
    eval_scores: dict[str, Any] | None  # faithfulness, relevance, etc.

    # ── Execution mode ────────────────────────────────────────────────────────
    fallback_mode: bool                 # True when deterministic fallback handled the request
    llm_error: str | None               # raw LLM error message (for UI transparency)
