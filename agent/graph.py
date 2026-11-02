"""
agent/graph.py
==============
LangGraph state machine definition.

Flow
----
    [START] → planner → executor → insight → evaluator → [END]

Error handling
--------------
If any node raises an exception, it is caught here and stored as
``state["error"]``. The graph transitions to END, allowing the caller
to inspect the error rather than crashing.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agent.evaluator import evaluator_node
from agent.nodes import executor_node, insight_node, planner_node
from agent.state import AgentState
from core.logging import get_logger

logger = get_logger(__name__)


def _safe_node(fn):
    """Wrap a node function so exceptions are stored in state instead of raised."""
    def wrapper(state: AgentState) -> dict:
        try:
            return fn(state)
        except Exception as exc:
            logger.error("node_error", node=fn.__name__, error=str(exc))
            return {"error": str(exc)}
    wrapper.__name__ = fn.__name__
    return wrapper


def _should_continue(state: AgentState) -> str:
    """Conditional edge: abort the graph if a previous node set state['error']."""
    if state.get("error"):
        return "end"
    return "continue"


def build_graph() -> StateGraph:
    """Build and compile the DataPilot LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", _safe_node(planner_node))
    graph.add_node("executor", _safe_node(executor_node))
    graph.add_node("insight", _safe_node(insight_node))
    graph.add_node("evaluator", _safe_node(evaluator_node))

    # Entry point
    graph.add_edge(START, "planner")

    # Conditional edges: stop on error, continue otherwise
    graph.add_conditional_edges(
        "planner",
        _should_continue,
        {"continue": "executor", "end": END},
    )
    graph.add_conditional_edges(
        "executor",
        _should_continue,
        {"continue": "insight", "end": END},
    )
    graph.add_conditional_edges(
        "insight",
        _should_continue,
        {"continue": "evaluator", "end": END},
    )
    graph.add_edge("evaluator", END)

    return graph.compile()


# ── Public convenience function ───────────────────────────────────────────────

def build_schema_summary(df: pd.DataFrame) -> str:
    """Build a compact text description of a DataFrame's schema for prompt injection."""
    lines = [f"Rows: {len(df)}, Columns: {len(df.columns)}", ""]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        sample_vals = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(str(v) for v in sample_vals)
        lines.append(f"  - {col} ({dtype}, {null_count} nulls) — e.g. {sample_str}")
    return "\n".join(lines)


def run_analysis(
    question: str,
    df: pd.DataFrame,
    session_id: str = "",
    evaluate: bool = True,
) -> AgentState:
    """Run a full analysis pipeline for a given question and DataFrame.

    Args:
        question: Natural language question.
        df: The dataset to analyse.
        session_id: Optional session identifier for history tracking.
        evaluate: Whether to run the evaluator node.

    Returns:
        The final ``AgentState`` with plan, tool_results, insight, and eval_scores.
    """
    schema_summary = build_schema_summary(df)

    initial_state: AgentState = {
        "question": question,
        "dataframe": df,
        "schema_summary": schema_summary,
        "session_id": session_id,
        "plan": [],
        "tool_results": [],
        "figures": [],
        "insight": "",
        "error": None,
        "eval_scores": None,
    }

    graph = build_graph()
    t0 = time.perf_counter()
    final_state: AgentState = graph.invoke(initial_state)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        "analysis_complete",
        question=question[:80],
        elapsed_ms=elapsed_ms,
        error=final_state.get("error"),
    )
    return final_state
