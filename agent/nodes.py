"""
agent/nodes.py
==============
LangGraph node implementations: Planner, Executor, Insight Generator.

Each node is a plain function ``(state: AgentState) -> dict`` that returns
a partial state update. LangGraph merges the returned dict into the running state.

Execution mode decision
-----------------------
When ``Settings.llm_provider == LLMProvider.fallback`` (the default), the
planner goes directly to the rule-based engine — NO network call, NO API
initialization, NO external provider invocation of any kind.

This submission build is offline-only. The rule-based planner and insight
synthesizer are always used; external LLM providers are never initialized.

Security reminder
-----------------
The Executor node NEVER calls eval() or exec().
It only dispatches to tools registered in ``tools.registry``.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from agent.state import AgentState
from core.exceptions import ExecutionError, PlanningError
from core.logging import get_logger
from llm.fallback import build_fallback_plan, synthesize_insight
from tools.registry import dispatch

logger = get_logger(__name__)

# Maximum number of tool steps to prevent runaway plans
MAX_PLAN_STEPS = 5


def _is_rule_based_mode() -> bool:
    """This capstone build is intentionally offline and rule-based only.

    Never initialize or invoke an external LLM, even if an old .env file
    contains an LLM_PROVIDER value from an earlier experiment.
    """
    return True


# ── Planner ───────────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """Build a deterministic plan without initializing any external model."""
    question = state.get("question", "")
    df: pd.DataFrame | None = state.get("dataframe")
    logger.info("planner_start", question=question[:100], mode="rule_based")

    if df is None:
        raise PlanningError("No DataFrame available for rule-based planning.")

    plan = build_fallback_plan(question, df)
    if not plan:
        raise PlanningError("Rule-based planner could not map the question to an analysis.")

    logger.info("planner_complete", steps=len(plan), mode="rule_based")
    return {"plan": plan, "fallback_mode": True, "llm_error": None}


# ── Executor ──────────────────────────────────────────────────────────────────

def executor_node(state: AgentState) -> dict:
    """Execute each step in the plan using the tool registry.

    Reads:  plan, dataframe
    Writes: tool_results, figures

    Chart-after-query fix
    ---------------------
    DuckDB queries produce computed column aliases (e.g. ``total_sales``) that
    do not exist in the original DataFrame.  When a ``plot_chart`` step
    immediately follows a successful ``duckdb_query`` step, the executor
    reconstructs a DataFrame from the query result and uses it as the chart's
    data source instead of the original ``df``.
    """
    plan = state.get("plan", [])
    df: pd.DataFrame = state.get("dataframe")

    if df is None or df.empty:
        raise ExecutionError("No DataFrame available for tool execution.")

    import plotly.graph_objects as go

    tool_results: list[dict[str, Any]] = []
    figures: list[go.Figure] = []

    # Holds the DataFrame reconstructed from the most recent duckdb_query result.
    # Used so that plot_chart can reference computed alias columns.
    last_query_df: pd.DataFrame | None = None

    for step in plan:
        tool_name = step["tool"]
        params = step.get("params", {})
        logger.info("executor_step", tool=tool_name)
        t0 = time.perf_counter()

        # For plot_chart: prefer the last query result DataFrame so computed
        # alias columns (e.g. total_sales, avg) are available.
        dispatch_df = (
            last_query_df
            if tool_name == "plot_chart" and last_query_df is not None
            else df
        )

        try:
            result = dispatch(tool_name, params, dispatch_df)
        except Exception as exc:
            logger.warning("executor_step_failed", tool=tool_name, error=str(exc))
            tool_results.append({
                "tool": tool_name,
                "params": params,
                "error": str(exc),
                "success": False,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            })
            continue

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        if isinstance(result, go.Figure):
            figures.append(result)
            tool_results.append({
                "tool": tool_name,
                "params": params,
                "result": "chart_generated",
                "success": True,
                "latency_ms": latency_ms,
            })
        elif isinstance(result, dict) and "dataframe" in result:
            # Capture the embedded DataFrame so subsequent plot_chart steps can
            # reference computed alias columns (e.g. total_sales) from the query.
            last_query_df = result["dataframe"]
            serialisable = {k: v for k, v in result.items() if k != "dataframe"}
            tool_results.append({
                "tool": tool_name,
                "params": params,
                "result": serialisable,
                "success": True,
                "latency_ms": latency_ms,
            })
        else:
            tool_results.append({
                "tool": tool_name,
                "params": params,
                "result": result,
                "success": True,
                "latency_ms": latency_ms,
            })

    logger.info("executor_complete", steps=len(plan), figures=len(figures))
    return {"tool_results": tool_results, "figures": figures}


# ── Insight Generator ─────────────────────────────────────────────────────────

def insight_node(state: AgentState) -> dict:
    """Create a grounded explanation from deterministic tool outputs only."""
    question = state.get("question", "")
    tool_results = state.get("tool_results", [])
    df: pd.DataFrame | None = state.get("dataframe")

    insight = synthesize_insight(
        question=question,
        tool_results=tool_results,
        df=df if df is not None else pd.DataFrame(),
    )
    logger.info("insight_complete", mode="rule_based")
    return {"insight": insight}
