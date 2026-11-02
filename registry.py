"""
tools/registry.py
=================
Tool registry and dispatcher.

The agent ONLY calls tools registered here. This is the single enforcement
point ensuring no arbitrary code is ever executed.

Each tool is described by a ``ToolSpec`` which includes:
- name          : identifier the LLM uses in its plan
- description   : shown in prompts
- handler       : the actual Python function
- param_model   : Pydantic model for parameter validation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from core.exceptions import ToolError
from core.logging import get_logger
from tools.anomaly_tool import detect_anomalies
from tools.duckdb_tool import run_duckdb_query
from tools.pandas_tool import pandas_describe
from tools.plotly_tool import plot_chart
from tools.stats_tool import run_correlation, run_ttest, run_anova, run_chi2

logger = get_logger(__name__)


# ── Parameter models (Pydantic validation) ────────────────────────────────────

class DuckDBParams(BaseModel):
    sql: str = Field(..., min_length=1, max_length=4096)

    @field_validator("sql")
    @classmethod
    def no_semicolons(cls, v: str) -> str:
        # Strip trailing semicolons; reject multiple statements
        v = v.rstrip(";").strip()
        if ";" in v:
            raise ValueError("Multiple SQL statements are not allowed.")
        return v


class PandasDescribeParams(BaseModel):
    columns: list[str] | None = None


class StatsParams(BaseModel):
    test: str = Field(..., pattern="^(correlation|ttest|anova|chi2)$")
    columns: list[str] | None = None
    value_col: str | None = None
    group_col: str | None = None
    col_a: str | None = None
    col_b: str | None = None
    method: str = "pearson"


class PlotParams(BaseModel):
    chart_type: str = Field(..., pattern="^(bar|line|scatter|histogram|box|heatmap|pie)$")
    x: str | None = None
    y: str | list[str] | None = None
    color: str | None = None
    title: str = "Chart"


class AnomalyParams(BaseModel):
    columns: list[str] | None = None
    contamination: float = Field(default=0.05, ge=0.01, le=0.5)


# ── Tool spec ─────────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    name: str
    description: str
    handler: Callable
    param_model: type[BaseModel]


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, ToolSpec] = {}


def _register(spec: ToolSpec) -> None:
    _REGISTRY[spec.name] = spec


_register(ToolSpec(
    name="duckdb_query",
    description="Run a SQL SELECT query on the dataset table named 'df'.",
    handler=run_duckdb_query,
    param_model=DuckDBParams,
))

_register(ToolSpec(
    name="pandas_describe",
    description="Get descriptive statistics for one or more columns.",
    handler=pandas_describe,
    param_model=PandasDescribeParams,
))

_register(ToolSpec(
    name="stats_test",
    description="Run a statistical test: correlation, ttest, anova, or chi2.",
    handler=None,  # dispatch below handles routing
    param_model=StatsParams,
))

_register(ToolSpec(
    name="plot_chart",
    description="Create a Plotly chart (bar, line, scatter, histogram, box, heatmap, pie).",
    handler=plot_chart,
    param_model=PlotParams,
))

_register(ToolSpec(
    name="detect_anomalies",
    description="Run IsolationForest anomaly detection on numeric columns.",
    handler=detect_anomalies,
    param_model=AnomalyParams,
))


def list_tools() -> list[str]:
    """Return names of all registered tools."""
    return list(_REGISTRY.keys())


# ── Dispatcher ────────────────────────────────────────────────────────────────

def dispatch(tool_name: str, params: dict[str, Any], df: pd.DataFrame) -> Any:
    """Validate and execute a registered tool call.

    Args:
        tool_name: Must be a key in the registry.
        params: Raw parameter dict from the agent plan.
        df: The active DataFrame passed to every tool.

    Returns:
        Tool result (type varies by tool).

    Raises:
        ``ToolError``: for unknown tools, invalid parameters, or execution failures.
    """
    if tool_name not in _REGISTRY:
        raise ToolError(
            f"Unknown tool '{tool_name}'. Registered tools: {list_tools()}",
            details={"tool_name": tool_name},
        )

    spec = _REGISTRY[tool_name]
    logger.info("tool_dispatch", tool=tool_name)

    # Validate parameters
    try:
        validated = spec.param_model(**params)
    except Exception as exc:
        raise ToolError(
            f"Invalid parameters for tool '{tool_name}': {exc}",
            details={"tool": tool_name, "params": params, "error": str(exc)},
        ) from exc

    # Special dispatch for stats_test (routes to sub-functions)
    if tool_name == "stats_test":
        return _dispatch_stats(validated, df)

    # Standard dispatch: pass df as first positional arg + validated params
    param_dict = validated.model_dump(exclude_none=True)
    return spec.handler(df, **param_dict)


def _dispatch_stats(params: StatsParams, df: pd.DataFrame) -> Any:
    """Route stats_test to the appropriate scipy function."""
    test = params.test
    if test == "correlation":
        return run_correlation(df, columns=params.columns, method=params.method)
    elif test == "ttest":
        if not params.value_col or not params.group_col:
            raise ToolError("ttest requires 'value_col' and 'group_col'.")
        return run_ttest(df, value_col=params.value_col, group_col=params.group_col)
    elif test == "anova":
        if not params.value_col or not params.group_col:
            raise ToolError("anova requires 'value_col' and 'group_col'.")
        return run_anova(df, value_col=params.value_col, group_col=params.group_col)
    elif test == "chi2":
        if not params.col_a or not params.col_b:
            raise ToolError("chi2 requires 'col_a' and 'col_b'.")
        return run_chi2(df, col_a=params.col_a, col_b=params.col_b)
    else:
        raise ToolError(f"Unknown stats test: {test}")
