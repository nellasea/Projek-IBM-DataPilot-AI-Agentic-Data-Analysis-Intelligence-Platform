"""
tests/test_tools.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from tools.duckdb_tool import run_duckdb_query, _validate_sql
from tools.pandas_tool import pandas_describe, pandas_value_counts, pandas_filter
from tools.stats_tool import run_correlation, run_ttest, run_anova, run_chi2
from tools.anomaly_tool import detect_anomalies
from tools.plotly_tool import plot_chart
from tools.registry import dispatch, list_tools
from core.exceptions import DuckDBToolError, PandasToolError, ToolError


# ── DuckDB ────────────────────────────────────────────────────────────────────

def test_duckdb_basic_select(sample_df):
    result = run_duckdb_query(sample_df, "SELECT * FROM df LIMIT 5")
    assert result["row_count"] == 5
    assert "price" in result["columns"]


def test_duckdb_aggregation(sample_df):
    result = run_duckdb_query(
        sample_df, "SELECT category, COUNT(*) as cnt FROM df GROUP BY category ORDER BY category"
    )
    assert result["row_count"] == 4


def test_duckdb_rejects_insert(sample_df):
    with pytest.raises(DuckDBToolError, match="SELECT"):
        run_duckdb_query(sample_df, "INSERT INTO df VALUES (1, 2)")


def test_duckdb_rejects_non_select(sample_df):
    with pytest.raises(DuckDBToolError, match="SELECT"):
        run_duckdb_query(sample_df, "DROP TABLE df")


def test_duckdb_validate_multiple_statements():
    with pytest.raises((DuckDBToolError, Exception)):
        _validate_sql("SELECT 1; DROP TABLE df")


# ── Pandas ────────────────────────────────────────────────────────────────────

def test_pandas_describe(sample_df):
    result = pandas_describe(sample_df)
    assert "price" in result
    assert "mean" in result["price"]


def test_pandas_describe_specific_columns(sample_df):
    result = pandas_describe(sample_df, columns=["price", "quantity"])
    assert set(result.keys()) == {"price", "quantity"}


def test_pandas_describe_missing_column(sample_df):
    with pytest.raises(PandasToolError, match="not found"):
        pandas_describe(sample_df, columns=["nonexistent"])


def test_pandas_value_counts(sample_df):
    result = pandas_value_counts(sample_df, "category")
    assert len(result["values"]) == 4
    assert sum(result["counts"]) == 20


def test_pandas_filter_eq(sample_df):
    result = pandas_filter(sample_df, "category", "eq", "A")
    assert result["matched_rows"] == 5


def test_pandas_filter_contains(sample_df):
    result = pandas_filter(sample_df, "name", "contains", "Item 1")
    assert result["matched_rows"] >= 1


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_correlation(large_numeric_df):
    result = run_correlation(large_numeric_df, columns=["x", "y", "z"])
    assert result["columns"] == ["x", "y", "z"]
    assert len(result["matrix"]) == 3


def test_ttest(large_numeric_df):
    result = run_ttest(large_numeric_df, value_col="x", group_col="group")
    assert "t_stat" in result
    assert "p_value" in result
    assert isinstance(result["significant"], bool)


def test_anova(large_numeric_df):
    result = run_anova(large_numeric_df, value_col="x", group_col="group")
    assert "f_stat" in result


def test_chi2(sample_df):
    result = run_chi2(sample_df, col_a="category", col_b="in_stock")
    assert "chi2_stat" in result
    assert "p_value" in result


# ── Anomaly ───────────────────────────────────────────────────────────────────

def test_detect_anomalies(large_numeric_df):
    result = detect_anomalies(large_numeric_df, columns=["x", "y", "z"])
    assert "anomaly_count" in result
    assert "anomaly_pct" in result
    assert result["anomaly_count"] >= 0


def test_detect_anomalies_no_numeric():
    df = pd.DataFrame({"name": ["a", "b", "c"]})
    result = detect_anomalies(df)
    assert "error" in result


# ── Plotly ────────────────────────────────────────────────────────────────────

def test_plot_bar_chart(sample_df):
    fig = plot_chart(sample_df, chart_type="bar", x="category", y="price", title="Test")
    assert fig is not None
    assert fig.layout.title.text == "Test"


def test_plot_histogram(sample_df):
    fig = plot_chart(sample_df, chart_type="histogram", x="price")
    assert fig is not None


def test_plot_invalid_type(sample_df):
    with pytest.raises(ToolError, match="Unknown chart type"):
        plot_chart(sample_df, chart_type="treemap")


def test_plot_missing_column(sample_df):
    with pytest.raises(ToolError, match="not found"):
        plot_chart(sample_df, chart_type="bar", x="nonexistent_col", y="price")


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_known_tools():
    tools = list_tools()
    assert "duckdb_query" in tools
    assert "pandas_describe" in tools
    assert "plot_chart" in tools
    assert "detect_anomalies" in tools
    assert "stats_test" in tools


def test_registry_dispatch_duckdb(sample_df):
    result = dispatch("duckdb_query", {"sql": "SELECT COUNT(*) as n FROM df"}, sample_df)
    assert result["row_count"] == 1


def test_registry_unknown_tool(sample_df):
    with pytest.raises(ToolError, match="Unknown tool"):
        dispatch("eval_code", {"code": "import os"}, sample_df)


def test_registry_invalid_params(sample_df):
    with pytest.raises(ToolError, match="Invalid parameters"):
        dispatch("duckdb_query", {"sql": ""}, sample_df)  # min_length=1 but empty after strip -> validation error


# ── Chart-after-query fix (executor_node last_query_df) ──────────────────────

def test_executor_chart_uses_query_result_df():
    """plot_chart must succeed when x/y reference computed DuckDB alias columns.

    Previously plot_chart received the original df (which has no 'total_sales'
    column) instead of the DuckDB result df, causing a ToolError every time a
    plan contained duckdb_query followed by plot_chart with an alias column.
    """
    from agent.graph import run_analysis
    from unittest.mock import patch

    df = pd.DataFrame({
        "product": ["Laptop", "Phone", "Laptop", "Tablet", "Phone"],
        "sales":   [100.0, 200.0, 300.0, 150.0, 100.0],
    })

    with patch("agent.nodes._is_rule_based_mode", return_value=True):
        state = run_analysis(
            question="Which product has the highest sales?",
            df=df,
            evaluate=False,
        )

    assert state.get("error") is None

    chart_results = [
        r for r in state.get("tool_results", [])
        if r.get("tool") == "plot_chart"
    ]
    assert chart_results, "Expected a plot_chart step in the plan"
    failed = [r for r in chart_results if not r.get("success")]
    assert not failed, (
        f"plot_chart failed — computed alias column not passed correctly. "
        f"Errors: {[r.get('error') for r in failed]}"
    )
    assert len(state.get("figures", [])) >= 1, "Expected at least one figure to be generated"
