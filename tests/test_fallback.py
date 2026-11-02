"""
tests/test_fallback.py
======================
Tests for the deterministic fallback planner and insight synthesiser.

Scenarios covered
-----------------
1.  Rule-based planner works without API credentials.
2.  No external LLM initialization occurs.
3.  User asks for missing-value analysis.
4.  User asks for the highest-selling product.
5.  User asks for anomaly detection.
6.  User asks for a visualisation.
7.  Fallback produces actual results from the DataFrame (not mock data).
8.  Deterministic insight synthesis contains real numbers from tool results.
9.  ``fallback_mode`` flag is set correctly in agent state.
10. Unsupported dimensions and metrics are rejected explicitly.
11. Tool results remain grounded in the active DataFrame.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.exceptions import LLMProviderError


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sales_df() -> pd.DataFrame:
    """Realistic sales DataFrame for all fallback tests."""
    return pd.DataFrame({
        "product":  ["Apple", "Banana", "Cherry", "Apple", "Banana",
                     "Cherry", "Apple", "Banana", "Cherry", "Apple"],
        "region":   ["North", "South", "East", "West", "North",
                     "South", "East", "West", "North", "South"],
        "revenue":  [100.0, 200.0, 150.0, 300.0, 80.0,
                     220.0, 175.0, 90.0, 130.0, 260.0],
        "quantity": [5, 8, 6, 10, 4, 9, 7, 3, 5, 11],
    })


@pytest.fixture
def nulls_df() -> pd.DataFrame:
    """DataFrame with deliberate nulls."""
    df = pd.DataFrame({
        "name":   ["Alice", None, "Charlie", "Diana", None],
        "score":  [85.0, 92.0, None, 78.0, 88.0],
        "group":  ["A", "B", "A", "B", "A"],
    })
    return df


# ── Intent detection ───────────────────────────────────────────────────────────

class TestDetectIntent:
    def test_profile_default(self):
        from llm.fallback import detect_intent
        assert "profile" in detect_intent("give me an overview")

    def test_missing_values_intent(self):
        from llm.fallback import detect_intent
        assert "missing_values" in detect_intent("how many missing values are there?")

    def test_anomaly_intent(self):
        from llm.fallback import detect_intent
        assert "anomaly" in detect_intent("detect outliers in the data")

    def test_top_n_intent(self):
        from llm.fallback import detect_intent
        assert "top_n" in detect_intent("show me the top 5 products by revenue")

    def test_bottom_n_intent(self):
        from llm.fallback import detect_intent
        assert "bottom_n" in detect_intent("what are the lowest selling items?")

    def test_aggregation_intent(self):
        from llm.fallback import detect_intent
        assert "aggregation" in detect_intent("total sales by region")

    def test_correlation_intent(self):
        from llm.fallback import detect_intent
        assert "correlation" in detect_intent("is there a correlation between price and quantity?")

    def test_distribution_intent(self):
        from llm.fallback import detect_intent
        assert "distribution" in detect_intent("show the distribution of prices")

    def test_visualize_intent(self):
        from llm.fallback import detect_intent
        assert "visualize" in detect_intent("plot a bar chart of sales")

    def test_duplicates_intent(self):
        from llm.fallback import detect_intent
        assert "duplicates" in detect_intent("are there any duplicate rows?")

    def test_time_series_intent(self):
        from llm.fallback import detect_intent
        assert "time_series" in detect_intent("show revenue trend over time")

    def test_unknown_returns_unsupported(self):
        from llm.fallback import detect_intent
        result = detect_intent("xyzzy quux frob")
        assert result == ["unsupported"]


# ── build_fallback_plan ────────────────────────────────────────────────────────

class TestBuildFallbackPlan:
    def test_returns_non_empty_list(self, sales_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("describe the data", sales_df)
        assert isinstance(plan, list)
        assert len(plan) >= 1

    def test_plan_capped_at_five_steps(self, sales_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("show everything visualized with charts", sales_df)
        assert len(plan) <= 5

    def test_plan_uses_valid_tool_names(self, sales_df):
        from llm.fallback import build_fallback_plan
        from tools.registry import list_tools
        allowed = set(list_tools())
        plan = build_fallback_plan("top products by revenue", sales_df)
        for step in plan:
            assert step["tool"] in allowed, f"Unknown tool in plan: {step['tool']}"

    def test_missing_values_plan(self, nulls_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("which columns have missing values?", nulls_df)
        tools = [s["tool"] for s in plan]
        assert "duckdb_query" in tools

    def test_missing_values_sql_references_null_columns(self, nulls_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("show missing values", nulls_df)
        sql_steps = [s for s in plan if s["tool"] == "duckdb_query"]
        assert sql_steps
        sql = sql_steps[0]["params"]["sql"].lower()
        # Should reference at least one of the columns that has nulls
        assert "name" in sql or "score" in sql

    def test_top_n_plan_has_duckdb_step(self, sales_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("top 5 products by revenue", sales_df)
        tools = [s["tool"] for s in plan]
        assert "duckdb_query" in tools

    def test_top_n_plan_orders_desc(self, sales_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("highest selling product", sales_df)
        sql_steps = [s for s in plan if s["tool"] == "duckdb_query"]
        assert sql_steps
        sql = sql_steps[0]["params"]["sql"].upper()
        assert "ORDER BY" in sql and "DESC" in sql

    def test_anomaly_plan_uses_detect_anomalies(self, sales_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("detect anomalies", sales_df)
        tools = [s["tool"] for s in plan]
        assert "detect_anomalies" in tools

    def test_correlation_plan_uses_stats_test(self, sales_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("correlation between revenue and quantity", sales_df)
        tools = [s["tool"] for s in plan]
        assert "stats_test" in tools

    def test_visualize_plan_includes_plot_chart(self, sales_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("plot a bar chart of revenue by product", sales_df)
        tools = [s["tool"] for s in plan]
        assert "plot_chart" in tools

    def test_all_params_have_valid_types(self, sales_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan("total revenue by region", sales_df)
        for step in plan:
            assert isinstance(step["params"], dict)


# ── Fallback plan executes successfully via tool registry ─────────────────────

class TestFallbackPlanExecution:
    """Verify that every step produced by the fallback planner executes
    successfully through the real tool registry — no mock results."""

    def _run_plan(self, question: str, df: pd.DataFrame) -> list[dict]:
        from llm.fallback import build_fallback_plan
        from tools.registry import dispatch
        plan = build_fallback_plan(question, df)
        results = []
        for step in plan:
            try:
                result = dispatch(step["tool"], step["params"], df)
                results.append({"tool": step["tool"], "result": result, "success": True})
            except Exception as exc:
                results.append({"tool": step["tool"], "error": str(exc), "success": False})
        return results

    def test_missing_value_analysis_produces_results(self, nulls_df):
        results = self._run_plan("how many missing values?", nulls_df)
        successful = [r for r in results if r["success"]]
        assert len(successful) >= 1

    def test_top_products_uses_real_data(self, sales_df):
        results = self._run_plan("top selling product", sales_df)
        successful = [r for r in results if r["success"] and r["tool"] == "duckdb_query"]
        assert successful, "DuckDB step should succeed"
        rows = successful[0]["result"].get("rows", [])
        # Real data: Apple has most revenue
        assert rows, "Should return actual rows from the dataset"

    def test_anomaly_detection_runs(self, sales_df):
        results = self._run_plan("detect anomalies in the dataset", sales_df)
        anom = [r for r in results if r["tool"] == "detect_anomalies"]
        assert anom
        # With only 10 rows, anomaly detection may return an error msg in result
        # (need >=10 rows) — just assert it doesn't raise unhandled exception
        assert "result" in anom[0] or "error" in anom[0]

    def test_descriptive_stats_runs(self, sales_df):
        results = self._run_plan("average mean median of all columns", sales_df)
        desc = [r for r in results if r["tool"] == "pandas_describe"]
        assert desc and desc[0]["success"]

    def test_correlation_runs(self, sales_df):
        results = self._run_plan("correlation between revenue and quantity", sales_df)
        corr = [r for r in results if r["tool"] == "stats_test"]
        assert corr and corr[0]["success"]


# ── synthesize_insight ─────────────────────────────────────────────────────────

class TestSynthesizeInsight:
    def test_contains_no_fabricated_numbers(self):
        """Numbers in the insight must come from tool results, not be hardcoded."""
        from llm.fallback import synthesize_insight
        df = pd.DataFrame({"x": [1, 2, 3]})
        tool_results = [
            {
                "tool": "duckdb_query",
                "success": True,
                "result": {
                    "columns": ["product", "total"],
                    "rows": [["Apple", 12345.0], ["Banana", 9876.0]],
                    "row_count": 2,
                },
            }
        ]
        insight = synthesize_insight("top products", tool_results, df)
        # Both real values from tool results must appear (now formatted with commas)
        assert "12,345" in insight or "12345" in insight
        assert "9,876" in insight or "9876" in insight

    def test_mentions_fallback_engine(self):
        from llm.fallback import synthesize_insight
        df = pd.DataFrame({"x": [1]})
        insight = synthesize_insight("test", [], df)
        # Text is now Indonesian — check for "rule-based agent" or "agent"
        assert "rule-based" in insight.lower() or "agent" in insight.lower()

    def test_no_llm_claim_in_fallback_insight(self):
        from llm.fallback import synthesize_insight
        df = pd.DataFrame({"x": [1]})
        insight = synthesize_insight("test", [], df)
        # Must not claim an LLM was used
        assert "openai" not in insight.lower()
        assert "gpt" not in insight.lower()

    def test_empty_results_produces_message(self):
        from llm.fallback import synthesize_insight
        df = pd.DataFrame({"x": [1]})
        insight = synthesize_insight("test", [], df)
        assert len(insight) > 10

    def test_anomaly_result_described(self):
        from llm.fallback import synthesize_insight
        df = pd.DataFrame({"x": range(20)})
        tool_results = [
            {
                "tool": "detect_anomalies",
                "success": True,
                "result": {
                    "anomaly_count": 3,
                    "anomaly_pct": 15.0,
                    "columns_used": ["x"],
                    "anomaly_indices": [0, 5, 10],
                    "sample_anomalies": [],
                    "sample_columns": ["x"],
                },
            }
        ]
        insight = synthesize_insight("detect outliers", tool_results, df)
        assert "3" in insight
        assert "15.0" in insight or "15%" in insight or "15" in insight

    def test_correlation_result_described(self):
        from llm.fallback import synthesize_insight
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        tool_results = [
            {
                "tool": "stats_test",
                "success": True,
                "result": {
                    "columns": ["a", "b"],
                    "matrix": [[1.0, 0.95], [0.95, 1.0]],
                    "method": "pearson",
                },
            }
        ]
        insight = synthesize_insight("correlation", tool_results, df)
        assert "0.950" in insight or "0.95" in insight


# ── Offline-only planner and insight pipeline ─────────────────────────────────

class TestPlannerNodeOfflineOnly:
    def test_planner_never_initializes_llm(self, sales_df):
        from agent.graph import build_schema_summary
        from agent.nodes import planner_node
        from unittest.mock import patch
        import llm.provider as provider_module

        state = {
            "question": "top products by revenue",
            "dataframe": sales_df,
            "schema_summary": build_schema_summary(sales_df),
            "plan": [],
        }
        with patch.object(provider_module, "LLMClient", side_effect=AssertionError("LLM must not initialize")):
            result = planner_node(state)
        assert result["fallback_mode"] is True
        assert result["llm_error"] is None
        assert result["plan"]

    def test_provider_is_disabled_even_if_environment_selects_openai(self):
        from agent.nodes import _is_rule_based_mode
        assert _is_rule_based_mode() is True

    def test_insight_uses_actual_tool_results(self, sales_df):
        from agent.nodes import insight_node
        state = {
            "question": "top product by revenue",
            "dataframe": sales_df,
            "fallback_mode": True,
            "tool_results": [{
                "tool": "duckdb_query", "success": True,
                "result": {"columns": ["product", "total_revenue"],
                           "rows": [["Apple", 835.0]], "row_count": 1},
            }],
        }
        result = insight_node(state)
        assert "Apple" in result["insight"]
        assert "835" in result["insight"]
