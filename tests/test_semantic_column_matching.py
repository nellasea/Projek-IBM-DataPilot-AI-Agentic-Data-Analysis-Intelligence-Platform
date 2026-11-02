"""
tests/test_semantic_column_matching.py
======================================
Regression tests for the semantic column selection bug.

The critical bug:
    User asks "Produk apa yang memiliki total penjualan tertinggi?"
    System grouped by `operator` instead of `produk`.

These tests verify that:
1. `produk` questions use the product column, NOT the operator column.
2. `operator` questions use the operator column, NOT the product column.
3. `wilayah` questions use the region column.
4. Missing column produces an understandable result (not a crash).
5. The agent runs without any LLM call (no API key needed).
6. Bilingual intent detection works for Indonesian questions.
7. End-to-end pipeline returns correct grouped results.
8. Missing requested dimensions/metrics never fall back to unrelated columns.
9. Transaction-count rankings use COUNT(*) rather than summing a year column.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

# ── Controlled test DataFrames ────────────────────────────────────────────────

@pytest.fixture
def sales_df() -> pd.DataFrame:
    """Exact DataFrame from the bug report."""
    return pd.DataFrame({
        "product":  ["Laptop", "Phone",   "Laptop", "Tablet", "Phone"],
        "operator": ["Company A", "Company B", "Company B", "Company A", "Company A"],
        "sales":    [100.0, 200.0, 300.0, 150.0, 100.0],
    })


@pytest.fixture
def indonesian_df() -> pd.DataFrame:
    """DataFrame with Indonesian column names (mirrors real-world dataset)."""
    return pd.DataFrame({
        "produk":   ["Laptop", "HP",    "Laptop", "Tablet", "HP"],
        "operator": ["PT. A",  "PT. B", "PT. B",  "PT. A",  "PT. A"],
        "penjualan": [100.0, 200.0, 300.0, 150.0, 100.0],
        "wilayah":  ["Utara", "Selatan", "Timur", "Utara", "Selatan"],
    })


@pytest.fixture
def no_product_df() -> pd.DataFrame:
    """DataFrame that has no product-like column at all."""
    return pd.DataFrame({
        "operator": ["PT. A", "PT. B", "PT. C"],
        "revenue":  [1000.0, 2000.0, 3000.0],
        "wilayah":  ["Utara", "Selatan", "Timur"],
    })


# ── Semantic column resolution unit tests ─────────────────────────────────────

class TestResolveGroupEntity:
    """Direct unit tests of _resolve_group_entity."""

    def test_produk_in_question_maps_to_product_column(self, sales_df):
        from llm.fallback import _resolve_group_entity
        col, entity = _resolve_group_entity(
            "Produk apa yang memiliki total penjualan tertinggi?",
            sales_df,
        )
        assert col == "product", (
            f"Expected 'product' but got '{col}'. "
            "The system MUST NOT map 'produk' → 'operator'."
        )
        assert entity == "product"

    def test_produk_question_does_not_return_operator(self, sales_df):
        from llm.fallback import _resolve_group_entity
        col, entity = _resolve_group_entity(
            "Produk apa yang memiliki total penjualan tertinggi?",
            sales_df,
        )
        assert col != "operator", (
            "REGRESSION: 'produk' question MUST NOT be resolved to 'operator' column."
        )

    def test_operator_in_question_maps_to_operator_column(self, sales_df):
        from llm.fallback import _resolve_group_entity
        col, entity = _resolve_group_entity(
            "Siapa operator dengan jumlah transaksi terbanyak?",
            sales_df,
        )
        assert col == "operator", f"Expected 'operator' but got '{col}'."

    def test_wilayah_maps_to_region_column(self, indonesian_df):
        from llm.fallback import _resolve_group_entity
        col, entity = _resolve_group_entity(
            "Total penjualan berdasarkan wilayah",
            indonesian_df,
        )
        assert col == "wilayah", f"Expected 'wilayah' but got '{col}'."

    def test_indonesian_column_names(self, indonesian_df):
        """produk column named 'produk' should resolve correctly."""
        from llm.fallback import _resolve_group_entity
        col, entity = _resolve_group_entity(
            "Produk apa yang memiliki total penjualan tertinggi?",
            indonesian_df,
        )
        assert col == "produk", f"Expected 'produk' but got '{col}'."

    def test_no_matching_entity_returns_none(self, no_product_df):
        """If question asks for 'produk' but no product column exists, return None."""
        from llm.fallback import _resolve_group_entity
        col, entity = _resolve_group_entity(
            "Produk apa yang memiliki total penjualan tertinggi?",
            no_product_df,
        )
        # no_product_df has no product column, so should return None
        assert col is None


class TestResolveMetricEntity:
    def test_penjualan_maps_to_sales_column(self, sales_df):
        from llm.fallback import _resolve_metric_entity
        col, entity = _resolve_metric_entity(
            "total penjualan tertinggi",
            sales_df,
        )
        assert col == "sales", f"Expected 'sales' but got '{col}'."

    def test_indonesian_penjualan_maps_to_penjualan_column(self, indonesian_df):
        from llm.fallback import _resolve_metric_entity
        col, entity = _resolve_metric_entity(
            "Produk apa yang memiliki total penjualan tertinggi?",
            indonesian_df,
        )
        assert col == "penjualan", f"Expected 'penjualan' but got '{col}'."


# ── Intent detection tests ─────────────────────────────────────────────────────

class TestIndonesianIntentDetection:
    def test_tertinggi_maps_to_top_n(self):
        from llm.fallback import detect_intent
        intents = detect_intent("Produk apa yang memiliki total penjualan tertinggi?")
        assert "top_n" in intents, f"'tertinggi' should map to top_n, got: {intents}"

    def test_terbanyak_maps_to_top_n(self):
        from llm.fallback import detect_intent
        assert "top_n" in detect_intent("Siapa operator dengan jumlah transaksi terbanyak?")

    def test_terendah_maps_to_bottom_n(self):
        from llm.fallback import detect_intent
        assert "bottom_n" in detect_intent("Produk apa yang memiliki penjualan terendah?")

    def test_penjualan_maps_to_aggregation(self):
        from llm.fallback import detect_intent
        intents = detect_intent("Total penjualan berdasarkan produk")
        assert "aggregation" in intents

    def test_kosong_maps_to_missing_values(self):
        from llm.fallback import detect_intent
        assert "missing_values" in detect_intent("Apakah ada kolom yang kosong?")

    def test_duplikat_maps_to_duplicates(self):
        from llm.fallback import detect_intent
        assert "duplicates" in detect_intent("Apakah terdapat data duplikat?")

    def test_anomali_maps_to_anomaly(self):
        from llm.fallback import detect_intent
        assert "anomaly" in detect_intent("Apakah ada data anomali?")

    def test_rata_rata_maps_to_descriptive(self):
        from llm.fallback import detect_intent
        assert "descriptive" in detect_intent("Berapa rata-rata penjualan?")

    def test_grafik_maps_to_visualize(self):
        from llm.fallback import detect_intent
        assert "visualize" in detect_intent("Buat grafik penjualan berdasarkan produk")

    def test_berapa_maps_to_count(self):
        from llm.fallback import detect_intent
        assert "count" in detect_intent("Berapa jumlah transaksi per operator?")

    def test_tren_maps_to_time_series(self):
        from llm.fallback import detect_intent
        assert "time_series" in detect_intent("Tampilkan tren penjualan per bulan")

    def test_korelasi_maps_to_correlation(self):
        from llm.fallback import detect_intent
        assert "correlation" in detect_intent("Apakah ada korelasi antara penjualan dan jumlah?")


# ── Plan correctness tests ─────────────────────────────────────────────────────

class TestFallbackPlanCorrectness:
    """Verify the generated SQL groups by the CORRECT column."""

    def _get_sql(self, question: str, df: pd.DataFrame) -> str | None:
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan(question, df)
        for step in plan:
            if step["tool"] == "duckdb_query":
                return step["params"].get("sql", "")
        return None

    def test_produk_question_sql_groups_by_product(self, sales_df):
        sql = self._get_sql(
            "Produk apa yang memiliki total penjualan tertinggi?",
            sales_df,
        )
        assert sql is not None, "Expected a duckdb_query step"
        assert '"product"' in sql or "product" in sql, (
            f"SQL should GROUP BY product, got: {sql}"
        )
        assert "operator" not in sql, (
            f"REGRESSION: SQL must NOT reference operator for a produk question. Got: {sql}"
        )

    def test_operator_question_sql_groups_by_operator(self, sales_df):
        sql = self._get_sql(
            "Siapa operator dengan total penjualan terbanyak?",
            sales_df,
        )
        assert sql is not None
        assert "operator" in sql, f"SQL should GROUP BY operator. Got: {sql}"

    def test_indonesian_column_sql_correct(self, indonesian_df):
        sql = self._get_sql(
            "Produk apa yang memiliki total penjualan tertinggi?",
            indonesian_df,
        )
        assert sql is not None
        assert "produk" in sql.lower(), f"SQL should GROUP BY produk. Got: {sql}"
        assert "operator" not in sql.lower(), (
            f"REGRESSION: SQL must NOT reference operator. Got: {sql}"
        )

    def test_wilayah_question_sql_correct(self, indonesian_df):
        sql = self._get_sql(
            "Total penjualan berdasarkan wilayah",
            indonesian_df,
        )
        assert sql is not None
        assert "wilayah" in sql.lower(), f"SQL should GROUP BY wilayah. Got: {sql}"


# ── End-to-end correctness: Laptop has highest sales (400) ────────────────────

class TestEndToEndCorrectness:
    """
    Verify that the COMPLETE pipeline returns the right answer.

    Expected:
        Laptop = 100 + 300 = 400
        Phone  = 200 + 100 = 300
        Tablet = 150

    The answer must name Laptop, not Company A or Company B.
    """

    def test_highest_product_is_laptop(self, sales_df):
        from agent.graph import run_analysis

        # Force rule-based mode regardless of what .env says
        with patch("agent.nodes._is_rule_based_mode", return_value=True):
            state = run_analysis(
                question="Produk apa yang memiliki total penjualan tertinggi?",
                df=sales_df,
                evaluate=False,
            )

        assert state.get("error") is None, f"Unexpected error: {state.get('error')}"
        assert state.get("fallback_mode") is True

        # Check tool results contain the correct grouping
        sql_results = [
            r for r in state.get("tool_results", [])
            if r.get("tool") == "duckdb_query" and r.get("success")
        ]
        assert sql_results, "Expected at least one successful duckdb_query result"
        result = sql_results[0]["result"]
        rows = result.get("rows", [])
        cols = result.get("columns", [])
        assert rows, "Expected non-empty query results"

        # First column should be the grouping column (product-like)
        group_col_idx = 0
        first_group_value = str(rows[0][group_col_idx]).lower()
        assert "laptop" in first_group_value, (
            f"REGRESSION: Top result should be 'Laptop', not '{rows[0][group_col_idx]}'. "
            f"Full results: {rows[:3]}"
        )

    def test_result_not_grouped_by_operator(self, sales_df):
        from agent.graph import run_analysis

        with patch("agent.nodes._is_rule_based_mode", return_value=True):
            state = run_analysis(
                question="Produk apa yang memiliki total penjualan tertinggi?",
                df=sales_df,
                evaluate=False,
            )

        sql_results = [
            r for r in state.get("tool_results", [])
            if r.get("tool") == "duckdb_query" and r.get("success")
        ]
        if sql_results:
            result = sql_results[0]["result"]
            rows = result.get("rows", [])
            if rows:
                first_value = str(rows[0][0])
                # "Company A" or "Company B" would mean wrong grouping
                assert "company" not in first_value.lower(), (
                    f"REGRESSION: Result grouped by operator, not product. "
                    f"First row: {rows[0]}"
                )
                assert "pt." not in first_value.lower(), (
                    f"REGRESSION: Result grouped by operator. First row: {rows[0]}"
                )

    def test_total_laptop_is_400(self, sales_df):
        """Laptop: 100+300=400. This must be the actual computed value."""
        from llm.fallback import build_fallback_plan
        from tools.registry import dispatch

        with patch("agent.nodes._is_rule_based_mode", return_value=True):
            plan = build_fallback_plan(
                "Produk apa yang memiliki total penjualan tertinggi?",
                sales_df,
            )
        sql_steps = [s for s in plan if s["tool"] == "duckdb_query"]
        assert sql_steps, "Plan must contain a duckdb_query step"

        result = dispatch("duckdb_query", sql_steps[0]["params"], sales_df)
        rows = result.get("rows", [])
        assert rows, "Query returned no rows"

        # First row = highest total
        top_name = str(rows[0][0])
        top_total = float(rows[0][1])
        assert top_name.lower() == "laptop", (
            f"Expected 'Laptop' as top product, got '{top_name}'"
        )
        assert top_total == pytest.approx(400.0), (
            f"Expected total=400.0, got {top_total}"
        )

    def test_no_llm_call_in_rule_based_mode(self, sales_df):
        """No LLMClient instantiation must occur in rule-based mode.

        LLMClient is a lazy import in nodes.py so we verify by patching
        llm.provider.LLMClient and confirming it is never constructed.
        """
        from agent.graph import run_analysis
        import llm.provider as provider_module

        with patch("agent.nodes._is_rule_based_mode", return_value=True):
            with patch.object(provider_module, "LLMClient") as mock_cls:
                run_analysis(
                    question="Produk apa yang memiliki total penjualan tertinggi?",
                    df=sales_df,
                    evaluate=False,
                )
                # In rule-based mode, LLMClient must never be instantiated
                mock_cls.assert_not_called()

    def test_works_without_openai_api_key(self, sales_df, monkeypatch):
        """Application must run with no OPENAI_API_KEY set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "fallback")

        from agent.graph import run_analysis

        with patch("agent.nodes._is_rule_based_mode", return_value=True):
            state = run_analysis(
                question="Produk apa yang memiliki total penjualan tertinggi?",
                df=sales_df,
                evaluate=False,
            )

        assert state.get("error") is None


# ── Invalid column handling ────────────────────────────────────────────────────

class TestMissingAndInvalidColumns:
    def test_no_product_column_does_not_crash(self, no_product_df):
        from llm.fallback import build_fallback_plan
        plan = build_fallback_plan(
            "Produk apa yang memiliki total penjualan tertinggi?",
            no_product_df,
        )
        # Must produce a plan (even if it falls back to describe)
        assert isinstance(plan, list)
        assert len(plan) >= 1

    def test_all_plans_use_valid_tool_names(self, sales_df, indonesian_df):
        from llm.fallback import build_fallback_plan
        from tools.registry import list_tools
        allowed = set(list_tools())
        questions = [
            "Produk apa yang memiliki total penjualan tertinggi?",
            "Siapa operator dengan transaksi terbanyak?",
            "Berapa rata-rata penjualan?",
            "Apakah ada data anomali?",
            "Tampilkan korelasi antar kolom",
            "Apakah ada nilai kosong?",
            "Apakah ada duplikat?",
        ]
        for q in questions:
            for df in [sales_df, indonesian_df]:
                plan = build_fallback_plan(q, df)
                for step in plan:
                    assert step["tool"] in allowed, (
                        f"Unknown tool '{step['tool']}' for question: {q}"
                    )

    def test_mean_calculation_correct(self, sales_df):
        """sales mean = (100+200+300+150+100)/5 = 170."""
        from tools.pandas_tool import pandas_describe
        result = pandas_describe(sales_df, columns=["sales"])
        assert "sales" in result
        assert result["sales"]["mean"] == pytest.approx(170.0)

    def test_sum_by_product_correct(self, sales_df):
        """Laptop=400, Phone=300, Tablet=150."""
        from tools.duckdb_tool import run_duckdb_query
        result = run_duckdb_query(
            sales_df,
            sql='SELECT "product", SUM("sales") AS total FROM df GROUP BY "product" ORDER BY total DESC',
        )
        rows = {str(r[0]): float(r[1]) for r in result["rows"]}
        assert rows["Laptop"] == pytest.approx(400.0)
        assert rows["Phone"] == pytest.approx(300.0)
        assert rows["Tablet"] == pytest.approx(150.0)
    def test_missing_product_does_not_fall_back_to_operator(self, no_product_df):
        from llm.fallback import build_fallback_plan
        with pytest.raises(ValueError, match="produk|product"):
            build_fallback_plan(
                "Produk apa yang memiliki total penjualan tertinggi?", no_product_df
            )

    def test_missing_sales_metric_does_not_fall_back_to_year(self):
        from llm.fallback import build_fallback_plan
        df = pd.DataFrame({
            "operator": ["A", "B"],
            "year": [2014, 2015],
        })
        with pytest.raises(ValueError, match="metrik|numerik"):
            build_fallback_plan("Operator dengan total penjualan tertinggi", df)

    def test_transaction_ranking_uses_count_not_first_numeric_column(self):
        from llm.fallback import build_fallback_plan
        df = pd.DataFrame({
            "operator": ["A", "A", "B"],
            "year": [2014, 2014, 2014],
        })
        plan = build_fallback_plan(
            "Operator dengan jumlah transaksi terbanyak", df
        )
        sql = next(step["params"]["sql"] for step in plan if step["tool"] == "duckdb_query")
        assert "COUNT(*)" in sql
        assert "SUM(\"year\")" not in sql

