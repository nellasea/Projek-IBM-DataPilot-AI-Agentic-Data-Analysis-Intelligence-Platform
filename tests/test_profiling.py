"""
tests/test_profiling.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.profiling import profile_dataframe, _profile_numeric, _profile_text


def test_profile_basic(sample_df):
    report = profile_dataframe(sample_df, "test")
    assert report.row_count == 20
    assert report.column_count == 6
    assert report.total_cells == 120
    assert report.total_nulls == 0
    assert len(report.columns) == 6


def test_profile_null_tracking(sample_df_with_nulls):
    report = profile_dataframe(sample_df_with_nulls, "nulls")
    assert report.total_nulls >= 2


def test_profile_numeric_stats():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = _profile_numeric(series)
    assert stats.mean == pytest.approx(3.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.min == pytest.approx(1.0)
    assert stats.max == pytest.approx(5.0)


def test_profile_text_stats():
    series = pd.Series(["apple", "banana", "apple", "cherry", None])
    stats = _profile_text(series)
    assert stats.unique_count == 3
    assert stats.max_length == 6  # "banana" or "cherry"
    assert len(stats.top_values) >= 1


def test_profile_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    report = profile_dataframe(df, "dups")
    assert report.duplicate_row_count == 1


def test_profile_column_categories(sample_df):
    report = profile_dataframe(sample_df, "cats")
    type_map = {c.name: c.dtype_category for c in report.columns}
    assert type_map["price"] == "numeric"
    assert type_map["name"] == "text"
    assert type_map["in_stock"] == "boolean"
