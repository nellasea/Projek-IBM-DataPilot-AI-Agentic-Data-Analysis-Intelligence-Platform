"""
tests/test_quality.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.quality import (
    inspect_quality,
    _check_missing,
    _check_duplicates,
    _check_constant_columns,
    _check_outliers,
    _check_type_mismatches,
    IssueSeverity,
)


def test_no_issues_clean_df(sample_df):
    report = inspect_quality(sample_df, "clean")
    assert report.score > 80
    # No missing values or duplicates expected
    missing_issues = [i for i in report.issues if i.check == "missing_values"]
    assert len(missing_issues) == 0


def test_detects_missing_values(sample_df_with_nulls):
    report = inspect_quality(sample_df_with_nulls, "nulls")
    checks = [i.check for i in report.issues]
    assert "missing_values" in checks


def test_detects_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    report = inspect_quality(df, "dups")
    checks = [i.check for i in report.issues]
    assert "duplicate_rows" in checks


def test_detects_constant_column():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5]})
    issues = _check_constant_columns(df)
    assert len(issues) == 1
    assert issues[0].column == "b"


def test_detects_outliers():
    import numpy as np
    values = list(range(100)) + [10000]  # one extreme outlier
    df = pd.DataFrame({"value": values})
    issues = _check_outliers(df)
    assert len(issues) == 1
    assert issues[0].details["outlier_count"] >= 1


def test_detects_type_mismatch():
    df = pd.DataFrame({"nums_as_str": ["1", "2", "3", "4", "5"] * 40})
    issues = _check_type_mismatches(df)
    assert len(issues) == 1
    assert issues[0].check == "type_mismatch"


def test_score_decrements_with_issues(sample_df_with_nulls):
    report = inspect_quality(sample_df_with_nulls, "score_test")
    assert report.score < 100.0


def test_error_vs_warning_severity():
    # >50% nulls → ERROR
    df = pd.DataFrame({"col": [None] * 60 + [1] * 40})
    issues = _check_missing(df)
    assert issues[0].severity == IssueSeverity.ERROR

    # <50% nulls → WARNING
    df2 = pd.DataFrame({"col": [None] * 10 + [1] * 90})
    issues2 = _check_missing(df2)
    assert issues2[0].severity == IssueSeverity.WARNING
