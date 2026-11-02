"""
tools/stats_tool.py
===================
Statistical analysis tools.

Supported tests
---------------
- correlation  : Pearson / Spearman correlation matrix
- ttest        : Independent samples t-test between two groups
- anova        : One-way ANOVA across groups
- chi2         : Chi-squared test of independence for categorical columns
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from core.exceptions import ToolError
from core.logging import get_logger

logger = get_logger(__name__)

CorrelationMethod = Literal["pearson", "spearman"]


def run_correlation(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: CorrelationMethod = "pearson",
) -> dict[str, Any]:
    """Compute a correlation matrix.

    Returns:
        ``{"columns": [...], "matrix": [[...], ...], "method": "pearson"}``
    """
    try:
        if columns:
            subset = df[columns].select_dtypes(include=[np.number])
        else:
            subset = df.select_dtypes(include=[np.number])

        if subset.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns for correlation."}

        corr = subset.corr(method=method)
        return {
            "columns": list(corr.columns),
            "matrix": [[_safe(v) for v in row] for row in corr.values.tolist()],
            "method": method,
        }
    except Exception as exc:
        raise ToolError(f"correlation failed: {exc}", details={"error": str(exc)}) from exc


def run_ttest(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
) -> dict[str, Any]:
    """Independent samples t-test.

    Compares the first two groups found in *group_col*.

    Returns:
        ``{"t_stat": ..., "p_value": ..., "groups": [...], "means": [...], "significant": bool}``
    """
    try:
        groups = df[group_col].dropna().unique()[:2]
        if len(groups) < 2:
            return {"error": f"Column '{group_col}' has fewer than 2 distinct groups."}

        a = df[df[group_col] == groups[0]][value_col].dropna().values
        b = df[df[group_col] == groups[1]][value_col].dropna().values

        if len(a) < 2 or len(b) < 2:
            return {"error": "Each group needs at least 2 observations."}

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        return {
            "t_stat": _safe(t_stat),
            "p_value": _safe(p_value),
            "groups": [str(g) for g in groups],
            "means": [_safe(float(a.mean())), _safe(float(b.mean()))],
            "significant": bool(p_value < 0.05),
        }
    except Exception as exc:
        raise ToolError(f"ttest failed: {exc}", details={"error": str(exc)}) from exc


def run_anova(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
) -> dict[str, Any]:
    """One-way ANOVA.

    Returns:
        ``{"f_stat": ..., "p_value": ..., "groups": n, "significant": bool}``
    """
    try:
        group_data = [
            group[value_col].dropna().values
            for _, group in df.groupby(group_col)
            if len(group[value_col].dropna()) >= 2
        ]
        if len(group_data) < 2:
            return {"error": "Need at least 2 groups with 2+ observations each."}

        f_stat, p_value = stats.f_oneway(*group_data)
        return {
            "f_stat": _safe(f_stat),
            "p_value": _safe(p_value),
            "group_count": len(group_data),
            "significant": bool(p_value < 0.05),
        }
    except Exception as exc:
        raise ToolError(f"anova failed: {exc}", details={"error": str(exc)}) from exc


def run_chi2(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict[str, Any]:
    """Chi-squared test of independence.

    Returns:
        ``{"chi2_stat": ..., "p_value": ..., "dof": ..., "significant": bool}``
    """
    try:
        contingency = pd.crosstab(df[col_a], df[col_b])
        if contingency.size == 0:
            return {"error": "Contingency table is empty."}

        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        return {
            "chi2_stat": _safe(chi2),
            "p_value": _safe(p_value),
            "dof": int(dof),
            "significant": bool(p_value < 0.05),
        }
    except Exception as exc:
        raise ToolError(f"chi2 failed: {exc}", details={"error": str(exc)}) from exc


def _safe(v: Any) -> Any:
    """Return None for NaN/Inf, else native Python float/int."""
    try:
        f = float(v)
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    except (TypeError, ValueError):
        return v
