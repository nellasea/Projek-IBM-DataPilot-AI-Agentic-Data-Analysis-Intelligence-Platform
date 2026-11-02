"""
tools/pandas_tool.py
====================
Safe Pandas operations exposed as discrete, parameterised functions.

All operations are read-only — they never mutate the source DataFrame.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.exceptions import PandasToolError
from core.logging import get_logger

logger = get_logger(__name__)


def pandas_describe(df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, Any]:
    """Return descriptive statistics for *columns* (or all numeric columns).

    Args:
        df: Source DataFrame.
        columns: Optional list of column names. Defaults to all numeric columns.

    Returns:
        Dict mapping column name → stats dict.

    Raises:
        ``PandasToolError``: if any column does not exist.
    """
    try:
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise PandasToolError(
                    f"Columns not found: {missing}",
                    details={"missing": missing, "available": list(df.columns)},
                )
            subset = df[columns]
        else:
            subset = df.select_dtypes(include=[np.number])

        if subset.empty or subset.shape[1] == 0:
            return {}

        desc = subset.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        # Convert numpy types to native Python for JSON serialisation
        return {
            col: {k: _to_python(v) for k, v in stats.items()}
            for col, stats in desc.items()
        }
    except PandasToolError:
        raise
    except Exception as exc:
        raise PandasToolError(
            f"pandas_describe failed: {exc}",
            details={"error": str(exc)},
        ) from exc


def pandas_value_counts(
    df: pd.DataFrame, column: str, top_n: int = 20
) -> dict[str, Any]:
    """Return value counts for *column*.

    Returns:
        ``{"values": [...], "counts": [...]}``
    """
    try:
        if column not in df.columns:
            raise PandasToolError(
                f"Column '{column}' not found.",
                details={"column": column, "available": list(df.columns)},
            )
        vc = df[column].value_counts(dropna=False).head(top_n)
        return {
            "values": [str(v) for v in vc.index.tolist()],
            "counts": vc.values.tolist(),
        }
    except PandasToolError:
        raise
    except Exception as exc:
        raise PandasToolError(
            f"pandas_value_counts failed: {exc}",
            details={"error": str(exc)},
        ) from exc


def pandas_filter(
    df: pd.DataFrame, column: str, operator: str, value: Any
) -> dict[str, Any]:
    """Filter DataFrame and return a summary of matching rows.

    Supported operators: ``eq``, ``ne``, ``gt``, ``gte``, ``lt``, ``lte``, ``contains``

    Returns:
        ``{"matched_rows": n, "sample": [[...], ...], "columns": [...]}``
    """
    _OPERATORS = {
        "eq": lambda s, v: s == v,
        "ne": lambda s, v: s != v,
        "gt": lambda s, v: s > v,
        "gte": lambda s, v: s >= v,
        "lt": lambda s, v: s < v,
        "lte": lambda s, v: s <= v,
        "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
    }
    try:
        if column not in df.columns:
            raise PandasToolError(f"Column '{column}' not found.")
        if operator not in _OPERATORS:
            raise PandasToolError(
                f"Unsupported operator '{operator}'. "
                f"Allowed: {list(_OPERATORS.keys())}"
            )
        mask = _OPERATORS[operator](df[column], value)
        filtered = df[mask].head(1000)
        return {
            "matched_rows": int(mask.sum()),
            "sample": filtered.head(20).values.tolist(),
            "columns": list(filtered.columns),
        }
    except PandasToolError:
        raise
    except Exception as exc:
        raise PandasToolError(
            f"pandas_filter failed: {exc}", details={"error": str(exc)}
        ) from exc


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_python(v: Any) -> Any:
    """Convert numpy scalars to native Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
        return None  # NaN / Inf → null in JSON
    return v
