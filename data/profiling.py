"""
data/profiling.py
=================
Automated data profiling — produces a rich summary of a DataFrame
without relying on external profiling libraries (ydata-profiling, etc.)
to keep the dependency footprint small and the output fully controlled.

Output: ``ProfileReport`` dataclass, JSON-serialisable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.exceptions import ProfilingError
from core.logging import get_logger

logger = get_logger(__name__)


# ── Output models ─────────────────────────────────────────────────────────────

@dataclass
class NumericStats:
    mean: float
    median: float
    std: float
    min: float
    max: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float
    zeros: int
    zeros_pct: float


@dataclass
class TextStats:
    avg_length: float
    min_length: int
    max_length: int
    unique_count: int
    unique_pct: float
    top_values: list[dict[str, Any]]   # [{"value": x, "count": n, "pct": p}, ...]


@dataclass
class DatetimeStats:
    min: str
    max: str
    range_days: float
    unique_count: int


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    dtype_category: str
    total_count: int
    null_count: int
    null_pct: float
    distinct_count: int
    distinct_pct: float
    is_unique: bool
    numeric: NumericStats | None = None
    text: TextStats | None = None
    datetime: DatetimeStats | None = None


@dataclass
class DatasetProfile:
    source_name: str
    row_count: int
    column_count: int
    total_cells: int
    total_nulls: int
    null_pct: float
    duplicate_row_count: int
    duplicate_row_pct: float
    memory_usage_bytes: int
    columns: list[ColumnProfile] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_float(v: Any) -> float:
    """Convert to float, replacing NaN/Inf with 0.0."""
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return 0.0


def _profile_numeric(series: pd.Series) -> NumericStats:
    s = series.dropna()
    if s.empty:
        return NumericStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)
    q1 = _safe_float(s.quantile(0.25))
    q3 = _safe_float(s.quantile(0.75))
    zeros = int((s == 0).sum())
    return NumericStats(
        mean=_safe_float(s.mean()),
        median=_safe_float(s.median()),
        std=_safe_float(s.std()),
        min=_safe_float(s.min()),
        max=_safe_float(s.max()),
        q1=q1,
        q3=q3,
        iqr=_safe_float(q3 - q1),
        skewness=_safe_float(s.skew()),
        kurtosis=_safe_float(s.kurtosis()),
        zeros=zeros,
        zeros_pct=_safe_float(zeros / max(len(s), 1) * 100),
    )


def _profile_text(series: pd.Series) -> TextStats:
    s = series.dropna().astype(str)
    lengths = s.str.len()
    vc = series.value_counts(normalize=False).head(10)
    total = max(len(series.dropna()), 1)
    top_values = [
        {"value": str(v), "count": int(c), "pct": round(c / total * 100, 2)}
        for v, c in vc.items()
    ]
    unique = int(series.nunique(dropna=True))
    return TextStats(
        avg_length=_safe_float(lengths.mean()) if not lengths.empty else 0.0,
        min_length=int(lengths.min()) if not lengths.empty else 0,
        max_length=int(lengths.max()) if not lengths.empty else 0,
        unique_count=unique,
        unique_pct=_safe_float(unique / max(len(series.dropna()), 1) * 100),
        top_values=top_values,
    )


def _profile_datetime(series: pd.Series) -> DatetimeStats:
    s = series.dropna()
    if s.empty:
        return DatetimeStats("N/A", "N/A", 0.0, 0)
    mn, mx = s.min(), s.max()
    delta = (mx - mn).total_seconds() / 86400
    return DatetimeStats(
        min=str(mn),
        max=str(mx),
        range_days=_safe_float(delta),
        unique_count=int(s.nunique()),
    )


def _profile_column(col: str, series: pd.Series, total_rows: int) -> ColumnProfile:
    null_count = int(series.isna().sum())
    distinct_count = int(series.nunique(dropna=True))

    profile = ColumnProfile(
        name=col,
        dtype=str(series.dtype),
        dtype_category="unknown",
        total_count=total_rows,
        null_count=null_count,
        null_pct=_safe_float(null_count / max(total_rows, 1) * 100),
        distinct_count=distinct_count,
        distinct_pct=_safe_float(distinct_count / max(total_rows - null_count, 1) * 100),
        is_unique=distinct_count == total_rows - null_count and null_count < total_rows,
    )

    if pd.api.types.is_bool_dtype(series):
        profile.dtype_category = "boolean"
        profile.text = _profile_text(series.astype(str))
    elif pd.api.types.is_numeric_dtype(series):
        profile.dtype_category = "numeric"
        profile.numeric = _profile_numeric(series)
    elif pd.api.types.is_datetime64_any_dtype(series):
        profile.dtype_category = "datetime"
        profile.datetime = _profile_datetime(series)
    else:
        profile.dtype_category = "text"
        profile.text = _profile_text(series)

    return profile


# ── Public API ────────────────────────────────────────────────────────────────

def profile_dataframe(df: pd.DataFrame, source_name: str = "dataset") -> DatasetProfile:
    """Compute a full profile of *df*.

    Args:
        df: The DataFrame to profile.
        source_name: Label for the report.

    Returns:
        ``DatasetProfile`` with per-column statistics.

    Raises:
        ``ProfilingError`` on unexpected failures.
    """
    logger.info("profiling_started", source=source_name, rows=len(df), cols=len(df.columns))
    try:
        total_cells = df.size
        total_nulls = int(df.isna().sum().sum())
        dup_rows = int(df.duplicated().sum())

        dataset_profile = DatasetProfile(
            source_name=source_name,
            row_count=len(df),
            column_count=len(df.columns),
            total_cells=total_cells,
            total_nulls=total_nulls,
            null_pct=_safe_float(total_nulls / max(total_cells, 1) * 100),
            duplicate_row_count=dup_rows,
            duplicate_row_pct=_safe_float(dup_rows / max(len(df), 1) * 100),
            memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
        )

        for col in df.columns:
            try:
                col_profile = _profile_column(col, df[col], len(df))
                dataset_profile.columns.append(col_profile)
            except Exception as exc:
                logger.warning("column_profile_failed", column=col, error=str(exc))
                # Partial failure — skip column rather than abort entire profile

        logger.info("profiling_complete", source=source_name)
        return dataset_profile

    except ProfilingError:
        raise
    except Exception as exc:
        raise ProfilingError(
            f"Failed to profile '{source_name}': {exc}",
            details={"error": str(exc)},
        ) from exc
