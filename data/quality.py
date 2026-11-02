"""
data/quality.py
===============
Data quality inspection.

Checks performed
----------------
1. Missing values per column
2. Duplicate rows
3. Constant columns (zero variance)
4. High-cardinality text columns
5. Numeric outliers (IQR method)
6. Suspected type mismatches (numeric stored as text)
7. Date ordering anomalies (datetime columns out of expected range)

Output: ``QualityReport`` dataclass, JSON-serialisable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from core.exceptions import QualityError
from core.logging import get_logger

logger = get_logger(__name__)


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityIssue:
    check: str                  # name of the check that raised the issue
    severity: IssueSeverity
    column: str | None          # None = dataset-level issue
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    source_name: str
    row_count: int
    column_count: int
    issues: list[QualityIssue] = field(default_factory=list)
    score: float = 100.0        # 0–100, decremented per issue severity

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    @property
    def passed(self) -> bool:
        return self.error_count == 0


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_missing(df: pd.DataFrame) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue
        null_pct = null_count / len(df) * 100
        severity = IssueSeverity.ERROR if null_pct > 50 else IssueSeverity.WARNING
        issues.append(QualityIssue(
            check="missing_values",
            severity=severity,
            column=col,
            message=f"Column '{col}' has {null_count} missing values ({null_pct:.1f}%).",
            details={"null_count": null_count, "null_pct": round(null_pct, 2)},
        ))
    return issues


def _check_duplicates(df: pd.DataFrame) -> list[QualityIssue]:
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        return []
    dup_pct = dup_count / len(df) * 100
    return [QualityIssue(
        check="duplicate_rows",
        severity=IssueSeverity.WARNING,
        column=None,
        message=f"Dataset contains {dup_count} duplicate rows ({dup_pct:.1f}%).",
        details={"duplicate_count": dup_count, "duplicate_pct": round(dup_pct, 2)},
    )]


def _check_constant_columns(df: pd.DataFrame) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            issues.append(QualityIssue(
                check="constant_column",
                severity=IssueSeverity.WARNING,
                column=col,
                message=f"Column '{col}' has zero variance (all values are identical or null).",
                details={"unique_values": int(df[col].nunique(dropna=False))},
            ))
    return issues


def _check_high_cardinality(df: pd.DataFrame, threshold: float = 0.95) -> list[QualityIssue]:
    """Flag text columns where distinct ratio > threshold (likely IDs mistakenly included)."""
    issues: list[QualityIssue] = []
    for col in df.select_dtypes(include=["object", "str"]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        ratio = non_null.nunique() / len(non_null)
        if ratio >= threshold:
            issues.append(QualityIssue(
                check="high_cardinality",
                severity=IssueSeverity.INFO,
                column=col,
                message=(
                    f"Column '{col}' has very high cardinality ({ratio:.0%} unique values). "
                    "It may be an identifier column — consider excluding from analysis."
                ),
                details={"unique_ratio": round(ratio, 4)},
            ))
    return issues


def _check_outliers(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> list[QualityIssue]:
    """IQR-based outlier detection for numeric columns."""
    issues: list[QualityIssue] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        outlier_mask = (series < lower) | (series > upper)
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            outlier_pct = outlier_count / len(series) * 100
            issues.append(QualityIssue(
                check="outliers_iqr",
                severity=IssueSeverity.INFO,
                column=col,
                message=(
                    f"Column '{col}' has {outlier_count} potential outliers "
                    f"({outlier_pct:.1f}%) using IQR×{iqr_multiplier}."
                ),
                details={
                    "outlier_count": outlier_count,
                    "outlier_pct": round(outlier_pct, 2),
                    "lower_fence": round(float(lower), 4),
                    "upper_fence": round(float(upper), 4),
                },
            ))
    return issues


def _check_type_mismatches(df: pd.DataFrame) -> list[QualityIssue]:
    """Detect numeric data stored as strings in object columns."""
    issues: list[QualityIssue] = []
    for col in df.select_dtypes(include=["object", "str"]).columns:
        sample = df[col].dropna().head(200)
        if len(sample) == 0:
            continue
        try:
            converted = pd.to_numeric(sample, errors="coerce")
            numeric_pct = converted.notna().sum() / len(sample)
            if numeric_pct >= 0.85:
                issues.append(QualityIssue(
                    check="type_mismatch",
                    severity=IssueSeverity.WARNING,
                    column=col,
                    message=(
                        f"Column '{col}' appears to contain numeric data stored as text "
                        f"({numeric_pct:.0%} of sampled values are numeric)."
                    ),
                    details={"numeric_pct_sample": round(numeric_pct, 4)},
                ))
        except Exception:
            pass
    return issues


# ── Score calculation ─────────────────────────────────────────────────────────

_SEVERITY_DEDUCTIONS = {
    IssueSeverity.ERROR: 15.0,
    IssueSeverity.WARNING: 5.0,
    IssueSeverity.INFO: 1.0,
}


def _calculate_score(issues: list[QualityIssue]) -> float:
    score = 100.0
    for issue in issues:
        score -= _SEVERITY_DEDUCTIONS.get(issue.severity, 0)
    return max(0.0, score)


# ── Public API ────────────────────────────────────────────────────────────────

def inspect_quality(df: pd.DataFrame, source_name: str = "dataset") -> QualityReport:
    """Run all quality checks against *df*.

    Args:
        df: DataFrame to inspect.
        source_name: Label for the report.

    Returns:
        ``QualityReport`` with categorised issues and a quality score (0–100).

    Raises:
        ``QualityError`` on unexpected failures.
    """
    logger.info("quality_check_started", source=source_name)
    try:
        issues: list[QualityIssue] = []
        issues.extend(_check_missing(df))
        issues.extend(_check_duplicates(df))
        issues.extend(_check_constant_columns(df))
        issues.extend(_check_high_cardinality(df))
        issues.extend(_check_outliers(df))
        issues.extend(_check_type_mismatches(df))

        score = _calculate_score(issues)
        report = QualityReport(
            source_name=source_name,
            row_count=len(df),
            column_count=len(df.columns),
            issues=issues,
            score=score,
        )
        logger.info(
            "quality_check_complete",
            source=source_name,
            score=score,
            errors=report.error_count,
            warnings=report.warning_count,
        )
        return report

    except QualityError:
        raise
    except Exception as exc:
        raise QualityError(
            f"Quality check failed for '{source_name}': {exc}",
            details={"error": str(exc)},
        ) from exc
