"""
data/ingestion.py
=================
CSV and XLSX ingestion with validation and schema inference.

Responsibilities
----------------
- Accept file paths or raw bytes.
- Detect file type (CSV / XLSX).
- Load into a validated pandas DataFrame.
- Infer and return column schema (name, dtype category, nullable).
- Enforce size limits from settings.
- Raise ``IngestionError`` on any failure — never propagate raw library exceptions.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from core.config import get_settings
from core.exceptions import IngestionError
from core.logging import get_logger

logger = get_logger(__name__)

DTypeCategory = Literal["numeric", "datetime", "boolean", "text", "unknown"]


@dataclass
class ColumnSchema:
    name: str
    dtype_raw: str          # pandas dtype string, e.g. "int64"
    dtype_category: DTypeCategory
    nullable: bool          # True if the column contains any NaN


@dataclass
class IngestionResult:
    dataframe: pd.DataFrame
    source_name: str        # original filename or label
    row_count: int
    column_count: int
    columns: list[ColumnSchema] = field(default_factory=list)
    file_size_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _categorise_dtype(dtype: pd.api.types.CategoricalDtypeType) -> DTypeCategory:
    """Map a pandas dtype to a broad category."""
    dtype_str = str(dtype)
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if dtype_str in ("object", "string", "category"):
        return "text"
    return "unknown"


def _build_column_schemas(df: pd.DataFrame) -> list[ColumnSchema]:
    return [
        ColumnSchema(
            name=col,
            dtype_raw=str(df[col].dtype),
            dtype_category=_categorise_dtype(df[col].dtype),
            nullable=bool(df[col].isna().any()),
        )
        for col in df.columns
    ]


def _check_size(size_bytes: int, source_name: str) -> None:
    settings = get_settings()
    if size_bytes > settings.max_upload_bytes:
        limit_mb = settings.max_upload_mb
        actual_mb = size_bytes / (1024 * 1024)
        raise IngestionError(
            f"File '{source_name}' is {actual_mb:.1f} MB, "
            f"which exceeds the {limit_mb} MB limit.",
            details={"size_bytes": size_bytes, "limit_bytes": settings.max_upload_bytes},
        )


def _try_parse_datetimes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Best-effort conversion of object columns that look like dates."""
    warnings: list[str] = []
    for col in df.select_dtypes(include=["object", "str"]).columns:
        sample = df[col].dropna().head(50)
        try:
            converted = pd.to_datetime(sample, infer_datetime_format=True, errors="raise")
            # Only convert if the majority of non-null values parse successfully
            full_converted = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            parse_rate = full_converted.notna().sum() / max(df[col].notna().sum(), 1)
            if parse_rate >= 0.8:
                df[col] = full_converted
                logger.debug("datetime_inferred", column=col, parse_rate=f"{parse_rate:.0%}")
        except Exception:
            pass  # Leave column as-is if parsing fails
    return df, warnings


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_file(source: str | Path | bytes, source_name: str = "dataset") -> IngestionResult:
    """Load a CSV or XLSX file into an ``IngestionResult``.

    Args:
        source: Filesystem path (str / Path) or raw file bytes.
        source_name: Human-readable label for error messages and metadata.

    Returns:
        ``IngestionResult`` with a validated DataFrame and column schemas.

    Raises:
        ``IngestionError``: on any parse or validation failure.
    """
    logger.info("ingestion_started", source=source_name)

    # ── Resolve bytes ─────────────────────────────────────────────────────────
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise IngestionError(f"File not found: {path}")
        raw_bytes = path.read_bytes()
        source_name = source_name or path.name
    else:
        raw_bytes = source

    _check_size(len(raw_bytes), source_name)

    # ── Detect format and parse ───────────────────────────────────────────────
    name_lower = source_name.lower()
    try:
        if name_lower.endswith(".xlsx") or name_lower.endswith(".xls"):
            df = _load_excel(raw_bytes, source_name)
        elif name_lower.endswith(".csv"):
            df = _load_csv(raw_bytes, source_name)
        else:
            # Try CSV first, then Excel
            df = _load_csv_or_excel_fallback(raw_bytes, source_name)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Unexpected error parsing '{source_name}': {exc}",
            details={"error": str(exc)},
        ) from exc

    if df.empty:
        raise IngestionError(
            f"'{source_name}' loaded an empty dataset (0 rows or 0 columns)."
        )

    # ── Post-processing ───────────────────────────────────────────────────────
    df = _sanitise_column_names(df)
    df, dt_warnings = _try_parse_datetimes(df)

    columns = _build_column_schemas(df)
    result = IngestionResult(
        dataframe=df,
        source_name=source_name,
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        file_size_bytes=len(raw_bytes),
        warnings=dt_warnings,
    )

    logger.info(
        "ingestion_complete",
        source=source_name,
        rows=result.row_count,
        cols=result.column_count,
    )
    return result


def _load_csv(raw_bytes: bytes, source_name: str) -> pd.DataFrame:
    """Parse bytes as CSV, trying common encodings."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(
                io.BytesIO(raw_bytes),
                encoding=encoding,
                low_memory=False,
                on_bad_lines="warn",
            )
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise IngestionError(
                f"CSV parse error in '{source_name}': {exc}",
                details={"error": str(exc)},
            ) from exc
    raise IngestionError(
        f"Could not decode '{source_name}' as CSV with utf-8, latin-1, or cp1252."
    )


def _load_excel(raw_bytes: bytes, source_name: str) -> pd.DataFrame:
    """Parse bytes as Excel (xlsx/xls)."""
    try:
        xl = pd.ExcelFile(io.BytesIO(raw_bytes), engine="openpyxl")
        if len(xl.sheet_names) > 1:
            logger.warning(
                "multi_sheet_excel",
                source=source_name,
                sheets=xl.sheet_names,
                note="Loading first sheet only.",
            )
        return xl.parse(xl.sheet_names[0])
    except Exception as exc:
        raise IngestionError(
            f"Excel parse error in '{source_name}': {exc}",
            details={"error": str(exc)},
        ) from exc


def _load_csv_or_excel_fallback(raw_bytes: bytes, source_name: str) -> pd.DataFrame:
    """Try CSV first, then Excel, when extension is unknown."""
    try:
        return _load_csv(raw_bytes, source_name)
    except IngestionError:
        pass
    try:
        return _load_excel(raw_bytes, source_name)
    except IngestionError:
        pass
    raise IngestionError(
        f"Could not parse '{source_name}' as CSV or Excel. "
        "Ensure the file has a .csv or .xlsx extension."
    )


def _sanitise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and deduplicate column names."""
    df.columns = [str(c).strip() for c in df.columns]
    # Deduplicate: col, col_1, col_2, ...
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df
