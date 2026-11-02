"""
tests/test_ingestion.py
"""
from __future__ import annotations

import io
import pytest
import pandas as pd

from data.ingestion import ingest_file, _sanitise_column_names
from core.exceptions import IngestionError


def test_ingest_csv_bytes(sample_csv_bytes):
    result = ingest_file(sample_csv_bytes, source_name="test.csv")
    assert result.row_count == 20
    assert result.column_count == 6
    assert result.source_name == "test.csv"


def test_ingest_csv_column_schemas(sample_csv_bytes):
    result = ingest_file(sample_csv_bytes, source_name="test.csv")
    col_names = [c.name for c in result.columns]
    assert "price" in col_names
    price_col = next(c for c in result.columns if c.name == "price")
    assert price_col.dtype_category == "numeric"


def test_ingest_empty_raises():
    empty_csv = b"col1,col2\n"
    with pytest.raises(IngestionError, match="empty"):
        ingest_file(empty_csv, source_name="empty.csv")


def test_ingest_file_not_found():
    with pytest.raises(IngestionError, match="not found"):
        ingest_file("/tmp/nonexistent_datapilot_xyz.csv")


def test_ingest_exceeds_size_limit(monkeypatch, sample_csv_bytes):
    # Patch _check_size to always raise, simulating an oversize file
    from data import ingestion as ing
    def mock_check(size_bytes, source_name):
        raise IngestionError(f"File '{source_name}' exceeds the size limit.", details={})
    monkeypatch.setattr(ing, "_check_size", mock_check)
    with pytest.raises(IngestionError, match="exceeds"):
        ingest_file(sample_csv_bytes, source_name="big.csv")


def test_sanitise_duplicate_columns():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    result = _sanitise_column_names(df)
    assert list(result.columns) == ["a", "a_1", "b"]


def test_sanitise_strips_whitespace():
    df = pd.DataFrame([[1]], columns=["  col  "])
    result = _sanitise_column_names(df)
    assert list(result.columns) == ["col"]


def test_ingest_xlsx(tmp_path):
    """Write a real XLSX and ingest it."""
    xlsx_path = tmp_path / "test.xlsx"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_excel(xlsx_path, index=False)
    result = ingest_file(str(xlsx_path), source_name="test.xlsx")
    assert result.row_count == 3
    assert result.column_count == 2
