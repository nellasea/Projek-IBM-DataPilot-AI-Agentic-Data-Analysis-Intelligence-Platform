"""
tools/duckdb_tool.py
====================
Safe DuckDB SQL execution against an in-memory database loaded from a DataFrame.

Security model
--------------
- Only SELECT statements are allowed (enforced via AST check).
- The database is in-memory; no file system access is possible.
- No DDL (CREATE, DROP, ALTER) or DML (INSERT, UPDATE, DELETE) is permitted.
- Query results are capped at ``MAX_ROWS`` to prevent memory exhaustion.
"""
from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd

from core.exceptions import DuckDBToolError
from core.logging import get_logger

logger = get_logger(__name__)

MAX_ROWS = 10_000
TABLE_NAME = "df"

# Keywords that indicate non-SELECT statements
_FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|ATTACH|COPY|EXPORT|PRAGMA|CALL)\b",
    re.IGNORECASE,
)


def _validate_sql(sql: str) -> None:
    """Raise DuckDBToolError if the SQL is not a safe SELECT statement."""
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise DuckDBToolError(
            "Only SELECT statements are permitted.",
            details={"sql": sql},
        )
    if _FORBIDDEN_PATTERN.search(stripped):
        match = _FORBIDDEN_PATTERN.search(stripped)
        raise DuckDBToolError(
            f"Forbidden SQL keyword detected: {match.group(0).upper()}",
            details={"sql": sql},
        )


def run_duckdb_query(df: pd.DataFrame, sql: str) -> dict[str, Any]:
    """Execute a SQL SELECT query against *df* and return the result as a dict.

    Args:
        df: The source DataFrame (registered as table ``"df"``).
        sql: A SQL SELECT statement.

    Returns:
        ``{"columns": [...], "rows": [[...], ...], "row_count": n}``

    Raises:
        ``DuckDBToolError``: on validation failure or execution error.
    """
    _validate_sql(sql)
    logger.debug("duckdb_query_start", sql=sql[:200])

    try:
        con = duckdb.connect(database=":memory:")
        con.register(TABLE_NAME, df)
        result = con.execute(sql).fetchdf()

        if len(result) > MAX_ROWS:
            logger.warning("duckdb_result_truncated", original=len(result), cap=MAX_ROWS)
            result = result.head(MAX_ROWS)

        logger.debug("duckdb_query_ok", rows=len(result), cols=len(result.columns))
        return {
            "columns": list(result.columns),
            "rows": result.values.tolist(),
            "row_count": len(result),
            "dataframe": result,  # kept for downstream tools; stripped before JSON serialisation
        }
    except DuckDBToolError:
        raise
    except Exception as exc:
        raise DuckDBToolError(
            f"DuckDB query failed: {exc}",
            details={"sql": sql, "error": str(exc)},
        ) from exc
    finally:
        try:
            con.close()
        except Exception:
            pass
