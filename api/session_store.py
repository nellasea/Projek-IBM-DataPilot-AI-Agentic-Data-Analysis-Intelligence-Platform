"""
api/session_store.py
====================
In-process session store: keeps ingested DataFrames in memory
keyed by session_id so subsequent API calls can reference them.

For production, replace with Redis or a shared object store.
"""
from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)

# session_id → {"df": DataFrame, "source_name": str, "meta": dict}
_STORE: dict[str, dict[str, Any]] = {}


def new_session_id() -> str:
    return str(uuid.uuid4())


def store_dataframe(session_id: str, df: pd.DataFrame, source_name: str, meta: dict | None = None) -> None:
    _STORE[session_id] = {"df": df, "source_name": source_name, "meta": meta or {}}
    logger.debug("session_stored", session_id=session_id, source=source_name)


def get_dataframe(session_id: str) -> pd.DataFrame | None:
    entry = _STORE.get(session_id)
    return entry["df"] if entry else None


def get_source_name(session_id: str) -> str:
    entry = _STORE.get(session_id)
    return entry["source_name"] if entry else "unknown"


def session_exists(session_id: str) -> bool:
    return session_id in _STORE


def delete_session(session_id: str) -> None:
    _STORE.pop(session_id, None)
    logger.debug("session_deleted", session_id=session_id)
