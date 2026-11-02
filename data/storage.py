"""
data/storage.py
===============
SQLite-backed session and analysis history persistence.
Uses SQLAlchemy (sync) so it works inside FastAPI background tasks
and plain Python scripts equally.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.config import get_settings
from core.exceptions import StorageError
from core.logging import get_logger

logger = get_logger(__name__)


# ── ORM Models ────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    source_name = Column(String(255), nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    meta_json = Column(Text, default="{}")


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    plan_json = Column(Text, default="[]")
    result_json = Column(Text, default="{}")
    insight_text = Column(Text, default="")
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ── Engine / Session factory ──────────────────────────────────────────────────

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        # Convert async URL to sync for SQLAlchemy core
        url = settings.sqlite_url.replace("sqlite+aiosqlite", "sqlite")
        _engine = create_engine(url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(_engine)
        logger.info("storage_initialised", url=url)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal()


# ── Public API ────────────────────────────────────────────────────────────────

def save_session(
    session_id: str,
    source_name: str,
    row_count: int,
    column_count: int,
    quality_score: float | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Persist an analysis session record."""
    try:
        with get_session() as db:
            record = AnalysisSession(
                session_id=session_id,
                source_name=source_name,
                row_count=row_count,
                column_count=column_count,
                quality_score=quality_score,
                meta_json=json.dumps(meta or {}),
            )
            db.add(record)
            db.commit()
            logger.debug("session_saved", session_id=session_id)
    except Exception as exc:
        raise StorageError(f"Failed to save session: {exc}", details={"error": str(exc)}) from exc


def save_analysis(
    session_id: str,
    query_text: str,
    plan: list[dict],
    result: dict[str, Any],
    insight: str,
    latency_ms: float | None = None,
) -> int:
    """Persist an analysis record and return its ID."""
    try:
        with get_session() as db:
            record = AnalysisRecord(
                session_id=session_id,
                query_text=query_text,
                plan_json=json.dumps(plan),
                result_json=json.dumps(result, default=str),
                insight_text=insight,
                latency_ms=latency_ms,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            logger.debug("analysis_saved", session_id=session_id, record_id=record.id)
            return record.id
    except Exception as exc:
        raise StorageError(
            f"Failed to save analysis: {exc}", details={"error": str(exc)}
        ) from exc


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent analysis sessions."""
    try:
        with get_session() as db:
            rows = db.execute(
                select(AnalysisSession)
                .order_by(AnalysisSession.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "session_id": r.session_id,
                    "source_name": r.source_name,
                    "row_count": r.row_count,
                    "column_count": r.column_count,
                    "quality_score": r.quality_score,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception as exc:
        raise StorageError(f"Failed to list sessions: {exc}", details={"error": str(exc)}) from exc


def list_analyses(session_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return analysis records for a session."""
    try:
        with get_session() as db:
            rows = db.execute(
                select(AnalysisRecord)
                .where(AnalysisRecord.session_id == session_id)
                .order_by(AnalysisRecord.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "query_text": r.query_text,
                    "insight_text": r.insight_text,
                    "latency_ms": r.latency_ms,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
    except Exception as exc:
        raise StorageError(
            f"Failed to list analyses: {exc}", details={"error": str(exc)}
        ) from exc
