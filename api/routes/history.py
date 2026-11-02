"""
api/routes/history.py
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import AnalysisOut, SessionOut
from data.storage import list_analyses, list_sessions

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/sessions", response_model=list[SessionOut])
async def get_sessions(limit: int = 50) -> list[SessionOut]:
    """Return the most recent analysis sessions."""
    rows = list_sessions(limit=limit)
    return [SessionOut(**r) for r in rows]


@router.get("/sessions/{session_id}/analyses", response_model=list[AnalysisOut])
async def get_analyses(session_id: str, limit: int = 100) -> list[AnalysisOut]:
    """Return analysis records for a specific session."""
    rows = list_analyses(session_id=session_id, limit=limit)
    if not rows:
        raise HTTPException(status_code=404, detail="No analyses found for this session.")
    return [AnalysisOut(**r) for r in rows]
