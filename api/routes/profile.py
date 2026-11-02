"""
api/routes/profile.py
"""
from __future__ import annotations

import dataclasses

from fastapi import APIRouter, HTTPException

from api.schemas import ProfileResponse, QualityResponse, QualityIssueOut
from api.session_store import get_dataframe, get_source_name, session_exists
from core.exceptions import ProfilingError, QualityError
from data.profiling import profile_dataframe
from data.quality import inspect_quality

router = APIRouter(prefix="/profile", tags=["Profiling & Quality"])


def _require_session(session_id: str):
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return get_dataframe(session_id)


@router.get("/{session_id}", response_model=ProfileResponse)
async def get_profile(session_id: str) -> ProfileResponse:
    """Return a full automated profile of the uploaded dataset."""
    df = _require_session(session_id)
    source_name = get_source_name(session_id)
    try:
        report = profile_dataframe(df, source_name=source_name)
    except ProfilingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Convert dataclass columns to plain dicts for JSON serialisation
    cols_dicts = [dataclasses.asdict(c) for c in report.columns]

    return ProfileResponse(
        session_id=session_id,
        source_name=report.source_name,
        row_count=report.row_count,
        column_count=report.column_count,
        total_cells=report.total_cells,
        total_nulls=report.total_nulls,
        null_pct=report.null_pct,
        duplicate_row_count=report.duplicate_row_count,
        duplicate_row_pct=report.duplicate_row_pct,
        memory_usage_bytes=report.memory_usage_bytes,
        columns=cols_dicts,
    )


@router.get("/{session_id}/quality", response_model=QualityResponse)
async def get_quality(session_id: str) -> QualityResponse:
    """Return a data quality inspection report for the uploaded dataset."""
    df = _require_session(session_id)
    source_name = get_source_name(session_id)
    try:
        report = inspect_quality(df, source_name=source_name)
    except QualityError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QualityResponse(
        session_id=session_id,
        source_name=report.source_name,
        row_count=report.row_count,
        column_count=report.column_count,
        score=report.score,
        error_count=report.error_count,
        warning_count=report.warning_count,
        passed=report.passed,
        issues=[
            QualityIssueOut(
                check=i.check,
                severity=i.severity.value,
                column=i.column,
                message=i.message,
                details=i.details,
            )
            for i in report.issues
        ],
    )
