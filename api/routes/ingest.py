"""
api/routes/ingest.py
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File

from api.schemas import ColumnSchemaOut, IngestResponse
from api.session_store import new_session_id, store_dataframe
from core.exceptions import IngestionError
from data.ingestion import ingest_file
from data.storage import save_session

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("/", response_model=IngestResponse)
async def ingest_dataset(file: UploadFile = File(...)) -> IngestResponse:
    """Upload a CSV or XLSX file and start a new analysis session."""
    raw = await file.read()
    try:
        result = ingest_file(raw, source_name=file.filename or "upload")
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session_id = new_session_id()
    store_dataframe(session_id, result.dataframe, result.source_name)

    # Persist session to SQLite (best-effort)
    try:
        save_session(
            session_id=session_id,
            source_name=result.source_name,
            row_count=result.row_count,
            column_count=result.column_count,
        )
    except Exception:
        pass  # Don't fail the upload if storage is unavailable

    return IngestResponse(
        session_id=session_id,
        source_name=result.source_name,
        row_count=result.row_count,
        column_count=result.column_count,
        columns=[
            ColumnSchemaOut(
                name=c.name,
                dtype_raw=c.dtype_raw,
                dtype_category=c.dtype_category,
                nullable=c.nullable,
            )
            for c in result.columns
        ],
        file_size_bytes=result.file_size_bytes,
        warnings=result.warnings,
    )
