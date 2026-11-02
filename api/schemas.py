"""
api/schemas.py
==============
Pydantic request and response models for all FastAPI endpoints.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── Ingest ────────────────────────────────────────────────────────────────────

class ColumnSchemaOut(BaseModel):
    name: str
    dtype_raw: str
    dtype_category: str
    nullable: bool


class IngestResponse(BaseModel):
    session_id: str
    source_name: str
    row_count: int
    column_count: int
    columns: list[ColumnSchemaOut]
    file_size_bytes: int
    warnings: list[str]


# ── Profile ───────────────────────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    session_id: str
    source_name: str
    row_count: int
    column_count: int
    total_cells: int
    total_nulls: int
    null_pct: float
    duplicate_row_count: int
    duplicate_row_pct: float
    memory_usage_bytes: int
    columns: list[dict[str, Any]]


# ── Quality ───────────────────────────────────────────────────────────────────

class QualityIssueOut(BaseModel):
    check: str
    severity: str
    column: str | None
    message: str
    details: dict[str, Any]


class QualityResponse(BaseModel):
    session_id: str
    source_name: str
    row_count: int
    column_count: int
    score: float
    error_count: int
    warning_count: int
    passed: bool
    issues: list[QualityIssueOut]


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=3, max_length=1000)
    evaluate: bool = True


class ToolResultOut(BaseModel):
    tool: str
    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: float | None = None


class QueryResponse(BaseModel):
    session_id: str
    question: str
    plan: list[dict[str, Any]]
    tool_results: list[ToolResultOut]
    insight: str
    eval_scores: dict[str, Any] | None
    error: str | None
    figures_count: int


# ── History ───────────────────────────────────────────────────────────────────

class SessionOut(BaseModel):
    session_id: str
    source_name: str
    row_count: int
    column_count: int
    quality_score: float | None
    created_at: str | None


class AnalysisOut(BaseModel):
    id: int
    session_id: str
    query_text: str
    insight_text: str
    latency_ms: float | None
    created_at: str | None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
