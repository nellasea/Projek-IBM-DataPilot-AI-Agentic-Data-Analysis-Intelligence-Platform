"""
api/routes/query.py
"""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from api.schemas import QueryRequest, QueryResponse, ToolResultOut
from api.session_store import get_dataframe, get_source_name, session_exists
from agent.graph import run_analysis
from data.storage import save_analysis

router = APIRouter(prefix="/query", tags=["Analysis"])


@router.post("/", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    """Run a natural language analysis query against an ingested dataset."""
    if not session_exists(request.session_id):
        raise HTTPException(
            status_code=404, detail=f"Session '{request.session_id}' not found."
        )

    df = get_dataframe(request.session_id)
    t0 = time.perf_counter()

    final_state = run_analysis(
        question=request.question,
        df=df,
        session_id=request.session_id,
        evaluate=request.evaluate,
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Persist to history (best-effort)
    try:
        save_analysis(
            session_id=request.session_id,
            query_text=request.question,
            plan=final_state.get("plan", []),
            result={
                "tool_results": [
                    {k: v for k, v in r.items() if k != "dataframe"}
                    for r in final_state.get("tool_results", [])
                ]
            },
            insight=final_state.get("insight", ""),
            latency_ms=latency_ms,
        )
    except Exception:
        pass

    tool_results = [
        ToolResultOut(
            tool=r.get("tool", ""),
            success=r.get("success", False),
            result=r.get("result"),
            error=r.get("error"),
            latency_ms=r.get("latency_ms"),
        )
        for r in final_state.get("tool_results", [])
    ]

    return QueryResponse(
        session_id=request.session_id,
        question=request.question,
        plan=final_state.get("plan", []),
        tool_results=tool_results,
        insight=final_state.get("insight", ""),
        eval_scores=final_state.get("eval_scores"),
        error=final_state.get("error"),
        figures_count=len(final_state.get("figures", [])),
    )
