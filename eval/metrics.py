"""
eval/metrics.py
===============
Evaluation metrics for the DataPilot AI agent.

Metrics computed
----------------
- faithfulness   : from LLM self-eval (0–10)
- relevance      : from LLM self-eval (0–10)
- groundedness   : from LLM self-eval (0–10)
- completeness   : from LLM self-eval (0–10)
- tool_success_rate : fraction of tool steps that succeeded
- plan_length       : number of steps in the plan
- total_latency_ms  : total wall-clock time
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationMetrics:
    session_id: str
    question: str

    # LLM self-evaluation scores (None if evaluation was skipped)
    faithfulness: int | None = None
    relevance: int | None = None
    groundedness: int | None = None
    completeness: int | None = None

    # Tool execution metrics
    plan_length: int = 0
    tool_success_count: int = 0
    tool_failure_count: int = 0
    tool_success_rate: float = 0.0
    total_latency_ms: float = 0.0

    # Step-level detail
    step_latencies_ms: list[float] = field(default_factory=list)

    @property
    def avg_eval_score(self) -> float | None:
        scores = [
            s for s in [self.faithfulness, self.relevance, self.groundedness, self.completeness]
            if s is not None
        ]
        return round(sum(scores) / len(scores), 2) if scores else None


def compute_metrics(
    session_id: str,
    question: str,
    plan: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    eval_scores: dict[str, Any] | None,
) -> EvaluationMetrics:
    """Compute evaluation metrics from agent state components."""
    success = [r for r in tool_results if r.get("success")]
    failures = [r for r in tool_results if not r.get("success")]
    latencies = [r.get("latency_ms", 0) for r in tool_results if r.get("latency_ms")]

    metrics = EvaluationMetrics(
        session_id=session_id,
        question=question,
        plan_length=len(plan),
        tool_success_count=len(success),
        tool_failure_count=len(failures),
        tool_success_rate=round(len(success) / max(len(tool_results), 1), 4),
        total_latency_ms=round(sum(latencies), 1),
        step_latencies_ms=latencies,
    )

    if eval_scores:
        metrics.faithfulness = eval_scores.get("faithfulness")
        metrics.relevance = eval_scores.get("relevance")
        metrics.groundedness = eval_scores.get("groundedness")
        metrics.completeness = eval_scores.get("completeness")

    return metrics
