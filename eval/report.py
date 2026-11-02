"""
eval/report.py
==============
Evaluation report generator.

Aggregates metrics across multiple analyses and produces a summary report.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from eval.metrics import EvaluationMetrics


def generate_report(metrics_list: list[EvaluationMetrics]) -> dict[str, Any]:
    """Aggregate a list of EvaluationMetrics into a summary report dict."""
    if not metrics_list:
        return {"error": "No metrics to report."}

    n = len(metrics_list)

    def avg(values):
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    faithfulness_scores = [m.faithfulness for m in metrics_list]
    relevance_scores = [m.relevance for m in metrics_list]
    groundedness_scores = [m.groundedness for m in metrics_list]
    completeness_scores = [m.completeness for m in metrics_list]
    success_rates = [m.tool_success_rate for m in metrics_list]
    latencies = [m.total_latency_ms for m in metrics_list]

    return {
        "total_analyses": n,
        "avg_faithfulness": avg(faithfulness_scores),
        "avg_relevance": avg(relevance_scores),
        "avg_groundedness": avg(groundedness_scores),
        "avg_completeness": avg(completeness_scores),
        "avg_tool_success_rate": avg(success_rates),
        "avg_latency_ms": avg(latencies),
        "total_tool_failures": sum(m.tool_failure_count for m in metrics_list),
        "analyses": [asdict(m) for m in metrics_list],
    }
