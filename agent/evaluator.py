"""Offline evaluation gate for the DataPilot AI capstone build."""
from __future__ import annotations

from agent.state import AgentState
from core.logging import get_logger

logger = get_logger(__name__)


def evaluator_node(state: AgentState) -> dict:
    """Skip LLM self-evaluation; quality is measured by deterministic tests."""
    logger.info("llm_evaluator_skipped", reason="offline_rule_based_mode")
    return {"eval_scores": None}
