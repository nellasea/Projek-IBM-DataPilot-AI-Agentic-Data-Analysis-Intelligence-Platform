"""External LLM providers are disabled in the offline capstone submission.

The project intentionally runs through the deterministic rule-based agent.
This module remains as a compatibility boundary for older imports, but it
cannot initialize a remote or local model.
"""
from __future__ import annotations

from typing import Any

from core.exceptions import LLMProviderError


def get_llm() -> Any:
    """Always reject model initialization; no network or model calls allowed."""
    raise LLMProviderError(
        "External LLM providers are disabled in this offline capstone build. "
        "Use the Rule-Based Agent workflow instead.",
        details={"provider": "disabled", "mode": "rule_based"},
    )


class LLMClient:
    """Compatibility stub that fails closed without initializing a provider."""

    def __init__(self) -> None:
        self._llm = get_llm()

    def chat(self, system: str, user: str) -> str:
        raise LLMProviderError("External LLM providers are disabled.")

    def chat_json(self, system: str, user: str) -> Any:
        raise LLMProviderError("External LLM providers are disabled.")
