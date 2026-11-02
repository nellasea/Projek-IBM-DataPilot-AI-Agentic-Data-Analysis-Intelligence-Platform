"""Offline capstone guard: external LLM providers must never initialize."""
from __future__ import annotations

import pytest

from core.config import Settings
from core.exceptions import LLMProviderError


def test_external_llm_is_disabled_even_if_provider_and_key_are_configured():
    from llm.provider import get_llm

    # The provider selection and credential must not bypass the offline guard.
    settings = Settings(llm_provider="openai", openai_api_key="sk-test-not-real")
    assert settings.openai_api_key
    with pytest.raises(LLMProviderError, match="disabled"):
        get_llm()


def test_rule_based_provider_does_not_initialize_an_llm():
    from llm.provider import get_llm

    with pytest.raises(LLMProviderError, match="disabled"):
        get_llm()
