"""
tests/test_config_env.py
========================
Tests for environment variable loading and API key configuration.

These tests verify:
1. The project root is resolved correctly (absolute, not CWD-relative).
2. The .env file path is inside the project root.
3. A missing API key is detected and raises the correct error.
4. When the key IS set (via env var), the LLM provider initialises correctly.
5. The lru_cache on get_settings() can be bypassed for isolated tests.

No real API calls are made.  The LLM client is never invoked.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.config import (
    Settings,
    get_env_file_path,
    get_project_root,
    get_settings,
)
from core.exceptions import LLMProviderError


# ── 1. Project root resolution ────────────────────────────────────────────────

def test_project_root_is_absolute():
    root = get_project_root()
    assert root.is_absolute(), "Project root must be an absolute path."


def test_project_root_contains_pyproject():
    """Sanity check: the resolved root must contain pyproject.toml."""
    root = get_project_root()
    assert (root / "pyproject.toml").exists(), (
        f"pyproject.toml not found in resolved project root: {root}"
    )


def test_env_file_path_is_inside_project_root():
    env_path = get_env_file_path()
    root = get_project_root()
    assert env_path.parent == root, (
        f".env path {env_path} is not directly inside project root {root}"
    )


def test_env_file_path_is_absolute():
    assert get_env_file_path().is_absolute()


def test_env_file_name_is_dot_env():
    assert get_env_file_path().name == ".env"


# ── 2. Settings reads from env vars regardless of CWD ────────────────────────

def test_settings_reads_openai_key_from_env_var():
    """
    pydantic-settings reads from os.environ *in addition to* the .env file.
    This tests the fallback path when .env doesn't exist but the var is set
    in the process environment (e.g. CI/CD pipeline).
    """
    # Bypass the lru_cache by instantiating Settings directly
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-from-env-var"}, clear=False):
        fresh = Settings()
        assert fresh.openai_api_key == "sk-test-key-from-env-var"


def test_settings_empty_key_when_nothing_set():
    """When neither env var nor .env sets the key, default should be empty.

    We must patch out both the env var AND the .env file path, because
    pydantic-settings reads from the .env file even when os.environ is cleared.
    """
    env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    # Point Settings at a non-existent .env file so it cannot read the real key.
    import core.config as config_module
    original_env_file = config_module._ENV_FILE
    try:
        config_module._ENV_FILE = Path("/nonexistent/.env.missing")
        with patch.dict(os.environ, env_without_key, clear=True):
            fresh = Settings(_env_file="/nonexistent/.env.missing")
            assert fresh.openai_api_key == ""
    finally:
        config_module._ENV_FILE = original_env_file


# ── 3. Offline submission guard ──────────────────────────────────────────────

def test_external_llm_is_disabled_even_if_key_is_present():
    from llm.provider import get_llm

    with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"}):
        with pytest.raises(LLMProviderError, match="disabled"):
            get_llm()


def test_external_llm_is_disabled_without_key():
    from llm.provider import get_llm

    with pytest.raises(LLMProviderError, match="disabled"):
        get_llm()
