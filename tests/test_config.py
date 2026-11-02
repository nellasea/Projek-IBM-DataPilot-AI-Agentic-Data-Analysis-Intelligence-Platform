"""
tests/test_config.py
"""
from __future__ import annotations

import pytest
from core.config import Settings, LLMProvider, AppEnv


def test_defaults():
    # Pass _env_file pointing at a non-existent file so the real .env on disk
    # (which may override defaults) cannot interfere with this test.
    s = Settings(_env_file="/nonexistent/.env.missing")
    assert s.app_env == AppEnv.development
    assert s.llm_provider == LLMProvider.fallback
    assert s.max_upload_mb == 100
    assert s.api_port == 8000


def test_max_upload_bytes():
    s = Settings(max_upload_mb=10)
    assert s.max_upload_bytes == 10 * 1024 * 1024


def test_is_development():
    s = Settings(app_env="development")
    assert s.is_development is True

    s2 = Settings(app_env="production")
    assert s2.is_development is False


def test_invalid_provider():
    with pytest.raises(Exception):
        Settings(llm_provider="unknown_provider")


def test_ensure_upload_dir(tmp_path):
    s = Settings(upload_dir=str(tmp_path / "uploads"))
    result = s.ensure_upload_dir()
    assert result.exists()
