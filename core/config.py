"""
core/config.py
==============
Centralised application settings loaded from environment variables / .env file.
Uses pydantic-settings so every value is type-validated at startup.

env_file resolution
-------------------
The path to .env is resolved as an *absolute* path anchored to this file's
own directory (core/ → project root).  This guarantees the correct .env is
found regardless of the working directory from which the application is
started (e.g. ``streamlit run ui/app.py`` from any directory).
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root as the parent of the directory that contains this file.
# core/config.py  → core/  → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class AppEnv(str, Enum):
    development = "development"
    production = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMProvider(str, Enum):
    openai = "openai"
    anthropic = "anthropic"
    ollama = "ollama"
    gemini = "gemini"
    fallback = "fallback"   # deterministic rule-based mode — no API key required


class Settings(BaseSettings):
    """All application settings.  Read from .env or environment variables."""

    model_config = SettingsConfigDict(
        # Absolute path — works regardless of the CWD at runtime.
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.development
    log_level: LogLevel = LogLevel.INFO
    secret_key: str = "change-me-in-production"

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Default is "fallback" so the application works without any API credentials.
    llm_provider: LLMProvider = LLMProvider.fallback

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # ── Storage ───────────────────────────────────────────────────────────────
    sqlite_url: str = "sqlite+aiosqlite:///./datapilot.db"
    upload_dir: Path = Path("./uploads")
    max_upload_mb: int = Field(default=100, gt=0, le=2048)

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, gt=0, lt=65536)

    # ── Streamlit ─────────────────────────────────────────────────────────────
    streamlit_server_port: int = Field(default=8501, gt=0, lt=65536)

    # ── Computed helpers ──────────────────────────────────────────────────────
    @field_validator("upload_dir", mode="before")
    @classmethod
    def coerce_upload_dir(cls, v: str | Path) -> Path:
        return Path(v)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.app_env == AppEnv.development

    def ensure_upload_dir(self) -> Path:
        """Create upload directory if it does not exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        return self.upload_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def get_env_file_path() -> Path:
    """Return the absolute path to the expected .env file."""
    return _ENV_FILE
