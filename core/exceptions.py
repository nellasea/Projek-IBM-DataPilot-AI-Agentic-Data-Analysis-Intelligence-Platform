"""
core/exceptions.py
==================
Custom exception hierarchy for DataPilot AI.

Hierarchy
---------
DataPilotError (base)
├── IngestionError       – file loading / parsing failures
├── ProfilingError       – profiling computation failures
├── QualityError         – quality check failures
├── ToolError            – tool execution failures
│   ├── DuckDBToolError
│   └── PandasToolError
├── AgentError           – LangGraph agent failures
│   ├── PlanningError
│   └── ExecutionError
├── LLMError             – LLM provider failures
│   ├── LLMProviderError
│   └── LLMResponseError
├── StorageError         – database / file system failures
└── ValidationError      – input / output schema validation failures
"""
from __future__ import annotations


class DataPilotError(Exception):
    """Base exception for all DataPilot AI errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details})"


# ── Data layer ────────────────────────────────────────────────────────────────

class IngestionError(DataPilotError):
    """Raised when a file cannot be loaded or parsed."""


class ProfilingError(DataPilotError):
    """Raised when automated profiling fails."""


class QualityError(DataPilotError):
    """Raised when a quality check cannot be completed."""


# ── Tools ─────────────────────────────────────────────────────────────────────

class ToolError(DataPilotError):
    """Raised when a registered tool fails during execution."""


class DuckDBToolError(ToolError):
    """Raised when a DuckDB query fails."""


class PandasToolError(ToolError):
    """Raised when a Pandas operation fails."""


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentError(DataPilotError):
    """Raised when the LangGraph agent encounters an unrecoverable error."""


class PlanningError(AgentError):
    """Raised when the planner node cannot produce a valid plan."""


class ExecutionError(AgentError):
    """Raised when the executor node fails to run a tool step."""


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLMError(DataPilotError):
    """Base LLM exception."""


class LLMProviderError(LLMError):
    """Raised when the LLM provider returns an API-level error."""


class LLMResponseError(LLMError):
    """Raised when the LLM response cannot be parsed into the expected schema."""


# ── Infrastructure ────────────────────────────────────────────────────────────

class StorageError(DataPilotError):
    """Raised when a database or file-system operation fails."""


class ValidationError(DataPilotError):
    """Raised when user input or model output fails schema validation."""
