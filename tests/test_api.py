"""
tests/test_api.py
=================
Integration tests for the FastAPI endpoints.
Uses TestClient — no real LLM calls are made.
"""
from __future__ import annotations

import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@pytest.fixture
def uploaded_session(sample_csv_bytes):
    """Upload a CSV and return the session_id."""
    response = client.post(
        "/ingest/",
        files={"file": ("test.csv", io.BytesIO(sample_csv_bytes), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# ── Ingest ────────────────────────────────────────────────────────────────────

def test_ingest_csv(sample_csv_bytes):
    response = client.post(
        "/ingest/",
        files={"file": ("test.csv", io.BytesIO(sample_csv_bytes), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["row_count"] == 20
    assert data["column_count"] == 6
    assert "session_id" in data
    assert len(data["columns"]) == 6


def test_ingest_invalid_file():
    response = client.post(
        "/ingest/",
        files={"file": ("bad.csv", io.BytesIO(b""), "text/csv")},
    )
    assert response.status_code == 422


# ── Profile ───────────────────────────────────────────────────────────────────

def test_get_profile(uploaded_session):
    response = client.get(f"/profile/{uploaded_session}")
    assert response.status_code == 200
    data = response.json()
    assert data["row_count"] == 20
    assert len(data["columns"]) == 6


def test_get_quality(uploaded_session):
    response = client.get(f"/profile/{uploaded_session}/quality")
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "issues" in data
    assert data["score"] <= 100


def test_profile_not_found():
    response = client.get("/profile/nonexistent-session-id")
    assert response.status_code == 404


# ── History ───────────────────────────────────────────────────────────────────

def test_list_sessions():
    response = client.get("/history/sessions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
