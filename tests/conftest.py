"""
tests/conftest.py
=================
Shared pytest fixtures.
"""
from __future__ import annotations

import io
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A simple 20-row DataFrame for testing."""
    return pd.DataFrame({
        "id": range(1, 21),
        "name": [f"Item {i}" for i in range(1, 21)],
        "category": (["A", "B", "C", "D"] * 5),
        "price": [10.5, 20.0, 15.75, 8.0, 30.0, 12.5, 22.0, 9.0, 18.5, 25.0,
                  11.0, 19.0, 14.5, 7.5, 28.0, 13.0, 21.0, 10.0, 17.5, 24.0],
        "quantity": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5,
                     1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "in_stock": ([True, False] * 10),
    })


@pytest.fixture
def sample_csv_bytes(sample_df) -> bytes:
    buf = io.BytesIO()
    sample_df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def sample_df_with_nulls(sample_df) -> pd.DataFrame:
    df = sample_df.copy()
    df.loc[0, "price"] = None
    df.loc[5, "name"] = None
    return df


@pytest.fixture
def large_numeric_df() -> pd.DataFrame:
    """100-row numeric DataFrame for stats and anomaly tests."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "x": rng.normal(0, 1, 100),
        "y": rng.normal(5, 2, 100),
        "z": rng.uniform(0, 10, 100),
        "group": (["A", "B"] * 50),
    })
