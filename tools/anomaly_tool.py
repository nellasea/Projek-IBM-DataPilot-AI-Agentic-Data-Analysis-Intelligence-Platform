"""
tools/anomaly_tool.py
=====================
ML-based anomaly detection using Scikit-learn's IsolationForest.

Only numeric columns are used. The function returns a DataFrame-level
anomaly flag and a per-row anomaly score.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from core.exceptions import ToolError
from core.logging import get_logger

logger = get_logger(__name__)


def detect_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    contamination: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """Detect anomalous rows using IsolationForest.

    Args:
        df: Source DataFrame.
        columns: Numeric columns to use. Defaults to all numeric columns.
        contamination: Expected proportion of anomalies (0.0–0.5).
        random_state: For reproducibility.

    Returns:
        ``{
            "anomaly_count": int,
            "anomaly_pct": float,
            "columns_used": [...],
            "anomaly_indices": [...],       # row index positions
            "sample_anomalies": [[...], ...],
            "sample_columns": [...]
        }``

    Raises:
        ``ToolError``: if not enough numeric data is available.
    """
    try:
        # Select columns
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ToolError(f"Columns not found: {missing}")
            numeric_df = df[columns].select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.shape[1] == 0:
            return {"error": "No numeric columns available for anomaly detection."}

        if len(numeric_df) < 10:
            return {"error": "Need at least 10 rows for anomaly detection."}

        # Drop rows where ALL selected columns are NaN, fill remaining NaN with median
        numeric_df = numeric_df.dropna(how="all")
        numeric_df = numeric_df.fillna(numeric_df.median())

        # Validate contamination
        contamination = max(0.01, min(contamination, 0.5))

        # Scale and fit
        scaler = StandardScaler()
        X = scaler.fit_transform(numeric_df.values)

        model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
        )
        preds = model.fit_predict(X)       # 1 = normal, -1 = anomaly
        scores = model.decision_function(X)  # higher = more normal

        anomaly_mask = preds == -1
        anomaly_indices = numeric_df.index[anomaly_mask].tolist()
        anomaly_count = int(anomaly_mask.sum())
        anomaly_pct = round(anomaly_count / len(numeric_df) * 100, 2)

        # Sample of anomalous rows for display
        sample = df.loc[anomaly_indices[:20]]

        logger.info(
            "anomaly_detection_complete",
            anomaly_count=anomaly_count,
            anomaly_pct=anomaly_pct,
            columns=list(numeric_df.columns),
        )

        return {
            "anomaly_count": anomaly_count,
            "anomaly_pct": anomaly_pct,
            "columns_used": list(numeric_df.columns),
            "anomaly_indices": [int(i) for i in anomaly_indices[:100]],
            "sample_anomalies": sample.head(20).values.tolist(),
            "sample_columns": list(sample.columns),
        }

    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            f"anomaly detection failed: {exc}",
            details={"error": str(exc)},
        ) from exc
