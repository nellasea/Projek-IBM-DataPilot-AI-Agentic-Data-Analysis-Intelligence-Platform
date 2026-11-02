"""
tools/plotly_tool.py
====================
Plotly chart generation.

All chart functions return a ``plotly.graph_objects.Figure`` instance.
Figures are JSON-serialisable via ``fig.to_json()`` for API transport
and renderable directly in Streamlit via ``st.plotly_chart(fig)``.
"""
from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.exceptions import ToolError
from core.logging import get_logger

logger = get_logger(__name__)

ChartType = Literal["bar", "line", "scatter", "histogram", "box", "heatmap", "pie"]


def plot_chart(
    df: pd.DataFrame,
    chart_type: ChartType,
    x: str | None = None,
    y: str | list[str] | None = None,
    color: str | None = None,
    title: str = "Chart",
    **kwargs: Any,
) -> go.Figure:
    """Create a Plotly chart from a DataFrame.

    Args:
        df: Source data.
        chart_type: One of bar | line | scatter | histogram | box | heatmap | pie.
        x: Column name for the x-axis.
        y: Column name (or list) for the y-axis.
        color: Column name for colour grouping.
        title: Chart title.
        **kwargs: Passed through to the Plotly Express function.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Raises:
        ``ToolError``: if columns are missing or chart type is unknown.
    """
    try:
        _validate_columns(df, [c for c in [x, y, color] if isinstance(c, str)])

        logger.debug("plot_chart", chart_type=chart_type, x=x, y=y, title=title)

        common = dict(title=title, template="plotly_white")

        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color=color, **common, **kwargs)

        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color, **common, **kwargs)

        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, **common, **kwargs)

        elif chart_type == "histogram":
            fig = px.histogram(df, x=x, color=color, **common, **kwargs)

        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color=color, **common, **kwargs)

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            corr = numeric_df.corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title=title,
            )

        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, **common, **kwargs)

        else:
            raise ToolError(
                f"Unknown chart type '{chart_type}'. "
                f"Supported: bar, line, scatter, histogram, box, heatmap, pie"
            )

        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        return fig

    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            f"plot_chart failed: {exc}",
            details={"chart_type": chart_type, "error": str(exc)},
        ) from exc


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise ToolError(
            f"Columns not found in DataFrame: {missing}",
            details={"missing": missing, "available": list(df.columns)},
        )
