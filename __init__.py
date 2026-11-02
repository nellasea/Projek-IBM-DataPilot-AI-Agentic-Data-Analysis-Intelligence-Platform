"""tools package."""
from tools.registry import dispatch, list_tools
from tools.duckdb_tool import run_duckdb_query
from tools.pandas_tool import pandas_describe
from tools.stats_tool import run_correlation, run_ttest, run_anova, run_chi2
from tools.plotly_tool import plot_chart
from tools.anomaly_tool import detect_anomalies

__all__ = [
    "dispatch",
    "list_tools",
    "run_duckdb_query",
    "pandas_describe",
    "run_correlation",
    "run_ttest",
    "run_anova",
    "run_chi2",
    "plot_chart",
    "detect_anomalies",
]
