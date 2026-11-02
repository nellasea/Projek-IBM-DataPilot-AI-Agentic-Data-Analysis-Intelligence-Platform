"""
llm/fallback.py
===============
Deterministic rule-based agent for DataPilot AI.

This is the PRIMARY execution path — not a "fallback" from an LLM.
When LLM_PROVIDER=fallback (the default), this module handles all analysis.

Design principles
-----------------
- **No fabricated results.**  Numbers come from tool handlers (DuckDB, Pandas, etc.).
- **No external calls.**  Zero network, zero API, zero model download.
- **Bilingual.**  Supports Indonesian and English question patterns.
- **Semantic column matching.**  Resolves question keywords to actual DataFrame
  columns using a rich alias table — never blindly picks the first column.
- **Transparent.**  Every response clearly states it was computed deterministically.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from core.logging import get_logger

logger = get_logger(__name__)

# ── Semantic entity aliases ────────────────────────────────────────────────────
# Maps a semantic "entity name" (what the user is asking about) to a list of
# keywords that may appear EITHER in the question OR in the actual column name.
# Order within each list matters: more specific terms come first.

_ENTITY_ALIASES: dict[str, list[str]] = {
    # ── Dimension entities (group-by) ─────────────────────────────────────────
    "product": [
        "produk", "product", "nama produk", "product_name", "nama_produk",
        "item", "barang", "goods", "nama barang", "commodity",
    ],
    "operator": [
        "operator", "perusahaan", "penyedia", "provider", "company",
        "perusahaan operator", "nama operator",
    ],
    "region": [
        "wilayah", "region", "daerah", "area", "kota", "city", "district",
        "kabupaten", "provinsi", "province", "lokasi", "location", "zone", "zona",
    ],
    "category": [
        "kategori", "category", "tipe", "type", "jenis", "class", "kelas",
        "grup", "group", "segment",
    ],
    "customer": [
        "pelanggan", "customer", "client", "nasabah", "konsumen",
        "nama pelanggan", "customer_name",
    ],
    "employee": [
        "karyawan", "pegawai", "staff", "employee", "petugas", "driver",
        "supir", "pengemudi",
    ],
    "route": [
        "rute", "route", "trayek", "jalur", "path", "line",
    ],
    # ── Metric entities (aggregate) ───────────────────────────────────────────
    "sales": [
        "penjualan", "sales", "total penjualan", "total_penjualan",
        "revenue", "pendapatan", "omzet", "omset", "income",
        "nilai penjualan", "nilai_penjualan", "total_revenue", "selling", "sold",
        "units sold", "unit terjual",
    ],
    "quantity": [
        "jumlah", "quantity", "qty", "unit", "jumlah transaksi",
        "jumlah_transaksi", "count", "volume", "banyak",
    ],
    "amount": [
        "amount", "nilai", "value", "total", "nominal", "harga", "price",
        "biaya", "cost", "tarif", "fare",
    ],
    "profit": [
        "profit", "laba", "keuntungan", "gain",
    ],
    # ── Temporal entities ─────────────────────────────────────────────────────
    "date": [
        "tanggal", "date", "waktu", "time", "bulan", "month",
        "tahun", "year", "hari", "day", "periode", "period",
        "transaction_date", "tanggal_transaksi", "tgl",
    ],
}

# ── Intent patterns (bilingual) ────────────────────────────────────────────────
_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Anomaly / outlier
    (re.compile(
        r"\b(anomal|outlier|unusual|irregular|abnormal|ekstrem|extreme|menyimpang)\w*\b", re.I
    ), "anomaly"),
    # Correlation
    (re.compile(
        r"\b(correlat|relationship|association|related|korelasi|hubungan|keterkaitan)\w*\b", re.I
    ), "correlation"),
    # Missing values
    (re.compile(
        r"\b(missing|null|nan|empty|blank|kosong|hilang|tidak ada|tidak terisi)\w*\b", re.I
    ), "missing_values"),
    # Duplicates
    (re.compile(
        r"\b(duplicate|dupe|repeated|repeat|duplikat|duplikasi|ganda|kembar)\w*\b", re.I
    ), "duplicates"),
    # Distribution / histogram
    (re.compile(
        r"\b(distribut|histogram|spread|frequen|sebaran|distribusi|frekuensi)\w*\b", re.I
    ), "distribution"),
    # Top / highest / ranking (Indonesian first to avoid false English positives)
    (re.compile(
        r"\b(tertinggi|terbesar|terbanyak|teratas|terlaris|terpopuler|terbaik"
        r"|top|highest|largest|maximum|most|best|leading|rank(?:ing)?)\w*\b", re.I
    ), "top_n"),
    # Bottom / lowest
    (re.compile(
        r"\b(terendah|terkecil|tersedikit|terbawah|terburuk"
        r"|lowest|smallest|minimum|least|worst|bottom)\w*\b", re.I
    ), "bottom_n"),
    # Time series
    (re.compile(
        r"\b(tren|trend|waktu|tanggal|over time|time series"
        r"|per (bulan|tahun|hari|tanggal|week|month|year|day|date))\b", re.I
    ), "time_series"),
    # Aggregation / total / sum
    (re.compile(
        r"\b(total|jumlah|penjualan|pendapatan|omzet|revenue|sales|aggregat|sum)\w*\b", re.I
    ), "aggregation"),
    # Descriptive stats
    (re.compile(
        r"\b(rata-rata|average|mean|median|typical|statistik|statistics)\w*\b", re.I
    ), "descriptive"),
    # Count
    (re.compile(
        r"\b(berapa|count|how many|number of|banyak|hitung)\b", re.I
    ), "count"),
    # Visualisation
    (re.compile(
        r"\b(grafik|buat grafik|chart|plot|visuali|graph|gambar|tampilkan)\w*\b", re.I
    ), "visualize"),
    # Profile / overview / summary
    (re.compile(
        r"\b(ringkasan|overview|summary|describe|profile|schema|gambaran|ikhtisar)\w*\b", re.I
    ), "profile"),
]


# ── Intent detection ───────────────────────────────────────────────────────────

def detect_intent(question: str) -> list[str]:
    """Return ordered list of intent labels matched in *question*.

    Returns ["unsupported"] when no supported intent is recognized.
    """
    seen: set[str] = set()
    intents: list[str] = []
    for pattern, label in _INTENT_PATTERNS:
        if pattern.search(question) and label not in seen:
            intents.append(label)
            seen.add(label)
    return intents or ["unsupported"]


# ── Schema helpers ─────────────────────────────────────────────────────────────

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    import numpy as np
    return list(df.select_dtypes(include=[np.number]).columns)


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["object", "category", "string"]).columns)


def _datetime_cols(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["datetime64", "datetimetz"]).columns)


def _col_lower_map(df: pd.DataFrame) -> dict[str, str]:
    """Return {lowercase_col_name: original_col_name} for all columns."""
    return {c.lower().strip(): c for c in df.columns}


def _match_entity_to_column(
    entity: str,
    df: pd.DataFrame,
    question: str,
    numeric_only: bool = False,
) -> str | None:
    """Find the best DataFrame column for a semantic entity.

    Strategy (in priority order):
    1. Exact case-insensitive column name match in aliases.
    2. Alias keyword appears inside a column name (e.g. "produk_name" → product).
    3. Alias keyword appears in the question AND is a substring of a column name.
    4. None — explicit signal that the entity does not exist.

    Never falls back to "first column of the right type".
    """
    aliases = _ENTITY_ALIASES.get(entity, [entity])
    col_map = _col_lower_map(df)
    q_lower = question.lower()

    candidates = list(df.columns)
    if numeric_only:
        numeric = set(_numeric_cols(df))
        candidates = [c for c in candidates if c in numeric]
    else:
        cats = set(_categorical_cols(df))
        candidates = [c for c in candidates if c in cats]

    # Pass 1: exact alias == column name (case-insensitive)
    for alias in aliases:
        if alias in col_map and col_map[alias] in candidates:
            return col_map[alias]

    # Pass 2: alias is a substring of a column name
    for alias in aliases:
        for col in candidates:
            if alias in col.lower():
                return col

    # Pass 3: alias appears in the question AND is a substring of a column name
    for alias in aliases:
        if alias in q_lower:
            for col in candidates:
                if any(a in col.lower() for a in aliases):
                    return col

    return None   # entity not found in this dataset


def _resolve_group_entity(question: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return (group_column, entity_name) by matching the question to dimension entities.

    Only checks dimension entities (product, operator, region, category, …).
    Returns (None, None) when no matching column is found.
    """
    q_lower = question.lower()
    dimension_entities = [
        "product", "operator", "region", "category", "customer",
        "employee", "route",
    ]
    for entity in dimension_entities:
        aliases = _ENTITY_ALIASES[entity]
        if any(alias in q_lower for alias in aliases):
            col = _match_entity_to_column(entity, df, question, numeric_only=False)
            if col:
                return col, entity
    # No specific entity mentioned → None (caller must decide what to do)
    return None, None


def _resolve_metric_entity(question: str, df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return (metric_column, entity_name) by matching the question to metric entities."""
    q_lower = question.lower()
    metric_entities = ["sales", "quantity", "amount", "profit"]
    for entity in metric_entities:
        aliases = _ENTITY_ALIASES[entity]
        if any(alias in q_lower for alias in aliases):
            col = _match_entity_to_column(entity, df, question, numeric_only=True)
            if col:
                return col, entity

    # Do not silently substitute an unrelated numeric column (e.g. year)
    # when the user explicitly requested sales/revenue/quantity/profit.
    return None, None


def _guess_date_col(df: pd.DataFrame) -> str | None:
    """Return the most likely date column."""
    dt_cols = _datetime_cols(df)
    if dt_cols:
        return dt_cols[0]
    aliases = _ENTITY_ALIASES["date"]
    for col in df.columns:
        if any(kw in col.lower() for kw in aliases):
            return col
    return None


# ── Plan builders ──────────────────────────────────────────────────────────────

def _plan_profile(df: pd.DataFrame) -> list[dict]:
    numeric = _numeric_cols(df)
    plan: list[dict] = [{"tool": "pandas_describe", "params": {}}]
    if numeric:
        plan.append({
            "tool": "plot_chart",
            "params": {
                "chart_type": "histogram",
                "x": numeric[0],
                "title": f"Distribution of {numeric[0]}",
            },
        })
    return plan


def _plan_missing_values(df: pd.DataFrame) -> list[dict]:
    null_counts = df.isnull().sum()
    cols_with_nulls = [str(c) for c in null_counts[null_counts > 0].index.tolist()]
    if cols_with_nulls:
        parts = [f"COUNT(*) - COUNT(\"{c}\") AS missing_{re.sub(r'[^a-zA-Z0-9_]', '_', c)}"
                 for c in cols_with_nulls[:8]]
        sql = "SELECT " + ", ".join(parts) + " FROM df"
    else:
        sql = "SELECT COUNT(*) AS total_rows FROM df"
    return [{"tool": "duckdb_query", "params": {"sql": sql}}]


def _plan_duplicates(df: pd.DataFrame) -> list[dict]:
    return [{
        "tool": "duckdb_query",
        "params": {"sql": (
            "SELECT COUNT(*) AS total_rows, "
            "COUNT(*) - COUNT(DISTINCT *) AS duplicate_rows FROM df"
        )},
    }]


def _plan_top_n(
    df: pd.DataFrame,
    question: str,
    ascending: bool = False,
    limit: int = 10,
) -> list[dict]:
    group_col, group_entity = _resolve_group_entity(question, df)
    metric_col, metric_entity = _resolve_metric_entity(question, df)
    order = "ASC" if ascending else "DESC"
    label = "bottom" if ascending else "top"

    # Never replace a requested semantic dimension with an unrelated column.
    q_lower = question.lower()
    requested_dimension = next((entity for entity in
        ("product", "operator", "region", "category", "customer", "employee", "route")
        if any(alias in q_lower for alias in _ENTITY_ALIASES[entity])), None)
    if requested_dimension and group_col is None:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"Kolom untuk entitas '{requested_dimension}' tidak ditemukan dalam dataset. "
            f"Kolom yang tersedia: {available}. Sistem tidak akan menggantinya dengan kolom lain."
        )
    if group_col is None:
        raise ValueError(
            "Dimensi pengelompokan belum jelas. Sebutkan kolom seperti produk, operator, atau wilayah. "
            f"Kolom tersedia: {', '.join(map(str, df.columns))}."
        )
    transaction_count = bool(re.search(
        r"\b(jumlah transaksi|banyak transaksi|transaksi|transaction count|transactions|number of transactions)\b",
        q_lower,
    ))
    if metric_col is None and not transaction_count:
        raise ValueError(
            "Kolom metrik numerik yang diminta tidak ditemukan. Sebutkan metrik yang tersedia. "
            f"Kolom numerik tersedia: {', '.join(map(str, _numeric_cols(df)))}."
        )

    # Singular superlative questions default to one result; explicit top N wins.
    limit_match = re.search(r"\b(?:top|teratas|lima|5|sepuluh|10)\s*(\d+)\b", q_lower)
    if limit_match:
        limit = max(1, min(50, int(limit_match.group(1))))
    elif re.search(r"\b(top|teratas|tertinggi|terbesar|terbanyak|terendah|terkecil|terlaris)\b", q_lower):
        limit = 1

    if group_col:
        # Quote identifiers to handle spaces and special characters.
        gc = '"' + str(group_col).replace('"', '""') + '"'
        if transaction_count:
            alias = "transaction_count"
            aggregate = "COUNT(*)"
        else:
            mc = '"' + str(metric_col).replace('"', '""') + '"'
            alias = re.sub(r'[^a-zA-Z0-9_]', '_', f"total_{metric_col}")
            aggregate = f"SUM({mc})"
        sql = (
            f"SELECT {gc}, {aggregate} AS {alias} "
            f"FROM df GROUP BY {gc} ORDER BY {alias} {order} LIMIT {limit}"
        )
        plan: list[dict] = [
            {"tool": "duckdb_query", "params": {"sql": sql}},
            {
                "tool": "plot_chart",
                "params": {
                    "chart_type": "bar",
                    "x": group_col,
                    "y": alias,
                    "title": f"{label.title()} {limit} {group_col} by {('transaction count' if transaction_count else metric_col)}",
                },
            },
        ]
    else:
        plan = [{"tool": "pandas_describe", "params": {}}]

    return plan


def _plan_aggregation(df: pd.DataFrame, question: str) -> list[dict]:
    group_col, _ = _resolve_group_entity(question, df)
    metric_col, _ = _resolve_metric_entity(question, df)

    if group_col is None:
        requested_dimension = next((entity for entity in
            ("product", "operator", "region", "category", "customer", "employee", "route")
            if any(alias in question.lower() for alias in _ENTITY_ALIASES[entity])), None)
        if requested_dimension:
            raise ValueError(
                f"Kolom untuk entitas '{requested_dimension}' tidak ditemukan. "
                f"Kolom tersedia: {', '.join(map(str, df.columns))}."
            )
    if metric_col is None:
        requested_metric = next((entity for entity in ("sales", "quantity", "amount", "profit")
            if any(alias in question.lower() for alias in _ENTITY_ALIASES[entity])), None)
        if requested_metric:
            raise ValueError(
                f"Kolom metrik '{requested_metric}' tidak ditemukan. "
                f"Kolom numerik tersedia: {', '.join(map(str, _numeric_cols(df)))}."
            )

    if group_col and metric_col:
        gc = f'"{group_col}"'
        mc = f'"{metric_col}"'
        sql = (
            f"SELECT {gc}, SUM({mc}) AS total, AVG({mc}) AS avg, COUNT(*) AS count "
            f"FROM df GROUP BY {gc} ORDER BY total DESC"
        )
        return [
            {"tool": "duckdb_query", "params": {"sql": sql}},
            {
                "tool": "plot_chart",
                "params": {
                    "chart_type": "bar",
                    "x": group_col,
                    "y": "total",
                    "title": f"Total {metric_col} by {group_col}",
                },
            },
        ]
    return [{"tool": "pandas_describe", "params": {}}]


def _plan_time_series(df: pd.DataFrame, question: str) -> list[dict]:
    date_col = _guess_date_col(df)
    metric_col, _ = _resolve_metric_entity(question, df)

    if date_col and metric_col:
        dc = f'"{date_col}"'
        mc = f'"{metric_col}"'
        sql = f"SELECT {dc}, SUM({mc}) AS total FROM df GROUP BY {dc} ORDER BY {dc}"
        return [
            {"tool": "duckdb_query", "params": {"sql": sql}},
            {
                "tool": "plot_chart",
                "params": {
                    "chart_type": "line",
                    "x": date_col,
                    "y": "total",
                    "title": f"{metric_col} over time",
                },
            },
        ]
    return [{"tool": "pandas_describe", "params": {}}]


def _plan_descriptive(df: pd.DataFrame) -> list[dict]:
    numeric = _numeric_cols(df)
    plan: list[dict] = [{"tool": "pandas_describe", "params": {}}]
    if numeric:
        plan.append({
            "tool": "plot_chart",
            "params": {
                "chart_type": "box",
                "y": numeric[:4],
                "title": "Distribution of numeric columns",
            },
        })
    return plan


def _plan_distribution(df: pd.DataFrame, question: str) -> list[dict]:
    metric_col, _ = _resolve_metric_entity(question, df)
    if metric_col:
        return [{
            "tool": "plot_chart",
            "params": {
                "chart_type": "histogram",
                "x": metric_col,
                "title": f"Distribution of {metric_col}",
            },
        }]
    return _plan_profile(df)


def _plan_correlation(df: pd.DataFrame) -> list[dict]:
    numeric = _numeric_cols(df)
    plan: list[dict] = [{"tool": "stats_test", "params": {"test": "correlation"}}]
    if len(numeric) >= 2:
        plan.append({
            "tool": "plot_chart",
            "params": {"chart_type": "heatmap", "title": "Correlation matrix"},
        })
    return plan


def _plan_anomaly(df: pd.DataFrame) -> list[dict]:
    return [{"tool": "detect_anomalies", "params": {"contamination": 0.05}}]


def _plan_count(df: pd.DataFrame, question: str) -> list[dict]:
    group_col, _ = _resolve_group_entity(question, df)
    if group_col:
        gc = f'"{group_col}"'
        sql = f"SELECT {gc}, COUNT(*) AS count FROM df GROUP BY {gc} ORDER BY count DESC"
        return [{"tool": "duckdb_query", "params": {"sql": sql}}]
    return [{"tool": "duckdb_query", "params": {"sql": "SELECT COUNT(*) AS total_rows FROM df"}}]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_fallback_plan(question: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a deterministic tool-call plan for *question* against *df*.

    The plan uses the same format expected by the executor:
    ``[{"tool": "<name>", "params": {...}}, ...]``

    No LLM is called. Results come from the real tool handlers.
    Column selection is based on semantic alias matching — never blindly picks
    the first available column.
    """
    intents = detect_intent(question)
    logger.info("rule_based_planner", intents=intents, question=question[:100])

    primary = intents[0]

    if primary == "unsupported":
        raise ValueError(
            "Pertanyaan belum didukung atau belum cukup spesifik. "
            "Coba tanyakan ringkasan dataset, statistik, agregasi per kolom, "
            "missing values, duplikasi, anomali, korelasi, atau visualisasi."
        )
    elif primary == "anomaly":
        plan = _plan_anomaly(df)
    elif primary == "correlation":
        plan = _plan_correlation(df)
    elif primary == "missing_values":
        plan = _plan_missing_values(df)
    elif primary == "duplicates":
        plan = _plan_duplicates(df)
    elif primary == "distribution":
        plan = _plan_distribution(df, question)
    elif primary == "top_n":
        plan = _plan_top_n(df, question, ascending=False)
    elif primary == "bottom_n":
        plan = _plan_top_n(df, question, ascending=True)
    elif primary == "time_series":
        plan = _plan_time_series(df, question)
    elif primary == "aggregation":
        plan = _plan_aggregation(df, question)
    elif primary == "descriptive":
        plan = _plan_descriptive(df)
    elif primary == "count":
        plan = _plan_count(df, question)
    elif primary == "visualize":
        plan = _plan_aggregation(df, question) or _plan_profile(df)
    else:
        plan = _plan_profile(df)

    # If the user also asked for visualisation and we don't already end with a chart
    if "visualize" in intents and primary != "visualize":
        if not plan or plan[-1].get("tool") != "plot_chart":
            extra = _plan_aggregation(df, question)
            for step in extra:
                if step["tool"] == "plot_chart" and step not in plan:
                    plan.append(step)
                    break

    return plan[:5]


# ── Insight synthesis ──────────────────────────────────────────────────────────

def synthesize_insight(
    question: str,
    tool_results: list[dict[str, Any]],
    df: pd.DataFrame,
) -> str:
    """Generate a text insight from tool results without calling any LLM.

    Every number cited is read directly from tool output.
    """
    if not tool_results:
        return (
            "Tidak ada hasil analisis yang dihasilkan. "
            "Dataset mungkin kosong atau pertanyaan tidak dapat dipetakan ke alat yang tersedia.\n\n"
            "*Analisis dilakukan oleh **Rule-Based Agent** (tanpa LLM).*"
        )

    lines: list[str] = [
        f"**Analisis:** *{question}*\n",
        f"Dataset: {len(df):,} baris × {len(df.columns)} kolom.\n",
    ]

    for r in tool_results:
        tool = r.get("tool", "")
        result = r.get("result")
        if not r.get("success") or result is None or result == "chart_generated":
            continue

        if tool == "pandas_describe" and isinstance(result, dict):
            lines.append("**Statistik deskriptif:**")
            for col, stats in list(result.items())[:6]:
                parts = []
                for stat_name in ("mean", "std", "min", "max"):
                    v = stats.get(stat_name)
                    if v is not None:
                        try:
                            parts.append(f"{stat_name}={float(v):.2f}")
                        except (TypeError, ValueError):
                            pass
                if parts:
                    lines.append(f"- **{col}**: {', '.join(parts)}")

        elif tool == "duckdb_query" and isinstance(result, dict):
            rows = result.get("rows", [])
            cols = result.get("columns", [])
            row_count = result.get("row_count", len(rows))
            lines.append(f"**Hasil query: {row_count} baris.**")
            if rows and cols:
                for row in rows[:10]:
                    pairs = []
                    for c, v in zip(cols, row):
                        if isinstance(v, float):
                            pairs.append(f"{c}={v:,.2f}")
                        else:
                            pairs.append(f"{c}={v}")
                    lines.append(f"- {', '.join(pairs)}")
                if row_count > 10:
                    lines.append(f"  *(dan {row_count - 10} baris lainnya)*")

        elif tool == "detect_anomalies" and isinstance(result, dict):
            if "error" in result:
                lines.append(f"Deteksi anomali: {result['error']}")
            else:
                count = result.get("anomaly_count", 0)
                pct = result.get("anomaly_pct", 0.0)
                cols_used = result.get("columns_used", [])
                lines.append(
                    f"**Deteksi anomali** menemukan **{count} baris anomali** "
                    f"({pct}% dari data) pada kolom: {', '.join(cols_used)}."
                )

        elif tool == "stats_test" and isinstance(result, dict):
            if "matrix" in result:
                method = result.get("method", "pearson")
                col_names = result.get("columns", [])
                lines.append(f"**Korelasi ({method})** pada {len(col_names)} kolom numerik:")
                matrix = result.get("matrix", [])
                for i, row in enumerate(matrix):
                    for j, val in enumerate(row):
                        if j > i and val is not None:
                            lines.append(f"- {col_names[i]} ↔ {col_names[j]}: r={val:.3f}")
            elif "t_stat" in result:
                groups = result.get("groups", [])
                means = result.get("means", [])
                sig = result.get("significant", False)
                lines.append(
                    f"**T-test**: grup {groups} — rata-rata {means} — "
                    f"{'signifikan (p<0.05)' if sig else 'tidak signifikan'}."
                )
            elif "f_stat" in result:
                sig = result.get("significant", False)
                lines.append(
                    f"**ANOVA**: {'perbedaan signifikan' if sig else 'tidak ada perbedaan signifikan'} "
                    f"di antara {result.get('group_count', '?')} grup."
                )

    lines.append(
        "\n---\n*Analisis dilakukan oleh **Rule-Based Agent** "
        "(tanpa model AI / LLM eksternal). Semua angka dihitung langsung dari dataset Anda.*"
    )
    return "\n".join(lines)
