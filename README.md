# DataPilot AI 🧭

> Agentic Data Analysis & Intelligence Platform

Upload a CSV or XLSX file and ask supported questions about it in Indonesian or English. This capstone build uses a LangGraph workflow with deterministic, rule-based intent detection and registered analytical tools. It does not use an LLM or external AI API.

---

## Implementation Mode

DataPilot AI operates in two modes:

| Mode | Description | Requires |
|------|-------------|----------|
| **Rule-Based Agent** *(only mode in this submission build)* | Deterministic intent detection, planning, registered-tool selection, execution, and grounded explanation | Nothing — no API key or external service |

**This submission build is intentionally offline. Legacy provider modules may remain in the repository for reference, but the active agent flow never initializes or invokes them.**

---

## Features

| # | Feature |
|---|---------|
| 1 | CSV & XLSX ingestion with validation |
| 2 | Automated data profiling |
| 3 | Data quality inspection & scoring |
| 4 | Natural language querying (LangGraph agent) |
| 5 | Rule-based intent detection (12 intent categories) |
| 6 | Safe tool execution (DuckDB, Pandas, Stats) |
| 7 | Statistical analysis (correlation, t-test, ANOVA, chi²) |
| 8 | Interactive Plotly visualisations |
| 9 | Grounded explanations from actual tool results |
| 10 | ML anomaly detection (IsolationForest, where sufficient numeric data is available) |
| 11 | Analysis history (session-based) |
| 12 | Offline rule-based agent workflow (no external LLM integration in this build) |

---

## Architecture

```
Streamlit UI  ←→  LangGraph Agent
                       ↓
               Rule-Based Planner         ← active, offline-only
                       ↓
                 Tool Registry
                (DuckDB, Pandas, Stats, Plotly, Anomaly)
                       ↓
               Grounded Explanation
               (deterministic, template-based)
```

### Rule-Based Agent Execution Flow

```
USER QUESTION
    ↓
QUESTION NORMALISATION
    ↓
INTENT DETECTION  (keyword/regex matching)
    ↓
PARAMETER & COLUMN VALIDATION
    ↓
TOOL SELECTION  (from registry)
    ↓
TOOL EXECUTION  (Pandas · DuckDB · SciPy · Scikit-learn)
    ↓
RESULT VALIDATION
    ↓
GROUNDED EXPLANATION  (numbers come from actual tool output)
    ↓
DISPLAY RESULT
```

### Supported Intents

| Intent | Example question |
|--------|-----------------|
| Dataset profile / overview | "Give me an overview of this dataset" |
| Descriptive statistics | "What is the average, mean, and median?" |
| Group-by aggregation | "Total revenue by product category" |
| Top / highest values | "Which product has the highest sales?" |
| Bottom / lowest values | "What are the lowest-selling items?" |
| Missing-value analysis | "How many missing values are there?" |
| Duplicate detection | "Are there any duplicate rows?" |
| Anomaly detection | "Detect outliers in the data" |
| Correlation analysis | "Is there a correlation between price and quantity?" |
| Distribution analysis | "Show the distribution of prices" |
| Time-series aggregation | "Revenue trend over time" |
| Visualisation requests | "Plot a bar chart of sales by region" |

---

## Project Structure

```
datapilot-ai/
├── agent/          LangGraph state machine (Planner → Executor → Insight → Evaluator)
├── api/            FastAPI app and routes (optional backend)
├── core/           Config, logging, exceptions
├── data/           Ingestion, profiling, quality
├── llm/            Legacy provider abstraction and offline rule-based planner
│   └── fallback.py     Rule-based planner & grounded insight synthesiser
├── tools/          DuckDB, Pandas, Stats, Plotly, Anomaly tools + registry
├── ui/             Streamlit pages
│   ├── app.py
│   └── pages/
│       ├── 1_upload.py
│       ├── 2_profile.py
│       ├── 3_query.py
│       ├── 4_visualize.py
│       └── 5_history.py
└── tests/          Pytest test suite (run locally; see Testing)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- No external API credentials required for Rule-Based Mode

### 1. Clone and install

```powershell
git clone https://github.com/your-org/datapilot-ai.git
cd datapilot-ai
pip install -e ".[dev]"
```

### 2. Configure environment (optional)

```powershell
Copy-Item .env.example .env
# LLM_PROVIDER=fallback — this submission build runs offline without API keys
```

### 3. Run the Streamlit UI

```powershell
streamlit run ui/app.py
```

UI available at http://localhost:8501

### 4. (Optional) Run the FastAPI backend

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs at http://localhost:8000/docs

---

## Environment Variables

See [`.env.example`](.env.example) for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `fallback` | Offline rule-based execution; external providers are disabled in this submission build |
| `APP_ENV` | `development` | `development` / `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `MAX_UPLOAD_MB` | `100` | Maximum file upload size |
| `SQLITE_URL` | `sqlite+aiosqlite:///./datapilot.db` | History database |

---

## Running Tests

```powershell
python -m pytest tests/ -v
```

**Test status:** execute the suite locally before submission; no unverified pass count is claimed.

Test coverage:
- Application configuration (with and without API keys)
- CSV and XLSX ingestion
- Data profiling
- Data quality inspection
- All tool implementations (DuckDB, Pandas, Stats, Anomaly, Plotly)
- Rule-based intent detection (12 intent categories)
- Fallback plan builder (all intent paths)
- Fallback plan execution against real data
- Insight synthesis from actual tool results (Indonesian & English)
- Rule-based execution without external API credentials
- Semantic column matching — bilingual alias resolution (33 regression cases)
- Agent planner/executor/insight nodes in offline rule-based mode
- Regression checks for missing requested columns and transaction-count ranking
- Chart rendering from DuckDB query result (alias column fix)

---

## Demonstration Questions

Upload any CSV or XLSX file to the **Upload** page, then use these questions on the **Ask DataPilot** page:

1. `Give me an overview of this dataset`
2. `Which category has the highest total revenue?`
3. `How many missing values are there in each column?`
4. `Detect anomalies in the data`
5. `What is the correlation between numeric columns?`
6. `Plot a bar chart of sales by region`

---

## Security Model

- **No `eval()` or `exec()`** anywhere in the codebase.
- The agent produces a **structured plan** — only registered tool names and typed parameters.
- DuckDB queries run against an **in-memory database** — no file system access.
- All tool parameters are validated by **Pydantic models** before execution.
- Only **SELECT** SQL statements are permitted (regex-validated).
- No API keys are needed in the offline submission build. Never include `.env` or credentials in a shared ZIP or Git repository.

---

## Known Limitations

- **Rule-Based Mode** responds to the primary intent only; multi-part questions with compound intents are handled by their dominant keyword match.
- **Anomaly detection** requires at least 10 rows of numeric data (IsolationForest minimum).
- **Time-series analysis** requires a column with `date`, `time`, `year`, `month`, or `day` in its name, or a datetime-typed column.
- **LLM self-evaluation is disabled** in this offline submission build. Use the deterministic regression tests and evaluation scripts for measurable quality checks.
- **The FastAPI backend** is optional for the Streamlit UI — the UI functions standalone without it.

---

## Technology Stack

Python · FastAPI · Streamlit · LangGraph · Pandas · DuckDB · SciPy · Scikit-learn · Plotly · SQLite · Pydantic · Structlog

---

## License

MIT


## Capstone implementation note

This version should be described accurately as a **Rule-Based Agentic Data Analysis Platform**. It demonstrates intent detection, planning, registered-tool selection, deterministic execution, and grounded result synthesis. It is not an LLM-powered agent.

Do not distribute a `.env` file or API credentials. The repository intentionally excludes local secrets, virtual environments, runtime databases, and cache files.
#   P r o j e k - I B M - D a t a P i l o t - A I - A g e n t i c - D a t a - A n a l y s i s - I n t e l l i g e n c e - P l a t f o r m  
 