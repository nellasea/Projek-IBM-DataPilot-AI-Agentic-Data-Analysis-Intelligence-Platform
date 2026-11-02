# DataPilot AI 🧭

**Agentic Data Analysis & Intelligence Platform**

DataPilot AI is a rule-based agentic data analysis platform that allows users to upload CSV or XLSX files and ask analytical questions in Indonesian or English.

Built with Python, Streamlit, and LangGraph, the platform uses deterministic intent detection, structured planning, registered analytical tools, and grounded result generation to automate common data analysis tasks.

This capstone implementation operates entirely offline. It does not require an LLM, external AI API, or API credentials.

---

## Implementation Mode

DataPilot AI currently supports one active execution mode:

| Mode | Description | Requirements |
|---|---|---|
| **Rule-Based Agent** | Deterministic intent detection, planning, registered-tool selection, execution, and grounded explanation. | No API key or external service required. |

**Important:** This submission build is intentionally offline. Legacy provider modules may remain in the repository for reference, but the active agent workflow does not initialize or invoke external LLM providers.

---

## Features

| No. | Feature | Description |
|---|---|---|
| 1 | CSV & XLSX Ingestion | Upload and validate structured datasets. |
| 2 | Automated Data Profiling | Generate dataset summaries and column information. |
| 3 | Data Quality Inspection | Identify missing values, duplicates, and data quality issues. |
| 4 | Natural Language Querying | Ask analytical questions through the LangGraph agent workflow. |
| 5 | Rule-Based Intent Detection | Recognize 12 categories of analytical intent. |
| 6 | Safe Tool Execution | Execute analytical operations using DuckDB, Pandas, and statistical tools. |
| 7 | Statistical Analysis | Perform correlation, t-test, ANOVA, and chi-square analysis. |
| 8 | Interactive Visualization | Generate interactive charts using Plotly. |
| 9 | Grounded Explanations | Generate explanations based on actual analytical tool results. |
| 10 | Anomaly Detection | Detect potential anomalies using IsolationForest when sufficient numeric data is available. |
| 11 | Analysis History | Maintain session-based analysis history. |
| 12 | Offline Agent Workflow | Run the complete rule-based analytical workflow without external LLM integration. |

---

## System Architecture

DataPilot AI uses a LangGraph-based workflow to coordinate intent detection, analytical planning, tool execution, and result explanation.

```text
                 Streamlit UI
                      ↕
                LangGraph Agent
                      ↓
              Rule-Based Planner
               (Offline Execution)
                      ↓
                 Tool Registry
                      ↓
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
    DuckDB         Pandas          Stats
       ↓              ↓              ↓
    Plotly       Anomaly Tools   Analysis
       └──────────────┼──────────────┘
                      ↓
             Result Validation
                      ↓
             Grounded Explanation
             (Template-Based)
                      ↓
                Display Result
```

### Rule-Based Agent Execution Flow

```text
USER QUESTION
     ↓
QUESTION NORMALIZATION
     ↓
INTENT DETECTION
(Keyword / Regex Matching)
     ↓
PARAMETER & COLUMN VALIDATION
     ↓
TOOL SELECTION
(Registered Tool Registry)
     ↓
TOOL EXECUTION
(Pandas · DuckDB · SciPy · Scikit-learn)
     ↓
RESULT VALIDATION
     ↓
GROUNDED EXPLANATION
(Values from Actual Tool Output)
     ↓
DISPLAY RESULT
```

### Supported Intents

The agent supports the following 12 categories of analytical intent:

| Intent | Example Question |
|---|---|
| Dataset Profile / Overview | "Give me an overview of this dataset." |
| Descriptive Statistics | "What are the average, mean, and median values?" |
| Group-by Aggregation | "What is the total revenue by product category?" |
| Top / Highest Values | "Which product has the highest sales?" |
| Bottom / Lowest Values | "What are the lowest-selling items?" |
| Missing-value Analysis | "How many missing values are there?" |
| Duplicate Detection | "Are there any duplicate rows?" |
| Anomaly Detection | "Detect outliers in the data." |
| Correlation Analysis | "Is there a correlation between price and quantity?" |
| Distribution Analysis | "Show the distribution of prices." |
| Time-series Aggregation | "What is the revenue trend over time?" |
| Visualization Requests | "Plot a bar chart of sales by region." |

---

## Project Structure

```text
datapilot-ai/
│
├── agent/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── graph.py
│   ├── nodes.py
│   └── state.py
│
├── api/
│   ├── routes/
│   ├── main.py
│   ├── schemas.py
│   └── session_store.py
│
├── core/
│   ├── config.py
│   ├── exceptions.py
│   └── logging.py
│
├── data/
│   ├── ingestion.py
│   ├── profiling.py
│   ├── quality.py
│   └── storage.py
│
├── eval/
│   ├── metrics.py
│   └── report.py
│
├── llm/
│   ├── fallback.py
│   ├── prompts.py
│   └── provider.py
│
├── tools/
│   ├── anomaly_tool.py
│   ├── duckdb_tool.py
│   ├── pandas_tool.py
│   ├── plotly_tool.py
│   ├── registry.py
│   └── stats_tool.py
│
├── ui/
│   ├── app.py
│   └── pages/
│       ├── 1_upload.py
│       ├── 2_profile.py
│       ├── 3_query.py
│       ├── 4_visualize.py
│       └── 5_history.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_config_env.py
│   ├── test_fallback.py
│   ├── test_ingestion.py
│   ├── test_llm_provider.py
│   ├── test_profiling.py
│   ├── test_quality.py
│   ├── test_semantic_column_matching.py
│   └── test_tools.py
│
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend API | FastAPI |
| Agent Orchestration | LangGraph |
| Data Processing | Pandas, NumPy |
| Query Engine | DuckDB |
| Statistical Analysis | SciPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Plotly |
| Database | SQLite |
| Data Validation | Pydantic |
| Logging | Structlog |
| Testing | Pytest |

---

## Quick Start

### Prerequisites

- Python 3.11 or newer
- Git
- No external API credentials required for the active Rule-Based Agent mode.

### 1. Clone the Repository

```powershell
git clone https://github.com/nellasea/Projek-IBM-DataPilot-AI-Agentic-Data-Analysis-Intelligence-Platform.git

cd Projek-IBM-DataPilot-AI-Agentic-Data-Analysis-Intelligence-Platform
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can run the application using the virtual environment's Python executable directly without activating it.

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. Configure Environment Variables (Optional)

Create a local environment file:

```powershell
Copy-Item .env.example .env
```

The active submission build runs in offline Rule-Based Agent mode and does not require an API key.

### 5. Run the Streamlit Application

```powershell
streamlit run ui/app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

### 6. Run the Optional FastAPI Backend

If the backend is configured and required:

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

The FastAPI backend is optional. The Streamlit interface can operate independently.

---

## Environment Variables

See [`.env.example`](.env.example) for the environment configuration used by the project.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `fallback` | Selects the offline rule-based execution mode in this submission build. |
| `APP_ENV` | `development` | Application environment. |
| `LOG_LEVEL` | `INFO` | Logging level: DEBUG, INFO, WARNING, or ERROR. |
| `MAX_UPLOAD_MB` | `100` | Maximum permitted upload size in MB. |
| `SQLITE_URL` | `sqlite+aiosqlite:///./datapilot.db` | SQLite database connection for history storage. |

---

## Running Tests

Run the test suite from the project root:

```powershell
python -m pytest tests/ -v
```

**Test status:** The test suite must be executed locally before submission. No unverified test pass count or success rate is claimed.

### Test Coverage

The existing test suite includes checks for:

- Application configuration, including configurations with and without API keys.
- CSV and XLSX ingestion.
- Automated data profiling.
- Data quality inspection.
- Analytical tool implementations: DuckDB, Pandas, Stats, Anomaly, and Plotly.
- Rule-based intent detection across 12 intent categories.
- Fallback plan generation and execution.
- Insight synthesis from actual tool results in Indonesian and English.
- Offline execution without external API credentials.
- Semantic column matching and bilingual alias resolution.
- Agent planner, executor, and insight nodes.
- Missing requested-column validation.
- Transaction-count ranking regression checks.
- Chart rendering from DuckDB query results.

---

## Demonstration Questions

Upload a CSV or XLSX dataset through the **Upload** page. Then open **Ask DataPilot** and try these example questions:

1. `Give me an overview of this dataset`
2. `Which category has the highest total revenue?`
3. `How many missing values are there in each column?`
4. `Detect anomalies in the data`
5. `What is the correlation between numeric columns?`
6. `Plot a bar chart of sales by region`

The questions are examples. Their results depend on the columns, data types, and contents of the uploaded dataset.

---

## Security Model

DataPilot AI applies the following safeguards:

- **No dynamic code execution:** The project does not rely on `eval()` or `exec()` for agent execution.
- **Structured analytical plans:** The agent generates structured plans using registered tool names and typed parameters.
- **Controlled tool execution:** Analytical operations are routed through the registered tool system.
- **Parameter validation:** Tool parameters are validated using Pydantic models.
- **Restricted SQL operations:** Only permitted `SELECT` queries are intended to be executed, subject to the application's SQL validation.
- **Offline execution:** The active submission workflow does not require external API credentials.
- **Secret protection:** Never commit `.env` files, API credentials, or other sensitive configuration files to GitHub.

---

## Known Limitations

- **Rule-Based Intent Detection:** The agent uses keyword and regex matching. It is not equivalent to unrestricted natural-language understanding.
- **Primary Intent Handling:** The current workflow prioritizes the dominant detected intent. Compound questions containing multiple analytical requests may not be fully supported.
- **Anomaly Detection:** IsolationForest requires sufficient numeric data. The current implementation is intended for datasets with at least 10 rows.
- **Time-series Analysis:** Requires a datetime-typed column or a recognizable date/time-related column name, such as `date`, `time`, `year`, `month`, or `day`.
- **No LLM Self-Evaluation:** LLM-based self-evaluation is disabled in the offline submission build. Deterministic tests and evaluation scripts are used for quality checks.
- **Optional API Backend:** The FastAPI backend is not required for the standalone Streamlit application.

---

## Future Development

Potential improvements include:

- More flexible natural-language understanding.
- Multi-step analytical planning and execution.
- Expanded semantic column matching.
- Predictive analytics and forecasting.
- Automated analytical report generation and export.
- Multi-dataset analysis.
- Additional database integrations.
- Optional local LLM integration when sufficient hardware resources are available.

---

## Capstone Implementation Note

This submission should be described accurately as a **Rule-Based Agentic Data Analysis Platform**.

It demonstrates an agentic workflow through:

1. User-question normalization.
2. Intent detection.
3. Analytical parameter and column validation.
4. Structured planning.
5. Registered-tool selection.
6. Deterministic tool execution.
7. Result validation.
8. Grounded insight generation.

It is **not an LLM-powered agent**. Its active analytical workflow does not depend on external generative AI services.

IBM Bob was used as a development assistant during project implementation, not as the runtime engine of DataPilot AI.

---

## License

This project is licensed under the MIT License.

---

**DataPilot AI**  
*Turning structured data into actionable insights through agentic workflows.*
