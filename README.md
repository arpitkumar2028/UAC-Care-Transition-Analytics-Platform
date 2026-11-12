# Care Transition Efficiency & Placement Outcome Analytics

Pipeline analytics, REST API, machine learning forecasting, and interactive Streamlit dashboard for the U.S. HHS/ORR and CBP Unaccompanied Children (UAC) program.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1.0-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-30%2F30%20Passing-success)](run_all_tests.py)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uac-care-transition-analytics-platform.streamlit.app/)

**Live Interactive Dashboard:** [Launch App on Streamlit Cloud](https://uac-care-transition-analytics-platform.streamlit.app/)

---

## Overview

Most reporting on the HHS Office of Refugee Resettlement (ORR) and Customs and Border Protection (CBP) Unaccompanied Children dataset treats daily headcounts as static totals. That makes it hard to see where children actually wait too long—whether at the border past the TVPRA 72-hour transfer limit, or inside licensed shelters during sponsor background checks.

This project models the data as a connected, two-stage operational pipeline:

```text
[Intake]                 [Stage 1]               [Stage 2]                [Outcome]
Border Apprehensions --> CBP Custody Queue ----> HHS Shelter Care ------> Sponsor Reunification
                         (72-hr Target)          (Vetting & Casework)
                               |                       |
                     Transfer Efficiency      Discharge Effectiveness
                     Ratio (T / Q_cbp)        Index (D / Q_hhs)
```

By tracking flow speed, day-of-week bottlenecks, dwell times, and backlog spikes across both stages, the app helps analysts and caseworkers spot delays early and test capacity changes before beds fill up.

---

## What the Platform Includes

1. **Multi-Stage Flow View**: Interactive Plotly Sankey diagram verifying end-to-end child movement from border encounters through CBP holding and HHS shelters to sponsor release, paired with daily census vs. bed-capacity area charts.
2. **Process Efficiency Tracking**: Trend views for Transfer Efficiency Ratio (TER), Discharge Effectiveness Index (DEI), Pipeline Throughput Rate (PTR), and 30-day Outcome Stability against statutory targets.
3. **Bottleneck & Anomaly Detection**: Day-of-week by month congestion heatmaps, rolling 30-day Z-score checks ($Z > 2.0$), and sector-level delay clustering across all six border sectors.
4. **Weekend Latency Analysis**: Day-of-week velocity charts and two-sample Welch's $t$-tests measuring how weekend administrative slowdowns create recurring Monday border backlogs.
5. **Local Operations Copilot (`care_pipeline/ai_copilot.py`)**: Rule-based diagnostic Q&A grounded in the active dataset—no external LLM API keys or paid subscriptions needed.
6. **ML Forecasting & Delay Risk Lab (`ml/models.py`)**: A Gradient Boosting forecaster with Fourier seasonal terms for 30- and 90-day shelter census projections, alongside a Random Forest classifier estimating whether an individual case will stay in care past 45 days.
7. **Interactive Policy Simulator (`care_pipeline/simulation_engine.py`)**: "What-If" stress controls for border arrival surges, transfer velocity, shelter bed caps, and sponsor vetting speeds.
8. **Sponsor Vetting Kanban Board (`care_pipeline/kanban_module.py`)**: Three-stage case tracker (*Vetting Queue*, *Home Study Pending*, *Ready for Discharge*) that flags children waiting more than 30 days.
9. **PDF Briefings & Webhook Alerts**: One-click ReportLab executive PDF exports (`care_pipeline/report_generator.py`) and fail-safe Slack/Discord webhook alerts (`care_pipeline/alerts_dispatcher.py`).
10. **FastAPI Backend & Dual Database (`backend/`, `db/`)**: REST endpoints under `/api/v1/*` with Pydantic schemas (and automatic Starlette/SQLite fallback), backed by SQLite (WAL mode) locally or PostgreSQL 15 in Docker.

---

## Core KPI Definitions

| KPI | Formula | Target | What It Measures |
| :--- | :--- | :--- | :--- |
| **Transfer Efficiency Ratio ($\text{TER}_t$)** | $\frac{\text{Transfers Out of CBP}_t}{\text{CBP Custody Queue}_t}$ | $\ge 0.85$ | Speed of handoffs from CBP holding into HHS care (TVPRA 72-hour standard). |
| **Discharge Effectiveness Index ($\text{DEI}_t$)** | $\frac{\text{Discharges to Sponsors}_t}{\text{Children in HHS Care}_t} \times 100$ | $\ge 2.5\%$ | Daily shelter bed turnover into sponsor placements. |
| **Pipeline Throughput Rate ($\text{PTR}_t$)** | $\frac{\text{Discharges to Sponsors}_t}{\text{Border Apprehensions}_t}$ | $\ge 1.00$ | System clearance balance ($\ge 1.0$ shrinks backlog; $< 1.0$ builds backlog). |
| **Cumulative Backlog ($\Sigma B_t$)** | $\sum_{\tau=1}^t (\text{Apprehensions}_\tau - \text{Discharges}_\tau)$ | $\le 0$ | Net change in children remaining in custody across the selected window. |
| **Little's Law Dwell Time ($W_t$)** | $\frac{\text{HHS Shelter Census}_t}{\text{MA}_7(\text{Discharges}_t)}$ | $30\text{–}35\text{ days}$ | Estimated average shelter stay in days at current placement speed. |
| **Backlog Accumulation Rate ($\text{BAR}_{t, 7}$)** | $\sum_{\tau=t-6}^t (\text{Apprehensions}_\tau - \text{Discharges}_\tau)$ | $\le 0$ | Seven-day rolling net inflow into federal custody. |
| **Backlog Anomaly Z-Score ($Z_t$)** | $\frac{\text{BAR}_{t, 7} - \mu_{30d}(\text{BAR})}{\sigma_{30d}(\text{BAR})}$ | $\le 2.0$ | Statistical process control flag for abnormal intake/outflow congestion. |

---

## Repository Structure

```text
.
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI REST API (/api/v1/summary, timeseries, bottlenecks, simulate, ml/*)
│   └── schemas.py              # Pydantic request/response models
│
├── care_pipeline/
│   ├── __init__.py
│   ├── ai_copilot.py           # Local deterministic Q&A and audit summary generator
│   ├── alerts_dispatcher.py    # Slack / Discord / HTTP webhook alert client
│   ├── data_loader.py          # SQLite DB & CSV archive ingestion + calendar features
│   ├── kanban_module.py        # 3-column sponsor vetting case tracker
│   ├── metrics_engine.py       # KPI calculations, Little's Law dwell time & health score
│   ├── ml_forecasting.py       # 14-day autoregressive forecast preview for PDF briefs
│   ├── report_generator.py     # ReportLab multi-page PDF exporter
│   ├── simulation_engine.py    # Forward surge & vetting stress-test engine
│   └── temporal_analysis.py    # Weekday vs. weekend t-tests, MoM trends & stagnation checks
│
├── core/
│   ├── __init__.py
│   └── analytics.py            # Process ratios & rolling 30-day Z-score anomaly detector
│
├── db/
│   ├── __init__.py
│   ├── connection.py           # SQLite (WAL) / PostgreSQL connection pool helper
│   ├── init_db.py              # Schema setup and 3-year synthetic dataset seeder
│   ├── models.py               # Typed dataclasses for macro and case-level records
│   └── schema.sql              # ANSI SQL schema with constraints and indexes
│
├── ml/
│   ├── __init__.py
│   ├── models.py               # Fourier Gradient Boosting forecaster & Random Forest delay classifier
│   └── saved_models/           # Pre-trained joblib model artifacts
│
├── docs/
│   ├── executive_summary.md    # Policy briefing memo for government stakeholders
│   └── research_paper.md       # Technical paper covering EDA, KPI math & ML benchmarks
│
├── tests/                      # 6 unit & integration test suites (30 checks total)
├── utils/
│   ├── __init__.py
│   └── data_generator.py       # 1,095-day macro & 27,375-case micro data generator
│
├── .streamlit/config.toml      # Streamlit light theme configuration
├── app.py                      # Main 6-tab Streamlit dashboard
├── styles.css                  # Warm off-white & forest-green UI stylesheet
├── Dockerfile                  # Multi-stage Docker image
├── docker-compose.yml          # PostgreSQL 15 + FastAPI + Streamlit stack
├── start_platform.ps1          # Windows one-click launcher
├── requirements.txt            # Python dependencies
├── run_all_tests.py            # Runs all 30 tests across 6 suites
├── HHS_Unaccompanied_Children_Program.csv
└── uac_analytics.db            # Pre-seeded SQLite database
```

---

## How to Run

### Option 1: Docker Compose (PostgreSQL 15 + FastAPI + Streamlit)

Requires Docker Desktop or Docker Engine:

```bash
docker compose up --build
```

* Streamlit Dashboard: `http://localhost:8501`
* FastAPI Swagger Docs: `http://localhost:8000/docs`
* PostgreSQL 15: port `5432` (automatically initialized from `db/schema.sql` and seeded on first boot via `--if-empty`)

### Option 2: Local Python Setup (SQLite, Python 3.10–3.12)

1. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Reset and seed the local SQLite database (optional, `uac_analytics.db` is already included):
   ```bash
   python db/init_db.py --reset
   ```
3. Run the test suite (30 tests across 6 suites):
   ```bash
   python run_all_tests.py
   ```
4. Start the backend and dashboard:
   ```powershell
   .\start_platform.ps1
   ```
   Or start both manually in separate terminals:
   ```bash
   # Terminal 1: REST API
   uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

   # Terminal 2: Streamlit UI
   streamlit run app.py
   ```

### Option 3: Host Live from GitHub (Free)

GitHub serves code and Markdown, so you can connect this repository to a free host to make all six tabs interactive in a browser:

1. **Streamlit Community Cloud (Recommended)**
   * **Live App**: [https://uac-care-transition-analytics-platform.streamlit.app/](https://uac-care-transition-analytics-platform.streamlit.app/)
   * To redeploy from your own fork: sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, select the repository, set branch to `main`, and set the main file to `app.py` (Python `3.11` under Advanced settings).
   * If no external FastAPI server is configured, `app.py` automatically falls back to the bundled SQLite database (`uac_analytics.db`), local analytics modules, and `.joblib` models (retraining cleanly in-memory if needed).
   * *(Optional)* If you host `backend/main.py` separately on Render or Railway, add `BACKEND_API_URL = "https://your-api-url"` in Streamlit Cloud Secrets.
2. **Hugging Face Spaces (Docker SDK)**
   * Create a Docker Space on [huggingface.co/spaces](https://huggingface.co/spaces) connected to this repository using the included `Dockerfile`.

---

## Documentation & Key Findings

* **[Executive Summary](docs/executive_summary.md)**: Policy briefing covering TVPRA standards, stage-wise bottlenecks, and operational recommendations.
* **[Technical Research Paper](docs/research_paper.md)**: Full breakdown of EDA, Little's Law dwell estimation, Welch's $t$-tests, rolling Z-score SPC, and ML metrics (`0.23%` in-sample MAPE forecaster, `0.968` ROC-AUC stagnation classifier).

Main empirical findings from the data:
1. **Weekend Discharge Slowdown**: Sponsor discharges drop sharply on weekends (`-44.3%` in the federal CSV archive, $p = 1.42 \times 10^{-28}$; `~48.8%` in the 3-year synthetic longitudinal database), while border arrivals stay steady—producing predictable Monday border queues.
2. **Variable Transfer Clearance**: Stage 1 TER ranges from `0.20` during arrival waves to above `1.20` during recovery periods, underscoring the value of surge transit capacity.
3. **Shelter Dwell Sensitivity**: Little's Law dwell estimates move between roughly `25` and `60` days depending on sponsor risk tiers and documentation completeness.
4. **Early Congestion Warnings**: Rolling 7-day backlog Z-scores above `2.0` flag developing bottlenecks up to two weeks before shelter capacity limits are reached.

---

## License

Developed as part of the UnifiedMentor Data Science & Analytics initiative. Released under the [MIT License](LICENSE).
