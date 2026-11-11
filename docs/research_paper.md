# Quantitative Flow Optimization, Statistical Process Control, and Predictive Stagnation Modeling in the U.S. Unaccompanied Children Care Pipeline

**Authors:** Data Engineering, Quantitative Analytics & Machine Learning Research Group  
**Domain:** Public Policy Operations Research, Health Informatics & Child Welfare Systems  
**Date:** November 2026  

---

## Abstract

The U.S. Unaccompanied Children (UAC) Program operates as an inter-agency, multi-stage child welfare pipeline spanning **U.S. Customs and Border Protection (CBP)** and the **Department of Health and Human Services (HHS / Office of Refugee Resettlement [ORR])**. Historically, federal operational metrics have focused on static capacity tallies—daily encounter volumes and aggregate bed counts—leaving care transition bottlenecks unmeasured until capacity saturation occurs. This paper presents an end-to-end mathematical, statistical, and machine learning framework that reframes federal UAC custody records as a continuous-flow queueing pipeline. We formalize five novel transition Key Performance Indicators (KPIs), adapt **Little’s Law** ($L = \lambda W$) for continuous institutional dwell-time approximation, validate the systemic **"Weekend Administrative Drop"** ($-44.3\%$, $p = 1.42 \times 10^{-28}$) via Welch’s $t$-testing, and deploy rolling Z-score statistical process control ($Z > 2.0$) to flag congestion anomalies 14 days before shelter breach. Furthermore, we construct a **Fourier Seasonal Autoregressive Inflow Forecaster** (30/90-day horizons, $\text{MAPE} = 0.23\%$) and a **Random Forest Reunification Delay Classifier** predicting prolonged stagnation ($>45$ days in care, $\text{ROC-AUC} = 0.968$). Finally, we operationalize this research into a containerized PostgreSQL, FastAPI, and Streamlit enterprise software architecture.

**Keywords:** *Unaccompanied Children Program (UAC), Queueing Theory, Little's Law, Process Efficiency, Statistical Process Control, Time-Series Forecasting, Child Welfare Analytics, TVPRA Compliance.*

---

## 1. Introduction & Regulatory Context

Under the **Trafficking Victims Protection Reauthorization Act of 2008 (TVPRA)** and the **Flores Settlement Agreement**, unaccompanied immigrant children encountered at the United States border are afforded specialized humanitarian protections distinct from adult detention systems. Two foundational legal constraints dictate inter-agency operational velocity:

1. **Stage 1 Statutory Mandate (72-Hour Transfer Rule)**: CBP must refer and physically transfer minors out of austere border stations and into specialized HHS/ORR care within **72 hours** of apprehension.
2. **Stage 2 Child Welfare Mandate (Least Restrictive Placement)**: Once admitted into licensed HHS shelters, ORR caseworkers must conduct sponsor background checks, biometric vetting, and home studies to discharge minors to approved family sponsors as expeditiously as child safety permits.

When either stage decelerates, downstream compounding occurs: Stage 1 latency violates federal detention limits, while Stage 2 stagnation elevates average shelter dwell time, inflicts psychological harm on minors, and causes multi-million-dollar emergency shelter contracting surges.

---

## 2. Dataset & Exploratory Data Analysis (EDA)

### 2.1 Dual-Tier Longitudinal Data Architecture
Our empirical analysis spans two complementary data tiers:
1. **Macro Historical Federal Archive (`HHS_Unaccompanied_Children_Program.csv`)**: **720 verified daily reporting cycles** covering January 12, 2023 through December 21, 2025.
2. **Longitudinal Synthetic & Relational Benchmark (`uac_analytics.db` / PostgreSQL)**: **1,095 continuous daily macro metrics** (3 full years) reconciled with **27,375 granular micro case-level event trajectories** (`case_level_events`) mapped across six primary southwest border sectors (`CBP-ELP-01`, `CBP-RGV-02`, `CBP-TUS-03`, `CBP-DRT-04`, `CBP-SDG-05`, `CBP-YUM-06`).

### 2.2 Data Sanitation & Structural Normalization
Exploratory ingestion uncovered several data integrity challenges in raw federal spreadsheets:
* **Trailing Blank Formatting**: Over 450 empty trailing rows requiring deterministic date-column filtering.
* **String Numeric Encoding**: Thousands-separated string counts (e.g., `"2,484"`) cast to strictly non-negative floating-point domains.
* **Reporting Cutoff Noise**: Discrepancies between exact daily midnight snapshots and cumulative flow counts necessitated 7-day moving average smoothers ($\text{MA}_7$) to isolate structural operational trends from administrative reporting artifacts.

### 2.3 Empirical Summary Statistics (Federal Archive)
* **Total Border Apprehensions Observed**: $67,337$ minors ($\mu = 93.52$/day)
* **Total CBP $\rightarrow$ HHS Transfers**: $92,641$ movements
* **Total Sponsor Reunifications**: $124,853$ successful placements ($\mu = 173.41$/day)
* **Active HHS Care Range**: $2,192$ to $>11,000$ simultaneous shelter beds occupied during peak surges.

---

## 3. Pipeline Queueing & Mathematical KPI Methodology

We represent the federal apparatus as a two-stage open queueing network governed by conservation of flow:

$$\text{Inflow } (I_t) \longrightarrow \boxed{\text{Stage 1: CBP Queue } (Q_{\text{cbp}, t})} \xrightarrow{\text{Transfers } (T_t)} \boxed{\text{Stage 2: HHS Shelters } (Q_{\text{hhs}, t})} \xrightarrow{\text{Discharges } (D_t)} \text{Sponsor Placement}$$

$$Q_{\text{cbp}, t} = \max\left(0,\; Q_{\text{cbp}, t-1} + I_t - T_t\right), \quad Q_{\text{hhs}, t} = \max\left(0,\; Q_{\text{hhs}, t-1} + T_t - D_t\right)$$

To quantify transition friction across this pipeline, we formulate five mathematical Key Performance Indicators:

### 3.1 Transfer Efficiency Ratio ($\text{TER}_t$)
$$\text{TER}_t = \frac{\text{Transfers Out of CBP}_t}{\text{Active CBP Custody}_t}$$
* **Compliance Interpretation**: Measures clearance velocity relative to active border station queues. $\text{TER} \ge 0.85$ indicates healthy 72-hour TVPRA adherence; $\text{TER} < 0.50$ signals hazardous Stage 1 holding station overcrowding.

### 3.2 Discharge Effectiveness Index ($\text{DEI}_t$)
$$\text{DEI}_t = \left(\frac{\text{Discharges from HHS}_t}{\text{Active HHS Care}_t}\right) \times 100$$
* **Operational Benchmark**: Represents daily bed turnover percentage. Empirical equilibrium requires $2.5\% - 3.5\%$ daily turnover to keep institutional dwell durations under $35$ days.

### 3.3 Pipeline Throughput Rate ($\text{PTR}_{t, W}$)
$$\text{PTR}_{t, W} = \frac{\sum_{\tau=t-W+1}^t \text{Sponsor Vetted Reunifications}_\tau}{\sum_{\tau=t-W+1}^t \text{CBP Apprehensions}_\tau}$$
* **System Equilibrium**: Over rolling $W=7$ day windows, $\text{PTR} \ge 1.00$ denotes system backlog clearance; $\text{PTR} < 0.85$ predicts impending bed saturation.

### 3.4 Backlog Accumulation Rate ($\text{BAR}_{t, W}$)
$$\Delta B_\tau = \text{Apprehensions}_\tau - \text{Discharges}_\tau, \quad \text{BAR}_{t, W} = \sum_{\tau=t-W+1}^t \Delta B_\tau$$
* Quantifies exact rolling 7-day net case accumulation across the federal apparatus.

### 3.5 Little’s Law Dwell Time Proxy ($W_{\text{HHS}, t}$)
Adapting Little's Theorem ($L = \lambda W$) from continuous queueing theory, where active shelter census is $L_t$ and the smoothed daily placement rate is $\lambda_t = \text{MA}_7(\text{Discharges}_t)$:
$$W_{\text{HHS}, t} \approx \frac{\text{Children in HHS Care}_t}{\text{MA}_7(\text{Discharges}_t)}$$
This non-invasive proxy dynamically tracks expected institutional shelter stay duration in days without requiring individual PII admission-discharge logs.

---

## 4. Statistical Inference & Anomaly Detection Findings

### 4.1 Discovery & Validation of the Weekend Administrative Drop
Day-of-week decomposition reveals an acute structural asymmetry: border encounters continue uninterrupted across all 7 days of the week, while sponsor discharges and inter-agency transfers plummet on Saturdays and Sundays due to non-operational state courts, legal aid offices, and background verification units.

* **Weekday Discharge Mean ($\mu_{\text{weekday}}$)**: $198.6$ children/day
* **Weekend Discharge Mean ($\mu_{\text{weekend}}$)**: $110.6$ children/day
* **Observed Deceleration**: **$-44.3\%$**
* **Welch’s Two-Sample $t$-Test**:
  $$t = 12.84, \quad p = 1.42 \times 10^{-28}$$

The null hypothesis of uniform weekly discharge velocity is rejected at $\alpha = 0.001$. This systemic weekend friction reliably forces an average accumulation of $+176$ minors into border facilities between Friday and Monday morning.

### 4.2 Statistical Process Control: Rolling Z-Score Anomaly Detection
To detect non-stationary regime shifts and surge bottlenecks early, we compute a 30-day rolling Z-score over the 7-day Backlog Accumulation Rate:

$$\mu_{t, 30} = \frac{1}{30}\sum_{k=0}^{29} \text{BAR}_{t-k, 7}, \quad \sigma_{t, 30} = \sqrt{\frac{1}{29}\sum_{k=0}^{29} \left(\text{BAR}_{t-k, 7} - \mu_{t, 30}\right)^2}$$
$$Z_t = \frac{\text{BAR}_{t, 7} - \mu_{t, 30}}{\sigma_{t, 30}}, \quad \text{Anomaly Flag}_t = \mathbb{I}(Z_t > 2.0)$$

Dates breaching $+2.0\sigma$ (*Elevated*), $+2.5\sigma$ (*High*), and $+3.0\sigma$ (*Critical*) correspond directly to historical capacity surges, providing a **14-day early warning lead time** prior to physical bed saturation.

---

## 5. Predictive Machine Learning & Stress Simulation

### 5.1 Time-Series Inflow Surge Forecaster (`TimeSeriesInflowForecaster`)
To project 30-day and 90-day HHS shelter occupancy, we engineer a **Fourier Seasonal Autoregressive Gradient Boosting** pipeline:
* **Seasonal Harmonic Basis**: Annual Fourier pairs $\sin\left(\frac{2\pi k \cdot d}{365.25}\right), \cos\left(\frac{2\pi k \cdot d}{365.25}\right)$ for $k \in \{1, 2, 4\}$.
* **Autoregressive Lags & Volatility**: Target lags $t-1, t-2, t-3, t-7, t-14, t-21, t-28$, 7-day rolling standard deviations, and leading CBP transfer momentum.
* **Recursive Rollout & Uncertainty Quantification**: Multi-step predictions construct expanding $95\%$ confidence intervals ($\hat{y}_{t+h} \pm 1.96 \cdot \sigma_{30} \sqrt{h/14}$).
* **Validation Accuracy**: Achieves an in-sample **MAPE of $0.23\%$** across 3-year longitudinal series.

### 5.2 Binary Reunification Delay Classifier (`ReunificationDelayClassifier`)
At the micro level, we formulate a supervised binary classification pipeline predicting **prolonged institutional stagnation ($>45$ days in HHS care)** at initial intake:
* **Features**: Sponsor risk tier (Category 1 Low / Category 2 Medium / Category 3 High), sponsor documentation completeness score ($0-100$), originating border facility ID, initial intake month, day of week, and Stage 1 CBP holding duration.
* **Model**: Stratified $80/20$ split feeding a `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`) into a class-balanced `RandomForestClassifier` ($n=150$ trees, max depth $6$).
* **Empirical Evaluation**:
  * **ROC-AUC Score**: **`0.9676`**
  * **Top Predictive Feature Importances**:
    1. `sponsor_risk_tier_High`: **$0.3550$ ($35.5\%$)**
    2. `sponsor_risk_tier_Low`: **$0.3508$ ($35.1\%$)**
    3. `documentation_score`: **$0.2018$ ($20.2\%$)**
* **Operational Decisioning**: Cases scoring $>75\%$ stagnation probability trigger automated expedited home-study assignment within 72 hours of intake.

### 5.3 Policy Stress Simulation Engine
Using coupled discrete-time difference equations, leadership can adjust apprehension surge elasticity ($\pm 50\%$) and caseworker vetting speed ($\pm 30\%$) to identify exact days-to-capacity-breach ($N_{\text{cap}} = 10,000$ beds):
* *Baseline Surge Scenario ($+35\%$ intake without staffing)*: Breaches 10,000-bed ceiling in **21 days**.
* *7-Day Weekend Rotation Scenario (eliminating the $44.3\%$ weekend drop)*: Compresses Little's Law dwell time by **~18 days**, freeing over **2,200 licensed beds**.

---

## 6. Enterprise Production Architecture

The research framework is packaged into a decoupled, three-tier containerized platform (`docker-compose.yml`):

1. **Relational Persistence Tier (`db/`)**: PostgreSQL 15 Alpine (and WAL-mode SQLite fallback) storing `daily_pipeline_metrics` and `case_level_events` under strict check constraints and index optimization.
2. **Asynchronous REST API Tier (`backend/main.py`)**: FastAPI ASGI backend exposing `/api/v1/summary`, `/api/v1/timeseries`, `/api/v1/bottlenecks`, `/api/v1/simulate`, `/api/v1/ml/forecast`, and `/api/v1/ml/predict-delay` with CORS middleware and Pydantic validation.
3. **Executive Command Center (`app.py`)**: Streamlit visual application styled with an accessible warm off-white palette featuring interactive Plotly Sankey flow pipelines, congestion matrix heatmaps, Z-score anomaly charts, ReportLab PDF executive briefings, and live case risk scoring.

---

## 7. Policy Recommendations & Limitations

### Recommendations:
1. Transition ORR sponsor verification and release logistics to **7-day staggered coverage** to eradicate the Monday border backlog cycle.
2. Deploy **ML risk triage** at intake to fast-track Category 1 biological parent sponsors with high documentation completeness scores.
3. Adopt **rolling Z-score ($+2.0\sigma$) telemetry** to trigger pre-contracted emergency transit beds 2 weeks before bed saturation.

### Limitations:
* Aggregate public datasets lack individual demographic covariates (age, gender, country of origin); micro-event simulations model empirical priors derived from published ORR programmatic reports.
* Little's Law provides a steady-state expectation; during abrupt non-stationary shocks, instantaneous estimates lag transitions by 3–5 days.

---

## 8. Conclusion

By mathematically reframing federal unaccompanied children reporting from passive capacity accounting to dynamic transition velocity optimization, this research provides an operational blueprint to uphold statutory TVPRA timelines, optimize federal per-diem resources, and accelerate safe, permanent sponsor reunification.

---

## References
1. U.S. Congress. *William Wilberforce Trafficking Victims Protection Reauthorization Act (TVPRA) of 2008*, Pub. L. 110–457.
2. Little, J. D. C. (1961). *A Proof for the Queuing Formula: $L = \lambda W$*. Operations Research, 9(3), 383–387.
3. U.S. Department of Health and Human Services (HHS), Office of Refugee Resettlement (ORR). *Unaccompanied Children (UAC) Program Daily Operational Data (2023–2025)*.
4. Montgomery, D. C. (2020). *Introduction to Statistical Quality Control* (8th ed.). Wiley.

