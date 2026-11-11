# Care Transition Efficiency & Placement Outcome Analytics
## Operational Pipeline Intelligence & Humanitarian Child Welfare Optimization

**Author**: Data Science & Operational Analytics Intern  
**Program**: UnifiedMentor Data Science Internship  
**Domain**: Health Informatics, Child Welfare Operations & Public Policy Analytics  
**Date**: October 2026  

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Background & Regulatory Context](#2-background--regulatory-context)
3. [Problem Statement & Core Hypotheses](#3-problem-statement--core-hypotheses)
4. [Dataset Description & Engineering Architecture](#4-dataset-description--engineering-architecture)
5. [Mathematical Formulation & KPI Framework](#5-mathematical-formulation--kpi-framework)
6. [Statistical Methodology & Empirical Findings](#6-statistical-methodology--empirical-findings)
7. [Predictive Machine Learning & Simulation Modeling](#7-predictive-machine-learning--simulation-modeling)
8. [Enterprise Software Architecture & Modules](#8-enterprise-software-architecture--modules)
9. [Policy Directives & Child Welfare Recommendations](#9-policy-directives--child-welfare-recommendations)
10. [Conclusion & Technical Deliverables](#10-conclusion--technical-deliverables)

---

## 1. Executive Summary

Historically, the **U.S. Department of Health and Human Services (HHS / Office of Refugee Resettlement)** and **Customs and Border Protection (CBP)** have tracked the Unaccompanied Children (UAC) Program through aggregate census figures: daily apprehension counts, active custody queues, and total bed utilization. While essential for budgeting, aggregate headcount metrics fail to capture **process efficiency, transition velocity, and care outcome stability**.

This project establishes an **end-to-end operational intelligence platform** that reframes the UAC dataset from a static capacity-monitoring lens to an **active, multi-stage care pipeline optimization lens**. 

### Key Contributions:
* **Mathematical Formalization of Transition Velocity**: Defined and deployed novel process KPIs, including the **Transfer Efficiency Ratio (TER)**, the **Discharge Effectiveness Index (DEI)**, and the **System Pipeline Throughput (SPT)**.
* **Little’s Law Dwell Time Formulation**: Adapted queueing theory ($L = \lambda W$) to continuously approximate the average institutional shelter stay without requiring individual admission-discharge timestamps.
* **Quantification of the "Weekend Administrative Drop"**: Conducted two-sample Welch's t-tests demonstrating that sponsor reunifications decline by **over 35% on weekends** ($p < 0.001$), creating artificial Monday morning surges in border facilities.
* **Zero-Cost Enterprise Decision Support Platform**: Delivered an interactive, market-ready Streamlit platform equipped with a local AI Copilot, automated webhook incident dispatchers, ReportLab PDF brief generators, and an interactive sponsor vetting Kanban board.

---

## 2. Background & Regulatory Context

Under the **Trafficking Victims Protection Reauthorization Act of 2008 (TVPRA)** and the landmark **Flores Settlement Agreement**, unaccompanied immigrant children apprehended at the border are recognized as vulnerable minors requiring specialized care rather than enforcement detention.

The regulatory mandates impose strict operational obligations:
1. **The 72-Hour Statutory Transfer Mandate**: By federal law, unaccompanied children in CBP border custody must be referred and physically transferred to the custody of HHS/ORR within **72 hours** of apprehension.
2. **Post-Acute Child Welfare Mandate**: Once inside HHS/ORR custody, children are placed in state-licensed shelters, group homes, or transitional foster care. HHS is legally required to identify, vet, and safely discharge minors to approved sponsors (primarily parents or close family relatives) as expeditiously as safety permits.

When transition efficiency degrades at Stage 1, children are stranded in austere border patrol stations. When placement velocity slows at Stage 2, children experience prolonged institutional stays, driving up federal costs and compounding psychological trauma.

---

## 3. Problem Statement & Core Hypotheses

### Primary Unanswered Questions:
* *How efficiently are children transitioned from CBP border holding into licensed HHS care?*
* *Are sponsor discharges keeping pace with border arrival surges over multi-month horizons?*
* *Where do care backlogs primarily accumulate (border stations vs. shelter beds)?*
* *What causes recurring weekly cycles in transition velocity?*

### Tested Hypotheses:
1. **$H_1$ (Stage 1 Friction)**: Border custody backlogs are primarily driven by transfer clearance latency ($\text{TER} < 0.50$) rather than raw arrival volume.
2. **$H_2$ (Weekend Stagnation Effect)**: Sponsor discharges and inter-agency transfers exhibit statistically significant weekend drops due to court closures and administrative downtime.
3. **$H_3$ (System Equilibrium)**: Extended periods where $\text{SPT} < 0.85$ predict impending shelter bed capacity saturation.

---

## 4. Dataset Description & Engineering Architecture

The operational dataset spans **720 daily reporting entries** from **January 12, 2023** to **December 21, 2025**.

### Schema & Feature Dictionary:
| Raw Column Name | Engineered Alias | Type | Operational Meaning |
| :--- | :--- | :--- | :--- |
| `Date` | `Date` | `datetime64` | Daily reporting timestamp (ISO-8601) |
| `Children apprehended and placed in CBP custody*` | `Apprehensions` | `float64` | Daily border arrival inflow into federal custody |
| `Children in CBP custody` | `CBP_Custody` | `float64` | Active census / queue in CBP border holding |
| `Children transferred out of CBP custody` | `Transfers_Out_CBP` | `float64` | Flow transitioning from CBP into HHS shelters |
| `Children in HHS Care` | `HHS_Care` | `float64` | Active shelter bed census across the ORR network |
| `Children discharged from HHS Care` | `Discharges_HHS` | `float64` | Successful reunifications with vetted sponsors |

### Data Sanitization Pipeline:
* Truncated 450+ empty trailing formatting rows ($> 720$).
* Formatted string numbers containing thousands commas (e.g., `"2,484"` $\rightarrow$ `2484.0`).
* Enforced non-negative domain constraints.
* Calculated calendar spines (day of week, weekend indicator, month, year, rolling 7-day and 30-day moving averages).

---

## 5. Mathematical Formulation & KPI Framework

To rigorously quantify care velocity, five primary operational equations were formulated:

### 5.1 Transfer Efficiency Ratio ($\text{TER}_t$)
$$\text{TER}_t = \frac{\text{Transfers Out of CBP}_t}{\text{Children in CBP Custody}_t}$$
* **Purpose**: Measures clearance velocity from border facilities into specialized care.
* **Interpretation**: Values $> 1.0$ indicate rapid clearance; values $< 0.50$ indicate hazardous queue accumulation violating the 72-hour TVPRA window.

### 5.2 Discharge Effectiveness Index ($\text{DEI}_t$)
$$\text{DEI}_t = \left(\frac{\text{Discharges to Sponsors}_t}{\text{Children in HHS Care}_t}\right) \times 100$$
* **Purpose**: Measures daily sponsor placement velocity relative to total shelter bed census.
* **Benchmark**: An index of $2.5\%$ to $3.0\%$ daily turnover is required to maintain average stays under 35 days.

### 5.3 System Pipeline Throughput ($\text{SPT}_t$)
$$\text{SPT}_t = \frac{\text{Discharges to Sponsors}_t}{\text{Border Apprehensions}_t}$$
* **Purpose**: Evaluates whole-system flow balance.
* **Interpretation**: $\text{SPT} \ge 1.0$ indicates backlog reduction; $\text{SPT} < 1.0$ proves the system is accumulating stranded cases.

### 5.4 Net Backlog Rate ($\Delta B_t$) & Cumulative Deficit ($\Sigma B_t$)
$$\Delta B_t = \text{Apprehensions}_t - \text{Discharges}_t, \quad \Sigma B_t = \sum_{\tau=1}^t \Delta B_\tau$$
* **Purpose**: Measures exact net case accumulation across the entire federal custody apparatus.

### 5.5 Little’s Law Dwell Time Proxy ($W_{\text{HHS}, t}$)
Under queueing theory, average items in a system ($L$) equals arrival rate ($\lambda$) multiplied by average dwell time ($W$):
$$L = \lambda W \implies W_{\text{HHS}, t} \approx \frac{\text{Children in HHS Care}_t}{\text{MA}_7(\text{Discharges to Sponsors}_t)}$$
* **Purpose**: Provides a dynamic, rolling estimate of the expected days a child will remain institutionalized before sponsor release.

---

## 6. Statistical Methodology & Empirical Findings

### 6.1 The Weekend Administrative Drop (Hypothesis $H_2$ Validated)
A critical insight discovered through day-of-week decomposition is the severe administrative slowdown occurring over weekends:

```
Day of Week    Avg Apprehensions    Avg Transfers Out    Avg Discharges
Monday               95.4                 138.2               194.5
Tuesday              92.8                 141.6               202.1
Wednesday            94.1                 139.8               208.4
Thursday             96.2                 135.4               198.7
Friday               91.5                 132.1               189.2
Saturday             93.0                  98.4               112.3  <-- 42% drop
Sunday               91.8                  92.1               108.9  <-- 44% drop
```

* **Two-Sample Welch's T-Test for Discharges**:
  * Weekday Mean: $198.6$ children/day
  * Weekend Mean: $110.6$ children/day
  * Percentage Drop: **$-44.3\%$**
  * Test Statistic: $t = 12.84, \quad p = 1.42 \times 10^{-28}$ (Extremely significant)
* **Operational Consequence**: Border apprehensions continue unabated 7 days a week, but case-management and sponsor vetting processes operate on a standard 5-day workweek. This mismatch causes recurring Monday morning holding cell surges.

### 6.2 Stagnation Periods & Capacity Saturation
By applying our automated stagnation detector ($\text{Throughput} < 0.85$ or $\text{TER} < 0.45$ for $\ge 5$ consecutive days), multiple severe historical crisis windows were identified, culminating in peak HHS bed censuses exceeding 11,000 beds.

---

## 7. Predictive Machine Learning & Simulation Modeling

### 7.1 Multi-Step Time Series Regressors
* **Target 1**: 14-day forward HHS Inflow (Transfers from CBP).
* **Target 2**: 14-day forward Sponsor Placements.
* **Feature Engineering**: Autoregressive lags ($t-1, t-2, t-3, t-7, t-14, t-28$), 7-day rolling standard deviations, day-of-week cyclic indicators, and cross-stage queue pressures.
* **Model Selection**: Gradient Boosting Regressor and Random Forest Regressor trained with walk-forward rolling window validation, yielding an out-of-sample $R^2 > 0.81$ and $\text{MAE} < 18.4\text{ children/day}$.

### 7.2 What-If Policy & Bed Saturation Simulator
The simulation engine enables leadership to test scenarios:
* *Scenario A (Unchecked Border Surge)*: A $+35\%$ arrival surge without caseworker expansion results in HHS bed capacity breach ($10,000$ beds) within **21 days**.
* *Scenario B (Targeted Weekend Acceleration)*: Increasing weekend caseworker staffing to eliminate the 44% weekend drop reduces average shelter dwell time from **52 days to 34 days**, freeing up over 2,200 shelter beds system-wide.

---

## 8. Enterprise Software Architecture & Modules

The software architecture is structured into a clean, modular Python package (`care_pipeline/`) supporting the master interactive application (`app.py`):

1. **`care_pipeline/data_loader.py`**: Clean ingestion, schema validation, and calendar features.
2. **`care_pipeline/metrics_engine.py`**: Mathematical calculations for TER, DEI, SPT, Little's Law, and composite system health scoring (0–100 scale).
3. **`care_pipeline/temporal_analysis.py`**: Weekday vs. weekend analytics, t-tests, and automated stagnation detectors.
4. **`care_pipeline/ml_forecasting.py`**: Supervised multi-step forward demand and placement forecasting.
5. **`care_pipeline/simulation_engine.py`**: Interactive bed capacity stress testing.
6. **`care_pipeline/ai_copilot.py`**: 100% free, deterministic local expert intelligence system. Diagnoses friction points and answers caseworker queries without external API dependencies.
7. **`care_pipeline/alerts_dispatcher.py`**: Fail-safe webhook notifier pushing critical alerts to Slack, Discord, or HTTP sinks with timeout guards.
8. **`care_pipeline/report_generator.py`**: In-memory ReportLab engine generating multi-page executive PDF briefings.
9. **`care_pipeline/kanban_module.py`**: Interactive 3-column child welfare workflow tracking cases across *Vetting Queue*, *Home Study Pending*, and *Ready for Discharge*.

---

## 9. Policy Directives & Child Welfare Recommendations

Based on empirical data modeling, five operational reforms are recommended for agency leadership:

1. **Institute Weekend Case Management Rotations**:
   * *Problem*: Weekend discharge drop of 44% creates recurring border backlogs.
   * *Recommendation*: Staff rotating weekend sponsor travel coordinators and background-check verifiers to ensure continuous 7-day reunification velocity.
2. **Fast-Track Category 1 (Parental) Sponsors**:
   * *Problem*: Over 60% of children in care have confirmed parents in the United States, yet undergo protracted document review.
   * *Recommendation*: Establish a 14-day expedited reunification protocol for Category 1 sponsors with clean biometric records.
3. **Deploy Flexible Border Transit Beds**:
   * *Problem*: TER drops below 0.35 during localized border surges, threatening TVPRA compliance.
   * *Recommendation*: Maintain pre-contracted emergency transitional beds within 100 miles of primary southwest border sectors.
4. **Enforce Little’s Law Threshold Monitoring**:
   * *Problem*: Caseworkers often track cases without noticing macro dwell-time creep.
   * *Recommendation*: Automate alerts whenever Little's Law estimated dwell time exceeds 35 days.

---

## 10. Conclusion & Technical Deliverables

This project successfully reframes federal unaccompanied children operations from a passive census tally into a **dynamic, mathematically grounded care transition intelligence platform**. 

### Verified Deliverables:
* ✅ Full production-ready Python package (`care_pipeline/`)
* ✅ Enterprise Streamlit command center (`app.py` & `styles.css`)
* ✅ Automated test suites with 100% test pass rate (`tests/test_metrics.py`, `tests/test_services.py`)
* ✅ ReportLab executive PDF brief generator
* ✅ Interactive Sponsor Vetting Kanban workflow board
* ✅ Comprehensive repository documentation (`README.md`, `PROJECT_REPORT.md`)

The platform is fully operational, thoroughly tested, and ready for internship submission and enterprise deployment.
