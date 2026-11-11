"""
app.py
Care Transition Efficiency & Placement Outcome Analytics
Production Executive Web Application | UAC Care Operations & Policy Intelligence

Frontend Architecture:
- Human-crafted warm off-white & buttercream-to-lime aesthetic (#FEFEEF -> #EFF6D2)
  with crisp white cards (#FFFFFF), sage borders (#DCE7CA), and deep forest green accents
- Full-stack integration consuming FastAPI REST endpoints (/api/v1/*) with resilient
  local database & analytical engine fallback on disconnection
- Six modular executive tabs:
  1. Executive Overview & KPI Strip (+ AI Copilot & ReportLab PDF Brief)
  2. Care Pipeline Flow Visualization (Plotly Sankey & Multi-Stage Area Chart)
  3. Transfer & Discharge Efficiency Panels (Dual-Axis Divergence, Welch's t-Test & Stability)
  4. Bottleneck Detection, Heatmaps & Sponsor Kanban (Congestion Matrix, Z-Scores & Case Board)
  5. Interactive Policy Simulator (Live Backlog Dissipation Curves via REST / Local Engine)
  6. ML Forecast & Case Risk Lab (30/90d Fourier Forecast, Feature Importance & Stagnation Scorer)
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Internal pipeline and analytics modules
from care_pipeline.data_loader import load_and_clean_data, load_data_from_db, load_case_events_from_db
from care_pipeline.metrics_engine import compute_pipeline_metrics, get_current_kpis
from care_pipeline.temporal_analysis import (
    analyze_day_of_week,
    analyze_monthly_trends,
    detect_stagnation_periods,
)
from care_pipeline.ml_forecasting import forecast_demand_and_placement
from care_pipeline.simulation_engine import run_stress_simulation
from care_pipeline.ai_copilot import PipelineCopilot
from care_pipeline.report_generator import generate_pdf_report
from care_pipeline.alerts_dispatcher import dispatch_webhook_alert
from care_pipeline.kanban_module import render_kanban_board
from core.analytics import detect_backlog_anomalies
from ml.models import (
    TimeSeriesInflowForecaster,
    ReunificationDelayClassifier,
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_FORECASTER_PATH,
)

# --- 1. STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Care Transition Efficiency & Placement Outcome Analytics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. INJECT HUMAN-CRAFTED WARM OFF-WHITE THEME CSS ---
css_path = Path(__file__).resolve().parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- 3. BACKEND REST API CLIENT & RESILIENT LOCAL ENGINE FALLBACK ---
def _resolve_backend_url() -> str:
    """Resolves FastAPI URL from environment or Streamlit Cloud Secrets safely."""
    env_url = os.getenv("BACKEND_API_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    try:
        secret_url = st.secrets.get("BACKEND_API_URL", "")
        if secret_url:
            return str(secret_url).strip().rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:8000"


BACKEND_API_URL = _resolve_backend_url()


@st.cache_data(ttl=45)
def check_backend_connection() -> Tuple[bool, str]:
    """Probes the FastAPI REST API health endpoint with non-blocking timeout."""
    try:
        resp = requests.get(f"{BACKEND_API_URL}/", timeout=1.5)
        if resp.status_code == 200:
            return True, "FastAPI REST Service (Live & Connected)"
    except Exception:
        pass
    return False, "Local Relational Database Engine (Fallback Active)"


@st.cache_data(ttl=60)
def api_fetch_bottlenecks(is_online: bool) -> Optional[Dict[str, Any]]:
    """Queries GET /api/v1/bottlenecks when backend REST API is active."""
    if not is_online:
        return None
    try:
        resp = requests.get(f"{BACKEND_API_URL}/api/v1/bottlenecks", timeout=3.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def api_post_simulation(
    transfer_mult: float,
    capacity_limit: int,
    surge_pct: float,
    sim_days: int,
) -> Optional[Dict[str, Any]]:
    """Executes POST /api/v1/simulate when backend REST API is active."""
    if not api_online:
        return None
    try:
        payload = {
            "transfer_velocity_multiplier": float(transfer_mult),
            "capacity_adjustment": int(capacity_limit),
            "apprehension_surge_pct": float(surge_pct),
            "sim_days": int(sim_days),
        }
        resp = requests.post(f"{BACKEND_API_URL}/api/v1/simulate", json=payload, timeout=4.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=120)
def api_fetch_ml_forecast(horizon_days: int, is_online: bool) -> Optional[Dict[str, Any]]:
    """Queries GET /api/v1/ml/forecast when backend REST API is active."""
    if not is_online:
        return None
    try:
        resp = requests.get(
            f"{BACKEND_API_URL}/api/v1/ml/forecast",
            params={"horizon_days": int(horizon_days)},
            timeout=5.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=120)
def build_cached_pdf_brief(df_len: int, last_date_str: str, _filtered_df: pd.DataFrame, kpis_dict: Dict[str, Any], audit_dict: Dict[str, Any]) -> bytes:
    """Caches PDF compilation keyed on data span to keep slider updates instantaneous."""
    try:
        fc_preview = forecast_demand_and_placement(_filtered_df, days_ahead=14)
    except Exception:
        fc_preview = None
    return generate_pdf_report({**kpis_dict, **audit_dict}, fc_preview)


def api_predict_case_risk(case_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Executes POST /api/v1/ml/predict-delay when backend REST API is active."""
    if not api_online:
        return None
    try:
        resp = requests.post(
            f"{BACKEND_API_URL}/api/v1/ml/predict-delay",
            json=case_payload,
            timeout=3.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=120)
def load_master_dataset(data_source: str = "Database") -> pd.DataFrame:
    """Ingests dataset from REST timeseries, local SQLite/PostgreSQL, or CSV archive."""
    df: Optional[pd.DataFrame] = None
    if "Live Database" in data_source:
        try:
            df = load_data_from_db()
        except Exception:
            df = None

    if df is None or df.empty:
        df = load_and_clean_data()

    return compute_pipeline_metrics(df)


@st.cache_resource
def get_trained_forecaster(_train_df: pd.DataFrame) -> TimeSeriesInflowForecaster:
    """Loads serialized Fourier forecaster with automatic re-fit fallback across sklearn versions."""
    if DEFAULT_FORECASTER_PATH.exists():
        try:
            return TimeSeriesInflowForecaster.load()
        except Exception:
            pass
    model = TimeSeriesInflowForecaster().fit(_train_df)
    try:
        model.save()
    except Exception:
        pass
    return model


@st.cache_resource
def get_trained_classifier() -> ReunificationDelayClassifier:
    """Loads serialized Random Forest classifier with automatic re-fit fallback across sklearn versions."""
    if DEFAULT_CLASSIFIER_PATH.exists():
        try:
            return ReunificationDelayClassifier.load()
        except Exception:
            pass
    try:
        cases_train = load_case_events_from_db(limit=5000)
    except Exception:
        cases_train = pd.DataFrame()
    if cases_train.empty:
        from utils.data_generator import generate_macro_pipeline_data, generate_micro_case_events

        macro_fallback = generate_macro_pipeline_data(num_days=365, seed=42)
        cases_train = generate_micro_case_events(macro_fallback, max_cases_per_day=15, seed=42)
    model = ReunificationDelayClassifier().fit(cases_train)
    try:
        model.save()
    except Exception:
        pass
    return model


@st.cache_data(ttl=120)
def load_micro_cases(facility_filter: Optional[str] = None) -> pd.DataFrame:
    """Loads granular case events for facility cluster analytics and classifier training."""
    try:
        cases = load_case_events_from_db(limit=5000)
        if facility_filter and facility_filter != "All Facilities":
            cases = cases[cases["processing_facility_id"] == facility_filter]
        return cases
    except Exception:
        return pd.DataFrame()


def style_plotly(fig: go.Figure, height: int = 320, y_title: Optional[str] = None) -> go.Figure:
    """Applies unified human-crafted off-white / forest-green Plotly visual polish."""
    layout_kwargs: Dict[str, Any] = dict(
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#173C19"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=12, r=12, t=28, b=24),
        xaxis=dict(gridcolor="#E9F0DA", showline=True, linecolor="#DCE7CA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    if y_title:
        layout_kwargs["yaxis"] = dict(
            title=y_title, gridcolor="#E9F0DA", showline=True, linecolor="#DCE7CA"
        )
    else:
        layout_kwargs["yaxis"] = dict(gridcolor="#E9F0DA", showline=True, linecolor="#DCE7CA")
    fig.update_layout(**layout_kwargs)
    return fig


# Initialize shared state defaults
api_online, backend_status_label = check_backend_connection()
if "sim_surge" not in st.session_state:
    st.session_state["sim_surge"] = 20
if "sim_vetting" not in st.session_state:
    st.session_state["sim_vetting"] = 15
if "sim_transfer_mult" not in st.session_state:
    st.session_state["sim_transfer_mult"] = 1.15
if "sim_days" not in st.session_state:
    st.session_state["sim_days"] = 30

# --- 4. SIDEBAR OPERATIONAL CONTROLS ---
with st.sidebar:
    st.markdown("### 🎛️ Analytics Console")
    st.caption("Federal Child Welfare Operations Parameter Management")
    st.markdown("---")

    # Backend Telemetry Pill Card
    status_bg = "#EAF7ED" if api_online else "#F9FCF2"
    status_dot = "#196C2E" if api_online else "#447948"
    st.markdown(
        f"""
        <div style="background:{status_bg}; border:1px solid #DCE7CA; border-radius:12px; padding:10px 14px; margin-bottom:14px;">
            <div style="font-size:10px; font-weight:700; color:#447948; text-transform:uppercase; letter-spacing:0.08em;">Backend Telemetry</div>
            <div style="font-size:12px; font-weight:700; color:#143311; display:flex; align-items:center; gap:6px; margin-top:3px;">
                <span style="color:{status_dot}; font-size:14px;">●</span> {backend_status_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh Backend Probe", width="stretch"):
        check_backend_connection.clear()
        st.rerun()

    st.markdown("#### 🗄️ Data Repository")
    data_source = st.selectbox(
        "Active Pipeline Source",
        ["Live Database (3-Year Longitudinal Records)", "Historical Federal Archive (CSV File)"],
        index=0,
    )

    raw_df = load_master_dataset(data_source)
    min_date = raw_df["Date"].min().to_pydatetime()
    max_date = raw_df["Date"].max().to_pydatetime()

    st.markdown("#### 🏢 Border Facility / Sector")
    facilities = [
        "All Facilities",
        "CBP-ELP-01",
        "CBP-RGV-02",
        "CBP-TUS-03",
        "CBP-DRT-04",
        "CBP-SDG-05",
        "CBP-YUM-06",
    ]
    facility_labels = {
        "All Facilities": "All Border Facilities (System-Wide)",
        "CBP-ELP-01": "CBP-ELP-01 (El Paso Sector)",
        "CBP-RGV-02": "CBP-RGV-02 (Rio Grande Valley)",
        "CBP-TUS-03": "CBP-TUS-03 (Tucson Sector)",
        "CBP-DRT-04": "CBP-DRT-04 (Del Rio Sector)",
        "CBP-SDG-05": "CBP-SDG-05 (San Diego Sector)",
        "CBP-YUM-06": "CBP-YUM-06 (Yuma Sector)",
    }
    selected_facility = st.selectbox(
        "Sector Routing",
        facilities,
        format_func=lambda x: facility_labels.get(x, x),
        index=0,
    )

    st.markdown("#### 📅 Reporting Span")
    horizon = st.radio(
        "Observation Horizon",
        [
            "Full Multi-Year Census",
            "Recent 90-Day Operational Window",
            "Recent 30-Day Critical Window",
            "Custom Date Range",
        ],
        index=0,
    )

    if horizon == "Recent 30-Day Critical Window":
        start_date = max_date - timedelta(days=30)
        end_date = max_date
    elif horizon == "Recent 90-Day Operational Window":
        start_date = max_date - timedelta(days=90)
        end_date = max_date
    elif horizon == "Custom Date Range":
        d_range = st.date_input(
            "Select Span", value=(min_date, max_date), min_value=min_date, max_value=max_date
        )
        if isinstance(d_range, (tuple, list)) and len(d_range) == 2:
            start_date, end_date = pd.to_datetime(d_range[0]), pd.to_datetime(d_range[1])
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date, end_date = min_date, max_date

    st.markdown("---")
    st.markdown("#### 📈 Metric Ratio Toggles & Smoothing")
    smoothing_option = st.selectbox(
        "Moving Window Averaging",
        [
            "7-Day Moving Average (Standard)",
            "Raw Daily Operational Observations",
            "30-Day Macro Trend Horizon",
        ],
        index=0,
    )
    show_ter_panel = st.checkbox("Show Transfer Efficiency (TER)", value=True)
    show_dei_panel = st.checkbox("Show Discharge Effectiveness (DEI)", value=True)
    show_ptr_panel = st.checkbox("Show Pipeline Throughput (PTR)", value=True)
    show_stability_panel = st.checkbox("Show Outcome Stability Score", value=True)

    st.markdown("#### 🎯 Statutory Benchmarks & Policy Speed")
    ter_warn_thresh = st.slider(
        "TER Compliance Target", 0.30, 1.20, 0.85, 0.05, help="TVPRA 72-Hour Compliance Benchmark"
    )
    bed_capacity_limit = st.slider("HHS Shelter Capacity Limit", 5000, 20000, 10000, 500)
    sidebar_vetting_speed = st.slider(
        "Policy Simulator: Vetting Speed (%)",
        -30,
        60,
        key="sim_vetting",
        step=5,
        help="Adjusts sponsor vetting velocity in Tab 5 Policy Simulator",
    )

# Apply Date Filtering
filtered_df = raw_df[
    (raw_df["Date"] >= pd.Timestamp(start_date)) & (raw_df["Date"] <= pd.Timestamp(end_date))
].copy()
if filtered_df.empty:
    st.warning("Selected date range returned no records. Automatically resetting to full reporting window.")
    filtered_df = raw_df.copy()

kpis = get_current_kpis(filtered_df)
copilot = PipelineCopilot(filtered_df)
audit_summary = copilot.generate_audit_summary()

# --- 5. EXECUTIVE HERO HEADER & REFERENCE ANNOUNCEMENT CARD ---
curr_ter = kpis["ter_ma7"]
curr_spt = kpis["spt_current"]
if curr_ter < ter_warn_thresh:
    ann_label = "ANNOUNCEMENT · STAGE 1 LATENCY ALERT"
    ann_text = (
        f"Transfer Efficiency ({curr_ter:.2f}) trails statutory benchmark ({ter_warn_thresh:.2f}). "
        f"Active CBP holding sits at {kpis['cbp_custody']:,} minors."
    )
    ann_bar_class = "is-critical"
    ann_bar_text = "Action Required: Prioritize Stage 1 Transfers"
    ann_badge = "badge-critical"
elif curr_spt < 0.85:
    ann_label = "ANNOUNCEMENT · ELEVATED SHELTER ACCUMULATION"
    ann_text = (
        f"Pipeline Throughput ({curr_spt:.2f}) shows sponsor discharges trailing arrivals. "
        f"Estimated shelter dwell time is {kpis['estimated_dwell_days']} days."
    )
    ann_bar_class = "is-warning"
    ann_bar_text = "Monitor Shelter Utilization & Vetting Velocity"
    ann_badge = "badge-warning"
else:
    ann_label = "ANNOUNCEMENT · NOMINAL PIPELINE VELOCITY"
    ann_text = (
        f"Care transitions and sponsor placements are balanced (TER: {curr_ter:.2f} | "
        f"Throughput: {curr_spt:.2f}). Stability Index at {kpis['stability_score']:.2f}."
    )
    ann_bar_class = ""
    ann_bar_text = "TVPRA 72-Hour & Placement Targets On Track"
    ann_badge = "badge-optimal"

hero_grade_short = kpis["health_grade"].split()[0]

st.markdown(
    f"""
    <div class="hero-wrapper">
        <div class="hero-header">
            <div class="hero-eyebrow">U.S. Department of Health &amp; Human Services · ORR &amp; CBP Pipeline Intelligence</div>
            <h1 class="hero-title">
                Care Transition Efficiency &amp;
                <span class="hero-underline-wrap">
                    Placement Outcome Analytics
                    <svg class="hero-underline-svg" viewBox="0 0 280 10" preserveAspectRatio="none" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2 7C68 2 185 2 278 6" stroke="#286B34" stroke-width="2.6" stroke-linecap="round"/>
                    </svg>
                </span>
            </h1>
            <p class="hero-subtitle">
                End-to-end statistical process control, Little's Law dwell-time estimation, and machine learning risk telemetry
                across federal border custody and licensed sponsor reunification pipelines.
            </p>
            <div class="hero-meta-strip">
                <span class="pill-badge {ann_badge}">● Health Grade: {hero_grade_short} ({kpis['health_score']:.1f}/100)</span>
                <span class="hero-reg-tag">TVPRA 72-HR STANDARD &bull; UPDATED {kpis['as_of_date'].upper()} &bull; {len(filtered_df):,} DAYS IN VIEW</span>
            </div>
        </div>
        <div class="announcement-card">
            <div class="announcement-top">
                <div class="announcement-icon-badge">🌱</div>
                <div>
                    <div class="announcement-label">{ann_label}</div>
                    <div class="announcement-text">{ann_text}</div>
                </div>
            </div>
            <div class="announcement-footer-bar {ann_bar_class}">
                {ann_bar_text}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-divider"><span>MULTI-STAGE CARE PIPELINE COMMAND CENTER</span></div>',
    unsafe_allow_html=True,
)

# --- 6. MODULAR TAB ARCHITECTURE ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Overview & KPI Strip",
    "🌊 Care Pipeline Flow Visualization",
    "⚡ Transfer & Discharge Efficiency",
    "⚠️ Bottleneck Heatmaps & Kanban",
    "🎛️ Interactive Policy Simulator",
    "🔮 ML Forecast & Case Risk Lab",
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & KPI STRIP
# ==============================================================================
with tab1:
    st.markdown('<div class="section-divider"><span>CORE PROCESS KPI SCORECARD</span></div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title"><span>Transfer Efficiency</span><span>⚡</span></div>
                <div class="kpi-value">{kpis['ter_ma7']:.2f}</div>
                <div class="kpi-delta {'kpi-delta-positive' if kpis['ter_delta'] >= 0 else 'kpi-delta-negative'}">
                    {'▲' if kpis['ter_delta'] >= 0 else '▼'} {kpis['ter_delta']:+0.2f} (30d)
                </div>
                <div class="kpi-benchmark">Transfers ÷ CBP | Goal ≥ {ter_warn_thresh:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title"><span>Discharge Index</span><span>🏡</span></div>
                <div class="kpi-value">{kpis['dei_current_pct']:.2f}%</div>
                <div class="kpi-delta {'kpi-delta-positive' if kpis['dei_delta'] >= 0 else 'kpi-delta-negative'}">
                    {'▲' if kpis['dei_delta'] >= 0 else '▼'} {kpis['dei_delta']:+0.2f}% (30d)
                </div>
                <div class="kpi-benchmark">Exits ÷ Shelter Care | Goal ≥ 2.5%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title"><span>Pipeline Throughput</span><span>⚖️</span></div>
                <div class="kpi-value">{kpis['spt_current']:.2f}</div>
                <div class="kpi-delta {'kpi-delta-positive' if kpis['spt_delta'] >= 0 else 'kpi-delta-negative'}">
                    {'▲' if kpis['spt_delta'] >= 0 else '▼'} {kpis['spt_delta']:+0.2f} (30d)
                </div>
                <div class="kpi-benchmark">Exits ÷ Intake | ≥ 1.0 Clearing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        dwell_good = kpis["dwell_delta"] <= 0
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title"><span>Little's Dwell Time</span><span>⏳</span></div>
                <div class="kpi-value">{kpis['estimated_dwell_days']} <span style="font-size:15px;">d</span></div>
                <div class="kpi-delta {'kpi-delta-positive' if dwell_good else 'kpi-delta-negative'}">
                    {'▼' if dwell_good else '▲'} {abs(kpis['dwell_delta']):0.1f}d (30d)
                </div>
                <div class="kpi-benchmark">Mean Shelter Duration | Goal ≤ 30d</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        stab_score = kpis["stability_score"]
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title"><span>Outcome Stability</span><span>📐</span></div>
                <div class="kpi-value">{stab_score:.2f}</div>
                <div class="kpi-delta {'kpi-delta-positive' if stab_score >= 0.70 else 'kpi-delta-negative'}">
                    {'● High Consistency' if stab_score >= 0.70 else '● Volatile Releases'}
                </div>
                <div class="kpi-benchmark">1 − CV(30d Exits) | Goal ≥ 0.70</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k6:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title"><span>HHS Census &amp; Backlog</span><span>📦</span></div>
                <div class="kpi-value">{kpis['hhs_care']:,}</div>
                <div class="kpi-delta kpi-delta-neutral">
                    Net Daily: {kpis['daily_backlog']:+d}
                </div>
                <div class="kpi-benchmark">Cumulative Δ: {kpis['cumulative_backlog']:+,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    ov_left, ov_right = st.columns([2.2, 1])

    with ov_left:
        with st.container(border=True):
            st.markdown(
                '<div class="surface-title">Executive Diagnostic Brief &amp; Local AI Operations Copilot</div>'
                '<div class="surface-subtitle">100% deterministic, zero-cost diagnostic audit generated directly from pipeline telemetry.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(audit_summary.get("briefing_markdown", "System operations running nominally."))
            st.markdown("---")
            st.markdown("**Interactive Operations Copilot Q&A**")
            q_col1, q_col2 = st.columns([3, 1])
            with q_col1:
                copilot_query = st.text_input(
                    "Ask a diagnostic question",
                    value="Where is the primary bottleneck right now and why do Monday surges occur?",
                    label_visibility="collapsed",
                )
            with q_col2:
                ask_clicked = st.button("Ask Copilot", width="stretch")

            if copilot_query or ask_clicked:
                st.info(copilot.ask_copilot(copilot_query))

    with ov_right:
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center; padding: 6px 0 12px 0;">
                    <div style="font-size: 32px; margin-bottom: 6px;">📑</div>
                    <div style="font-size: 16px; font-weight: 700; color: #143311; margin-bottom: 4px;">Executive Briefing Pack</div>
                    <p style="font-size: 12.5px; color: #485A4A; margin: 0;">
                        Compile an executive ReportLab PDF briefing containing live KPIs, findings, and 10-day ML bed projections.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            pdf_bytes = build_cached_pdf_brief(
                len(filtered_df), str(filtered_df["Date"].max()), filtered_df, kpis, audit_summary
            )
            st.download_button(
                label="📥 Download Executive PDF Brief",
                data=pdf_bytes,
                file_name=f"HHS_UAC_Executive_Brief_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                width="stretch",
            )

            st.markdown("---")
            st.markdown("**Snapshot Summary**")
            st.metric("Current CBP Border Queue", f"{kpis['cbp_custody']:,} minors")
            st.metric("Daily Apprehensions", f"{kpis['apprehensions']:,} children/day")
            st.metric("Daily Sponsor Discharges", f"{kpis['discharges']:,} children/day")

# ==============================================================================
# TAB 2: CARE PIPELINE FLOW VISUALIZATION
# ==============================================================================
with tab2:
    with st.container(border=True):
        st.markdown(
            '<div class="surface-title">🌊 Inter-Agency Multi-Stage Care &amp; Custody Flow Pipeline</div>'
            '<div class="surface-subtitle">Volume conservation tracking from border apprehension through CBP holding, HHS licensed sheltering, and vetted sponsor placement.</div>',
            unsafe_allow_html=True,
        )

        tot_app = int(filtered_df["Apprehensions"].sum())
        tot_trans = int(filtered_df["Transfers_Out_CBP"].sum())
        tot_disc = int(filtered_df["Discharges_HHS"].sum())
        curr_cbp = int(filtered_df["CBP_Custody"].iloc[-1])
        curr_hhs = int(filtered_df["HHS_Care"].iloc[-1])

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Cumulative Apprehensions", f"{tot_app:,}")
        s2.metric("CBP → HHS Transfers", f"{tot_trans:,}")
        s3.metric("Sponsor Reunifications", f"{tot_disc:,}")
        s4.metric("Active Federal Census", f"{curr_cbp + curr_hhs:,}")

        sankey_fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=22,
                        thickness=28,
                        line=dict(color="#DCE7CA", width=1.5),
                        label=[
                            f"Border Encounters<br>({tot_app:,})",
                            f"CBP Custody Holding<br>(Active: {curr_cbp:,})",
                            f"Transfers to HHS Care<br>({tot_trans:,})",
                            f"HHS Licensed Shelters<br>(Active: {curr_hhs:,})",
                            f"Sponsor Reunification<br>({tot_disc:,})",
                        ],
                        color=["#286B34", "#D97706", "#447948", "#143311", "#10B981"],
                    ),
                    link=dict(
                        source=[0, 1, 2, 3],
                        target=[1, 2, 3, 4],
                        value=[max(1, tot_app), max(1, tot_trans), max(1, tot_trans), max(1, tot_disc)],
                        color=[
                            "rgba(40, 107, 52, 0.25)",
                            "rgba(217, 119, 6, 0.25)",
                            "rgba(68, 121, 72, 0.25)",
                            "rgba(16, 185, 129, 0.28)",
                        ],
                    ),
                )
            ]
        )
        sankey_fig.update_layout(
            font=dict(family="Plus Jakarta Sans, sans-serif", size=13, color="#173C19"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=370,
            margin=dict(l=15, r=15, t=20, b=20),
        )
        st.plotly_chart(sankey_fig, width="stretch")

    with st.container(border=True):
        st.markdown(
            '<div class="surface-title">Multi-Stage Active Careload Census Trajectory</div>'
            '<div class="surface-subtitle">Continuous active minor occupancy across Stage 1 CBP holding and Stage 2 HHS shelters against bed capacity limits.</div>',
            unsafe_allow_html=True,
        )
        area_fig = go.Figure()
        area_fig.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=filtered_df["HHS_Care"],
                name="HHS Licensed Shelter Census",
                line=dict(color="#143311", width=2.6),
                fill="tozeroy",
                fillcolor="rgba(40, 107, 52, 0.14)",
            )
        )
        area_fig.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=filtered_df["CBP_Custody"],
                name="CBP Border Station Custody",
                line=dict(color="#D97706", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(217, 119, 6, 0.18)",
            )
        )
        area_fig.add_hline(
            y=bed_capacity_limit,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text=f"Shelter Capacity Target ({bed_capacity_limit:,})",
            annotation_position="top left",
        )
        st.plotly_chart(style_plotly(area_fig, height=340, y_title="Active Minor Census"), width="stretch")

# ==============================================================================
# TAB 3: TRANSFER & DISCHARGE EFFICIENCY PANELS
# ==============================================================================
with tab3:
    with st.container(border=True):
        st.markdown(
            '<div class="surface-title">⚡ Inflow vs. Outflow Divergence &amp; Process Ratios</div>'
            '<div class="surface-subtitle">Contrasting border apprehension inflows against successful sponsor discharges across selected smoothing horizons.</div>',
            unsafe_allow_html=True,
        )

        app_series = (
            filtered_df["Apprehensions_MA7"]
            if "7-Day" in smoothing_option
            else (filtered_df["Apprehensions_MA30"] if "30-Day" in smoothing_option else filtered_df["Apprehensions"])
        )
        disc_series = (
            filtered_df["Discharges_HHS_MA7"]
            if "7-Day" in smoothing_option
            else (filtered_df["Discharges_HHS_MA30"] if "30-Day" in smoothing_option else filtered_df["Discharges_HHS"])
        )

        div_fig = go.Figure()
        div_fig.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=app_series,
                name="Border Apprehensions (Inflow)",
                line=dict(color="#DC2626", width=2.3),
            )
        )
        div_fig.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=disc_series,
                name="Sponsor Discharges (Outflow)",
                line=dict(color="#196C2E", width=2.3),
                fill="tonexty",
                fillcolor="rgba(220, 38, 38, 0.08)",
            )
        )
        st.plotly_chart(style_plotly(div_fig, height=320, y_title="Children / Day"), width="stretch")

    # Toggled Metric Ratio Charts
    ratio_charts: List[Tuple[str, go.Figure]] = []
    if show_ter_panel:
        ter_series = (
            filtered_df["TER_MA7"]
            if "7-Day" in smoothing_option
            else (
                filtered_df["TER_MA30"]
                if "30-Day" in smoothing_option
                else filtered_df["Transfer_Efficiency_Ratio"]
            )
        )
        fig_ter = go.Figure()
        fig_ter.add_trace(
            go.Scatter(x=filtered_df["Date"], y=ter_series, name="TER", line=dict(color="#286B34", width=2.4))
        )
        fig_ter.add_hline(
            y=ter_warn_thresh,
            line_dash="dash",
            line_color="#D97706",
            annotation_text=f"TVPRA Target ({ter_warn_thresh:.2f})",
            annotation_position="top left",
        )
        ratio_charts.append(("Transfer Efficiency Ratio (TER)", style_plotly(fig_ter, height=290, y_title="Transfers ÷ CBP Custody")))

    if show_dei_panel:
        dei_series = (
            filtered_df["DEI_MA7"] * 100
            if "7-Day" in smoothing_option
            else filtered_df["Discharge_Effectiveness_Pct"]
        )
        fig_dei = go.Figure()
        fig_dei.add_trace(
            go.Scatter(x=filtered_df["Date"], y=dei_series, name="DEI (%)", line=dict(color="#0D9488", width=2.4))
        )
        fig_dei.add_hline(
            y=2.5,
            line_dash="dash",
            line_color="#D97706",
            annotation_text="Sustainable Turnover (2.5%)",
            annotation_position="top left",
        )
        ratio_charts.append(("Discharge Effectiveness Index (DEI %)", style_plotly(fig_dei, height=290, y_title="Daily Discharges / HHS Care (%)")))

    if show_ptr_panel:
        fig_ptr = go.Figure()
        fig_ptr.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=filtered_df["Pipeline_Throughput_MA7"],
                name="PTR (7d)",
                line=dict(color="#143311", width=2.4),
            )
        )
        fig_ptr.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#196C2E",
            annotation_text="Equilibrium (1.00)",
            annotation_position="top left",
        )
        ratio_charts.append(("Pipeline Throughput Rate (PTR 7-Day)", style_plotly(fig_ptr, height=290, y_title="Discharges ÷ Inflow")))

    if show_stability_panel:
        fig_stab = go.Figure()
        fig_stab.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=filtered_df["Outcome_Stability_Score"],
                name="Outcome Stability",
                line=dict(color="#447948", width=2.4),
                fill="tozeroy",
                fillcolor="rgba(82, 155, 87, 0.14)",
            )
        )
        fig_stab.add_hline(
            y=0.70,
            line_dash="dash",
            line_color="#D97706",
            annotation_text="Stability Benchmark (0.70)",
            annotation_position="bottom left",
        )
        ratio_charts.append(("Outcome Stability Score (1 − CV)", style_plotly(fig_stab, height=290, y_title="Consistency Score (0–1)")))

    for i in range(0, len(ratio_charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(ratio_charts):
                title, fig = ratio_charts[i + j]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown(
            '<div class="surface-title">Weekday vs. Weekend Transition Velocity (Administrative Latency Proof)</div>'
            '<div class="surface-subtitle">Empirical comparison of Monday–Friday case processing speeds against Saturday–Sunday administrative slowdowns.</div>',
            unsafe_allow_html=True,
        )
        dow_res = analyze_day_of_week(filtered_df)
        dow_table = dow_res["dow_table"]
        disc_test = dow_res["discharges_test"]
        trans_test = dow_res["transfers_test"]

        wt1, wt2, wt3, wt4 = st.columns(4)
        wt1.metric("Weekday Mean Exits", f"{disc_test['weekday_mean']:.1f} / day")
        wt2.metric("Weekend Mean Exits", f"{disc_test['weekend_mean']:.1f} / day", delta=f"{disc_test['pct_change']}%")
        wt3.metric("Weekend Transfer Shift", f"{trans_test['pct_change']}%", delta=f"t = {trans_test['t_statistic']}")
        wt4.metric("Welch's t-Test (Exits)", f"t = {disc_test['t_statistic']}", delta=f"p = {disc_test['p_value']}")

        dow_fig = go.Figure()
        dow_fig.add_trace(
            go.Bar(
                x=dow_table["DayOfWeek"],
                y=dow_table["avg_discharges"],
                name="Mean Sponsor Discharges",
                marker_color=[
                    "#143311" if d not in ["Saturday", "Sunday"] else "#DC2626"
                    for d in dow_table["DayOfWeek"]
                ],
            )
        )
        dow_fig.add_trace(
            go.Bar(
                x=dow_table["DayOfWeek"],
                y=dow_table["avg_transfers"],
                name="Mean CBP → HHS Transfers",
                marker_color=[
                    "#529B57" if d not in ["Saturday", "Sunday"] else "#D97706"
                    for d in dow_table["DayOfWeek"]
                ],
            )
        )
        dow_fig = style_plotly(dow_fig, height=310, y_title="Mean Daily Children Moved")
        dow_fig.update_layout(barmode="group")
        st.plotly_chart(dow_fig, width="stretch")

        with st.expander("📅 Month-over-Month Pipeline Placement Trends"):
            mom_df = analyze_monthly_trends(filtered_df)
            st.dataframe(mom_df, width="stretch", hide_index=True)

# ==============================================================================
# TAB 4: BOTTLENECK DETECTION, HEATMAPS & SPONSOR KANBAN
# ==============================================================================
with tab4:
    with st.container(border=True):
        st.markdown(
            '<div class="surface-title">⚠️ Congestion Heatmap Matrix &amp; Rolling Z-Score Anomalies</div>'
            '<div class="surface-subtitle">Pinpointing seasonal/weekly net backlog congestion windows and automated statistical process control breaches (Z &gt; 2.0).</div>',
            unsafe_allow_html=True,
        )

        heat_df = filtered_df.copy()
        heat_df["DayOfWeek"] = heat_df["Date"].dt.day_name()
        heat_df["MonthPeriod"] = heat_df["Date"].dt.strftime("%Y-%m")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot_heat = heat_df.pivot_table(
            index="DayOfWeek", columns="MonthPeriod", values="Daily_Net_Backlog", aggfunc="mean"
        ).reindex(dow_order)

        heat_fig = px.imshow(
            pivot_heat,
            labels=dict(x="Reporting Month", y="Day of Week", color="Net Backlog Δ"),
            color_continuous_scale=[[0.0, "#EAF7ED"], [0.5, "#FEF9EB"], [1.0, "#DC2626"]],
            aspect="auto",
        )
        heat_fig.update_layout(
            font=dict(family="Plus Jakarta Sans, sans-serif", color="#173C19"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            height=300,
            margin=dict(l=12, r=12, t=20, b=20),
        )
        st.plotly_chart(heat_fig, width="stretch")

        # Rolling Z-score anomaly chart
        anomalies_df = detect_backlog_anomalies(filtered_df, z_threshold=2.0)
        z_fig = go.Figure()
        z_fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(anomalies_df["date"]),
                y=anomalies_df["z_score"],
                name="Rolling Backlog Z-Score",
                line=dict(color="#286B34", width=2.2),
            )
        )
        z_fig.add_hline(
            y=2.0,
            line_dash="dash",
            line_color="#DC2626",
            annotation_text="Anomaly Threshold (Z = +2.0)",
            annotation_position="top left",
        )

        flagged = anomalies_df[anomalies_df["is_anomaly"] == 1]
        if not flagged.empty:
            z_fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(flagged["date"]),
                    y=flagged["z_score"],
                    mode="markers",
                    name="Congestion Anomaly Breach",
                    marker=dict(color="#DC2626", size=8, symbol="triangle-up"),
                )
            )
        st.plotly_chart(style_plotly(z_fig, height=290, y_title="Z-Score (Std Deviations)"), width="stretch")

        if not flagged.empty:
            st.markdown(f"**Identified Congestion Anomaly Incidents ({len(flagged)} Dates Flagged)**")
            st.dataframe(
                flagged[[
                    "date",
                    "backlog_accumulation_rate",
                    "baseline_mean_30d",
                    "threshold_upper_bound",
                    "z_score",
                    "severity",
                ]].rename(
                    columns={
                        "date": "Anomaly Date",
                        "backlog_accumulation_rate": "7-Day Net Inflow",
                        "baseline_mean_30d": "30-Day Baseline Mean",
                        "threshold_upper_bound": "Upper Bound (μ+2σ)",
                        "z_score": "Z-Score",
                        "severity": "Severity Grade",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Zero Z-score congestion anomalies detected within the selected reporting window.")

    with st.container(border=True):
        st.markdown(
            '<div class="surface-title">🏢 Facility Delay Clusters &amp; Stagnation Episodes</div>'
            '<div class="surface-subtitle">Micro-event sector breakdown paired with sustained stagnation periods (&ge;5 consecutive stressed days).</div>',
            unsafe_allow_html=True,
        )

        api_bn = api_fetch_bottlenecks(api_online)
        if api_bn and api_bn.get("facility_delay_clusters") and selected_facility == "All Facilities":
            fac_api_df = pd.DataFrame(api_bn["facility_delay_clusters"]).rename(
                columns={
                    "processing_facility_id": "Facility ID",
                    "active_cases": "Active in Queue",
                    "high_risk_cases": "High-Risk Cases",
                    "avg_cbp_dwell_days": "Avg CBP Stay (Days)",
                    "avg_hhs_dwell_days": "Avg HHS Stay (Days)",
                    "delay_severity": "Delay Severity",
                }
            )
            st.dataframe(fac_api_df, width="stretch", hide_index=True)
        else:
            cases = load_micro_cases(selected_facility)
            if not cases.empty:
                cases["apprehension_date"] = pd.to_datetime(cases["apprehension_date"])
                cases["cbp_transfer_date"] = pd.to_datetime(cases["cbp_transfer_date"])
                cases["discharge_date"] = pd.to_datetime(cases["discharge_date"])
                cases["cbp_dwell"] = (cases["cbp_transfer_date"] - cases["apprehension_date"]).dt.days.fillna(2.5)
                cases["hhs_dwell"] = (cases["discharge_date"] - cases["cbp_transfer_date"]).dt.days.fillna(35.0)

                fac_summary = (
                    cases.groupby("processing_facility_id")
                    .agg(
                        total_cases=("case_id", "count"),
                        active_cases=("status", lambda s: s.isin(["CBP_Custody", "HHS_Care"]).sum()),
                        high_risk=("sponsor_risk_tier", lambda r: (r == "High").sum()),
                        avg_cbp_dwell=("cbp_dwell", "mean"),
                        avg_hhs_dwell=("hhs_dwell", "mean"),
                    )
                    .reset_index()
                )
                fac_summary["avg_cbp_dwell"] = fac_summary["avg_cbp_dwell"].round(1)
                fac_summary["avg_hhs_dwell"] = fac_summary["avg_hhs_dwell"].round(1)
                fac_summary["delay_severity"] = np.where(
                    fac_summary["avg_cbp_dwell"] > 3.0,
                    "Critical Delay",
                    np.where(fac_summary["avg_cbp_dwell"] > 2.5, "Elevated", "Nominal"),
                )
                st.dataframe(
                    fac_summary.rename(
                        columns={
                            "processing_facility_id": "Facility ID",
                            "total_cases": "Total Minors Tracked",
                            "active_cases": "Active in Queue",
                            "high_risk": "High-Risk Cases",
                            "avg_cbp_dwell": "Avg CBP Stay (Days)",
                            "avg_hhs_dwell": "Avg HHS Stay (Days)",
                            "delay_severity": "Delay Severity",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

        stagnation_events = detect_stagnation_periods(filtered_df, min_consecutive_days=5)
        if stagnation_events:
            with st.expander(f"📉 Sustained Historical Stagnation Periods ({len(stagnation_events)} Episodes)"):
                st.dataframe(pd.DataFrame(stagnation_events), width="stretch", hide_index=True)

        with st.expander("🔔 Operational Webhook Alert Dispatcher"):
            wh_col1, wh_col2 = st.columns([3, 1])
            with wh_col1:
                webhook_url = st.text_input("Slack / Discord / HTTP Alert Sink URL", placeholder="https://hooks.slack.com/services/...")
            with wh_col2:
                st.write("")
                st.write("")
                if st.button("Dispatch Alert", width="stretch"):
                    res_wh = dispatch_webhook_alert(
                        webhook_url=webhook_url,
                        alert_level="WARNING" if curr_ter < ter_warn_thresh else "INFO",
                        message=ann_text,
                        metric_details={"TER": curr_ter, "Throughput": curr_spt, "HHS_Census": kpis["hhs_care"]},
                    )
                    if res_wh["success"]:
                        st.success(f"Alert dispatched cleanly (HTTP {res_wh['status_code']}).")
                    else:
                        st.warning(f"Alert simulation returned status {res_wh['status_code']}: {res_wh['error']}")

    with st.container(border=True):
        render_kanban_board()

# ==============================================================================
# TAB 5: INTERACTIVE POLICY SIMULATOR
# ==============================================================================
with tab5:
    with st.container(border=True):
        st.markdown(
            '<div class="surface-title">🎛️ Interactive Policy Surge &amp; Backlog Dissipation Simulator</div>'
            '<div class="surface-subtitle">Model forward CBP holding queues and HHS bed saturation under simulated arrival shocks, transfer multipliers, and sponsor vetting accelerations.</div>',
            unsafe_allow_html=True,
        )

        sim_left, sim_right = st.columns([1, 2.2])
        with sim_left:
            st.markdown("##### Policy Stress Controls")
            sim_surge = st.slider(
                "Border Apprehension Surge (%)",
                -40,
                100,
                key="sim_surge",
                step=5,
                help="Simulates arrival influx variance relative to recent 14-day baseline",
            )
            sim_transfer_mult = st.slider(
                "CBP Transfer Velocity Multiplier",
                0.50,
                2.00,
                key="sim_transfer_mult",
                step=0.05,
                help="Adjusts Stage 1 border handoff speed (consumed by REST /api/v1/simulate)",
            )
            sim_days = st.slider("Simulation Horizon (Days)", 14, 90, key="sim_days", step=7)
            st.metric("Synced Vetting Adjustment", f"{sidebar_vetting_speed:+d}%", help="Adjust via sidebar slider")
            st.metric("Max Shelter Bed Limit", f"{bed_capacity_limit:,} beds")

        with sim_right:
            api_sim = api_post_simulation(
                transfer_mult=sim_transfer_mult,
                capacity_limit=bed_capacity_limit,
                surge_pct=float(sim_surge),
                sim_days=int(sim_days),
            )

            # Local stress engine keeps full vetting + surge parity and serves seamless fallback
            local_sim = run_stress_simulation(
                filtered_df,
                apprehension_pct_change=float(sim_surge),
                vetting_speed_pct_change=float(sidebar_vetting_speed),
                sim_days=int(sim_days),
                bed_capacity_limit=int(bed_capacity_limit),
            )

            if api_sim and api_sim.get("projected_trajectory") and sidebar_vetting_speed == 0:
                traj_df = pd.DataFrame(api_sim["projected_trajectory"]).rename(
                    columns={
                        "date": "Date",
                        "projected_apprehensions": "Apprehensions_Proj",
                        "projected_transfers": "Transfers_Proj",
                        "projected_discharges": "Discharges_Proj",
                        "cbp_queue": "CBP_Queue_Proj",
                        "hhs_census": "HHS_Census_Proj",
                        "net_backlog_delta": "Net_Delta_Proj",
                    }
                )
                cap_breached = api_sim["capacity_breached"]
                breach_day = api_sim["breach_day"]
                sim_source_tag = "REST Backend (/api/v1/simulate)"
            else:
                raw_traj = local_sim["trajectory_df"].copy()
                raw_traj["Date"] = pd.to_datetime(raw_traj["Date"]).dt.strftime("%Y-%m-%d")
                raw_traj["Net_Delta_Proj"] = (
                    raw_traj["Simulated_Apprehensions"] - raw_traj["Simulated_Discharges"]
                ).round(1)
                traj_df = raw_traj.rename(
                    columns={
                        "Simulated_Apprehensions": "Apprehensions_Proj",
                        "Simulated_Transfers_In": "Transfers_Proj",
                        "Simulated_Discharges": "Discharges_Proj",
                        "Simulated_CBP_Queue": "CBP_Queue_Proj",
                        "Simulated_HHS_Census": "HHS_Census_Proj",
                    }
                )
                cap_breached = local_sim["capacity_breached"]
                breach_day = local_sim["days_to_breach"]
                sim_source_tag = "Interactive Stress Engine (Live Vetting + Surge)"

            sim_fig = go.Figure()
            sim_fig.add_trace(
                go.Scatter(
                    x=traj_df["Date"],
                    y=traj_df["HHS_Census_Proj"],
                    name="Projected HHS Shelter Census",
                    line=dict(color="#143311", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(40, 107, 52, 0.10)",
                )
            )
            sim_fig.add_trace(
                go.Scatter(
                    x=traj_df["Date"],
                    y=traj_df["CBP_Queue_Proj"],
                    name="Projected CBP Custody Queue",
                    line=dict(color="#D97706", width=2.4),
                )
            )
            sim_fig.add_hline(
                y=bed_capacity_limit,
                line_dash="dash",
                line_color="#DC2626",
                annotation_text=f"Capacity Ceiling ({bed_capacity_limit:,})",
                annotation_position="top left",
            )
            st.plotly_chart(style_plotly(sim_fig, height=340, y_title="Projected Minor Census"), width="stretch")
            st.caption(f"Simulation Engine Source: {sim_source_tag}")

            peak_census = int(traj_df["HHS_Census_Proj"].max())
            if cap_breached:
                st.error(
                    f"⚠️ Capacity Breach Alert: Shelter census breaches the {bed_capacity_limit:,}-bed ceiling on **Day {breach_day}** (Peak: {peak_census:,})."
                )
            else:
                st.success(
                    f"✅ System Remains Within Capacity over {sim_days} days (Peak Census: {peak_census:,} vs {bed_capacity_limit:,} ceiling)."
                )

        st.markdown("#### Projected Forward Daily Trajectory")
        st.dataframe(
            traj_df[[
                "Date",
                "Apprehensions_Proj",
                "Transfers_Proj",
                "Discharges_Proj",
                "CBP_Queue_Proj",
                "HHS_Census_Proj",
                "Net_Delta_Proj",
            ]].rename(
                columns={
                    "Apprehensions_Proj": "Intake (Inflow)",
                    "Transfers_Proj": "CBP Transfers",
                    "Discharges_Proj": "Discharges (Outflow)",
                    "CBP_Queue_Proj": "CBP Queue",
                    "HHS_Census_Proj": "HHS Bed Census",
                    "Net_Delta_Proj": "Net Daily Delta",
                }
            ),
            width="stretch",
            hide_index=True,
        )

# ==============================================================================
# TAB 6: ML FORECAST & CASE RISK LAB
# ==============================================================================
with tab6:
    with st.container(border=True):
        st.markdown(
            '<div class="surface-title">🔮 Machine Learning Surge Forecaster &amp; Stagnation Risk Lab</div>'
            '<div class="surface-subtitle">Fourier Seasonal Autoregressive 30/90-day volume forecasting paired with a Random Forest binary classifier predicting prolonged stay (&gt;45 days).</div>',
            unsafe_allow_html=True,
        )

        ml_c1, ml_c2 = st.columns([1, 2])
        with ml_c1:
            st.markdown("#### Time-Series Forecast Horizon")
            fc_horizon = st.radio(
                "Select Prediction Horizon",
                [30, 90],
                format_func=lambda d: f"{d}-Day Forward Forecast",
                index=0,
            )

            forecaster = get_trained_forecaster(raw_df)

            fc_meta = forecaster.training_metadata
            st.markdown(
                f"""
                <div style="background:#F9FCF2; border:1px solid #DCE7CA; border-radius:12px; padding:14px; margin-top:12px;">
                    <div style="font-size:11px; font-weight:700; color:#447948; text-transform:uppercase; letter-spacing:0.06em;">Model Diagnostics</div>
                    <div style="font-size:13px; margin-top:6px;"><b>Algorithm:</b> Fourier Seasonal Gradient Boosting</div>
                    <div style="font-size:13px;"><b>Training Span:</b> {fc_meta.get('observations_used', 0):,} days</div>
                    <div style="font-size:13px;"><b>In-Sample MAPE:</b> {fc_meta.get('mape_pct', 0)}%</div>
                    <div style="font-size:13px;"><b>MAE:</b> ±{fc_meta.get('mae', 0)} minors</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with ml_c2:
            api_fc = api_fetch_ml_forecast(fc_horizon, api_online)
            if api_fc and api_fc.get("forecast"):
                fc_df = pd.DataFrame(api_fc["forecast"])
            else:
                fc_df = forecaster.predict_surges(raw_df, horizon_days=fc_horizon)

            hist_tail = raw_df.tail(60)
            fc_fig = go.Figure()
            fc_fig.add_trace(
                go.Scatter(
                    x=hist_tail["Date"],
                    y=hist_tail["HHS_Care"],
                    name="Historical HHS Care",
                    line=dict(color="#447948", width=2.2),
                )
            )
            fc_fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(fc_df["date"]),
                    y=fc_df["projected_hhs_census"],
                    name=f"{fc_horizon}-Day Forecast",
                    line=dict(color="#143311", width=3),
                )
            )
            fc_fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(fc_df["date"]),
                    y=fc_df["upper_95ci"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fc_fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(fc_df["date"]),
                    y=fc_df["lower_95ci"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(82, 155, 87, 0.20)",
                    name="95% Confidence Band",
                )
            )
            st.plotly_chart(style_plotly(fc_fig, height=330, y_title="Shelter Bed Census"), width="stretch")

    with st.container(border=True):
        st.markdown("#### Binary Reunification Delay Classifier (>45 Days in Care)")
        clf = get_trained_classifier()

        clf_metrics = clf.evaluation_metrics
        clf_c1, clf_c2 = st.columns([1.25, 1])

        with clf_c1:
            st.markdown(
                f"""
                <div style="display:flex; gap:10px; margin-bottom:14px;">
                    <div style="flex:1; background:#EAF7ED; border:1px solid #C6E9CD; border-radius:12px; padding:12px; text-align:center;">
                        <div style="font-size:10.5px; font-weight:700; color:#196C2E;">ROC-AUC SCORE</div>
                        <div style="font-size:23px; font-weight:800; color:#143311;">{clf_metrics.get('roc_auc', 0):.3f}</div>
                    </div>
                    <div style="flex:1; background:#F9FCF2; border:1px solid #DCE7CA; border-radius:12px; padding:12px; text-align:center;">
                        <div style="font-size:10.5px; font-weight:700; color:#447948;">PRECISION (DELAYED)</div>
                        <div style="font-size:23px; font-weight:800; color:#143311;">{clf_metrics.get('precision_delayed', 0):.1%}</div>
                    </div>
                    <div style="flex:1; background:#F9FCF2; border:1px solid #DCE7CA; border-radius:12px; padding:12px; text-align:center;">
                        <div style="font-size:10.5px; font-weight:700; color:#447948;">RECALL (DELAYED)</div>
                        <div style="font-size:23px; font-weight:800; color:#143311;">{clf_metrics.get('recall_delayed', 0):.1%}</div>
                    </div>
                    <div style="flex:1; background:#F9FCF2; border:1px solid #DCE7CA; border-radius:12px; padding:12px; text-align:center;">
                        <div style="font-size:10.5px; font-weight:700; color:#447948;">F1 (DELAYED)</div>
                        <div style="font-size:23px; font-weight:800; color:#143311;">{clf_metrics.get('f1_delayed', 0):.1%}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            imp_dict = clf.feature_importances
            top_feats = list(imp_dict.keys())[:8]
            top_vals = [imp_dict[k] for k in top_feats]

            imp_fig = go.Figure(
                go.Bar(x=top_vals[::-1], y=top_feats[::-1], orientation="h", marker_color="#286B34")
            )
            imp_fig = style_plotly(imp_fig, height=275)
            imp_fig.update_layout(
                title=dict(text="Random Forest Feature Importance Drivers", font=dict(size=14, color="#143311")),
                xaxis_title="Gini Importance Weight",
                hovermode="y unified",
            )
            st.plotly_chart(imp_fig, width="stretch")

            cm = clf_metrics.get("confusion_matrix")
            if cm and len(cm) == 2:
                with st.expander(f"📊 Classification Confusion Matrix (Test Accuracy: {clf_metrics.get('accuracy', 0):.1%})"):
                    cm_fig = px.imshow(
                        cm,
                        x=["Predicted: ≤45d", "Predicted: >45d Delayed"],
                        y=["Actual: ≤45d", "Actual: >45d Delayed"],
                        text_auto=True,
                        color_continuous_scale=[[0.0, "#F9FCF2"], [1.0, "#1E4620"]],
                    )
                    cm_fig.update_layout(
                        font=dict(family="Plus Jakarta Sans, sans-serif", color="#173C19"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#FFFFFF",
                        height=240,
                        margin=dict(l=10, r=10, t=15, b=15),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(cm_fig, width="stretch")

        with clf_c2:
            st.markdown("##### Interactive Caseworker Stagnation Risk Calculator")
            st.caption("Evaluate probability that a child remains in HHS shelter care >45 days based on intake & sponsor attributes.")

            in_tier = st.selectbox("Sponsor Risk Tier", ["Low", "Medium", "High"], index=1)
            in_doc = st.slider("Sponsor Documentation Completeness Score", 0, 100, 60, 5)
            in_fac = st.selectbox(
                "Originating Processing Facility",
                ["CBP-ELP-01", "CBP-RGV-02", "CBP-TUS-03", "CBP-DRT-04", "CBP-SDG-05", "CBP-YUM-06"],
            )
            in_month = st.slider("Initial Intake Month", 1, 12, 6)
            in_cbp = st.slider("Initial CBP Holding Latency (Days)", 1.0, 10.0, 2.5, 0.5)

            case_payload = {
                "sponsor_risk_tier": in_tier,
                "documentation_score": float(in_doc),
                "processing_facility": in_fac,
                "initial_intake_month": int(in_month),
                "cbp_holding_days": float(in_cbp),
            }
            pred_res = api_predict_case_risk(case_payload) or clf.predict_single_case(case_payload)

            prob_pct = pred_res["prolonged_delay_probability"] * 100
            box_bg = "#FDF2F2" if prob_pct >= 50 else "#EAF7ED"
            box_border = "#F8B4B4" if prob_pct >= 50 else "#C6E9CD"
            box_color = "#9B1C1C" if prob_pct >= 50 else "#196C2E"

            st.markdown(
                f"""
                <div style="background:{box_bg}; border:1px solid {box_border}; border-radius:14px; padding:16px; margin-top:10px;">
                    <div style="font-size:11px; font-weight:700; color:{box_color}; text-transform:uppercase; letter-spacing:0.06em;">
                        Predicted Stagnation Risk (&gt;45 Days)
                    </div>
                    <div style="font-size:32px; font-weight:800; color:{box_color}; margin:4px 0;">{prob_pct:.1f}%</div>
                    <div style="font-size:13px; font-weight:700; color:#143311; margin-bottom:6px;">{pred_res['risk_tier']}</div>
                    <div style="font-size:12px; color:#485A4A;"><b>Recommended Caseworker Action:</b> {pred_res['caseworker_recommendation']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# --- 7. FOOTER ---
st.markdown(
    """
    <div style="text-align: center; font-size: 12px; color: #526655; margin-top: 32px; padding: 18px 0; border-top: 1px solid #DCE7CA;">
        U.S. Department of Health and Human Services (HHS.gov) &bull; Administration for Children and Families (ACF) &bull; Office of Refugee Resettlement (ORR)<br>
        Unaccompanied Children Program Operational Analytics &bull; Designed in compliance with TVPRA and Flores Settlement Standards
    </div>
    """,
    unsafe_allow_html=True,
)
