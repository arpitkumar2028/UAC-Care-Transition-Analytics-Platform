"""
backend/main.py
High-Performance REST API for UAC Care Transition Efficiency & Placement Outcome Analytics.

Exposes REST endpoints:
- GET  /api/v1/summary      : Latest KPI scorecards with 30-day delta and status
- GET  /api/v1/timeseries   : Date-filtered time-series with transition ratios
- GET  /api/v1/bottlenecks  : Anomaly logs and facility delay clusters via rolling Z-scores
- POST /api/v1/simulate     : Forward policy stress testing and backlog dissipation curves

Includes:
- CORS middleware
- Pydantic validation schemas
- Database dependency injection
- Asynchronous execution
"""

import sys
import os
import sqlite3
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Generator

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from db.connection import get_connection, get_db_path, is_postgres_connection
from core.analytics import (
    compute_full_analytics_suite,
    detect_backlog_anomalies,
    compute_transfer_efficiency_ratio,
    compute_discharge_effectiveness_index,
    compute_pipeline_throughput_rate,
    compute_backlog_accumulation_rate
)
from backend.schemas import (
    SummaryResponse, KPIMetric, BacklogMetric, HealthMetric,
    TimeSeriesResponse, TimeSeriesRecord,
    BottlenecksResponse, BottleneckAnomalyEvent, FacilityDelayCluster,
    SimulationRequest, SimulationResponse, SimulationTrajectoryPoint,
    CaseRiskRequest, CaseRiskResponse, ForecastResponse, ForecastPoint
)
from ml.models import get_inflow_forecast, predict_single_case_risk

# --- FASTAPI / ASGI INITIALIZATION ---
try:
    from fastapi import FastAPI, Depends, Query, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    USE_FASTAPI = True
except ImportError:
    # Resilient fallback using Starlette (installed) if fastapi is being provisioned
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse
    from starlette.exceptions import HTTPException
    USE_FASTAPI = False

# Database Dependency Injection
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Provides a thread-safe database connection per request."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()

# App factory
def create_app():
    if USE_FASTAPI:
        app = FastAPI(
            title="UAC Care Transition Efficiency API",
            description="Official REST API for UAC care pipeline velocity, dwell-time estimation, and bottleneck detection.",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS Middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/", tags=["Health"])
        async def root():
            return {
                "service": "UAC Care Transition Efficiency REST API",
                "version": "1.0.0",
                "status": "OPERATIONAL",
                "endpoints": [
                    "/api/v1/summary",
                    "/api/v1/timeseries",
                    "/api/v1/bottlenecks",
                    "/api/v1/simulate"
                ]
            }
            
        @app.get("/api/v1/summary", response_model=SummaryResponse, tags=["Analytics"])
        async def get_summary(db: sqlite3.Connection = Depends(get_db)):
            return fetch_summary_data(db)

        @app.get("/api/v1/timeseries", response_model=TimeSeriesResponse, tags=["Analytics"])
        async def get_timeseries(
            start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
            end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
            db: sqlite3.Connection = Depends(get_db)
        ):
            return fetch_timeseries_data(db, start_date, end_date)

        @app.get("/api/v1/bottlenecks", response_model=BottlenecksResponse, tags=["Analytics"])
        async def get_bottlenecks(db: sqlite3.Connection = Depends(get_db)):
            return fetch_bottlenecks_data(db)

        @app.post("/api/v1/simulate", response_model=SimulationResponse, tags=["Simulation"])
        async def post_simulate(
            payload: SimulationRequest,
            db: sqlite3.Connection = Depends(get_db)
        ):
            return run_policy_simulation(db, payload)

        @app.get("/api/v1/ml/forecast", response_model=ForecastResponse, tags=["Machine Learning"])
        async def get_ml_forecast(
            horizon_days: int = Query(30, ge=7, le=90, description="Forecast horizon (30 or 90 days)")
        ):
            return fetch_ml_forecast(horizon_days)

        @app.post("/api/v1/ml/predict-delay", response_model=CaseRiskResponse, tags=["Machine Learning"])
        async def post_predict_delay(payload: CaseRiskRequest):
            return evaluate_case_delay_risk(payload)

        return app
    else:
        # Fallback Starlette app wrapper
        async def root(request):
            return JSONResponse({
                "service": "UAC Care Transition Efficiency REST API (Starlette Engine)",
                "version": "1.0.0",
                "status": "OPERATIONAL"
            })
            
        async def summary_endpoint(request):
            conn = get_connection()
            try:
                res = fetch_summary_data(conn)
                return JSONResponse(res.dict() if hasattr(res, 'dict') else res)
            finally:
                conn.close()

        async def timeseries_endpoint(request):
            conn = get_connection()
            try:
                start_date = request.query_params.get("start_date")
                end_date = request.query_params.get("end_date")
                res = fetch_timeseries_data(conn, start_date, end_date)
                return JSONResponse(res.dict() if hasattr(res, 'dict') else res)
            finally:
                conn.close()

        async def bottlenecks_endpoint(request):
            conn = get_connection()
            try:
                res = fetch_bottlenecks_data(conn)
                return JSONResponse(res.dict() if hasattr(res, 'dict') else res)
            finally:
                conn.close()

        async def simulate_endpoint(request):
            conn = get_connection()
            try:
                body = await request.json()
                req = SimulationRequest(**body)
                res = run_policy_simulation(conn, req)
                return JSONResponse(res.dict() if hasattr(res, 'dict') else res)
            finally:
                conn.close()

        async def ml_forecast_endpoint(request):
            horizon = int(request.query_params.get("horizon_days", 30))
            res = fetch_ml_forecast(horizon)
            return JSONResponse(res.dict() if hasattr(res, 'dict') else res)

        async def ml_predict_delay_endpoint(request):
            body = await request.json()
            req = CaseRiskRequest(**body)
            res = evaluate_case_delay_risk(req)
            return JSONResponse(res.dict() if hasattr(res, 'dict') else res)

        routes = [
            Route("/", root, methods=["GET"]),
            Route("/api/v1/summary", summary_endpoint, methods=["GET"]),
            Route("/api/v1/timeseries", timeseries_endpoint, methods=["GET"]),
            Route("/api/v1/bottlenecks", bottlenecks_endpoint, methods=["GET"]),
            Route("/api/v1/simulate", simulate_endpoint, methods=["POST"]),
            Route("/api/v1/ml/forecast", ml_forecast_endpoint, methods=["GET"]),
            Route("/api/v1/ml/predict-delay", ml_predict_delay_endpoint, methods=["POST"]),
        ]
        
        app = Starlette(routes=routes)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return app

# --- BUSINESS LOGIC HELPERS ---

def _load_macro_df(conn) -> pd.DataFrame:
    """Loads macro pipeline metrics from DB with resilient CSV archive fallback."""
    df = pd.read_sql("SELECT * FROM daily_pipeline_metrics ORDER BY date ASC;", conn)
    if df.empty:
        csv_path = ROOT_DIR / "HHS_Unaccompanied_Children_Program.csv"
        df = pd.read_csv(csv_path)
    return df

def fetch_summary_data(conn: sqlite3.Connection) -> SummaryResponse:
    """Calculates latest KPI scorecard with 30-day percentage changes."""
    df = _load_macro_df(conn)
        
    enriched = compute_full_analytics_suite(df, as_json=False)
    latest = enriched.iloc[-1]
    
    # 30-day prior comparison index
    prior_idx = max(0, len(enriched) - 31)
    prior_30d = enriched.iloc[prior_idx]
    
    def calc_delta(curr: float, prev: float):
        diff = curr - prev
        pct = (diff / prev * 100) if prev != 0 else 0.0
        return round(diff, 3), round(pct, 1)

    ter_diff, ter_pct = calc_delta(latest["transfer_efficiency_ratio"], prior_30d["transfer_efficiency_ratio"])
    dei_diff, dei_pct = calc_delta(latest["discharge_effectiveness_pct"], prior_30d["discharge_effectiveness_pct"])
    ptr_diff, ptr_pct = calc_delta(latest["pipeline_throughput_rate_7d"], prior_30d["pipeline_throughput_rate_7d"])

    # Health Score Heuristic (0 - 100)
    health_score = 100.0
    if latest["transfer_efficiency_ratio"] < 0.85:
        health_score -= 20.0 * (1.0 - latest["transfer_efficiency_ratio"])
    if latest["discharge_effectiveness_pct"] < 2.5:
        health_score -= 15.0
    if latest["pipeline_throughput_rate_7d"] < 1.0:
        health_score -= 25.0 * (1.0 - latest["pipeline_throughput_rate_7d"])
    health_score = max(10.0, min(100.0, round(health_score, 1)))
    
    health_grade = "A (Optimal)" if health_score >= 88 else ("B (Nominal)" if health_score >= 75 else ("C (Stressed)" if health_score >= 60 else "D (Critical)"))

    return SummaryResponse(
        as_of_date=str(latest["date"])[:10],
        transfer_efficiency=KPIMetric(
            current=round(float(latest["transfer_efficiency_ratio"]), 2),
            prior_30d=round(float(prior_30d["transfer_efficiency_ratio"]), 2),
            delta=ter_diff,
            pct_change=ter_pct,
            status="OPTIMAL" if latest["transfer_efficiency_ratio"] >= 0.85 else "BOTTLENECK"
        ),
        discharge_effectiveness=KPIMetric(
            current=round(float(latest["discharge_effectiveness_pct"]), 2),
            prior_30d=round(float(prior_30d["discharge_effectiveness_pct"]), 2),
            delta=dei_diff,
            pct_change=dei_pct,
            status="OPTIMAL" if latest["discharge_effectiveness_pct"] >= 2.5 else "BELOW_BENCHMARK"
        ),
        pipeline_throughput=KPIMetric(
            current=round(float(latest["pipeline_throughput_rate_7d"]), 2),
            prior_30d=round(float(prior_30d["pipeline_throughput_rate_7d"]), 2),
            delta=ptr_diff,
            pct_change=ptr_pct,
            status="CLEARING" if latest["pipeline_throughput_rate_7d"] >= 1.0 else "ACCUMULATING"
        ),
        active_backlogs=BacklogMetric(
            daily_net=int(latest["daily_flow_balance"]),
            cumulative=int(latest["cumulative_backlog"]),
            cbp_custody=int(latest["cbp_active_custody"]),
            hhs_care=int(latest["hhs_active_care"])
        ),
        operational_health=HealthMetric(
            score=health_score,
            grade=health_grade
        )
    )

def fetch_timeseries_data(
    conn: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> TimeSeriesResponse:
    """Queries and returns time-series data filtered by date window."""
    placeholder = "%s" if is_postgres_connection(conn) else "?"
    query = "SELECT * FROM daily_pipeline_metrics"
    conditions = []
    params = []
    if start_date:
        conditions.append(f"date >= {placeholder}")
        params.append(start_date)
    if end_date:
        conditions.append(f"date <= {placeholder}")
        params.append(end_date)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY date ASC;"
    
    df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return TimeSeriesResponse(
            start_date=start_date or "",
            end_date=end_date or "",
            total_records=0,
            data=[]
        )
        
    enriched = compute_full_analytics_suite(df, as_json=False)
    
    records = []
    for _, r in enriched.iterrows():
        records.append(TimeSeriesRecord(
            date=str(r["date"])[:10],
            cbp_apprehensions=int(r["cbp_apprehensions"]),
            cbp_active_custody=int(r["cbp_active_custody"]),
            cbp_transfers_out=int(r["cbp_transfers_out"]),
            hhs_active_care=int(r["hhs_active_care"]),
            hhs_discharges=int(r["hhs_discharges"]),
            transfer_efficiency_ratio=float(r["transfer_efficiency_ratio"]),
            discharge_effectiveness_pct=float(r["discharge_effectiveness_pct"]),
            pipeline_throughput_rate=float(r["pipeline_throughput_rate_7d"]),
            backlog_accumulation_rate=float(r["backlog_accumulation_rate_7d"])
        ))
        
    return TimeSeriesResponse(
        start_date=str(enriched["date"].iloc[0])[:10],
        end_date=str(enriched["date"].iloc[-1])[:10],
        total_records=len(records),
        data=records
    )

def fetch_bottlenecks_data(conn: sqlite3.Connection) -> BottlenecksResponse:
    """Returns anomaly logs and facility delay clusters using rolling Z-scores."""
    # 1. Macro Anomalies
    macro_df = _load_macro_df(conn)
    anomalies_df = detect_backlog_anomalies(macro_df, z_threshold=2.0)
    
    anomaly_rows = anomalies_df[anomalies_df["is_anomaly"] == 1]
    congestion_events = []
    for _, r in anomaly_rows.iterrows():
        congestion_events.append(BottleneckAnomalyEvent(
            date=str(r["date"]),
            z_score=float(r["z_score"]),
            backlog_accumulation_rate=float(r["backlog_accumulation_rate"]),
            baseline_mean_30d=float(r["baseline_mean_30d"]),
            threshold_upper_bound=float(r["threshold_upper_bound"]),
            severity=str(r["severity"])
        ))
        
    # 2. Facility Delay Clusters from micro case events
    try:
        cases_df = pd.read_sql("SELECT * FROM case_level_events;", conn)
        cases_df["apprehension_date"] = pd.to_datetime(cases_df["apprehension_date"])
        cases_df["cbp_transfer_date"] = pd.to_datetime(cases_df["cbp_transfer_date"])
        cases_df["discharge_date"] = pd.to_datetime(cases_df["discharge_date"])
        
        # Calculate Dwells
        cases_df["cbp_dwell"] = (cases_df["cbp_transfer_date"] - cases_df["apprehension_date"]).dt.days.fillna(2.5)
        cases_df["hhs_dwell"] = (cases_df["discharge_date"] - cases_df["cbp_transfer_date"]).dt.days.fillna(35.0)
        
        facility_clusters = []
        for fac_id, group in cases_df.groupby("processing_facility_id"):
            active_cases = int(group["status"].isin(["CBP_Custody", "HHS_Care"]).sum())
            high_risk = int((group["sponsor_risk_tier"] == "High").sum())
            avg_cbp = round(float(group["cbp_dwell"].mean()), 1)
            avg_hhs = round(float(group["hhs_dwell"].mean()), 1)
            
            # Severity classification
            if avg_cbp > 3.5 or active_cases > 500:
                severity = "Critical Delay"
            elif avg_cbp > 2.8:
                severity = "Elevated Dwell"
            else:
                severity = "Nominal"
                
            facility_clusters.append(FacilityDelayCluster(
                processing_facility_id=str(fac_id),
                active_cases=active_cases,
                high_risk_cases=high_risk,
                avg_cbp_dwell_days=avg_cbp,
                avg_hhs_dwell_days=avg_hhs,
                delay_severity=severity
            ))
    except Exception:
        facility_clusters = []
        
    critical_count = sum(1 for e in congestion_events if e.severity in ["Critical", "High"])

    return BottlenecksResponse(
        total_anomalies=len(congestion_events),
        active_critical_anomalies=critical_count,
        congestion_events=congestion_events,
        facility_delay_clusters=facility_clusters
    )

def run_policy_simulation(
    conn: sqlite3.Connection,
    request: SimulationRequest
) -> SimulationResponse:
    """
    Simulates forward backlog dissipation curves under user-defined policy parameters.
    """
    df = _load_macro_df(conn)
    last_row = df.iloc[-1]
    last_date = datetime.strptime(str(last_row["date"])[:10], "%Y-%m-%d").date()
    
    # Baseline rates (14-day rolling mean)
    recent = df.iloc[-14:]
    base_apprehensions = recent["cbp_apprehensions"].mean()
    base_ter = (recent["cbp_transfers_out"] / recent["cbp_active_custody"].replace(0, 1)).mean()
    base_discharges = recent["hhs_discharges"].mean()
    
    # Apply policy multipliers
    sim_apprehensions = max(10, base_apprehensions * (1.0 + request.apprehension_surge_pct / 100.0))
    sim_ter = base_ter * request.transfer_velocity_multiplier
    
    cbp_q = float(last_row["cbp_active_custody"])
    hhs_c = float(last_row["hhs_active_care"])
    cap_limit = request.capacity_adjustment
    
    trajectory: List[SimulationTrajectoryPoint] = []
    breach_day = None
    
    for d in range(1, request.sim_days + 1):
        cur_date = last_date + timedelta(days=d)
        is_weekend = (cur_date.weekday() >= 5)
        
        # Weekend administrative drop
        wk_factor = 0.60 if is_weekend else 1.0
        
        # Stage 1: Inflow & Transfers
        intake = int(round(sim_apprehensions * (0.9 if is_weekend else 1.05)))
        avail_cbp = cbp_q + intake
        transfers = int(min(avail_cbp, round(avail_cbp * min(0.95, sim_ter * wk_factor))))
        cbp_q = max(0, avail_cbp - transfers)
        
        # Stage 2: HHS care & Discharges
        avail_hhs = hhs_c + transfers
        # Discharges turnover rate
        discharge_turnover = 0.028 * wk_factor
        discharges = int(min(avail_hhs, round(avail_hhs * discharge_turnover)))
        hhs_c = max(0, avail_hhs - discharges)
        
        net_delta = intake - discharges
        
        if hhs_c > cap_limit and breach_day is None:
            breach_day = d
            
        trajectory.append(SimulationTrajectoryPoint(
            day=d,
            date=cur_date.strftime("%Y-%m-%d"),
            projected_apprehensions=int(intake),
            projected_transfers=int(transfers),
            projected_discharges=int(discharges),
            cbp_queue=int(cbp_q),
            hhs_census=int(hhs_c),
            net_backlog_delta=int(net_delta)
        ))
        
    return SimulationResponse(
        sim_days=request.sim_days,
        initial_cbp_queue=int(last_row["cbp_active_custody"]),
        initial_hhs_census=int(last_row["hhs_active_care"]),
        capacity_limit=cap_limit,
        capacity_breached=(breach_day is not None),
        breach_day=breach_day,
        projected_trajectory=trajectory
    )

def fetch_ml_forecast(horizon_days: int = 30) -> ForecastResponse:
    """Runs serialized TimeSeriesInflowForecaster for 30 or 90 day projections."""
    fc_df = get_inflow_forecast(horizon_days=horizon_days)
    points = []
    for _, r in fc_df.iterrows():
        points.append(ForecastPoint(
            date=str(r["date"]),
            projected_hhs_census=int(r["projected_hhs_census"]),
            lower_95ci=int(r["lower_95ci"]),
            upper_95ci=int(r["upper_95ci"]),
            is_surge_alert=int(r["is_surge_alert"]),
            projected_net_delta=int(r["projected_net_delta"])
        ))
    return ForecastResponse(
        horizon_days=horizon_days,
        total_surge_alerts=int(fc_df["is_surge_alert"].sum()),
        forecast=points
    )

def evaluate_case_delay_risk(request: CaseRiskRequest) -> CaseRiskResponse:
    """Runs serialized ReunificationDelayClassifier for single case risk prediction."""
    case_data = request.dict() if hasattr(request, "dict") else request.__dict__
    result = predict_single_case_risk(case_data)
    return CaseRiskResponse(**result)

# Master ASGI App Instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    print("Starting UAC Analytics FastAPI REST Service on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

