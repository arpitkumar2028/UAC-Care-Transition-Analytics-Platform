"""
backend/schemas.py
Pydantic schemas and request/response models for the UAC Analytics REST API.
Includes resilient fallback compatibility if Pydantic is not yet installed.
"""

from typing import List, Dict, Any, Optional
from datetime import date

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Lightweight fallback shim for environments where pydantic is being installed
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        @staticmethod
        def _serialize(val):
            if hasattr(val, "dict") and callable(val.dict):
                return val.dict()
            if isinstance(val, list):
                return [BaseModel._serialize(item) for item in val]
            if isinstance(val, dict):
                return {k: BaseModel._serialize(v) for k, v in val.items()}
            return val

        def dict(self, *args, **kwargs):
            return {
                k: BaseModel._serialize(v)
                for k, v in self.__dict__.items()
                if not k.startswith('_')
            }

        def model_dump(self, *args, **kwargs):
            return self.dict()
            
    def Field(default=None, **kwargs):
        return default

class KPIMetric(BaseModel):
    current: float
    prior_30d: float
    delta: float
    pct_change: float
    status: str

class BacklogMetric(BaseModel):
    daily_net: int
    cumulative: int
    cbp_custody: int
    hhs_care: int

class HealthMetric(BaseModel):
    score: float
    grade: str

class SummaryResponse(BaseModel):
    as_of_date: str
    transfer_efficiency: KPIMetric
    discharge_effectiveness: KPIMetric
    pipeline_throughput: KPIMetric
    active_backlogs: BacklogMetric
    operational_health: HealthMetric

class TimeSeriesRecord(BaseModel):
    date: str
    cbp_apprehensions: int
    cbp_active_custody: int
    cbp_transfers_out: int
    hhs_active_care: int
    hhs_discharges: int
    transfer_efficiency_ratio: float
    discharge_effectiveness_pct: float
    pipeline_throughput_rate: float
    backlog_accumulation_rate: float

class TimeSeriesResponse(BaseModel):
    start_date: str
    end_date: str
    total_records: int
    data: List[TimeSeriesRecord]

class FacilityDelayCluster(BaseModel):
    processing_facility_id: str
    active_cases: int
    high_risk_cases: int
    avg_cbp_dwell_days: float
    avg_hhs_dwell_days: float
    delay_severity: str

class BottleneckAnomalyEvent(BaseModel):
    date: str
    z_score: float
    backlog_accumulation_rate: float
    baseline_mean_30d: float
    threshold_upper_bound: float
    severity: str

class BottlenecksResponse(BaseModel):
    total_anomalies: int
    active_critical_anomalies: int
    congestion_events: List[BottleneckAnomalyEvent]
    facility_delay_clusters: List[FacilityDelayCluster]

class SimulationRequest(BaseModel):
    transfer_velocity_multiplier: float = Field(default=1.0, ge=0.1, le=3.0, description="Multiplier for transfer velocity (1.0 = baseline)")
    capacity_adjustment: int = Field(default=10000, ge=1000, le=50000, description="HHS Shelter bed capacity limit")
    apprehension_surge_pct: float = Field(default=0.0, ge=-50.0, le=100.0, description="Percentage change in daily apprehensions")
    sim_days: int = Field(default=30, ge=7, le=90, description="Simulation horizon in days")

class SimulationTrajectoryPoint(BaseModel):
    day: int
    date: str
    projected_apprehensions: int
    projected_transfers: int
    projected_discharges: int
    cbp_queue: int
    hhs_census: int
    net_backlog_delta: int

class SimulationResponse(BaseModel):
    sim_days: int
    initial_cbp_queue: int
    initial_hhs_census: int
    capacity_limit: int
    capacity_breached: bool
    breach_day: Optional[int]
    projected_trajectory: List[SimulationTrajectoryPoint]

class CaseRiskRequest(BaseModel):
    sponsor_risk_tier: str = Field(default="Medium", description="Low, Medium, or High")
    documentation_score: float = Field(default=60.0, ge=0.0, le=100.0)
    processing_facility: str = Field(default="CBP-ELP-01")
    initial_intake_month: int = Field(default=6, ge=1, le=12)
    intake_day_of_week: int = Field(default=2, ge=0, le=6)
    cbp_holding_days: float = Field(default=2.5, ge=0.5, le=15.0)

class CaseRiskResponse(BaseModel):
    prolonged_delay_probability: float
    predicted_delayed_gt_45d: bool
    risk_tier: str
    caseworker_recommendation: str
    top_risk_drivers: List[str]

class ForecastPoint(BaseModel):
    date: str
    projected_hhs_census: int
    lower_95ci: int
    upper_95ci: int
    is_surge_alert: int
    projected_net_delta: int

class ForecastResponse(BaseModel):
    horizon_days: int
    total_surge_alerts: int
    forecast: List[ForecastPoint]

