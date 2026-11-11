"""
utils/data_generator.py
Synthetic data generation engine for UAC Care Transition Efficiency platform.
Generates 3 years of daily macro time-series data and micro case-level events.

Features:
- Strict mathematical conservation of flow:
    Q_cbp[t] = max(0, Q_cbp[t-1] + Inflow[t] - Transfers[t])
    Q_hhs[t] = max(0, Q_hhs[t-1] + Transfers[t] - Discharges[t])
- Realistic weekly seasonality (35-45% drop in weekend transfers & discharges)
- Simulated policy-driven volume surges (e.g. spring waves, regulatory transition shocks)
- Micro case-level records with UUIDs, sponsor risk tiers, facilities, and lifecycle statuses
"""

import uuid
from datetime import datetime, date, timedelta
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

SECTORS_FACILITIES = [
    "CBP-ELP-01",  # El Paso Sector
    "CBP-RGV-02",  # Rio Grande Valley
    "CBP-TUS-03",  # Tucson Sector
    "CBP-DRT-04",  # Del Rio Sector
    "CBP-SDG-05",  # San Diego Sector
    "CBP-YUM-06"   # Yuma Sector
]

RISK_TIERS = ["Low", "Medium", "High"]
RISK_PROBABILITIES = [0.60, 0.25, 0.15]

def generate_macro_pipeline_data(
    num_days: int = 1095,
    end_date: Optional[date] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates 3 years of daily time-series macro data maintaining strict
    flow conservation and queueing dynamics.
    
    Returns DataFrame matching schema:
    [date, cbp_apprehensions, cbp_active_custody, cbp_transfers_out,
     hhs_active_care, hhs_discharges, sponsor_vetted_reunifications]
    """
    np.random.seed(seed)
    
    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=num_days - 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    n = len(date_range)
    
    # 1. Base Inflow (Apprehensions) with Annual Seasonality and Weekly Rhythm
    day_indices = np.arange(n)
    
    # Annual wave: peak in Spring (March-May), trough in Winter (Dec-Jan)
    annual_cycle = np.sin(2 * np.pi * (day_indices - 60) / 365.25)
    base_intake = 90.0 + 25.0 * annual_cycle
    
    # Weekly variation: arrivals continue 7 days, slight bump mid-week
    day_of_week = date_range.dayofweek.values
    dow_intake_factor = np.where(day_of_week < 5, 1.05, 0.90)
    
    # Policy Surges:
    # Surge 1: Year 1 Spring Surge (around day 120-160)
    surge_1 = 35.0 * np.exp(-0.5 * ((day_indices - 140) / 15) ** 2)
    # Surge 2: Year 2 Policy Transition Shock (around day 500-540)
    surge_2 = 60.0 * np.exp(-0.5 * ((day_indices - 520) / 20) ** 2)
    # Surge 3: Year 3 Late Summer Wave (around day 880-920)
    surge_3 = 40.0 * np.exp(-0.5 * ((day_indices - 900) / 18) ** 2)
    
    # Random Gaussian noise
    noise = np.random.normal(0, 8.0, n)
    
    apprehensions = np.maximum(
        15,
        np.round((base_intake * dow_intake_factor) + surge_1 + surge_2 + surge_3 + noise)
    ).astype(int)
    
    # 2. Sequential Queue Balancing (CBP Holding and HHS Shelter Care)
    cbp_active_custody = np.zeros(n, dtype=int)
    cbp_transfers_out = np.zeros(n, dtype=int)
    hhs_active_care = np.zeros(n, dtype=int)
    hhs_discharges = np.zeros(n, dtype=int)
    sponsor_vetted_reunifications = np.zeros(n, dtype=int)
    
    # Initial state initialization
    curr_cbp = 45
    curr_hhs = 2400
    
    for t in range(n):
        intake_t = apprehensions[t]
        dow = day_of_week[t]
        is_weekend = (dow >= 5)
        
        # Available in CBP before transfer
        available_cbp = curr_cbp + intake_t
        
        # Stage 1: Transfers out of CBP
        # Target: clear queue quickly to comply with 72-hr limit.
        # Weekdays: high efficiency (mean transfer rate ~0.85-0.95 of available)
        # Weekends: 35-45% administrative transfer drop
        if is_weekend:
            transfer_rate = np.random.uniform(0.40, 0.55)
        else:
            transfer_rate = np.random.uniform(0.80, 0.95)
            
        transfers_t = int(np.clip(
            np.round(available_cbp * transfer_rate),
            0,
            available_cbp
        ))
        curr_cbp = available_cbp - transfers_t
        
        # Stage 2: HHS Care & Discharges
        available_hhs = curr_hhs + transfers_t
        
        # Target daily turnover rate: ~2.5% to 3.5% of active census on weekdays
        # Weekend administrative drop: 40-50% drop on Sat/Sun
        if is_weekend:
            discharge_rate = np.random.uniform(0.012, 0.018)
        else:
            discharge_rate = np.random.uniform(0.024, 0.034)
            
        discharges_t = int(np.clip(
            np.round(available_hhs * discharge_rate + np.random.normal(0, 3)),
            0,
            available_hhs
        ))
        curr_hhs = available_hhs - discharges_t
        
        # Sponsor vetted reunifications: 92% - 96% of total discharges
        vetted_pct = np.random.uniform(0.92, 0.96)
        vetted_t = min(discharges_t, int(np.round(discharges_t * vetted_pct)))
        
        # Store values
        cbp_active_custody[t] = curr_cbp
        cbp_transfers_out[t] = transfers_t
        hhs_active_care[t] = curr_hhs
        hhs_discharges[t] = discharges_t
        sponsor_vetted_reunifications[t] = vetted_t

    macro_df = pd.DataFrame({
        "date": date_range.strftime("%Y-%m-%d"),
        "cbp_apprehensions": apprehensions,
        "cbp_active_custody": cbp_active_custody,
        "cbp_transfers_out": cbp_transfers_out,
        "hhs_active_care": hhs_active_care,
        "hhs_discharges": hhs_discharges,
        "sponsor_vetted_reunifications": sponsor_vetted_reunifications
    })
    
    return macro_df

def generate_micro_case_events(
    macro_df: pd.DataFrame,
    max_cases_per_day: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates granular child welfare case event trajectories consistent with
    macro inflow volumes and risk-tier dwell time distributions.
    
    Parameters:
    - macro_df: The generated macro daily time series.
    - max_cases_per_day: Optional cap on micro cases per day for lightweight testing.
    """
    np.random.seed(seed)
    
    case_records: List[Dict[str, Any]] = []
    
    dates = pd.to_datetime(macro_df["date"])
    apprehensions = macro_df["cbp_apprehensions"].values
    sim_end_date = dates.max().date()
    
    for i, app_date_ts in enumerate(dates):
        app_date = app_date_ts.date()
        daily_count = apprehensions[i]
        if max_cases_per_day and daily_count > max_cases_per_day:
            daily_count = max_cases_per_day
            
        if daily_count <= 0:
            continue
            
        facilities = np.random.choice(SECTORS_FACILITIES, size=daily_count)
        risk_tiers = np.random.choice(RISK_TIERS, size=daily_count, p=RISK_PROBABILITIES)
        
        for k in range(daily_count):
            case_id = str(uuid.uuid4())
            facility = facilities[k]
            risk = risk_tiers[k]
            
            # CBP Dwell Time: 1 to 4 days (72-hour regulatory target)
            cbp_dwell = int(np.clip(np.random.geometric(p=0.45), 1, 6))
            cbp_transfer_dt = app_date + timedelta(days=cbp_dwell)
            hhs_intake_dt = cbp_transfer_dt
            
            # HHS Shelter Dwell Time conditioned on sponsor risk tier:
            # Low: Parents/guardians -> ~18-28 days
            # Medium: Close relatives -> ~32-48 days
            # High: Distant sponsors/Home Study -> ~60-85 days
            if risk == "Low":
                hhs_dwell = int(np.clip(np.random.normal(22, 5), 10, 45))
            elif risk == "Medium":
                hhs_dwell = int(np.clip(np.random.normal(38, 8), 20, 65))
            else:
                hhs_dwell = int(np.clip(np.random.normal(70, 14), 35, 110))
                
            discharge_dt = hhs_intake_dt + timedelta(days=hhs_dwell)
            
            # Determine real-time lifecycle status relative to simulation end date
            if app_date > sim_end_date:
                continue
            elif cbp_transfer_dt > sim_end_date:
                status = "CBP_Custody"
                cbp_tr_str = None
                hhs_in_str = None
                dis_str = None
            elif discharge_dt > sim_end_date:
                status = "HHS_Care"
                cbp_tr_str = cbp_transfer_dt.strftime("%Y-%m-%d")
                hhs_in_str = hhs_intake_dt.strftime("%Y-%m-%d")
                dis_str = None
            else:
                # Discharged or Escalated (if dwell exceeded 90 days for high risk)
                if risk == "High" and hhs_dwell > 85:
                    status = "Escalated"
                else:
                    status = "Discharged"
                cbp_tr_str = cbp_transfer_dt.strftime("%Y-%m-%d")
                hhs_in_str = hhs_intake_dt.strftime("%Y-%m-%d")
                dis_str = discharge_dt.strftime("%Y-%m-%d")
                
            case_records.append({
                "case_id": case_id,
                "apprehension_date": app_date.strftime("%Y-%m-%d"),
                "cbp_transfer_date": cbp_tr_str,
                "hhs_intake_date": hhs_in_str,
                "discharge_date": dis_str,
                "processing_facility_id": facility,
                "sponsor_risk_tier": risk,
                "status": status
            })

    micro_df = pd.DataFrame(case_records)
    return micro_df

def generate_complete_dataset(
    num_days: int = 1095,
    max_cases_per_day: Optional[int] = 30,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience orchestrator generating both macro time-series
    and micro event ledger.
    """
    macro_df = generate_macro_pipeline_data(num_days=num_days, seed=seed)
    micro_df = generate_micro_case_events(macro_df, max_cases_per_day=max_cases_per_day, seed=seed)
    return macro_df, micro_df

