"""
simulation_engine.py
Interactive 'What-If' policy and operational stress testing simulator.
Enables executive leadership to model downstream effects of apprehension surges
and caseworker velocity enhancements on shelter bed saturation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

DEFAULT_BED_CAPACITY = 10000

def run_stress_simulation(
    df: pd.DataFrame,
    apprehension_pct_change: float = 0.0,
    vetting_speed_pct_change: float = 0.0,
    sim_days: int = 30,
    bed_capacity_limit: int = DEFAULT_BED_CAPACITY
) -> Dict[str, Any]:
    """
    Simulates daily forward trajectory of CBP queue, HHS care load, and discharges
    under user-defined operational stress conditions.
    
    Parameters:
    - apprehension_pct_change: Percentage change in border arrivals (-50% to +100%)
    - vetting_speed_pct_change: Percentage change in daily discharges (-30% to +50%)
    - sim_days: Number of days to simulate into the future (7 to 90)
    - bed_capacity_limit: Maximum safe operational shelter bed capacity (default 10,000)
    """
    last_date = df['Date'].max()
    sim_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=sim_days)
    
    # Baseline recent rates (last 14 days)
    recent_apprehensions = df['Apprehensions'].iloc[-14:].mean()
    recent_ter = df['Transfer_Efficiency_Ratio'].iloc[-14:].mean()
    recent_discharges = df['Discharges_HHS'].iloc[-14:].mean()
    
    # Adjusted stress rates
    sim_apprehensions = max(1, recent_apprehensions * (1.0 + apprehension_pct_change / 100.0))
    sim_discharges = max(1, recent_discharges * (1.0 + vetting_speed_pct_change / 100.0))
    
    cbp_queue = float(df['CBP_Custody'].iloc[-1])
    hhs_census = float(df['HHS_Care'].iloc[-1])
    
    trajectory = []
    breach_day = None
    
    for day_idx, sim_date in enumerate(sim_dates):
        # Stage 1: Inflow to CBP
        cbp_queue += sim_apprehensions
        
        # Transfers out of CBP (capped by transfer efficiency)
        transfers_out = min(cbp_queue, cbp_queue * recent_ter)
        cbp_queue = max(0, cbp_queue - transfers_out)
        
        # Stage 2: HHS Care Intake & Discharge
        hhs_census += transfers_out
        actual_discharges = min(hhs_census, sim_discharges)
        hhs_census = max(0, hhs_census - actual_discharges)
        
        utilization_pct = (hhs_census / bed_capacity_limit) * 100
        
        if hhs_census >= bed_capacity_limit and breach_day is None:
            breach_day = day_idx + 1
            
        trajectory.append({
            'Date': sim_date,
            'Day': day_idx + 1,
            'Simulated_Apprehensions': round(sim_apprehensions, 1),
            'Simulated_Transfers_In': round(transfers_out, 1),
            'Simulated_Discharges': round(actual_discharges, 1),
            'Simulated_CBP_Queue': round(cbp_queue),
            'Simulated_HHS_Census': round(hhs_census),
            'Bed_Utilization_Pct': round(utilization_pct, 1)
        })
        
    traj_df = pd.DataFrame(trajectory)
    
    net_total_growth = round(hhs_census - df['HHS_Care'].iloc[-1])
    required_discharge_for_balance = round(sim_apprehensions)
    
    return {
        "trajectory_df": traj_df,
        "sim_apprehensions_daily": round(sim_apprehensions, 1),
        "sim_discharges_daily": round(sim_discharges, 1),
        "initial_hhs_census": int(df['HHS_Care'].iloc[-1]),
        "ending_hhs_census": int(round(hhs_census)),
        "net_bed_change": int(net_total_growth),
        "bed_capacity_limit": bed_capacity_limit,
        "capacity_breached": breach_day is not None,
        "days_to_breach": breach_day,
        "required_daily_discharges_to_balance": required_discharge_for_balance
    }
