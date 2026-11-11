"""
metrics_engine.py
Core mathematical and operational engine for computing process efficiency,
throughput, backlog accumulation, dwell time, and stability indices across
the UAC care pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def compute_pipeline_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the cleaned dataframe with all primary and secondary operational metrics:
    - Transfer Efficiency Ratio (TER)
    - Discharge Effectiveness Index (DEI)
    - Pipeline Throughput Rate (SPT)
    - Stage Net Flows (Stage 1 CBP, Stage 2 HHS)
    - Daily Net Backlog & Cumulative Backlog Deficit
    - Little's Law Estimated Dwell Time (Days in Care)
    - Outcome Stability Score (30-day Rolling)
    - Composite System Health Score (0 - 100)
    """
    m_df = df.copy()
    
    # 1. Transfer Efficiency Ratio (TER) = Transfers out of CBP / Children in CBP custody
    # High TER (> 1.0) indicates rapid clearance; low (< 0.5) indicates bottlenecking
    m_df['Transfer_Efficiency_Ratio'] = np.where(
        m_df['CBP_Custody'] > 0,
        m_df['Transfers_Out_CBP'] / m_df['CBP_Custody'],
        np.where(m_df['Transfers_Out_CBP'] > 0, 1.0, 0.0)
    )
    
    # 2. Discharge Effectiveness Index (DEI) = Discharges / Children in HHS Care
    m_df['Discharge_Effectiveness_Index'] = np.where(
        m_df['HHS_Care'] > 0,
        m_df['Discharges_HHS'] / m_df['HHS_Care'],
        0.0
    )
    # Scaled to percentage points for dashboard readability
    m_df['Discharge_Effectiveness_Pct'] = m_df['Discharge_Effectiveness_Index'] * 100
    
    # 3. Pipeline Throughput Rate (SPT) = Total Exits (Discharges) / Total Entries (Apprehensions)
    m_df['Pipeline_Throughput_Rate'] = np.where(
        m_df['Apprehensions'] > 0,
        m_df['Discharges_HHS'] / m_df['Apprehensions'],
        np.where(m_df['Discharges_HHS'] > 0, 1.0, 1.0)
    )
    
    # 7-Day Smoothed Throughput (reduces daily reporting artifact noise)
    m_df['Pipeline_Throughput_MA7'] = np.where(
        m_df['Apprehensions_MA7'] > 0,
        m_df['Discharges_HHS_MA7'] / m_df['Apprehensions_MA7'],
        1.0
    )
    
    # 4. Stage-Wise Net Flows
    # Stage 1: CBP Queue Flow = Apprehensions - Transfers Out
    m_df['Stage1_CBP_Net_Flow'] = m_df['Apprehensions'] - m_df['Transfers_Out_CBP']
    
    # Stage 2: HHS Care Net Flow = Transfers In (Transfers out of CBP) - Discharges
    m_df['Stage2_HHS_Net_Flow'] = m_df['Transfers_Out_CBP'] - m_df['Discharges_HHS']
    
    # Total System Net Backlog = Apprehensions - Discharges
    m_df['Daily_Net_Backlog'] = m_df['Apprehensions'] - m_df['Discharges_HHS']
    m_df['Cumulative_Backlog'] = m_df['Daily_Net_Backlog'].cumsum()
    
    # 5. Little's Law Dwell Time Proxy: W ≈ L / λ (Shelter Census / Daily Discharge Rate)
    # We use 7-day smoothed discharges to prevent division by zero or extreme daily spikes
    safe_discharges_ma7 = m_df['Discharges_HHS_MA7'].replace(0, np.nan)
    m_df['Estimated_Dwell_Time_Days'] = (m_df['HHS_Care'] / safe_discharges_ma7).fillna(
        m_df['HHS_Care'] / (m_df['Discharges_HHS'].mean() or 1.0)
    ).clip(lower=1, upper=180)
    
    # 6. Outcome Stability Score: 1 - CV(30-day Discharges)
    # CV = std / mean. Higher score (closer to 1.0) means steady, predictable placement outcomes.
    rolling_mean_30 = m_df['Discharges_HHS'].rolling(window=30, min_periods=7).mean()
    rolling_std_30 = m_df['Discharges_HHS'].rolling(window=30, min_periods=7).std().fillna(0)
    cv_30 = np.where(rolling_mean_30 > 0, rolling_std_30 / rolling_mean_30, 0.0)
    m_df['Outcome_Stability_Score'] = np.clip(1.0 - cv_30, 0.0, 1.0)
    
    # 7. Rolling Smoothed Ratios
    m_df['TER_MA7'] = m_df['Transfer_Efficiency_Ratio'].rolling(window=7, min_periods=1).mean()
    m_df['TER_MA30'] = m_df['Transfer_Efficiency_Ratio'].rolling(window=30, min_periods=1).mean()
    m_df['DEI_MA7'] = m_df['Discharge_Effectiveness_Index'].rolling(window=7, min_periods=1).mean()
    
    # 8. Composite System Health Score (0 to 100)
    # Weights: TER component (30%), Throughput component (35%), Stability component (20%), CBP backlog buffer (15%)
    ter_score = np.clip(m_df['TER_MA7'] / 1.0, 0, 1.0) * 30
    spt_score = np.clip(m_df['Pipeline_Throughput_MA7'] / 1.0, 0, 1.0) * 35
    stability_score = m_df['Outcome_Stability_Score'] * 20
    cbp_pressure = np.clip(1.0 - (m_df['CBP_Custody'] / 500.0), 0, 1.0) * 15
    
    m_df['System_Health_Score'] = (ter_score + spt_score + stability_score + cbp_pressure).round(1)
    
    # Assign Status Grade
    m_df['Health_Grade'] = m_df['System_Health_Score'].apply(get_letter_grade)
    
    return m_df

def get_letter_grade(score: float) -> str:
    """Maps composite score to operational letter grade."""
    if score >= 85:
        return 'A (Optimal)'
    elif score >= 70:
        return 'B (Stable)'
    elif score >= 55:
        return 'C (Elevated Stress)'
    elif score >= 40:
        return 'D (Severe Bottlenecks)'
    else:
        return 'F (Critical Gridlock)'

def get_current_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts the latest KPI metrics along with 30-day delta changes.
    """
    latest = df.iloc[-1]
    prev_30 = df.iloc[-31] if len(df) > 30 else df.iloc[0]
    
    def calc_delta(curr, prev):
        diff = curr - prev
        pct = (diff / prev * 100) if prev != 0 else 0
        return round(diff, 2), round(pct, 1)

    ter_diff, ter_pct = calc_delta(latest['TER_MA7'], prev_30['TER_MA7'])
    dei_diff, dei_pct = calc_delta(latest['Discharge_Effectiveness_Pct'], prev_30['Discharge_Effectiveness_Pct'])
    spt_diff, spt_pct = calc_delta(latest['Pipeline_Throughput_MA7'], prev_30['Pipeline_Throughput_MA7'])
    dwell_diff, dwell_pct = calc_delta(latest['Estimated_Dwell_Time_Days'], prev_30['Estimated_Dwell_Time_Days'])
    health_diff, health_pct = calc_delta(latest['System_Health_Score'], prev_30['System_Health_Score'])

    return {
        "as_of_date": latest['Date'].strftime('%B %d, %Y'),
        "ter_current": round(float(latest['Transfer_Efficiency_Ratio']), 2),
        "ter_ma7": round(float(latest['TER_MA7']), 2),
        "ter_delta": ter_diff,
        "ter_delta_pct": ter_pct,
        
        "dei_current_pct": round(float(latest['Discharge_Effectiveness_Pct']), 2),
        "dei_delta": dei_diff,
        
        "spt_current": round(float(latest['Pipeline_Throughput_MA7']), 2),
        "spt_delta": spt_diff,
        
        "estimated_dwell_days": int(round(latest['Estimated_Dwell_Time_Days'])),
        "dwell_delta": dwell_diff,
        
        "daily_backlog": int(latest['Daily_Net_Backlog']),
        "cumulative_backlog": int(latest['Cumulative_Backlog']),
        
        "stability_score": round(float(latest['Outcome_Stability_Score']), 2),
        
        "health_score": float(latest['System_Health_Score']),
        "health_grade": latest['Health_Grade'],
        "health_delta": health_diff,
        
        "cbp_custody": int(latest['CBP_Custody']),
        "hhs_care": int(latest['HHS_Care']),
        "apprehensions": int(latest['Apprehensions']),
        "transfers": int(latest['Transfers_Out_CBP']),
        "discharges": int(latest['Discharges_HHS'])
    }
