"""
temporal_analysis.py
Statistical and pattern analysis module for evaluating:
1. Day-of-week & Weekend operational latency (administrative drop).
2. Month-over-Month (MoM) care pipeline dynamics.
3. Automated identification of prolonged stagnation and crisis periods.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List

DAYS_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def analyze_day_of_week(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes day-of-week averages and performs two-sample t-tests
    comparing Weekday (Mon-Fri) vs. Weekend (Sat-Sun) velocity.
    """
    dow_df = df.groupby('DayOfWeek').agg(
        avg_apprehensions=('Apprehensions', 'mean'),
        avg_transfers=('Transfers_Out_CBP', 'mean'),
        avg_discharges=('Discharges_HHS', 'mean'),
        avg_ter=('Transfer_Efficiency_Ratio', 'mean'),
        avg_dei_pct=('Discharge_Effectiveness_Pct', 'mean'),
        avg_cbp_custody=('CBP_Custody', 'mean'),
        avg_hhs_care=('HHS_Care', 'mean'),
        count=('Date', 'count')
    ).reindex(DAYS_ORDER).reset_index()

    weekday_mask = df['IsWeekend'] == 0
    weekend_mask = df['IsWeekend'] == 1
    
    # Statistical t-tests for weekend drop
    def perform_ttest(col):
        wkday_vals = df.loc[weekday_mask, col].dropna()
        wkend_vals = df.loc[weekend_mask, col].dropna()
        t_stat, p_val = stats.ttest_ind(wkday_vals, wkend_vals, equal_var=False)
        wkday_mean = float(wkday_vals.mean())
        wkend_mean = float(wkend_vals.mean())
        pct_diff = ((wkend_mean - wkday_mean) / wkday_mean * 100) if wkday_mean > 0 else 0
        return {
            "weekday_mean": round(wkday_mean, 2),
            "weekend_mean": round(wkend_mean, 2),
            "pct_change": round(pct_diff, 1),
            "t_statistic": round(float(t_stat), 3),
            "p_value": float(f"{p_val:.2e}"),
            "statistically_significant": bool(p_val < 0.05)
        }

    return {
        "dow_table": dow_df,
        "discharges_test": perform_ttest('Discharges_HHS'),
        "transfers_test": perform_ttest('Transfers_Out_CBP'),
        "ter_test": perform_ttest('Transfer_Efficiency_Ratio'),
        "apprehensions_test": perform_ttest('Apprehensions')
    }

def analyze_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates metrics by Year-Month to evaluate long-term trends and MoM growth.
    """
    monthly = df.groupby('YearMonth').agg(
        start_date=('Date', 'min'),
        total_apprehensions=('Apprehensions', 'sum'),
        total_transfers=('Transfers_Out_CBP', 'sum'),
        total_discharges=('Discharges_HHS', 'sum'),
        avg_cbp_custody=('CBP_Custody', 'mean'),
        avg_hhs_care=('HHS_Care', 'mean'),
        avg_ter=('Transfer_Efficiency_Ratio', 'mean'),
        avg_throughput=('Pipeline_Throughput_Rate', 'mean'),
        days_reported=('Date', 'count')
    ).reset_index()

    monthly['net_system_flow'] = monthly['total_apprehensions'] - monthly['total_discharges']
    monthly['apprehensions_mom_pct'] = monthly['total_apprehensions'].pct_change() * 100
    monthly['discharges_mom_pct'] = monthly['total_discharges'].pct_change() * 100
    monthly['transfer_mom_pct'] = monthly['total_transfers'].pct_change() * 100

    return monthly.round(2)

def detect_stagnation_periods(df: pd.DataFrame, min_consecutive_days: int = 5) -> List[Dict[str, Any]]:
    """
    Detects continuous historical periods of operational stagnation:
    Condition: Pipeline Throughput (MA7) < 0.85 OR Transfer Efficiency (MA7) < 0.45.
    """
    condition = (df['Pipeline_Throughput_MA7'] < 0.85) | (df['TER_MA7'] < 0.45)
    
    events = []
    in_event = False
    start_idx = 0
    
    for i, is_stressed in enumerate(condition):
        if is_stressed and not in_event:
            in_event = True
            start_idx = i
        elif not is_stressed and in_event:
            in_event = False
            duration = i - start_idx
            if duration >= min_consecutive_days:
                sub_df = df.iloc[start_idx:i]
                events.append({
                    "start_date": sub_df['Date'].min().strftime('%Y-%m-%d'),
                    "end_date": sub_df['Date'].max().strftime('%Y-%m-%d'),
                    "duration_days": duration,
                    "avg_throughput": round(float(sub_df['Pipeline_Throughput_MA7'].mean()), 2),
                    "avg_ter": round(float(sub_df['TER_MA7'].mean()), 2),
                    "net_backlog_accumulated": int(sub_df['Daily_Net_Backlog'].sum()),
                    "peak_cbp_custody": int(sub_df['CBP_Custody'].max()),
                    "peak_hhs_care": int(sub_df['HHS_Care'].max()),
                    "severity": "CRITICAL" if duration >= 14 or sub_df['TER_MA7'].mean() < 0.35 else "WARNING"
                })
                
    # Check if currently ongoing at the end of the series
    if in_event:
        duration = len(df) - start_idx
        if duration >= min_consecutive_days:
            sub_df = df.iloc[start_idx:]
            events.append({
                "start_date": sub_df['Date'].min().strftime('%Y-%m-%d'),
                "end_date": sub_df['Date'].max().strftime('%Y-%m-%d'),
                "duration_days": duration,
                "avg_throughput": round(float(sub_df['Pipeline_Throughput_MA7'].mean()), 2),
                "avg_ter": round(float(sub_df['TER_MA7'].mean()), 2),
                "net_backlog_accumulated": int(sub_df['Daily_Net_Backlog'].sum()),
                "peak_cbp_custody": int(sub_df['CBP_Custody'].max()),
                "peak_hhs_care": int(sub_df['HHS_Care'].max()),
                "severity": "CRITICAL" if duration >= 14 or sub_df['TER_MA7'].mean() < 0.35 else "WARNING"
            })
            
    return events
