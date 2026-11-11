"""
Unit tests for data loader, metrics engine, and temporal analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from care_pipeline.data_loader import load_and_clean_data, get_data_summary
from care_pipeline.metrics_engine import compute_pipeline_metrics, get_current_kpis
from care_pipeline.temporal_analysis import analyze_day_of_week, analyze_monthly_trends, detect_stagnation_periods

def test_data_loader():
    df = load_and_clean_data()
    assert len(df) == 720, f"Expected 720 rows, got {len(df)}"
    assert 'Date' in df.columns
    assert 'Apprehensions' in df.columns
    assert 'CBP_Custody' in df.columns
    assert 'Transfers_Out_CBP' in df.columns
    assert 'HHS_Care' in df.columns
    assert 'Discharges_HHS' in df.columns
    assert df['Apprehensions'].min() >= 0
    assert df['CBP_Custody'].min() >= 0
    assert df['Date'].is_monotonic_increasing

def test_metrics_calculation():
    df = load_and_clean_data()
    m_df = compute_pipeline_metrics(df)
    
    assert 'Transfer_Efficiency_Ratio' in m_df.columns
    assert 'Discharge_Effectiveness_Index' in m_df.columns
    assert 'Pipeline_Throughput_Rate' in m_df.columns
    assert 'Cumulative_Backlog' in m_df.columns
    assert 'Estimated_Dwell_Time_Days' in m_df.columns
    assert 'Outcome_Stability_Score' in m_df.columns
    assert 'System_Health_Score' in m_df.columns
    assert 'Health_Grade' in m_df.columns
    
    # Check bounds
    assert (m_df['Transfer_Efficiency_Ratio'] >= 0).all()
    assert (m_df['Discharge_Effectiveness_Index'] >= 0).all()
    assert (m_df['Outcome_Stability_Score'] >= 0).all()
    assert (m_df['Outcome_Stability_Score'] <= 1.0).all()
    assert (m_df['System_Health_Score'] >= 0).all()
    assert (m_df['System_Health_Score'] <= 100).all()

def test_current_kpis():
    df = load_and_clean_data()
    m_df = compute_pipeline_metrics(df)
    kpis = get_current_kpis(m_df)
    
    assert 'as_of_date' in kpis
    assert 'ter_current' in kpis
    assert 'spt_current' in kpis
    assert 'health_grade' in kpis
    assert 'estimated_dwell_days' in kpis
    assert kpis['estimated_dwell_days'] > 0

def test_temporal_analysis():
    df = load_and_clean_data()
    m_df = compute_pipeline_metrics(df)
    
    dow_res = analyze_day_of_week(m_df)
    assert 'dow_table' in dow_res
    assert len(dow_res['dow_table']) == 7
    assert 'discharges_test' in dow_res
    assert 'statistically_significant' in dow_res['discharges_test']
    
    monthly = analyze_monthly_trends(m_df)
    assert len(monthly) > 0
    assert 'net_system_flow' in monthly.columns
    
    stagnations = detect_stagnation_periods(m_df)
    assert isinstance(stagnations, list)
    if len(stagnations) > 0:
        assert 'start_date' in stagnations[0]
        assert 'duration_days' in stagnations[0]

if __name__ == '__main__':
    test_data_loader()
    test_metrics_calculation()
    test_current_kpis()
    test_temporal_analysis()
    print("ALL PHASE 1 TESTS PASSED SUCCESSFULLY!")
