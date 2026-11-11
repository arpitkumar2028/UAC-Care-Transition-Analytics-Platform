"""
Unit tests for Phase 2 and Phase 3: ML, Simulator, Copilot, Webhooks, PDF Generator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from care_pipeline.data_loader import load_and_clean_data
from care_pipeline.metrics_engine import compute_pipeline_metrics, get_current_kpis
from care_pipeline.ml_forecasting import forecast_demand_and_placement
from care_pipeline.simulation_engine import run_stress_simulation
from care_pipeline.ai_copilot import PipelineCopilot
from care_pipeline.alerts_dispatcher import dispatch_webhook_alert
from care_pipeline.report_generator import generate_pdf_report

def test_copilot_and_metrics():
    df = load_and_clean_data()
    m_df = compute_pipeline_metrics(df)
    
    copilot = PipelineCopilot(m_df)
    summary = copilot.generate_audit_summary()
    assert 'health_grade' in summary
    assert 'primary_bottleneck' in summary
    assert 'briefing_markdown' in summary
    
    # Test queries
    ans_bottleneck = copilot.ask_copilot("Where is the bottleneck?")
    assert "Diagnostic Finding" in ans_bottleneck
    
    ans_weekend = copilot.ask_copilot("What is the weekend effect?")
    assert "Weekend" in ans_weekend

def test_forecasting_and_simulation():
    df = load_and_clean_data()
    m_df = compute_pipeline_metrics(df)
    
    # Forecast 7 days
    fc_df = forecast_demand_and_placement(m_df, days_ahead=7)
    assert len(fc_df) == 7
    assert 'Projected_Transfers_In' in fc_df.columns
    assert 'Projected_Discharges' in fc_df.columns
    assert 'Projected_HHS_Census' in fc_df.columns
    
    # Simulation
    sim_res = run_stress_simulation(m_df, apprehension_pct_change=20.0, sim_days=14)
    assert 'trajectory_df' in sim_res
    assert len(sim_res['trajectory_df']) == 14
    assert 'net_bed_change' in sim_res

def test_webhook_dispatcher():
    # Test with empty/safe URL
    res = dispatch_webhook_alert("", "CRITICAL", "Test alert")
    assert res['success'] is False
    assert res['status_code'] == 400
    
    # Test with dummy endpoint
    res2 = dispatch_webhook_alert("http://localhost:9999/dummy_sink", "WARNING", "Test", timeout_sec=0.5)
    assert res2['success'] is False # Expected fail, but handled safely without crashing!

def test_pdf_generation():
    df = load_and_clean_data()
    m_df = compute_pipeline_metrics(df)
    copilot = PipelineCopilot(m_df)
    summary = copilot.generate_audit_summary()
    kpis = get_current_kpis(m_df)
    summary.update(kpis)
    
    fc_df = forecast_demand_and_placement(m_df, days_ahead=7)
    pdf_bytes = generate_pdf_report(summary, fc_df)
    
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b'%PDF'), "PDF file should begin with %PDF header"

if __name__ == '__main__':
    test_copilot_and_metrics()
    print("Copilot & Metrics passed!")
    test_forecasting_and_simulation()
    print("Forecasting & Simulation passed!")
    test_webhook_dispatcher()
    print("Webhook Dispatcher passed!")
    test_pdf_generation()
    print("ReportLab PDF Generator passed!")
    print("ALL ENTERPRISE SERVICE TESTS PASSED 100%!")
