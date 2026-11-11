"""
tests/test_backend.py
Unit and endpoint integration test suite for the UAC Analytics REST API (backend/main.py).
Tests /api/v1/summary, /api/v1/timeseries, /api/v1/bottlenecks, and /api/v1/simulate.
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import get_connection
from backend.main import (
    fetch_summary_data,
    fetch_timeseries_data,
    fetch_bottlenecks_data,
    run_policy_simulation,
    fetch_ml_forecast,
    evaluate_case_delay_risk
)
from backend.schemas import (
    SummaryResponse,
    TimeSeriesResponse,
    BottlenecksResponse,
    SimulationRequest,
    SimulationResponse,
    CaseRiskRequest,
    CaseRiskResponse,
    ForecastResponse
)

class TestBackendAPI(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.conn = get_connection()
        
    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_get_summary_endpoint(self):
        """Validates /api/v1/summary calculates latest KPI scorecards with delta percentages."""
        summary: SummaryResponse = fetch_summary_data(self.conn)
        
        # Verify schema structure
        self.assertIsNotNone(summary.as_of_date)
        self.assertGreater(summary.transfer_efficiency.current, 0)
        self.assertIn(summary.transfer_efficiency.status, ["OPTIMAL", "BOTTLENECK"])
        
        self.assertGreater(summary.discharge_effectiveness.current, 0)
        self.assertIn(summary.discharge_effectiveness.status, ["OPTIMAL", "BELOW_BENCHMARK"])
        
        self.assertGreater(summary.pipeline_throughput.current, 0)
        self.assertIn(summary.pipeline_throughput.status, ["CLEARING", "ACCUMULATING"])
        
        self.assertGreater(summary.active_backlogs.hhs_care, 0)
        self.assertGreater(summary.operational_health.score, 0)
        self.assertIn(summary.operational_health.grade.split()[0], ["A", "B", "C", "D", "F"])

    def test_get_timeseries_endpoint_with_date_filters(self):
        """Validates /api/v1/timeseries date range filtering."""
        # 1. Unfiltered query
        ts_all: TimeSeriesResponse = fetch_timeseries_data(self.conn)
        self.assertGreater(ts_all.total_records, 700)
        
        # 2. Filtered 90-day window
        start_dt = "2024-01-01"
        end_dt = "2024-03-31"
        ts_filtered: TimeSeriesResponse = fetch_timeseries_data(self.conn, start_date=start_dt, end_date=end_dt)
        self.assertEqual(ts_filtered.total_records, 91)
        self.assertEqual(ts_filtered.start_date, start_dt)
        self.assertEqual(ts_filtered.end_date, end_dt)
        
        # Verify first record fields
        first_record = ts_filtered.data[0]
        self.assertEqual(first_record.date, start_dt)
        self.assertGreaterEqual(first_record.cbp_apprehensions, 0)
        self.assertGreaterEqual(first_record.transfer_efficiency_ratio, 0.0)

    def test_get_bottlenecks_endpoint(self):
        """Validates /api/v1/bottlenecks anomaly detection and facility delay clusters."""
        bottlenecks: BottlenecksResponse = fetch_bottlenecks_data(self.conn)
        
        self.assertGreaterEqual(bottlenecks.total_anomalies, 0)
        self.assertGreaterEqual(bottlenecks.active_critical_anomalies, 0)
        
        if bottlenecks.total_anomalies > 0:
            first_event = bottlenecks.congestion_events[0]
            self.assertGreater(first_event.z_score, 2.0, "Anomaly events must have Z-score > 2.0")
            self.assertIn(first_event.severity, ["Elevated", "High", "Critical"])
            
        # Verify facility delay clusters
        self.assertGreater(len(bottlenecks.facility_delay_clusters), 0)
        first_facility = bottlenecks.facility_delay_clusters[0]
        self.assertTrue(first_facility.processing_facility_id.startswith("CBP-"))
        self.assertGreaterEqual(first_facility.active_cases, 0)
        self.assertGreaterEqual(first_facility.avg_cbp_dwell_days, 0.0)

    def test_post_simulate_endpoint(self):
        """Validates /api/v1/simulate policy projections under stress parameters."""
        req = SimulationRequest(
            transfer_velocity_multiplier=1.25,
            capacity_adjustment=10000,
            apprehension_surge_pct=30.0,
            sim_days=30
        )
        sim: SimulationResponse = run_policy_simulation(self.conn, req)
        
        self.assertEqual(sim.sim_days, 30)
        self.assertEqual(sim.capacity_limit, 10000)
        self.assertEqual(len(sim.projected_trajectory), 30)
        
        # Verify trajectory continuity
        for pt in sim.projected_trajectory:
            self.assertGreater(pt.projected_apprehensions, 0)
            self.assertGreaterEqual(pt.cbp_queue, 0)
            self.assertGreaterEqual(pt.hhs_census, 0)

    def test_ml_forecast_endpoint(self):
        """Validates /api/v1/ml/forecast returns 30-day forward surge forecasts."""
        fc: ForecastResponse = fetch_ml_forecast(horizon_days=30)
        self.assertEqual(fc.horizon_days, 30)
        self.assertEqual(len(fc.forecast), 30)
        self.assertGreater(fc.forecast[0].projected_hhs_census, 0)

    def test_ml_predict_delay_endpoint(self):
        """Validates /api/v1/ml/predict-delay classifies prolonged case stagnation risk."""
        req = CaseRiskRequest(
            sponsor_risk_tier="High",
            documentation_score=25.0,
            processing_facility="CBP-ELP-01",
            initial_intake_month=5
        )
        res: CaseRiskResponse = evaluate_case_delay_risk(req)
        self.assertGreater(res.prolonged_delay_probability, 0.5)
        self.assertTrue(res.predicted_delayed_gt_45d)

if __name__ == "__main__":
    unittest.main()

