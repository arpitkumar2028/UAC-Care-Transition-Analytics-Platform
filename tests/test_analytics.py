"""
tests/test_analytics.py
Unit tests for the quantitative analytics engine (core/analytics.py).
Tests process ratios, rolling throughput, backlog accumulation, and Z-score anomaly detection.
"""

import sys
import json
import unittest
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analytics import (
    compute_transfer_efficiency_ratio,
    compute_discharge_effectiveness_index,
    compute_pipeline_throughput_rate,
    compute_backlog_accumulation_rate,
    detect_backlog_anomalies,
    compute_full_analytics_suite,
    normalize_pipeline_dataframe
)
from utils.data_generator import generate_macro_pipeline_data

class TestAnalyticsEngine(unittest.TestCase):
    
    def setUp(self):
        """Builds both synthetic test fixture and controlled edge-case fixtures."""
        # 1. 180-day realistic synthetic dataset
        self.macro_df = generate_macro_pipeline_data(num_days=180, seed=42)
        
        # 2. Controlled toy dataset for exact mathematical assertion
        self.toy_df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "cbp_apprehensions": [100, 120, 80, 90, 110, 130, 95, 105, 115, 85],
            "cbp_active_custody": [50, 60, 40, 45, 55, 65, 48, 52, 58, 42],
            "cbp_transfers_out": [75, 90, 60, 70, 85, 100, 72, 80, 88, 64],
            "hhs_active_care": [2000, 2020, 2010, 2030, 2040, 2060, 2050, 2070, 2080, 2060],
            "hhs_discharges": [60, 70, 50, 55, 65, 80, 58, 62, 70, 52],
            "sponsor_vetted_reunifications": [56, 66, 47, 52, 61, 75, 55, 58, 66, 49]
        })

    def test_transfer_efficiency_ratio(self):
        """Tests TER = Transfers / CBP Custody calculation and boundary conditions."""
        ter = compute_transfer_efficiency_ratio(self.toy_df)
        self.assertEqual(len(ter), 10)
        
        # Expected day 0: 75 / 50 = 1.50
        self.assertAlmostEqual(ter.iloc[0], 1.50, places=4)
        # Expected day 1: 90 / 60 = 1.50
        self.assertAlmostEqual(ter.iloc[1], 1.50, places=4)
        
        # Test zero custody safety
        zero_df = pd.DataFrame({
            "date": ["2026-01-01"],
            "cbp_apprehensions": [10],
            "cbp_active_custody": [0],
            "cbp_transfers_out": [15],
            "hhs_active_care": [100],
            "hhs_discharges": [5],
            "sponsor_vetted_reunifications": [5]
        })
        ter_zero = compute_transfer_efficiency_ratio(zero_df)
        self.assertEqual(ter_zero.iloc[0], 1.0)

    def test_discharge_effectiveness_index(self):
        """Tests DEI = Discharges / HHS Care and percentage scaling."""
        dei_raw = compute_discharge_effectiveness_index(self.toy_df, as_percentage=False)
        dei_pct = compute_discharge_effectiveness_index(self.toy_df, as_percentage=True)
        
        # Day 0: 60 / 2000 = 0.030 (3.0%)
        self.assertAlmostEqual(dei_raw.iloc[0], 0.030, places=4)
        self.assertAlmostEqual(dei_pct.iloc[0], 3.0, places=2)
        self.assertTrue((dei_raw >= 0).all())

    def test_pipeline_throughput_rate(self):
        """Tests PTR rolling window calculation."""
        ptr_7d = compute_pipeline_throughput_rate(self.toy_df, window=7)
        self.assertEqual(len(ptr_7d), 10)
        
        # Day 0: only 1 day available -> 56 / 100 = 0.56
        self.assertAlmostEqual(ptr_7d.iloc[0], 0.56, places=4)
        
        # Day 6 (index 6, 7 full days 0-6):
        sum_reunif = sum([56, 66, 47, 52, 61, 75, 55]) # 412
        sum_app = sum([100, 120, 80, 90, 110, 130, 95])  # 725
        expected_ptr = sum_reunif / sum_app
        self.assertAlmostEqual(ptr_7d.iloc[6], expected_ptr, places=4)

    def test_backlog_accumulation_rate(self):
        """Tests rolling delta between Apprehensions and Discharges."""
        bar_7d = compute_backlog_accumulation_rate(self.toy_df, window=7)
        self.assertEqual(len(bar_7d), 10)
        
        # Day 0: 100 - 60 = 40
        self.assertEqual(bar_7d.iloc[0], 40)
        
        # Day 6: sum of (app - dis) over days 0-6
        daily_deltas = [100-60, 120-70, 80-50, 90-55, 110-65, 130-80, 95-58] # [40, 50, 30, 35, 45, 50, 37]
        self.assertEqual(bar_7d.iloc[6], sum(daily_deltas))

    def test_rolling_z_score_anomaly_detection(self):
        """
        Tests automated rolling Z-score anomaly detector.
        Artificially introduces a severe surge and tests detection.
        """
        # Inject artificial massive surge at day 100
        test_df = self.macro_df.copy()
        test_df.loc[100:105, "cbp_apprehensions"] += 250 # Extreme influx
        test_df.loc[100:105, "hhs_discharges"] = 10     # Bottleneck discharge
        
        anomalies_df = detect_backlog_anomalies(test_df, z_threshold=2.0, baseline_window=30, backlog_window=7)
        
        self.assertIn("z_score", anomalies_df.columns)
        self.assertIn("is_anomaly", anomalies_df.columns)
        self.assertIn("severity", anomalies_df.columns)
        
        # Surge window must contain flagged anomalies
        surge_flags = anomalies_df.loc[100:105, "is_anomaly"]
        self.assertGreater(surge_flags.sum(), 0, "Massive surge must be detected as anomaly")
        
        # Severity must be High or Critical during extreme surge
        surge_severities = anomalies_df.loc[100:105, "severity"]
        self.assertTrue(any(s in ["High", "Critical"] for s in surge_severities))

    def test_full_analytics_suite_and_json_serialization(self):
        """Tests that full analytics suite returns valid DataFrame and valid JSON."""
        # 1. DataFrame output
        enriched_df = compute_full_analytics_suite(self.macro_df, as_json=False)
        self.assertIsInstance(enriched_df, pd.DataFrame)
        self.assertIn("transfer_efficiency_ratio", enriched_df.columns)
        self.assertIn("discharge_effectiveness_pct", enriched_df.columns)
        self.assertIn("pipeline_throughput_rate_7d", enriched_df.columns)
        self.assertIn("backlog_accumulation_rate_7d", enriched_df.columns)
        self.assertIn("is_backlog_anomaly", enriched_df.columns)
        
        # 2. JSON API payload output
        json_payload = compute_full_analytics_suite(self.macro_df, as_json=True)
        self.assertIsInstance(json_payload, dict)
        self.assertIn("metadata", json_payload)
        self.assertIn("latest_kpis", json_payload)
        self.assertIn("time_series", json_payload)
        self.assertIn("anomaly_events", json_payload)
        
        # Validate JSON serialization without TypeError
        serialized = json.dumps(json_payload)
        self.assertTrue(len(serialized) > 100)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["metadata"]["total_days_evaluated"], len(self.macro_df))

    def test_column_name_alias_resilience(self):
        """Verifies engine seamlessly handles CSV alias column names."""
        alias_df = pd.DataFrame({
            "Date": ["2026-01-01", "2026-01-02"],
            "Apprehensions": [100, 110],
            "CBP_Custody": [40, 45],
            "Transfers_Out_CBP": [80, 85],
            "HHS_Care": [2200, 2210],
            "Discharges_HHS": [65, 70]
        })
        normalized = normalize_pipeline_dataframe(alias_df)
        self.assertIn("cbp_apprehensions", normalized.columns)
        self.assertIn("cbp_active_custody", normalized.columns)
        self.assertIn("sponsor_vetted_reunifications", normalized.columns)
        
        ter = compute_transfer_efficiency_ratio(alias_df)
        self.assertEqual(len(ter), 2)

if __name__ == "__main__":
    unittest.main()

