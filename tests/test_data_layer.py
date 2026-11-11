"""
tests/test_data_layer.py
Unit and integrity test suite for database schema, connection,
and synthetic data generation pipeline.
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import get_connection, execute_script
from db.init_db import init_schema, seed_database, validate_seeded_data
from utils.data_generator import (
    generate_macro_pipeline_data,
    generate_micro_case_events,
    generate_complete_dataset,
    SECTORS_FACILITIES,
    RISK_TIERS
)

class TestDataLayerAndGenerator(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Creates an in-memory or temporary database for testing."""
        cls.test_db_path = Path(__file__).resolve().parent / "test_uac.db"
        if cls.test_db_path.exists():
            cls.test_db_path.unlink()
        cls.stats = seed_database(
            num_days=1095,
            max_cases_per_day=15,
            seed=42,
            reset=True,
            db_path=cls.test_db_path
        )
        
    @classmethod
    def tearDownClass(cls):
        """Cleans up test database."""
        if cls.test_db_path.exists():
            cls.test_db_path.unlink()

    def test_database_seeding_integrity(self):
        """Asserts database initializes with valid rows and zero errors."""
        self.assertTrue(self.stats["is_valid"])
        self.assertEqual(self.stats["macro_records"], 1095)
        self.assertGreater(self.stats["micro_cases"], 10000)
        self.assertEqual(self.stats["invalid_metric_rows"], 0)
        self.assertEqual(self.stats["chronology_errors"], 0)

    def test_flow_conservation_macro(self):
        """
        Validates mathematical conservation of flow:
        Q_cbp[t] = Q_cbp[t-1] + Inflow[t] - Transfers[t]
        Q_hhs[t] = Q_hhs[t-1] + Transfers[t] - Discharges[t]
        """
        conn = get_connection(str(self.test_db_path))
        df = pd.read_sql("SELECT * FROM daily_pipeline_metrics ORDER BY date ASC;", conn)
        conn.close()
        
        # Verify 3 years of daily continuity
        self.assertEqual(len(df), 1095)
        
        # Verify non-negativity
        self.assertTrue((df["cbp_apprehensions"] >= 0).all())
        self.assertTrue((df["cbp_active_custody"] >= 0).all())
        self.assertTrue((df["cbp_transfers_out"] >= 0).all())
        self.assertTrue((df["hhs_active_care"] >= 0).all())
        self.assertTrue((df["hhs_discharges"] >= 0).all())
        self.assertTrue((df["sponsor_vetted_reunifications"] >= 0).all())
        
        # Verify vetted reunifications <= total discharges
        self.assertTrue((df["sponsor_vetted_reunifications"] <= df["hhs_discharges"]).all())

    def test_weekend_administrative_slowdown_seasonality(self):
        """
        Validates the weekly seasonality requirement:
        Weekend transfer and discharge rates must exhibit statistically significant
        reductions (p < 0.001) due to court/administrative downtime.
        """
        df = generate_macro_pipeline_data(num_days=1095, seed=42)
        df["date"] = pd.to_datetime(df["date"])
        df["is_weekend"] = df["date"].dt.dayofweek >= 5
        
        # Transfer rate = transfers_out / (active_custody + apprehensions)
        denom_cbp = (df["cbp_active_custody"] + df["cbp_transfers_out"]).replace(0, np.nan)
        df["transfer_rate"] = df["cbp_transfers_out"] / denom_cbp
        
        weekday_transfers = df[~df["is_weekend"]]["transfer_rate"].dropna()
        weekend_transfers = df[df["is_weekend"]]["transfer_rate"].dropna()
        
        # Welch's t-test on transfers
        t_stat, p_val = stats.ttest_ind(weekday_transfers, weekend_transfers, equal_var=False)
        self.assertGreater(t_stat, 0, "Weekday transfer rate must be higher than weekend")
        self.assertLess(p_val, 0.001, "Weekend transfer slowdown must be statistically significant (p < 0.001)")
        
        # Welch's t-test on discharges
        weekday_discharges = df[~df["is_weekend"]]["hhs_discharges"]
        weekend_discharges = df[df["is_weekend"]]["hhs_discharges"]
        t_stat_d, p_val_d = stats.ttest_ind(weekday_discharges, weekend_discharges, equal_var=False)
        self.assertGreater(t_stat_d, 0, "Weekday discharges must exceed weekend discharges")
        self.assertLess(p_val_d, 0.001, "Weekend discharge drop must be statistically significant (p < 0.001)")

    def test_micro_case_lifecycle_integrity(self):
        """
        Validates micro case ledger constraints:
        - UUID uniqueness
        - Permitted sponsor risk tiers
        - Valid processing facility IDs
        - Chronological validity: apprehension <= transfer <= discharge
        """
        conn = get_connection(str(self.test_db_path))
        cases_df = pd.read_sql("SELECT * FROM case_level_events LIMIT 2000;", conn)
        conn.close()
        
        # UUID uniqueness
        self.assertEqual(len(cases_df["case_id"].unique()), len(cases_df))
        
        # Facilities
        self.assertTrue(cases_df["processing_facility_id"].isin(SECTORS_FACILITIES).all())
        
        # Risk Tiers
        self.assertTrue(cases_df["sponsor_risk_tier"].isin(RISK_TIERS).all())
        
        # Valid Statuses
        valid_statuses = {"CBP_Custody", "HHS_Care", "Discharged", "Escalated"}
        self.assertTrue(cases_df["status"].isin(valid_statuses).all())
        
        # Chronology validation
        for _, row in cases_df.iterrows():
            app_dt = datetime.strptime(str(row["apprehension_date"])[:10], "%Y-%m-%d")
            if pd.notnull(row["cbp_transfer_date"]) and row["cbp_transfer_date"]:
                tr_dt = datetime.strptime(str(row["cbp_transfer_date"])[:10], "%Y-%m-%d")
                self.assertGreaterEqual(tr_dt, app_dt, "Transfer date must be >= Apprehension date")
            if pd.notnull(row["discharge_date"]) and row["discharge_date"]:
                dis_dt = datetime.strptime(str(row["discharge_date"])[:10], "%Y-%m-%d")
                if pd.notnull(row["hhs_intake_date"]) and row["hhs_intake_date"]:
                    intake_dt = datetime.strptime(str(row["hhs_intake_date"])[:10], "%Y-%m-%d")
                    self.assertGreaterEqual(dis_dt, intake_dt, "Discharge date must be >= Intake date")

if __name__ == "__main__":
    unittest.main()

