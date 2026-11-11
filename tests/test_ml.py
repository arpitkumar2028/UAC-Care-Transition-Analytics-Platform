"""
tests/test_ml.py
Unit and validation test suite for Machine Learning models (ml/models.py).
Tests:
1. Time-Series Inflow Forecaster (30-day & 90-day surge forecasting)
2. Reunification Delay Classifier (>45-day prolonged stagnation)
3. Model serialization and deserialization via joblib
4. Production inference functions
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.models import (
    TimeSeriesInflowForecaster,
    ReunificationDelayClassifier,
    get_inflow_forecast,
    predict_single_case_risk,
    SAVED_MODELS_DIR
)
from utils.data_generator import generate_macro_pipeline_data, generate_micro_case_events

class TestMachineLearningModule(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Generates synthetic datasets for ML testing."""
        cls.macro_df = generate_macro_pipeline_data(num_days=365, seed=42)
        # Add required aliases
        cls.macro_df["Date"] = pd.to_datetime(cls.macro_df["date"])
        cls.macro_df["HHS_Care"] = cls.macro_df["hhs_active_care"]
        cls.macro_df["Apprehensions"] = cls.macro_df["cbp_apprehensions"]
        cls.macro_df["Transfers_Out_CBP"] = cls.macro_df["cbp_transfers_out"]
        
        cls.cases_df = generate_micro_case_events(cls.macro_df, max_cases_per_day=15, seed=42)

    def test_inflow_forecaster_fit_and_predict(self):
        """Validates TimeSeriesInflowForecaster fitting, metrics, and multi-step rollout."""
        forecaster = TimeSeriesInflowForecaster(target_col="HHS_Care")
        forecaster.fit(self.macro_df)
        
        self.assertTrue(forecaster.is_fitted)
        self.assertIn("mae", forecaster.training_metadata)
        self.assertIn("rmse", forecaster.training_metadata)
        self.assertIn("mape_pct", forecaster.training_metadata)
        self.assertLess(forecaster.training_metadata["mape_pct"], 10.0, "Forecaster MAPE should be under 10%")
        
        # Test 30-day forecast
        fc_30 = forecaster.predict_surges(self.macro_df, horizon_days=30)
        self.assertEqual(len(fc_30), 30)
        self.assertIn("projected_hhs_census", fc_30.columns)
        self.assertIn("lower_95ci", fc_30.columns)
        self.assertIn("upper_95ci", fc_30.columns)
        self.assertIn("is_surge_alert", fc_30.columns)
        
        # Verify confidence interval bounds
        self.assertTrue((fc_30["upper_95ci"] >= fc_30["lower_95ci"]).all())
        
        # Test 90-day forecast
        fc_90 = forecaster.predict_surges(self.macro_df, horizon_days=90)
        self.assertEqual(len(fc_90), 90)

    def test_inflow_forecaster_serialization(self):
        """Validates model saving and loading with joblib."""
        test_path = SAVED_MODELS_DIR / "test_forecaster.joblib"
        forecaster = TimeSeriesInflowForecaster(target_col="HHS_Care").fit(self.macro_df)
        forecaster.save(test_path)
        self.assertTrue(test_path.exists())
        
        loaded = TimeSeriesInflowForecaster.load(test_path)
        self.assertTrue(loaded.is_fitted)
        self.assertEqual(loaded.feature_names, forecaster.feature_names)
        
        # Clean up
        if test_path.exists():
            test_path.unlink()

    def test_reunification_delay_classifier_fit_and_metrics(self):
        """Validates Random Forest classifier training, ROC-AUC, and feature importances."""
        clf = ReunificationDelayClassifier()
        clf.fit(self.cases_df)
        
        self.assertTrue(clf.is_fitted)
        metrics = clf.evaluation_metrics
        
        # Performance assertions
        self.assertIn("roc_auc", metrics)
        self.assertGreater(metrics["roc_auc"], 0.80, "ROC-AUC must exceed 0.80 benchmark")
        self.assertIn("precision_delayed", metrics)
        self.assertIn("recall_delayed", metrics)
        self.assertIn("f1_delayed", metrics)
        
        # Feature importances
        importances = clf.feature_importances
        self.assertGreater(len(importances), 0)
        # Top features should include sponsor risk tier or documentation score
        top_feature = list(importances.keys())[0]
        self.assertTrue(any(sub in top_feature for sub in ["sponsor_risk_tier", "documentation"]))

    def test_single_case_and_batch_inference(self):
        """Validates inference for single cases and batch DataFrames."""
        clf = ReunificationDelayClassifier().fit(self.cases_df)
        
        # Case A: Low risk, high documentation (should have low delay probability)
        case_low = {
            "sponsor_risk_tier": "Low",
            "documentation_score": 90.0,
            "processing_facility": "CBP-ELP-01",
            "initial_intake_month": 5
        }
        res_low = clf.predict_single_case(case_low)
        self.assertIn("prolonged_delay_probability", res_low)
        self.assertIn("risk_tier", res_low)
        self.assertIn("caseworker_recommendation", res_low)
        self.assertLess(res_low["prolonged_delay_probability"], 0.50, "Low-risk case should have <50% delay prob")
        
        # Case B: High risk, low documentation (should have high delay probability)
        case_high = {
            "sponsor_risk_tier": "High",
            "documentation_score": 25.0,
            "processing_facility": "CBP-RGV-02",
            "initial_intake_month": 7
        }
        res_high = clf.predict_single_case(case_high)
        self.assertGreater(res_high["prolonged_delay_probability"], 0.50, "High-risk case should have >50% delay prob")
        self.assertTrue(res_high["predicted_delayed_gt_45d"])
        
        # Batch inference
        batch_res = clf.batch_predict(self.cases_df.head(50))
        self.assertEqual(len(batch_res), 50)
        self.assertIn("delay_probability", batch_res.columns)
        self.assertIn("predicted_delayed_gt_45d", batch_res.columns)

    def test_classifier_serialization(self):
        """Validates classifier serialization via joblib."""
        test_path = SAVED_MODELS_DIR / "test_delay_clf.joblib"
        clf = ReunificationDelayClassifier().fit(self.cases_df)
        clf.save(test_path)
        self.assertTrue(test_path.exists())
        
        loaded_clf = ReunificationDelayClassifier.load(test_path)
        self.assertTrue(loaded_clf.is_fitted)
        self.assertEqual(loaded_clf.feature_names, clf.feature_names)
        
        # Clean up
        if test_path.exists():
            test_path.unlink()

if __name__ == "__main__":
    unittest.main()

