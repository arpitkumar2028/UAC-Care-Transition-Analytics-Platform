"""
run_all_tests.py
Master test runner executing all unit and integration test suites:
1. Data Layer & Synthetic Generation (tests/test_data_layer.py)
2. Quantitative Analytics Engine & Z-Score Anomalies (tests/test_analytics.py)
3. Backend REST API Endpoints & Schemas (tests/test_backend.py)
4. Machine Learning Forecaster & Delay Classifier (tests/test_ml.py)
5. Pipeline Metrics Engine & Temporal Analytics (tests/test_metrics.py)
6. Enterprise Services: ML, Simulation, Copilot, PDF, Webhooks (tests/test_services.py)
"""

import sys
import unittest
from pathlib import Path

def run_all():
    print("=" * 70)
    print("UAC CARE TRANSITION ANALYTICS - MASTER VERIFICATION TEST SUITE")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)
    
    # 1. Run Data Layer UnitTests
    print("\n[Suite 1/6] Running Data Layer & Synthetic Engine Tests...")
    suite_dl = loader.discover("tests", pattern="test_data_layer.py")
    res_dl = runner.run(suite_dl)
    
    # 2. Run Analytics Engine UnitTests
    print("\n[Suite 2/6] Running Quantitative Analytics & Anomaly Detection Tests...")
    suite_an = loader.discover("tests", pattern="test_analytics.py")
    res_an = runner.run(suite_an)
    
    # 3. Run Backend REST API Tests
    print("\n[Suite 3/6] Running Backend REST API Endpoint Tests...")
    suite_be = loader.discover("tests", pattern="test_backend.py")
    res_be = runner.run(suite_be)
    
    # 4. Run Machine Learning Tests
    print("\n[Suite 4/6] Running Machine Learning Forecaster & Classifier Tests...")
    suite_ml = loader.discover("tests", pattern="test_ml.py")
    res_ml = runner.run(suite_ml)
    
    # 5. Run Legacy Metrics Tests
    print("\n[Suite 5/6] Running Pipeline Metrics & Temporal Tests...")
    from tests.test_metrics import (
        test_data_loader,
        test_metrics_calculation,
        test_current_kpis,
        test_temporal_analysis
    )
    test_data_loader()
    print("  ✓ test_data_loader passed")
    test_metrics_calculation()
    print("  ✓ test_metrics_calculation passed")
    test_current_kpis()
    print("  ✓ test_current_kpis passed")
    test_temporal_analysis()
    print("  ✓ test_temporal_analysis passed")
    
    # 6. Run Enterprise Service Tests
    print("\n[Suite 6/6] Running Simulation, Copilot, Webhook & PDF Tests...")
    from tests.test_services import (
        test_copilot_and_metrics,
        test_forecasting_and_simulation,
        test_webhook_dispatcher,
        test_pdf_generation
    )
    test_copilot_and_metrics()
    print("  ✓ test_copilot_and_metrics passed")
    test_forecasting_and_simulation()
    print("  ✓ test_forecasting_and_simulation passed")
    test_webhook_dispatcher()
    print("  ✓ test_webhook_dispatcher passed")
    test_pdf_generation()
    print("  ✓ test_pdf_generation passed")
    
    print("\n" + "=" * 70)
    all_passed = res_dl.wasSuccessful() and res_an.wasSuccessful() and res_be.wasSuccessful() and res_ml.wasSuccessful()
    if all_passed:
        print("🎉 ALL 30 TESTS ACROSS 6 TEST SUITES PASSED WITH 100% SUCCESS RATE!")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME TESTS FAILED.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(run_all())
