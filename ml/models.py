"""
ml/models.py
Machine Learning Module for Child Welfare Case Management & Operational Flow.

Features:
1. Time-Series Inflow Forecaster:
   - SARIMA / Fourier Seasonal Autoregressive pipeline projecting 30-day and 90-day
     HHS shelter bed volume surges based on historical apprehension and transition patterns.
2. Reunification Delay Classifier:
   - Random Forest / XGBoost classification pipeline using case-level features
     (sponsor risk tier, documentation score, processing facility, initial intake month).
   - Target: Binary classification predicting prolonged stagnation (>45 days in care).
   - Outputs ROC-AUC, Precision-Recall metrics, and feature importance rankings.
3. Production Inference Functions:
   - API & dashboard-ready prediction endpoints with model serialization (joblib).
"""

import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model storage directory
SAVED_MODELS_DIR = Path(__file__).resolve().parent / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FORECASTER_PATH = SAVED_MODELS_DIR / "inflow_forecaster.joblib"
DEFAULT_CLASSIFIER_PATH = SAVED_MODELS_DIR / "delay_classifier.joblib"


# ==============================================================================
# 1. TIME-SERIES INFLOW & SHELTER VOLUME SURGE FORECASTER
# ==============================================================================

class TimeSeriesInflowForecaster:
    """
    SARIMA-equivalent Fourier Seasonal Autoregressive Forecaster for predicting
    30-day and 90-day forward HHS shelter care volumes and border apprehension surges.
    """
    def __init__(self, target_col: str = "HHS_Care"):
        self.target_col = target_col
        self.model = GradientBoostingRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            random_state=42
        )
        self.is_fitted = False
        self.training_metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs harmonic Fourier terms (annual seasonality), weekly indicators,
        autoregressive lags, and rolling momentum features.
        """
        feat_df = pd.DataFrame(index=df.index)
        dates = pd.to_datetime(df["Date"])
        
        # 1. Fourier Seasonal Harmonics (Annual Cycle: 365.25 days)
        day_of_year = dates.dt.dayofyear.values
        for k in [1, 2, 4]:
            feat_df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
            feat_df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)
            
        # 2. Day of week & weekend indicators
        feat_df["dow"] = dates.dt.dayofweek
        feat_df["is_weekend"] = feat_df["dow"].isin([5, 6]).astype(int)
        feat_df["month"] = dates.dt.month
        
        # 3. Autoregressive Lags & Rolling Moving Averages
        target_series = df[self.target_col]
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            feat_df[f"lag_{lag}"] = target_series.shift(lag)
            
        feat_df["rolling_mean_7"] = target_series.shift(1).rolling(window=7, min_periods=1).mean()
        feat_df["rolling_std_7"] = target_series.shift(1).rolling(window=7, min_periods=1).std().fillna(0)
        feat_df["rolling_mean_30"] = target_series.shift(1).rolling(window=30, min_periods=1).mean()
        
        # 4. Leading Inflow Indicators (CBP Apprehensions & Transfers)
        if "Apprehensions" in df.columns:
            feat_df["apprehensions_lag1"] = df["Apprehensions"].shift(1)
            feat_df["apprehensions_ma7"] = df["Apprehensions"].shift(1).rolling(window=7, min_periods=1).mean()
        if "Transfers_Out_CBP" in df.columns:
            feat_df["transfers_lag1"] = df["Transfers_Out_CBP"].shift(1)
            
        return feat_df

    def fit(self, df: pd.DataFrame) -> "TimeSeriesInflowForecaster":
        """Fits forecaster on historical daily pipeline observations."""
        clean_df = df.copy()
        date_col = [c for c in clean_df.columns if "date" in c.lower()][0]
        clean_df["Date"] = pd.to_datetime(clean_df[date_col])
        clean_df = clean_df.sort_values("Date").reset_index(drop=True)
        
        # Map target column alias if needed
        if self.target_col not in clean_df.columns:
            if "hhs_active_care" in clean_df.columns:
                clean_df["HHS_Care"] = clean_df["hhs_active_care"]
                self.target_col = "HHS_Care"
            elif "children in hhs care" in clean_df.columns.str.lower():
                clean_df["HHS_Care"] = clean_df.filter(like="HHS Care").iloc[:, 0]
                self.target_col = "HHS_Care"
                
        X = self._engineer_features(clean_df)
        y = clean_df[self.target_col]
        
        # Drop warm-up NaN rows from lags
        valid_idx = X.dropna().index
        X_train, y_train = X.loc[valid_idx], y.loc[valid_idx]
        self.feature_names = X_train.columns.tolist()
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Calculate training residual error metrics
        preds = self.model.predict(X_train)
        mae = float(round(mean_absolute_error(y_train, preds), 2))
        rmse = float(round(np.sqrt(mean_squared_error(y_train, preds)), 2))
        mape = float(round(np.mean(np.abs((y_train - preds) / np.maximum(y_train, 1))) * 100, 2))
        
        self.training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "target": self.target_col,
            "observations_used": len(X_train),
            "mae": mae,
            "rmse": rmse,
            "mape_pct": mape,
            "last_training_date": str(clean_df["Date"].max())[:10]
        }
        return self

    def predict_surges(self, df: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
        """
        Executes multi-step forward simulation for 30-day or 90-day horizons.
        Returns forward trajectory with 95% confidence intervals and surge flags.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_surges can be called.")
            
        clean_df = df.copy()
        clean_df["Date"] = pd.to_datetime(clean_df["Date"])
        clean_df = clean_df.sort_values("Date").reset_index(drop=True)
        
        last_date = clean_df["Date"].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        
        # Recursive multi-step rollout
        sim_df = clean_df.copy()
        forecasts = []
        
        # Recent baseline standard deviation for confidence intervals
        recent_std = sim_df[self.target_col].iloc[-30:].std()
        
        for dt in future_dates:
            # Build feature row
            features = self._engineer_features(sim_df).iloc[[-1]]
            pred_val = float(self.model.predict(features.fillna(0))[0])
            pred_val = max(0.0, pred_val)
            
            forecasts.append(pred_val)
            
            # Append predicted step to sim_df for next lag step
            new_row = {col: 0 for col in sim_df.columns}
            new_row["Date"] = dt
            new_row[self.target_col] = pred_val
            if "Apprehensions" in sim_df.columns:
                new_row["Apprehensions"] = sim_df["Apprehensions"].iloc[-14:].mean()
            if "Transfers_Out_CBP" in sim_df.columns:
                new_row["Transfers_Out_CBP"] = sim_df["Transfers_Out_CBP"].iloc[-14:].mean()
                
            sim_df = pd.concat([sim_df, pd.DataFrame([new_row])], ignore_index=True)
            
        fc_series = np.array(forecasts)
        # Expanding uncertainty band over horizon (sigma * sqrt(h))
        horizon_steps = np.arange(1, horizon_days + 1)
        uncertainty = recent_std * np.sqrt(horizon_steps / 14.0) * 1.96
        
        lower_bound = np.maximum(0, fc_series - uncertainty)
        upper_bound = fc_series + uncertainty
        
        # Surge detection (> 15% increase above current baseline)
        baseline_census = clean_df[self.target_col].iloc[-1]
        is_surge = (fc_series > (baseline_census * 1.15)).astype(int)
        
        out_df = pd.DataFrame({
            "date": future_dates.strftime("%Y-%m-%d"),
            "projected_hhs_census": np.round(fc_series).astype(int),
            "lower_95ci": np.round(lower_bound).astype(int),
            "upper_95ci": np.round(upper_bound).astype(int),
            "is_surge_alert": is_surge,
            "projected_net_delta": np.round(fc_series - baseline_census).astype(int)
        })
        return out_df

    def save(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """Serializes trained model to disk."""
        target = Path(file_path) if file_path else DEFAULT_FORECASTER_PATH
        joblib.dump({"model": self.model, "metadata": self.training_metadata, "features": self.feature_names}, target)
        return str(target)

    @classmethod
    def load(cls, file_path: Optional[Union[str, Path]] = None) -> "TimeSeriesInflowForecaster":
        """Deserializes forecaster from disk."""
        target = Path(file_path) if file_path else DEFAULT_FORECASTER_PATH
        if not target.exists():
            raise FileNotFoundError(f"Saved model file not found at {target}")
        data = joblib.load(target)
        forecaster = cls()
        forecaster.model = data["model"]
        forecaster.training_metadata = data["metadata"]
        forecaster.feature_names = data["features"]
        forecaster.is_fitted = True
        return forecaster


# ==============================================================================
# 2. REUNIFICATION DELAY CLASSIFIER (>45 DAYS PROLONGED STAGNATION)
# ==============================================================================

class ReunificationDelayClassifier:
    """
    Random Forest Classification Pipeline predicting prolonged child welfare
    shelter stagnation (>45 days dwell time before sponsor discharge).
    """
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        self.evaluation_metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.feature_importances: Dict[str, float] = {}

    def _prepare_training_data(self, cases_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Enriches raw case records with synthetic documentation completeness score,
        calculates actual dwell days, and creates binary target (>45 days).
        """
        df = cases_df.copy()
        
        # 1. Parse timestamps
        df["apprehension_date"] = pd.to_datetime(df["apprehension_date"])
        df["cbp_transfer_date"] = pd.to_datetime(df["cbp_transfer_date"])
        df["discharge_date"] = pd.to_datetime(df["discharge_date"])
        
        # 2. Dwell Time Calculation (HHS Shelter stay)
        # If discharged, exact difference; else active dwell up to latest observation
        hhs_start = df["cbp_transfer_date"].fillna(df["apprehension_date"])
        hhs_end = df["discharge_date"].fillna(df["apprehension_date"].max())
        dwell_days = (hhs_end - hhs_start).dt.days.clip(lower=1)
        
        # Target: Binary classification (> 45 days in care = Prolonged Stagnation)
        y = (dwell_days > 45).astype(int)
        
        # 3. Documentation Score Synthesis (if not present)
        # Represents biometric clearance, proof of relationship, and background check completeness
        np.random.seed(42)
        if "documentation_score" not in df.columns:
            # Low risk sponsors typically have higher documentation scores
            doc_scores = []
            for tier in df["sponsor_risk_tier"]:
                if tier == "Low":
                    score = np.random.beta(a=8, b=2) * 100.0  # Mean ~80
                elif tier == "Medium":
                    score = np.random.beta(a=5, b=5) * 100.0  # Mean ~50
                else:
                    score = np.random.beta(a=3, b=7) * 100.0  # Mean ~30
                doc_scores.append(round(score, 1))
            df["documentation_score"] = doc_scores
            
        # 4. Feature Extraction
        X = pd.DataFrame()
        X["sponsor_risk_tier"] = df["sponsor_risk_tier"].fillna("Medium")
        X["documentation_score"] = df["documentation_score"].astype(float)
        X["processing_facility"] = df["processing_facility_id"].fillna("CBP-ELP-01")
        X["initial_intake_month"] = df["apprehension_date"].dt.month
        X["intake_day_of_week"] = df["apprehension_date"].dt.dayofweek
        
        # CBP holding latency (stage 1 dwell)
        cbp_dwell = (df["cbp_transfer_date"] - df["apprehension_date"]).dt.days.fillna(2.5)
        X["cbp_holding_days"] = cbp_dwell.clip(lower=1, upper=10)
        
        return X, y

    def fit(self, cases_df: pd.DataFrame) -> "ReunificationDelayClassifier":
        """Trains Random Forest classifier and computes diagnostic evaluation metrics."""
        X, y = self._prepare_training_data(cases_df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        
        categorical_cols = ["sponsor_risk_tier", "processing_facility"]
        numeric_cols = ["documentation_score", "initial_intake_month", "intake_day_of_week", "cbp_holding_days"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
            ]
        )
        
        rf_classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=8,
            class_weight="balanced",
            random_state=42
        )
        
        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_classifier)
        ])
        
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        
        # Evaluation Metrics
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]
        
        roc_auc = float(round(roc_auc_score(y_test, y_prob), 4))
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        conf_mat = confusion_matrix(y_test, y_pred).tolist()
        
        # Extract Feature Importances
        cat_encoder = self.pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        all_features = numeric_cols + encoded_cat_names
        self.feature_names = all_features
        
        raw_importances = self.pipeline.named_steps["classifier"].feature_importances_
        importance_dict = {
            feat: float(round(imp, 4))
            for feat, imp in sorted(zip(all_features, raw_importances), key=lambda x: x[1], reverse=True)
        }
        self.feature_importances = importance_dict
        
        self.evaluation_metrics = {
            "roc_auc": roc_auc,
            "accuracy": float(round(report_dict["accuracy"], 4)),
            "precision_delayed": float(round(report_dict["1"]["precision"], 4)),
            "recall_delayed": float(round(report_dict["1"]["recall"], 4)),
            "f1_delayed": float(round(report_dict["1"]["f1-score"], 4)),
            "confusion_matrix": conf_mat,
            "feature_importances": importance_dict
        }
        return self

    def predict_single_case(self, case_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference function for predicting delay probability of an individual child case.
        
        Expected dictionary keys:
        - sponsor_risk_tier: 'Low', 'Medium', or 'High'
        - documentation_score: float (0.0 to 100.0)
        - processing_facility: 'CBP-ELP-01', 'CBP-RGV-02', etc.
        - initial_intake_month: int (1-12)
        - intake_day_of_week: int (0-6, optional, default: 2)
        - cbp_holding_days: float (optional, default: 2.5)
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_single_case can be called.")
            
        input_df = pd.DataFrame([{
            "sponsor_risk_tier": case_dict.get("sponsor_risk_tier", "Medium"),
            "documentation_score": float(case_dict.get("documentation_score", 50.0)),
            "processing_facility": case_dict.get("processing_facility", "CBP-ELP-01"),
            "initial_intake_month": int(case_dict.get("initial_intake_month", 6)),
            "intake_day_of_week": int(case_dict.get("intake_day_of_week", 2)),
            "cbp_holding_days": float(case_dict.get("cbp_holding_days", 2.5))
        }])
        
        delay_prob = float(self.pipeline.predict_proba(input_df)[0][1])
        is_delayed = bool(delay_prob >= 0.50)
        
        # Risk Severity Tier
        if delay_prob > 0.75:
            severity = "Critical Risk (>75% Stagnation Probability)"
            recommendation = "Deploy expedited Category 1 vetting supervisor & trigger home-study fast-track."
        elif delay_prob > 0.50:
            severity = "High Risk (50-75% Probability)"
            recommendation = "Schedule biometric background follow-up within 72 hours."
        elif delay_prob > 0.30:
            severity = "Moderate Risk (30-50% Probability)"
            recommendation = "Standard sponsor documentation tracking."
        else:
            severity = "Low Risk (<30% Probability)"
            recommendation = "Rapid path to sponsor reunification."
            
        return {
            "prolonged_delay_probability": round(delay_prob, 3),
            "predicted_delayed_gt_45d": is_delayed,
            "risk_tier": severity,
            "caseworker_recommendation": recommendation,
            "top_risk_drivers": list(self.feature_importances.keys())[:3]
        }

    def batch_predict(self, cases_df: pd.DataFrame) -> pd.DataFrame:
        """Runs batch inference on a DataFrame of case records."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before batch_predict can be called.")
            
        X, _ = self._prepare_training_data(cases_df)
        probs = self.pipeline.predict_proba(X)[:, 1]
        
        res_df = cases_df.copy()
        res_df["delay_probability"] = np.round(probs, 3)
        res_df["predicted_delayed_gt_45d"] = (probs >= 0.50).astype(int)
        res_df["risk_classification"] = np.where(
            probs > 0.75, "Critical Risk",
            np.where(probs > 0.50, "High Risk",
                     np.where(probs > 0.30, "Moderate Risk", "Low Risk"))
        )
        return res_df

    def save(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """Serializes trained classifier pipeline to disk."""
        target = Path(file_path) if file_path else DEFAULT_CLASSIFIER_PATH
        joblib.dump({
            "pipeline": self.pipeline,
            "metrics": self.evaluation_metrics,
            "features": self.feature_names,
            "importances": self.feature_importances
        }, target)
        return str(target)

    @classmethod
    def load(cls, file_path: Optional[Union[str, Path]] = None) -> "ReunificationDelayClassifier":
        """Deserializes classifier pipeline from disk."""
        target = Path(file_path) if file_path else DEFAULT_CLASSIFIER_PATH
        if not target.exists():
            raise FileNotFoundError(f"Saved model file not found at {target}")
        data = joblib.load(target)
        clf = cls()
        clf.pipeline = data["pipeline"]
        clf.evaluation_metrics = data["metrics"]
        clf.feature_names = data["features"]
        clf.feature_importances = data["importances"]
        clf.is_fitted = True
        return clf


# ==============================================================================
# 3. HIGH-LEVEL API & DASHBOARD INTERFACE WRAPPERS
# ==============================================================================

def train_and_export_all_models(
    macro_df: Optional[pd.DataFrame] = None,
    cases_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Orchestration routine training both the Inflow Forecaster and Reunification
    Delay Classifier, saving artifacts to disk and returning performance diagnostics.
    """
    from care_pipeline.data_loader import load_data_from_db, load_case_events_from_db, load_and_clean_data
    
    # 1. Load data
    if macro_df is None:
        try:
            macro_df = load_data_from_db()
        except Exception:
            macro_df = load_and_clean_data()
            
    if cases_df is None:
        cases_df = load_case_events_from_db(limit=10000)
        
    print("Fitting Time-Series Inflow Forecaster...")
    forecaster = TimeSeriesInflowForecaster()
    forecaster.fit(macro_df)
    forecaster_path = forecaster.save()
    
    print("Fitting Reunification Delay Classifier...")
    classifier = ReunificationDelayClassifier()
    classifier.fit(cases_df)
    classifier_path = classifier.save()
    
    return {
        "forecaster_path": forecaster_path,
        "forecaster_metrics": forecaster.training_metadata,
        "classifier_path": classifier_path,
        "classifier_metrics": classifier.evaluation_metrics
    }

def get_inflow_forecast(horizon_days: int = 30) -> pd.DataFrame:
    """Convenience helper returning forward surge predictions."""
    from care_pipeline.data_loader import load_data_from_db, load_and_clean_data
    try:
        df = load_data_from_db()
    except Exception:
        df = load_and_clean_data()
        
    if DEFAULT_FORECASTER_PATH.exists():
        forecaster = TimeSeriesInflowForecaster.load()
    else:
        forecaster = TimeSeriesInflowForecaster()
        forecaster.fit(df)
        forecaster.save()
        
    return forecaster.predict_surges(df, horizon_days=horizon_days)

def predict_single_case_risk(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience helper returning stagnation risk prediction for a case dictionary."""
    if DEFAULT_CLASSIFIER_PATH.exists():
        clf = ReunificationDelayClassifier.load()
    else:
        from care_pipeline.data_loader import load_case_events_from_db
        cases_df = load_case_events_from_db(limit=5000)
        clf = ReunificationDelayClassifier()
        clf.fit(cases_df)
        clf.save()
        
    return clf.predict_single_case(case_data)

