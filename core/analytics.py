"""
core/analytics.py
Quantitative Analytics & Process Efficiency Engine for the UAC Care Pipeline.

Implements core transition ratios, throughput dynamics, backlog accumulation rates,
and rolling Z-score anomaly detection algorithms.

Mathematical Formulations (LaTeX):
1. Transfer Efficiency Ratio (TER):
   \\text{TER}_t = \\frac{\\text{Transfers Out of CBP}_t}{\\text{Active CBP Custody}_t}
   
2. Discharge Effectiveness Index (DEI):
   \\text{DEI}_t = \\frac{\\text{Discharges from HHS}_t}{\\text{Active HHS Care}_t}

3. Pipeline Throughput Rate (PTR) over rolling window W:
   \\text{PTR}_{t, W} = \\frac{\\sum_{\\tau=t-W+1}^t \\text{Reunifications}_\\tau}{\\sum_{\\tau=t-W+1}^t \\text{Apprehensions}_\\tau}

4. Backlog Accumulation Rate (BAR) over rolling window W:
   \\Delta B_\\tau = \\text{Apprehensions}_\\tau - \\text{Discharges}_\\tau
   \\text{BAR}_{t, W} = \\sum_{\\tau=t-W+1}^t \\Delta B_\\tau

5. Rolling Z-Score Anomaly Detection:
   \\mu_{t, 30} = \\frac{1}{30} \\sum_{k=0}^{29} \\text{BAR}_{t-k, 7}
   \\sigma_{t, 30} = \\sqrt{\\frac{1}{29} \\sum_{k=0}^{29} (\\text{BAR}_{t-k, 7} - \\mu_{t, 30})^2}
   Z_t = \\frac{\\text{BAR}_{t, 7} - \\mu_{t, 30}}{\\sigma_{t, 30}}
   \\text{Anomaly Flag}_t = \\mathbb{I}(Z_t > 2.0)
"""

import json
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

# Canonical column dictionary supporting both database and CSV alias naming
CANONICAL_COLUMNS = {
    "apprehensions": ["cbp_apprehensions", "Apprehensions", "children_apprehended"],
    "cbp_custody": ["cbp_active_custody", "CBP_Custody", "children_in_cbp_custody"],
    "transfers_out": ["cbp_transfers_out", "Transfers_Out_CBP", "children_transferred_out"],
    "hhs_care": ["hhs_active_care", "HHS_Care", "children_in_hhs_care"],
    "discharges": ["hhs_discharges", "Discharges_HHS", "children_discharged"],
    "reunifications": ["sponsor_vetted_reunifications", "Vetted_Reunifications", "reunifications"],
    "date": ["date", "Date"]
}

def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Finds first matching column from candidate list (case-insensitive)."""
    col_lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in col_lookup:
            return col_lookup[cand.lower()]
    return None

def normalize_pipeline_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes input DataFrame into canonical column schema:
    [date, cbp_apprehensions, cbp_active_custody, cbp_transfers_out,
     hhs_active_care, hhs_discharges, sponsor_vetted_reunifications]
    """
    clean_df = df.copy()
    
    date_col = _resolve_column(clean_df, CANONICAL_COLUMNS["date"])
    if date_col is None:
        raise ValueError("Input DataFrame missing required 'date' column.")
    clean_df["date"] = pd.to_datetime(clean_df[date_col], errors="coerce")
    clean_df = clean_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    
    mapping = {
        "apprehensions": "cbp_apprehensions",
        "cbp_custody": "cbp_active_custody",
        "transfers_out": "cbp_transfers_out",
        "hhs_care": "hhs_active_care",
        "discharges": "hhs_discharges",
        "reunifications": "sponsor_vetted_reunifications"
    }
    
    renamed = {}
    missing_keys = []
    for key, target in mapping.items():
        found = _resolve_column(clean_df, CANONICAL_COLUMNS[key])
        if found:
            renamed[found] = target
        else:
            missing_keys.append((key, target))
            
    clean_df = clean_df.rename(columns=renamed)
    
    # Handle missing optional fields (e.g. reunifications approximated from discharges)
    for key, target in missing_keys:
        if key == "reunifications" and "hhs_discharges" in clean_df.columns:
            clean_df[target] = (clean_df["hhs_discharges"] * 0.94).round().astype(int)
        else:
            raise ValueError(f"Missing essential pipeline column for '{key}'.")
    
    # Cast to numeric non-negative floats
    numeric_targets = list(mapping.values())
    for col in numeric_targets:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        
    return clean_df

def compute_transfer_efficiency_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Transfer Efficiency Ratio (TER):
    $$\\text{TER}_t = \\frac{\\text{Transfers out of CBP}_t}{\\text{Active CBP Custody}_t}$$
    
    Safe division: If Active CBP Custody is 0, returns 1.0 when Transfers > 0, else 0.0.
    """
    clean_df = normalize_pipeline_dataframe(df)
    transfers = clean_df["cbp_transfers_out"].values
    custody = clean_df["cbp_active_custody"].values
    
    safe_custody = np.maximum(custody, 1e-9)
    ter = np.where(
        custody > 0,
        transfers / safe_custody,
        np.where(transfers > 0, 1.0, 0.0)
    )
    return pd.Series(ter, index=clean_df.index, name="transfer_efficiency_ratio")

def compute_discharge_effectiveness_index(df: pd.DataFrame, as_percentage: bool = False) -> pd.Series:
    """
    Calculates Discharge Effectiveness Index (DEI):
    $$\\text{DEI}_t = \\frac{\\text{Discharges from HHS}_t}{\\text{Active HHS Care}_t}$$
    
    Parameters:
    - as_percentage: If True, multiplies ratio by 100 for percentage points.
    """
    clean_df = normalize_pipeline_dataframe(df)
    discharges = clean_df["hhs_discharges"].values
    care = clean_df["hhs_active_care"].values
    
    safe_care = np.maximum(care, 1e-9)
    dei = np.where(
        care > 0,
        discharges / safe_care,
        0.0
    )
    if as_percentage:
        dei = dei * 100.0
    return pd.Series(dei, index=clean_df.index, name="discharge_effectiveness_index")

def compute_pipeline_throughput_rate(
    df: pd.DataFrame,
    window: int = 7
) -> pd.Series:
    """
    Calculates Pipeline Throughput Rate (PTR) over a rolling window:
    $$\\text{PTR}_{t, W} = \\frac{\\sum_{\\tau=t-W+1}^t \\text{Reunifications}_\\tau}{\\sum_{\\tau=t-W+1}^t \\text{Apprehensions}_\\tau}$$
    
    Parameters:
    - window: Rolling window in days (default: 7 days)
    """
    clean_df = normalize_pipeline_dataframe(df)
    rolling_reunifications = clean_df["sponsor_vetted_reunifications"].rolling(window=window, min_periods=1).sum()
    rolling_apprehensions = clean_df["cbp_apprehensions"].rolling(window=window, min_periods=1).sum()
    
    ptr = np.where(
        rolling_apprehensions > 0,
        rolling_reunifications / rolling_apprehensions,
        np.where(rolling_reunifications > 0, 1.0, 1.0)
    )
    return pd.Series(ptr, index=clean_df.index, name=f"pipeline_throughput_rate_{window}d")

def compute_backlog_accumulation_rate(
    df: pd.DataFrame,
    window: int = 7
) -> pd.Series:
    """
    Calculates Backlog Accumulation Rate (BAR):
    $$\\Delta B_\\tau = \\text{CBP Apprehensions}_\\tau - \\text{HHS Discharges}_\\tau$$
    $$\\text{BAR}_{t, W} = \\sum_{\\tau=t-W+1}^t \\Delta B_\\tau$$
    
    Positive values represent net system accumulation (inflow > outflow);
    Negative values indicate system backlog clearance.
    """
    clean_df = normalize_pipeline_dataframe(df)
    daily_delta = clean_df["cbp_apprehensions"] - clean_df["hhs_discharges"]
    rolling_bar = daily_delta.rolling(window=window, min_periods=1).sum()
    return pd.Series(rolling_bar, index=clean_df.index, name=f"backlog_accumulation_rate_{window}d")

def detect_backlog_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.0,
    baseline_window: int = 30,
    backlog_window: int = 7
) -> pd.DataFrame:
    """
    Automated anomaly detection algorithm using rolling Z-scores.
    
    Identifies systemic bottlenecks by flagging dates where the Backlog
    Accumulation Rate exceeds `z_threshold` (default 2.0) standard deviations
    above its rolling `baseline_window` (default 30-day) moving average.
    
    Mathematical Formulation:
    $$\\mu_t = \\text{MA}_{30}(\\text{BAR}_7)_t$$
    $$\\sigma_t = \\text{Std}_{30}(\\text{BAR}_7)_t$$
    $$Z_t = \\frac{\\text{BAR}_{7, t} - \\mu_t}{\\sigma_t}$$
    $$\\text{is\\_anomaly}_t = (Z_t > z\\_threshold)$$
    """
    clean_df = normalize_pipeline_dataframe(df)
    bar = compute_backlog_accumulation_rate(clean_df, window=backlog_window)
    
    # 30-day rolling baseline statistics
    rolling_mean = bar.rolling(window=baseline_window, min_periods=5).mean()
    rolling_std = bar.rolling(window=baseline_window, min_periods=5).std().replace(0, np.nan)
    
    z_scores = (bar - rolling_mean) / rolling_std
    z_scores = z_scores.fillna(0.0)
    
    threshold_value = rolling_mean + (z_threshold * rolling_std)
    is_anomaly = (z_scores > z_threshold).astype(int)
    
    # Severity classification:
    # High: Z > 3.0, Medium: 2.5 < Z <= 3.0, Low: 2.0 < Z <= 2.5
    severity = np.where(
        z_scores > 3.0,
        "Critical",
        np.where(
            z_scores > 2.5,
            "High",
            np.where(z_scores > z_threshold, "Elevated", "Normal")
        )
    )
    
    results_df = pd.DataFrame({
        "date": clean_df["date"].dt.strftime("%Y-%m-%d"),
        "backlog_accumulation_rate": bar.round(2),
        "baseline_mean_30d": rolling_mean.round(2),
        "baseline_std_30d": rolling_std.round(2),
        "threshold_upper_bound": threshold_value.round(2),
        "z_score": z_scores.round(2),
        "is_anomaly": is_anomaly,
        "severity": severity
    })
    return results_df

def compute_full_analytics_suite(
    df: pd.DataFrame,
    as_json: bool = False
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Computes all process ratios, throughput rates, and anomaly detection flags.
    Returns either an enriched DataFrame or an API-ready JSON-serializable dictionary.
    """
    clean_df = normalize_pipeline_dataframe(df)
    
    clean_df["transfer_efficiency_ratio"] = compute_transfer_efficiency_ratio(clean_df).round(4)
    clean_df["discharge_effectiveness_index"] = compute_discharge_effectiveness_index(clean_df, as_percentage=False).round(4)
    clean_df["discharge_effectiveness_pct"] = (clean_df["discharge_effectiveness_index"] * 100).round(2)
    clean_df["pipeline_throughput_rate_7d"] = compute_pipeline_throughput_rate(clean_df, window=7).round(4)
    clean_df["backlog_accumulation_rate_7d"] = compute_backlog_accumulation_rate(clean_df, window=7).round(2)
    
    # Cumulative system backlog
    clean_df["daily_flow_balance"] = clean_df["cbp_apprehensions"] - clean_df["hhs_discharges"]
    clean_df["cumulative_backlog"] = clean_df["daily_flow_balance"].cumsum()
    
    # Anomaly detection results
    anomalies_df = detect_backlog_anomalies(clean_df, z_threshold=2.0, baseline_window=30, backlog_window=7)
    clean_df["anomaly_z_score"] = anomalies_df["z_score"]
    clean_df["is_backlog_anomaly"] = anomalies_df["is_anomaly"]
    clean_df["anomaly_severity"] = anomalies_df["severity"]
    
    if as_json:
        # Convert to JSON-serializable structure
        records = clean_df.copy()
        records["date"] = records["date"].dt.strftime("%Y-%m-%d")
        
        latest = records.iloc[-1]
        active_anomalies = records[records["is_backlog_anomaly"] == 1]
        
        payload = {
            "metadata": {
                "total_days_evaluated": len(records),
                "start_date": records["date"].iloc[0],
                "end_date": records["date"].iloc[-1],
                "total_anomalies_detected": int(active_anomalies["is_backlog_anomaly"].sum())
            },
            "latest_kpis": {
                "date": latest["date"],
                "transfer_efficiency_ratio": float(latest["transfer_efficiency_ratio"]),
                "discharge_effectiveness_pct": float(latest["discharge_effectiveness_pct"]),
                "pipeline_throughput_rate_7d": float(latest["pipeline_throughput_rate_7d"]),
                "backlog_accumulation_rate_7d": float(latest["backlog_accumulation_rate_7d"]),
                "cumulative_backlog": int(latest["cumulative_backlog"]),
                "is_current_anomaly": bool(latest["is_backlog_anomaly"]),
                "current_severity": str(latest["anomaly_severity"])
            },
            "time_series": records.to_dict(orient="records"),
            "anomaly_events": active_anomalies[[
                "date", "backlog_accumulation_rate_7d", "anomaly_z_score", "anomaly_severity"
            ]].to_dict(orient="records")
        }
        return payload
        
    return clean_df
