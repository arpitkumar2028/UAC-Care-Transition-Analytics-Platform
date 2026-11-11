"""
data_loader.py
Module for robust ingestion, cleaning, validation, and temporal structuring
of the HHS / CBP Unaccompanied Children Program dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "HHS_Unaccompanied_Children_Program.csv"
if not DEFAULT_DATA_PATH.exists():
    DEFAULT_DATA_PATH = Path("HHS_Unaccompanied_Children_Program.csv")

COLUMN_MAPPING = {
    "Date": "Date",
    "Children apprehended and placed in CBP custody*": "Apprehensions",
    "Children apprehended and placed in CBP custody": "Apprehensions",
    "Children in CBP custody": "CBP_Custody",
    "Children transferred out of CBP custody": "Transfers_Out_CBP",
    "Children in HHS Care": "HHS_Care",
    "Children discharged from HHS Care": "Discharges_HHS"
}

def load_and_clean_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Ingests and cleans the daily operational dataset.
    
    Operations performed:
    1. Removes trailing empty formatting rows.
    2. Parses dates into datetime objects and orders chronologically.
    3. Cleans string numbers with commas (e.g., '2,484' -> 2484.0).
    4. Enforces non-negative numeric constraints.
    5. Standardizes column names while retaining mapping aliases.
    6. Adds calendar features (DayOfWeek, IsWeekend, Month, Year).
    """
    path = Path(file_path) if file_path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    
    df = pd.read_csv(path)
    
    # 1. Drop completely empty rows (e.g. rows > 720)
    df = df.dropna(how='all').copy()
    
    # Drop rows where Date is null
    date_col = [c for c in df.columns if 'date' in c.lower()][0]
    df = df.dropna(subset=[date_col]).copy()
    
    # 2. Clean and rename columns
    renamed_cols = {}
    for col in df.columns:
        stripped_col = col.strip()
        matched = False
        for k, v in COLUMN_MAPPING.items():
            if k.lower() == stripped_col.lower() or k.strip('*').lower() == stripped_col.strip('*').lower():
                renamed_cols[col] = v
                matched = True
                break
        if not matched:
            renamed_cols[col] = stripped_col
            
    df = df.rename(columns=renamed_cols)
    
    # 3. Clean and parse Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # 4. Clean numerical columns
    numeric_cols = ['Apprehensions', 'CBP_Custody', 'Transfers_Out_CBP', 'HHS_Care', 'Discharges_HHS']
    for col in numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            # Enforce non-negative constraint
            df[col] = df[col].clip(lower=0)
    
    # 5. Calendar features
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['DayOfWeekNum'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6
    df['IsWeekend'] = df['DayOfWeekNum'].isin([5, 6]).astype(int)
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.strftime('%B')
    df['Year'] = df['Date'].dt.year
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    # 6. Add rolling 7-day and 30-day baseline smoothers
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col}_MA7'] = df[col].rolling(window=7, min_periods=1).mean()
            df[f'{col}_MA30'] = df[col].rolling(window=30, min_periods=1).mean()
            
    return df

def load_data_from_db(db_path: Optional[str] = None) -> pd.DataFrame:
    """
    Ingests daily macro operational metrics directly from the SQLite database.
    Standardizes schema to align with analytics and forecasting modules.
    """
    import sqlite3
    from db.connection import get_connection, get_db_path
    
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM daily_pipeline_metrics ORDER BY date ASC;"
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
        
    db_mapping = {
        "date": "Date",
        "cbp_apprehensions": "Apprehensions",
        "cbp_active_custody": "CBP_Custody",
        "cbp_transfers_out": "Transfers_Out_CBP",
        "hhs_active_care": "HHS_Care",
        "hhs_discharges": "Discharges_HHS",
        "sponsor_vetted_reunifications": "Vetted_Reunifications"
    }
    df = df.rename(columns=db_mapping)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    numeric_cols = ['Apprehensions', 'CBP_Custody', 'Transfers_Out_CBP', 'HHS_Care', 'Discharges_HHS']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(lower=0)
            
    # Calendar features
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['DayOfWeekNum'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeekNum'].isin([5, 6]).astype(int)
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.strftime('%B')
    df['Year'] = df['Date'].dt.year
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    
    # Rolling smoothers
    for col in numeric_cols:
        if col in df.columns:
            df[f'{col}_MA7'] = df[col].rolling(window=7, min_periods=1).mean()
            df[f'{col}_MA30'] = df[col].rolling(window=30, min_periods=1).mean()
            
    return df

def load_case_events_from_db(db_path: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Ingests micro case-level events ledger from the SQLite database.
    """
    from db.connection import get_connection
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM case_level_events ORDER BY apprehension_date DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns high-level structural and health summary of the loaded dataset.
    """
    return {
        "total_records": len(df),
        "start_date": df['Date'].min().strftime('%Y-%m-%d'),
        "end_date": df['Date'].max().strftime('%Y-%m-%d'),
        "date_span_days": (df['Date'].max() - df['Date'].min()).days,
        "total_apprehensions": int(df['Apprehensions'].sum()),
        "total_transfers": int(df['Transfers_Out_CBP'].sum()),
        "total_discharges": int(df['Discharges_HHS'].sum()),
        "avg_daily_apprehensions": float(round(df['Apprehensions'].mean(), 2)),
        "avg_daily_discharges": float(round(df['Discharges_HHS'].mean(), 2)),
        "current_cbp_custody": int(df['CBP_Custody'].iloc[-1]),
        "current_hhs_care": int(df['HHS_Care'].iloc[-1]),
    }
