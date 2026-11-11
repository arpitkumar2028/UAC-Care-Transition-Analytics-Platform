"""
db/init_db.py
Database initialization and seeding script for UAC Care Transition Efficiency.
Provisions schema, executes data generation, performs bulk inserts, and runs integrity checks.
"""

import os
import sys
import argparse
import sqlite3
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from db.connection import get_connection, get_db_path, execute_script, is_postgres_connection
    from utils.data_generator import generate_macro_pipeline_data, generate_micro_case_events
except (ImportError, ModuleNotFoundError):
    from connection import get_connection, get_db_path, execute_script, is_postgres_connection
    from utils.data_generator import generate_macro_pipeline_data, generate_micro_case_events

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

def init_schema(db_path: Path = None) -> None:
    """Reads schema.sql and creates tables/indexes."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    execute_script(schema_sql, db_path=str(db_path) if db_path else None)

def _insert_dataframe_postgres(conn, table_name: str, df: pd.DataFrame) -> None:
    """Efficient bulk insert into PostgreSQL via psycopg2.extras.execute_values."""
    from psycopg2.extras import execute_values

    clean_df = df.where(pd.notnull(df), None)
    cols = list(clean_df.columns)
    col_list = ", ".join(cols)
    tuples = [
        tuple(None if (pd.isna(val) if not isinstance(val, (str, bytes, type(None))) else False) else val for val in row)
        for row in clean_df.itertuples(index=False, name=None)
    ]
    query = f"INSERT INTO {table_name} ({col_list}) VALUES %s ON CONFLICT DO NOTHING;"
    with conn.cursor() as cur:
        execute_values(cur, query, tuples, page_size=2000)


def seed_database(
    num_days: int = 1095,
    max_cases_per_day: int = 25,
    seed: int = 42,
    reset: bool = False,
    if_empty: bool = False,
    db_path: Path = None
) -> dict:
    """
    Initializes database schema and populates tables with synthetic data.
    Supports both local SQLite and PostgreSQL (via DATABASE_URL).
    
    Returns a dictionary of ingestion statistics and validation flags.
    """
    db_url = os.getenv("DATABASE_URL", "")
    using_postgres = (db_path is None and db_url.startswith(("postgresql://", "postgres://")))
    target_path = Path(db_path) if db_path else get_db_path()
    
    if reset and not using_postgres and target_path.exists():
        target_path.unlink()
        
    # 1. Initialize schema
    init_schema(target_path if db_path is not None else None)

    if if_empty:
        check_conn = get_connection(str(target_path) if db_path is not None else None)
        try:
            cur = check_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM daily_pipeline_metrics;")
            existing_rows = cur.fetchone()[0]
        finally:
            check_conn.close()
        if existing_rows and existing_rows > 0:
            print(f"Database already contains {existing_rows:,} macro rows; skipping seed (--if-empty).")
            return validate_seeded_data(target_path if db_path is not None else None)
    
    # 2. Generate Synthetic Datasets
    print(f"Generating {num_days} days of macro pipeline metrics...")
    macro_df = generate_macro_pipeline_data(num_days=num_days, seed=seed)
    
    print(f"Generating micro case events (capped at {max_cases_per_day}/day)...")
    micro_df = generate_micro_case_events(
        macro_df=macro_df,
        max_cases_per_day=max_cases_per_day,
        seed=seed
    )
    
    # 3. Insert into Database
    conn = get_connection(str(target_path) if db_path is not None else None)
    try:
        if is_postgres_connection(conn):
            try:
                if reset:
                    with conn.cursor() as cur:
                        cur.execute("TRUNCATE TABLE case_level_events, daily_pipeline_metrics;")
                _insert_dataframe_postgres(conn, "daily_pipeline_metrics", macro_df)
                _insert_dataframe_postgres(conn, "case_level_events", micro_df)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        else:
            with conn:
                if reset:
                    conn.execute("DELETE FROM case_level_events;")
                    conn.execute("DELETE FROM daily_pipeline_metrics;")
                macro_df.to_sql(
                    name="daily_pipeline_metrics",
                    con=conn,
                    if_exists="append",
                    index=False
                )
                micro_df.to_sql(
                    name="case_level_events",
                    con=conn,
                    if_exists="append",
                    index=False
                )
    finally:
        conn.close()
        
    # 4. Run Integrity Validations
    stats = validate_seeded_data(target_path if db_path is not None else None)
    return stats

def validate_seeded_data(db_path: Path = None) -> dict:
    """Runs automated post-seeding sanity and flow consistency checks."""
    conn = get_connection(str(db_path) if db_path else None)
    try:
        cursor = conn.cursor()
        
        # Row counts
        cursor.execute("SELECT COUNT(*) FROM daily_pipeline_metrics;")
        macro_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM case_level_events;")
        micro_count = cursor.fetchone()[0]
        
        # Non-negativity check
        cursor.execute("""
            SELECT COUNT(*) FROM daily_pipeline_metrics 
            WHERE cbp_apprehensions < 0 
               OR cbp_active_custody < 0 
               OR cbp_transfers_out < 0 
               OR hhs_active_care < 0 
               OR hhs_discharges < 0 
               OR sponsor_vetted_reunifications < 0;
        """)
        invalid_metric_rows = cursor.fetchone()[0]
        
        # Micro chronological order check
        cursor.execute("""
            SELECT COUNT(*) FROM case_level_events
            WHERE (cbp_transfer_date IS NOT NULL AND cbp_transfer_date < apprehension_date)
               OR (discharge_date IS NOT NULL AND discharge_date < hhs_intake_date);
        """)
        chronology_errors = cursor.fetchone()[0]
        
        # Risk tier distribution
        cursor.execute("""
            SELECT sponsor_risk_tier, COUNT(*) 
            FROM case_level_events 
            GROUP BY sponsor_risk_tier;
        """)
        risk_dist = dict(cursor.fetchall())
        
        # Status distribution
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM case_level_events 
            GROUP BY status;
        """)
        status_dist = dict(cursor.fetchall())
        
        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM daily_pipeline_metrics;")
        min_date, max_date = cursor.fetchone()
        
        is_valid = (
            macro_count > 0 and 
            micro_count > 0 and 
            invalid_metric_rows == 0 and 
            chronology_errors == 0
        )
        
        return {
            "is_valid": is_valid,
            "macro_records": macro_count,
            "micro_cases": micro_count,
            "date_range": f"{min_date} to {max_date}",
            "invalid_metric_rows": invalid_metric_rows,
            "chronology_errors": chronology_errors,
            "risk_distribution": risk_dist,
            "status_distribution": status_dist
        }
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize and seed the UAC database.")
    parser.add_argument("--days", type=int, default=1095, help="Number of historical days to simulate (default: 1095 = 3 years)")
    parser.add_argument("--max-cases", type=int, default=25, help="Max micro cases per day (default: 25)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild database tables")
    parser.add_argument("--if-empty", action="store_true", help="Only seed when daily_pipeline_metrics has 0 rows")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite database file path")
    
    args = parser.parse_args()
    db_target = Path(args.db_path) if args.db_path else None
    
    print("=" * 60)
    print("UAC CARE TRANSITION ANALYTICS - DATABASE INITIALIZATION")
    print("=" * 60)
    
    stats = seed_database(
        num_days=args.days,
        max_cases_per_day=args.max_cases,
        seed=args.seed,
        reset=args.reset,
        if_empty=args.if_empty,
        db_path=db_target
    )
    
    print("\nInitialization Results:")
    print(f"  Valid Database State   : {stats['is_valid']}")
    print(f"  Macro Metric Records   : {stats['macro_records']:,}")
    print(f"  Micro Case Events      : {stats['micro_cases']:,}")
    print(f"  Date Span              : {stats['date_range']}")
    print(f"  Chronology Errors      : {stats['chronology_errors']}")
    print(f"  Risk Tiers             : {stats['risk_distribution']}")
    print(f"  Lifecycle Statuses     : {stats['status_distribution']}")
    print("=" * 60)
    print("Database seeding completed successfully.")
