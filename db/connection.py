"""
db/connection.py
Database connection management for SQLite and PostgreSQL.
Provides thread-safe connections with WAL mode and robust error handling.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "uac_analytics.db"

def get_db_path() -> Path:
    """Returns the configured database path or default."""
    custom_path = os.getenv("SQLITE_DB_PATH")
    if custom_path:
        return Path(custom_path)
    return DEFAULT_SQLITE_PATH

def is_postgres_connection(conn) -> bool:
    """Returns True if the given DB connection is a PostgreSQL psycopg2 connection."""
    return type(conn).__module__.startswith("psycopg2")


def get_connection(db_path: Optional[str] = None):
    """
    Creates and configures a database connection.
    Supports PostgreSQL when DATABASE_URL is configured (with psycopg2),
    and defaults to high-concurrency SQLite with WAL mode.
    """
    db_url = os.getenv("DATABASE_URL", "")
    if db_path is None and db_url.startswith(("postgresql://", "postgres://")):
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            return conn
        except Exception:
            pass  # Fall back gracefully to local SQLite if driver/network unreachable

    target_path = Path(db_path) if db_path else get_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(
        str(target_path),
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        timeout=30.0
    )
    conn.row_factory = sqlite3.Row
    
    # SQLite optimization pragmas
    with conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        
    return conn

@contextmanager
def get_db_cursor(db_path: Optional[str] = None):
    """
    Context manager providing an active database cursor with automatic commit/rollback.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def execute_script(script_text: str, db_path: Optional[str] = None) -> None:
    """Executes a multi-statement SQL DDL/seed script on SQLite or PostgreSQL."""
    conn = get_connection(db_path)
    try:
        if is_postgres_connection(conn):
            with conn.cursor() as cur:
                cur.execute(script_text)
            conn.commit()
        else:
            with conn:
                conn.executescript(script_text)
    finally:
        conn.close()

