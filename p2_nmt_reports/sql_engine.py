"""DuckDB SQL Execution Engine for Project 2.

Loads contract tables into DuckDB, executes modular .sql window queries,
and returns analytics DataFrames.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Union
import duckdb
import pandas as pd

from shared.contract import validate_sessions_df, validate_wearable_daily_df
from p2_nmt_reports.loader import load_sessions, load_wearable_daily

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent / "sql"


def create_connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection and initialize contract schema."""
    con = duckdb.connect(database=":memory:")
    schema_path = SQL_DIR / "schema.sql"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            con.execute(f.read())
    return con


def load_tables(
    con: duckdb.DuckDBPyConnection,
    sessions: Union[str, Path, pd.DataFrame],
    wearable: Optional[Union[str, Path, pd.DataFrame]] = None,
) -> None:
    """Load validated sessions and wearable dataframes into DuckDB tables."""
    if isinstance(sessions, (str, Path, pd.DataFrame)):
        sessions_df = load_sessions(sessions, filter_valid=False)
    else:
        sessions_df = sessions

    # Keep only contract columns for sessions table
    cols_sessions = [
        "session_id", "date", "athlete_id", "test", "trial",
        "metric", "value", "unit", "side", "valid", "flags", "block"
    ]
    avail_sess_cols = [c for c in cols_sessions if c in sessions_df.columns]
    sess_to_load = sessions_df[avail_sess_cols].copy()

    con.execute("DELETE FROM sessions")
    con.register("temp_sessions", sess_to_load)
    con.execute(f"INSERT INTO sessions ({', '.join(avail_sess_cols)}) SELECT * FROM temp_sessions")
    con.unregister("temp_sessions")

    if wearable is not None:
        if isinstance(wearable, (str, Path, pd.DataFrame)):
            wearable_df = load_wearable_daily(wearable)
        else:
            wearable_df = wearable

        cols_wearable = [
            "date", "athlete_id", "recovery_pct", "hrv_ms",
            "rhr_bpm", "day_strain", "sleep_hours", "proxy_used"
        ]
        avail_wear_cols = [c for c in cols_wearable if c in wearable_df.columns]
        wear_to_load = wearable_df[avail_wear_cols].copy()

        con.execute("DELETE FROM wearable_daily")
        con.register("temp_wearable", wear_to_load)
        con.execute(f"INSERT INTO wearable_daily ({', '.join(avail_wear_cols)}) SELECT * FROM temp_wearable")
        con.unregister("temp_wearable")


def run_session_aggregates_sql(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Run session_aggregates.sql query in DuckDB."""
    query_path = SQL_DIR / "session_aggregates.sql"
    with open(query_path, "r", encoding="utf-8") as f:
        query = f.read()
    return con.execute(query).df()


def run_rolling_baselines_sql(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Run rolling_baselines.sql window query in DuckDB."""
    query_path = SQL_DIR / "rolling_baselines.sql"
    with open(query_path, "r", encoding="utf-8") as f:
        query = f.read()
    df = con.execute(query).df()
    df["date"] = pd.to_datetime(df["date"])
    return df
