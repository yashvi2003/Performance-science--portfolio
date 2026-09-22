"""Unit tests for Project 2 Milestone M1b: DuckDB SQL Window Parity against Pandas Reference."""

import pytest
import numpy as np
import pandas as pd

from shared.synth import make_sessions_table, make_wearable_daily
from p2_nmt_reports.loader import load_sessions, aggregate_session_metrics
from p2_nmt_reports.baselines import compute_rolling_baselines
from p2_nmt_reports.sql_engine import (
    create_connection,
    load_tables,
    run_session_aggregates_sql,
    run_rolling_baselines_sql,
)


def test_sql_session_aggregates_matches_pandas():
    """Test that SQL session aggregates query exactly matches Pandas aggregation."""
    df_raw = make_sessions_table(n_athletes=3, n_weeks=5, random_seed=42)
    # Inject some invalid trials to verify WHERE valid = TRUE filtering
    df_raw.loc[5, "valid"] = False
    df_raw.loc[12, "valid"] = False

    # Pandas reference
    sessions_valid = load_sessions(df_raw, filter_valid=True)
    df_pandas = aggregate_session_metrics(sessions_valid, method="mean")

    # DuckDB SQL execution
    con = create_connection()
    load_tables(con, df_raw)
    df_sql = run_session_aggregates_sql(con)

    # Sort identically
    sort_cols = ["athlete_id", "metric", "side", "date"]
    df_pandas["date"] = pd.to_datetime(df_pandas["date"])
    df_sql["date"] = pd.to_datetime(df_sql["date"])
    
    df_pandas = df_pandas.sort_values(sort_cols).reset_index(drop=True)
    df_sql = df_sql.sort_values(sort_cols).reset_index(drop=True)

    assert len(df_sql) == len(df_pandas)
    assert np.all(df_sql["athlete_id"] == df_pandas["athlete_id"])
    assert np.all(df_sql["metric"] == df_pandas["metric"])
    assert np.all(df_sql["n_trials"] == df_pandas["n_trials"])
    np.testing.assert_allclose(df_sql["value"].values, df_pandas["value"].values, rtol=1e-7, atol=1e-7)


def test_sql_rolling_baselines_matches_pandas_exact():
    """Test that SQL window functions produce numerically identical rolling baselines to Pandas.

    Validates:
    - baseline_mean: |diff| < 1e-6
    - baseline_sd: |diff| < 1e-6
    - baseline_n: exact integer equality
    - baseline_established: exact boolean equality
    - swc: |diff| < 1e-6
    - typical_error: |diff| < 1e-6
    - pct_change: |diff| < 1e-6
    - z_score: |diff| < 1e-6
    """
    df_raw = make_sessions_table(n_athletes=4, n_weeks=8, random_seed=123)

    # 1. Pandas reference calculation
    sessions_valid = load_sessions(df_raw, filter_valid=True)
    agg_pandas = aggregate_session_metrics(sessions_valid, method="mean")
    df_pandas = compute_rolling_baselines(agg_pandas)

    # 2. DuckDB SQL execution
    con = create_connection()
    load_tables(con, df_raw)
    df_sql = run_rolling_baselines_sql(con)

    # Align sort orders
    sort_cols = ["athlete_id", "metric", "side", "date"]
    df_pandas["date"] = pd.to_datetime(df_pandas["date"])
    df_sql["date"] = pd.to_datetime(df_sql["date"])

    df_pandas = df_pandas.sort_values(sort_cols).reset_index(drop=True)
    df_sql = df_sql.sort_values(sort_cols).reset_index(drop=True)

    assert len(df_sql) == len(df_pandas)

    # Structural / boolean columns
    assert np.all(df_sql["athlete_id"] == df_pandas["athlete_id"])
    assert np.all(df_sql["metric"] == df_pandas["metric"])
    assert np.all(df_sql["baseline_n"] == df_pandas["baseline_n"])
    assert np.all(df_sql["baseline_established"] == df_pandas["baseline_established"])

    # Numerical metrics comparison where baseline is established
    mask = df_pandas["baseline_established"] == True
    assert mask.sum() > 0

    np.testing.assert_allclose(
        df_sql.loc[mask, "baseline_mean"].values,
        df_pandas.loc[mask, "baseline_mean"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL baseline_mean does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "baseline_sd"].values,
        df_pandas.loc[mask, "baseline_sd"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL baseline_sd does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "swc"].values,
        df_pandas.loc[mask, "swc"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL swc does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "typical_error"].values,
        df_pandas.loc[mask, "typical_error"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL typical_error does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "pct_change"].values,
        df_pandas.loc[mask, "pct_change"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL pct_change does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "z_score"].values,
        df_pandas.loc[mask, "z_score"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL direction-aware z_score does not match Pandas reference",
    )
