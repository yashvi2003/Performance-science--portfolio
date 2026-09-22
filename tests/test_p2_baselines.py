"""Unit tests for Project 2 Milestone M1a: Loader, Session Aggregation, and Rolling Baselines."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from shared.synth import make_sessions_table, make_wearable_daily
from p2_nmt_reports.config import load_config, BaselineConfig
from p2_nmt_reports.loader import load_sessions, load_wearable_daily, aggregate_session_metrics
from p2_nmt_reports.baselines import compute_rolling_baselines


def test_load_sessions_and_filter_invalid(tmp_path):
    """Test session loading, contract validation, and invalid trial filtering."""
    df_raw = make_sessions_table(n_athletes=2, n_weeks=3, random_seed=42)
    # Mark one trial as invalid
    df_raw.loc[0, "valid"] = False
    csv_path = tmp_path / "test_sessions.csv"
    df_raw.to_csv(csv_path, index=False)

    # Load with filter
    df_loaded = load_sessions(csv_path, filter_valid=True)
    assert len(df_loaded) == len(df_raw) - 1
    assert False not in df_loaded["valid"].values
    assert pd.api.types.is_datetime64_any_dtype(df_loaded["date"])

    # Load without filter
    df_all = load_sessions(csv_path, filter_valid=False)
    assert len(df_all) == len(df_raw)


def test_load_wearable_daily(tmp_path):
    """Test wearable daily loading and validation."""
    df_raw = make_wearable_daily(n_athletes=2, n_days=10, random_seed=42)
    csv_path = tmp_path / "test_wearable.csv"
    df_raw.to_csv(csv_path, index=False)

    df_loaded = load_wearable_daily(csv_path)
    assert len(df_loaded) == len(df_raw)
    assert "recovery_pct" in df_loaded.columns
    assert "sleep_hours" in df_loaded.columns


def test_aggregate_session_metrics():
    """Test aggregation of multi-trial sessions to session mean."""
    data = [
        {"athlete_id": "ATH_01", "date": "2026-03-01", "session_id": "S1", "test": "CMJ", "trial": 1, "metric": "cmj_jump_height_imp", "value": 0.40, "unit": "m", "side": "both", "valid": True, "flags": ""},
        {"athlete_id": "ATH_01", "date": "2026-03-01", "session_id": "S1", "test": "CMJ", "trial": 2, "metric": "cmj_jump_height_imp", "value": 0.42, "unit": "m", "side": "both", "valid": True, "flags": ""},
        {"athlete_id": "ATH_01", "date": "2026-03-01", "session_id": "S1", "test": "CMJ", "trial": 3, "metric": "cmj_jump_height_imp", "value": 0.44, "unit": "m", "side": "both", "valid": True, "flags": ""},
    ]
    df = pd.DataFrame(data)
    agg_df = aggregate_session_metrics(df, method="mean")
    assert len(agg_df) == 1
    assert np.isclose(agg_df.loc[0, "value"], 0.42)
    assert agg_df.loc[0, "n_trials"] == 3


def test_rolling_baseline_math_and_swc():
    """Test exact mathematical correctness of rolling baseline mean, SD, SWC, TE, pct_change, and z-score."""
    # Athlete with 3 sessions on consecutive days: values = 0.40, 0.42, 0.44
    data = [
        {"athlete_id": "ATH_01", "date": "2026-03-01", "session_id": "S1", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.40, "n_trials": 1},
        {"athlete_id": "ATH_01", "date": "2026-03-02", "session_id": "S2", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.42, "n_trials": 1},
        {"athlete_id": "ATH_01", "date": "2026-03-03", "session_id": "S3", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.44, "n_trials": 1},
    ]
    df = pd.DataFrame(data)
    config = BaselineConfig(rolling_window_days=28, min_sessions_for_baseline=3, swc_multiplier=0.5)
    res = compute_rolling_baselines(df, config=config)

    # Session 1: N=1 < 3 -> unestablished
    assert not res.loc[0, "baseline_established"]
    assert np.isnan(res.loc[0, "z_score"])

    # Session 2: N=2 < 3 -> unestablished
    assert not res.loc[1, "baseline_established"]
    assert np.isnan(res.loc[1, "z_score"])

    # Session 3: N=3 >= 3 -> established
    s3 = res.loc[2]
    assert s3["baseline_established"]
    assert s3["baseline_n"] == 3
    # Mean: (0.40 + 0.42 + 0.44) / 3 = 0.42
    assert np.isclose(s3["baseline_mean"], 0.42)
    # Sample SD (ddof=1): sqrt(((0.40-0.42)^2 + 0 + (0.44-0.42)^2)/2) = sqrt(0.0008/2) = 0.02
    assert np.isclose(s3["baseline_sd"], 0.02)
    # SWC = 0.5 * 0.02 = 0.01
    assert np.isclose(s3["swc"], 0.01)
    # TE = 0.02 / sqrt(2) = 0.0141421356
    assert np.isclose(s3["typical_error"], 0.02 / np.sqrt(2))
    # pct_change = (0.44 - 0.42) / 0.42 * 100 = 4.76190476%
    assert np.isclose(s3["pct_change"], (0.44 - 0.42) / 0.42 * 100.0)
    # z_score (higher_is_better): +(0.44 - 0.42) / 0.02 = +1.0
    assert np.isclose(s3["z_score"], 1.0)


def test_direction_aware_z_score():
    """Test direction-awareness of z-scores for lower_is_better metrics (e.g., dj_gct)."""
    # GCT baseline = [0.20, 0.22, 0.24], mean = 0.22, sd = 0.02
    # Current value = 0.20 (faster contact -> better performance -> should yield positive z-score)
    data = [
        {"athlete_id": "ATH_01", "date": "2026-03-01", "session_id": "S1", "test": "DJ", "metric": "dj_gct", "side": "both", "unit": "s", "value": 0.24, "n_trials": 1},
        {"athlete_id": "ATH_01", "date": "2026-03-02", "session_id": "S2", "test": "DJ", "metric": "dj_gct", "side": "both", "unit": "s", "value": 0.22, "n_trials": 1},
        {"athlete_id": "ATH_01", "date": "2026-03-03", "session_id": "S3", "test": "DJ", "metric": "dj_gct", "side": "both", "unit": "s", "value": 0.20, "n_trials": 1},
    ]
    df = pd.DataFrame(data)
    config = BaselineConfig(rolling_window_days=28, min_sessions_for_baseline=3)
    res = compute_rolling_baselines(df, config=config)

    s3 = res.loc[2]
    # For lower_is_better, z = -1.0 * (0.20 - 0.22) / 0.02 = +1.0 (positive readiness)
    assert np.isclose(s3["z_score"], 1.0)


def test_28d_window_expiry_and_gap_handling():
    """Test that sessions outside the 28-day rolling window expire from the calculation."""
    # Sessions on Day 0, Day 10, and Day 40
    data = [
        {"athlete_id": "ATH_01", "date": "2026-01-01", "session_id": "S1", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.40, "n_trials": 1},
        {"athlete_id": "ATH_01", "date": "2026-01-10", "session_id": "S2", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.42, "n_trials": 1},
        {"athlete_id": "ATH_01", "date": "2026-02-15", "session_id": "S3", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.45, "n_trials": 1},
    ]
    df = pd.DataFrame(data)
    config = BaselineConfig(rolling_window_days=28, min_sessions_for_baseline=3)
    res = compute_rolling_baselines(df, config=config)

    # Day 40 (2026-02-15): 2026-01-01 (45 days prior) and 2026-01-10 (36 days prior) are outside 28d window
    # Window for S3 contains only S3 itself (N=1 < 3) -> baseline_established is False
    s3 = res.loc[2]
    assert not s3["baseline_established"]
    assert s3["baseline_n"] == 1
    assert np.isnan(s3["z_score"])
