"""Unit tests for Project 2 Milestones M2a, M2b, M2d, and M3."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from shared.synth import make_sessions_table, make_wearable_daily
from p2_nmt_reports.loader import load_sessions, load_wearable_daily, aggregate_session_metrics
from p2_nmt_reports.baselines import compute_rolling_baselines
from p2_nmt_reports.readiness import compute_composite_readiness
from p2_nmt_reports.wearable import join_wearable_to_readiness, classify_recovery_zone
from p2_nmt_reports.stats import fit_within_athlete_change_model
from p2_nmt_reports.charts import plot_metric_baseline_ribbon, plot_readiness_wearable_overlay


def test_composite_readiness_weights_and_direction():
    """Test composite readiness index correctly combines weighted direction-aware z-scores."""
    # Athlete with established baseline (N=3)
    data = [
        {"athlete_id": "A01", "date": "2026-03-01", "session_id": "S1", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.40, "n_trials": 1},
        {"athlete_id": "A01", "date": "2026-03-01", "session_id": "S1", "test": "CMJ", "metric": "cmj_rsimod", "side": "both", "unit": "ratio", "value": 0.50, "n_trials": 1},
        {"athlete_id": "A01", "date": "2026-03-01", "session_id": "S1", "test": "CMJ", "metric": "cmj_peak_power", "side": "both", "unit": "W", "value": 4000.0, "n_trials": 1},

        {"athlete_id": "A01", "date": "2026-03-02", "session_id": "S2", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.42, "n_trials": 1},
        {"athlete_id": "A01", "date": "2026-03-02", "session_id": "S2", "test": "CMJ", "metric": "cmj_rsimod", "side": "both", "unit": "ratio", "value": 0.52, "n_trials": 1},
        {"athlete_id": "A01", "date": "2026-03-02", "session_id": "S2", "test": "CMJ", "metric": "cmj_peak_power", "side": "both", "unit": "W", "value": 4100.0, "n_trials": 1},

        # Day 3: jump_height z=+1.0 (mean=0.42, sd=0.02, val=0.44)
        # rsimod z=+1.0 (mean=0.52, sd=0.02, val=0.54)
        # peak_power z=+1.0 (mean=4100, sd=100, val=4200)
        {"athlete_id": "A01", "date": "2026-03-03", "session_id": "S3", "test": "CMJ", "metric": "cmj_jump_height_imp", "side": "both", "unit": "m", "value": 0.44, "n_trials": 1},
        {"athlete_id": "A01", "date": "2026-03-03", "session_id": "S3", "test": "CMJ", "metric": "cmj_rsimod", "side": "both", "unit": "ratio", "value": 0.54, "n_trials": 1},
        {"athlete_id": "A01", "date": "2026-03-03", "session_id": "S3", "test": "CMJ", "metric": "cmj_peak_power", "side": "both", "unit": "W", "value": 4200.0, "n_trials": 1},
    ]
    df = pd.DataFrame(data)
    baselines = compute_rolling_baselines(df)
    readiness = compute_composite_readiness(baselines)

    s3 = readiness[readiness["session_id"] == "S3"].iloc[0]
    # All 3 metrics have z=+1.0, so weighted composite should be exactly +1.0
    assert np.isclose(s3["composite_readiness_z"], 1.0)
    assert not s3["acute_drop_flag"]
    assert "Experimental Heuristic Index" in s3["heuristic_disclaimer"]


def test_wearable_join_and_recovery_zones():
    """Test joining wearable daily telemetry, recovery zone tagging, and deficit flags."""
    df_sess = make_sessions_table(n_athletes=2, n_weeks=5, random_seed=42, fatigue_athlete="A01", fatigue_week=4)
    df_wear = make_wearable_daily(n_athletes=2, n_days=35, random_seed=42, fatigue_athlete="A01", fatigue_start_day=21)

    sessions_valid = load_sessions(df_sess)
    agg_sess = aggregate_session_metrics(sessions_valid)
    baselines = compute_rolling_baselines(agg_sess)
    readiness = compute_composite_readiness(baselines)

    merged = join_wearable_to_readiness(readiness, df_wear)

    assert "recovery_pct" in merged.columns
    assert "recovery_zone" in merged.columns
    assert "day_strain" in merged.columns
    assert set(merged["recovery_zone"].unique()).issubset({"Red", "Yellow", "Green", "Unknown"})


def test_classify_recovery_zone():
    """Test recovery zone boundaries (Red <34, Yellow 34-66, Green >=67)."""
    assert classify_recovery_zone(25.0) == "Red"
    assert classify_recovery_zone(33.0) == "Red"
    assert classify_recovery_zone(34.0) == "Yellow"
    assert classify_recovery_zone(55.0) == "Yellow"
    assert classify_recovery_zone(66.0) == "Yellow"
    assert classify_recovery_zone(67.0) == "Green"
    assert classify_recovery_zone(95.0) == "Green"
    assert classify_recovery_zone(None) == "Unknown"


def test_within_athlete_mixed_effects_model():
    """Test fitting Linear Mixed-Effects Model on longitudinal data."""
    df_sess = make_sessions_table(n_athletes=5, n_weeks=6, random_seed=42)
    df_wear = make_wearable_daily(n_athletes=5, n_days=42, random_seed=42)

    sessions_valid = load_sessions(df_sess)
    agg_sess = aggregate_session_metrics(sessions_valid)
    baselines = compute_rolling_baselines(agg_sess)
    readiness = compute_composite_readiness(baselines)
    merged = join_wearable_to_readiness(readiness, df_wear)

    results = fit_within_athlete_change_model(merged, outcome_col="composite_readiness_z")

    assert "fixed_effects" in results
    assert "variance_components" in results
    assert "mdc95" in results["variance_components"]
    assert results["n_groups"] == 5
    assert results["variance_components"]["mdc95"] > 0
    assert "Demonstration" in results["disclaimer"]


def test_chart_generation(tmp_path):
    """Test generation and saving of baseline ribbon and readiness overlay plots."""
    df_sess = make_sessions_table(n_athletes=2, n_weeks=5, random_seed=42)
    df_wear = make_wearable_daily(n_athletes=2, n_days=35, random_seed=42)

    sessions_valid = load_sessions(df_sess)
    agg_sess = aggregate_session_metrics(sessions_valid)
    baselines = compute_rolling_baselines(agg_sess)
    readiness = compute_composite_readiness(baselines)
    merged = join_wearable_to_readiness(readiness, df_wear)

    ribbon_path = tmp_path / "ribbon.png"
    overlay_path = tmp_path / "overlay.png"

    fig1 = plot_metric_baseline_ribbon(baselines, athlete_id="A01", save_path=ribbon_path)
    fig2 = plot_readiness_wearable_overlay(merged, athlete_id="A01", save_path=overlay_path)

    assert ribbon_path.exists()
    assert overlay_path.exists()
    assert ribbon_path.stat().st_size > 1000
    assert overlay_path.stat().st_size > 1000
