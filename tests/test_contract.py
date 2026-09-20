"""Unit tests for shared/contract.py."""
import pandas as pd
import pytest

from shared.contract import (
    LimbSide,
    MetricDirection,
    METRIC_REGISTRY,
    SESSIONS_COLUMNS,
    TestType,
    validate_sessions_df,
    validate_wearable_daily_df,
    WEARABLE_DAILY_COLUMNS,
)


def test_metric_registry_integrity():
    """Verify every metric in the registry has complete and valid metadata."""
    assert len(METRIC_REGISTRY) > 15
    for name, metric_def in METRIC_REGISTRY.items():
        assert metric_def.name == name
        assert isinstance(metric_def.test, TestType)
        assert isinstance(metric_def.direction, MetricDirection)
        assert len(metric_def.unit) > 0
        assert len(metric_def.description) > 0


def test_valid_sessions_df():
    """Verify that a valid sessions DataFrame passes validation."""
    valid_data = {
        "session_id": ["A01_2026-03-02", "A01_2026-03-02"],
        "date": ["2026-03-02", "2026-03-02"],
        "athlete_id": ["A01", "A01"],
        "test": ["CMJ", "CMJ"],
        "trial": [1, 1],
        "metric": ["cmj_jump_height_imp", "cmj_rsimod"],
        "value": [0.352, 0.521],
        "unit": ["m", "ratio"],
        "side": ["both", "both"],
        "valid": [True, True],
        "flags": ["", ""],
        "block": ["week_1", "week_1"],
    }
    df = pd.DataFrame(valid_data)
    is_valid, errors = validate_sessions_df(df)
    assert is_valid is True
    assert len(errors) == 0


def test_sessions_df_missing_columns():
    """Verify missing required columns are flagged."""
    df = pd.DataFrame({"session_id": ["A01"], "date": ["2026-03-02"]})
    is_valid, errors = validate_sessions_df(df)
    assert is_valid is False
    assert any("Missing required columns" in err for err in errors)


def test_sessions_df_invalid_date():
    """Verify malformed dates fail validation."""
    df = pd.DataFrame({
        "session_id": ["S1"],
        "date": ["02/03/2026"],  # Non-ISO format
        "athlete_id": ["A01"],
        "test": ["CMJ"],
        "trial": [1],
        "metric": ["cmj_jump_height_imp"],
        "value": [0.35],
        "unit": ["m"],
        "side": ["both"],
        "valid": [True],
        "flags": [""],
        "block": [""],
    })
    is_valid, errors = validate_sessions_df(df)
    assert is_valid is False
    assert any("Invalid ISO dates" in err for err in errors)


def test_sessions_df_invalid_test_type():
    """Verify unsupported test types are rejected."""
    df = pd.DataFrame({
        "session_id": ["S1"],
        "date": ["2026-03-02"],
        "athlete_id": ["A01"],
        "test": ["INVALID_TEST"],
        "trial": [1],
        "metric": ["cmj_jump_height_imp"],
        "value": [0.35],
        "unit": ["m"],
        "side": ["both"],
        "valid": [True],
        "flags": [""],
        "block": [""],
    })
    is_valid, errors = validate_sessions_df(df)
    assert is_valid is False
    assert any("Invalid test types" in err for err in errors)


def test_sessions_df_unit_mismatch():
    """Verify metric unit mismatch is flagged."""
    df = pd.DataFrame({
        "session_id": ["S1"],
        "date": ["2026-03-02"],
        "athlete_id": ["A01"],
        "test": ["CMJ"],
        "trial": [1],
        "metric": ["cmj_jump_height_imp"],
        "value": [35.0],
        "unit": ["cm"],  # Expected 'm'
        "side": ["both"],
        "valid": [True],
        "flags": [""],
        "block": [""],
    })
    is_valid, errors = validate_sessions_df(df)
    assert is_valid is False
    assert any("Unit mismatch" in err for err in errors)


def test_valid_wearable_daily_df():
    """Verify valid wearable telemetry passes validation."""
    valid_data = {
        "date": ["2026-03-02", "2026-03-03"],
        "athlete_id": ["A01", "A01"],
        "recovery_pct": [68.0, 32.5],
        "hrv_ms": [82.0, 54.0],
        "rhr_bpm": [50.0, 58.0],
        "day_strain": [12.4, 16.8],
        "sleep_hours": [7.8, 5.2],
        "proxy_used": ["", ""],
    }
    df = pd.DataFrame(valid_data)
    is_valid, errors = validate_wearable_daily_df(df)
    assert is_valid is True
    assert len(errors) == 0


def test_wearable_daily_out_of_range():
    """Verify out-of-range recovery_pct or negative values are flagged."""
    invalid_data = {
        "date": ["2026-03-02"],
        "athlete_id": ["A01"],
        "recovery_pct": [120.0],  # > 100
        "hrv_ms": [-10.0],       # Negative
        "rhr_bpm": [50.0],
        "day_strain": [12.0],
        "sleep_hours": [8.0],
        "proxy_used": [""],
    }
    df = pd.DataFrame(invalid_data)
    is_valid, errors = validate_wearable_daily_df(df)
    assert is_valid is False
    assert any("recovery_pct" in err for err in errors)
    assert any("hrv_ms" in err for err in errors)
