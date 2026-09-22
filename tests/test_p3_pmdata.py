"""Unit tests for Project 3 PMData Wearable Adapter and Contract Validation."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from p3_nmt_profiler.config import ProfilerConfig
from p3_nmt_profiler.pmdata_adapter import (
    adapt_single_participant,
    load_participant_sleep,
    load_participant_srpe,
    load_participant_wellness,
)
from shared.contract import (
    WEARABLE_DAILY_COLUMNS,
    validate_wearable_daily_df,
)


@pytest.fixture
def mock_participant_dir(tmp_path):
    """Create a mock PMData directory structure with synthetic records."""
    p_dir = tmp_path / "p99"
    pmsys = p_dir / "pmsys"
    fitbit = p_dir / "fitbit"
    pmsys.mkdir(parents=True)
    fitbit.mkdir(parents=True)

    # 1. Mock wellness.csv
    wellness_csv = (
        "effective_time_frame,fatigue,mood,readiness,sleep_duration_h,sleep_quality,soreness,soreness_area,stress\n"
        "2020-01-01T07:00:00.000Z,2,4,7,7.5,4,2,[],2\n"
        "2020-01-02T07:15:00.000Z,3,3,5,6.0,3,3,[],3\n"
        "2020-01-03T07:30:00.000Z,1,5,9,8.0,5,1,[],1\n"
    )
    (pmsys / "wellness.csv").write_text(wellness_csv, encoding="utf-8")

    # 2. Mock srpe.csv (Training on Jan 1 & 2; rest on Jan 3)
    srpe_csv = (
        "end_date_time,activity_names,perceived_exertion,duration_min\n"
        "2020-01-01T18:00:00.000Z,\"['running']\",6,45\n"       # 6 * 45 = 270
        "2020-01-02T19:00:00.000Z,\"['strength']\",8,60\n"      # 8 * 60 = 480
        "2020-01-02T21:00:00.000Z,\"['stretching']\",2,15\n"    # 2 * 15 = 30 -> Jan 2 total = 510
    )
    (pmsys / "srpe.csv").write_text(srpe_csv, encoding="utf-8")

    # 3. Mock sleep_score.csv
    sleep_csv = (
        "timestamp,sleep_log_entry_id,overall_score,composition_score,revitalization_score,duration_score,deep_sleep_in_minutes,resting_heart_rate,restlessness\n"
        "2020-01-01T06:30:00Z,1001,78,18,20,40,45,54,0.06\n"
        "2020-01-02T06:45:00Z,1002,68,15,16,37,30,57,0.11\n"
        "2020-01-03T07:00:00Z,1003,88,22,25,41,60,51,0.05\n"
    )
    (fitbit / "sleep_score.csv").write_text(sleep_csv, encoding="utf-8")

    return p_dir


def test_pmdata_adapter_mock_pipeline(mock_participant_dir):
    """Verify adapt_single_participant transforms multi-source logs into valid wearable_daily rows."""
    cfg = ProfilerConfig.load()
    df = adapt_single_participant("p99", mock_participant_dir, cfg=cfg)

    assert len(df) == 3
    assert list(df.columns) == WEARABLE_DAILY_COLUMNS

    # Jan 1: readiness=7 -> recovery_pct=70.0; sRPE=270; sleep=7.5; rhr=54
    row1 = df[df["date"] == "2020-01-01"].iloc[0]
    assert row1["athlete_id"] == "p99"
    assert row1["recovery_pct"] == pytest.approx(70.0)
    assert row1["day_strain"] == pytest.approx(270.0)
    assert row1["sleep_hours"] == pytest.approx(7.5)
    assert row1["rhr_bpm"] == pytest.approx(54.0)
    assert pd.isna(row1["hrv_ms"])
    assert "sRPE" in row1["proxy_used"]

    # Jan 2: multiple training sessions sum correctly (480 + 30 = 510)
    row2 = df[df["date"] == "2020-01-02"].iloc[0]
    assert row2["day_strain"] == pytest.approx(510.0)
    assert row2["recovery_pct"] == pytest.approx(50.0)
    assert row2["rhr_bpm"] == pytest.approx(57.0)

    # Jan 3: rest day (no sRPE entry) -> day_strain filled with 0.0
    row3 = df[df["date"] == "2020-01-03"].iloc[0]
    assert row3["day_strain"] == pytest.approx(0.0)
    assert row3["recovery_pct"] == pytest.approx(90.0)
    assert row3["sleep_hours"] == pytest.approx(8.0)
    assert row3["rhr_bpm"] == pytest.approx(51.0)

    # Strict contract schema validation
    is_valid, errors = validate_wearable_daily_df(df)
    assert is_valid, f"Contract validation failed: {errors}"


def test_pmdata_adapter_real_downloaded_participant():
    """Verify adapter works on real downloaded PMData participant (p01)."""
    p01_dir = Path("data/raw/pmdata/p01")
    if not p01_dir.exists() or not (p01_dir / "pmsys" / "wellness.csv").exists():
        pytest.skip("PMData p01 not downloaded yet")

    cfg = ProfilerConfig.load()
    df = adapt_single_participant("p01", p01_dir, cfg=cfg)

    assert len(df) > 50  # Over 50 days of real recordings
    assert set(df.columns) == set(WEARABLE_DAILY_COLUMNS)

    # Verify plausible ranges
    assert (df["recovery_pct"].dropna() >= 0).all()
    assert (df["recovery_pct"].dropna() <= 100).all()
    assert (df["sleep_hours"].dropna() >= 0).all()
    assert (df["sleep_hours"].dropna() <= 24).all()
    assert (df["rhr_bpm"].dropna() >= 30).all()
    assert (df["rhr_bpm"].dropna() <= 120).all()
    assert (df["day_strain"].dropna() >= 0).all()
    assert df["hrv_ms"].isna().all()  # All NaN per contract

    # Strict contract schema validation
    is_valid, errors = validate_wearable_daily_df(df)
    assert is_valid, f"Contract validation failed on real PMData: {errors}"


def test_build_wearable_daily_dataset():
    """Verify build_wearable_daily_dataset combines participants and strictly adheres to contract."""
    from p3_nmt_profiler.pmdata_adapter import build_wearable_daily_dataset

    raw_dir = Path("data/raw/pmdata")
    if not raw_dir.exists():
        pytest.skip("PMData raw directory not found")

    df = build_wearable_daily_dataset(raw_dir)
    assert len(df) > 100
    assert list(df.columns) == WEARABLE_DAILY_COLUMNS

    is_valid, errors = validate_wearable_daily_df(df)
    assert is_valid, f"Combined dataset validation failed: {errors}"


def test_build_modeling_dataset_lags():
    """Verify build_modeling_dataset correctly constructs 1-day temporal lag features."""
    from p3_nmt_profiler.pmdata_adapter import build_modeling_dataset

    raw_dir = Path("data/raw/pmdata")
    if not raw_dir.exists():
        pytest.skip("PMData raw directory not found")

    model_df = build_modeling_dataset(raw_dir)
    assert len(model_df) > 100

    # Required lag features must exist
    assert "prior_day_strain" in model_df.columns
    assert "prior_sleep_hours" in model_df.columns

    # Verify lag alignment logic on first participant: day t's prior_day_strain equals day t-1's day_strain
    p1 = model_df[model_df["athlete_id"] == "p01"].sort_values("date").reset_index(drop=True)
    if len(p1) >= 2:
        assert p1.loc[1, "prior_day_strain"] == pytest.approx(p1.loc[0, "day_strain"])
        assert p1.loc[1, "prior_sleep_hours"] == pytest.approx(p1.loc[0, "sleep_duration_h"])

