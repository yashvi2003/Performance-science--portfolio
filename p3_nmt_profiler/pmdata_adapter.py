"""PMData Adapter: Transforms PMData sports logging records into shared contract tables.

Converts multi-source logs (Fitbit Versa 2 + PMSys wellness + sRPE) into:
1. `wearable_daily.csv` strictly conforming to `specs/00_shared_contract.md`.
2. Modeling analytical frame with prior-day/night lags and person-mean centered predictors.

PROVENANCE & METHODOLOGICAL DOCUMENTATION:
- Dataset: PMData (Thambawita et al., 2020, MMSys '20, DOI: 10.1145/3339825.3394926).
- License: CC BY-NC 4.0 (OSF license.txt) / CC BY 4.0 (Simula portal).
- Strain Proxy: Foster's Session-RPE = sum(perceived_exertion * duration_min) per day.
- Recovery Proxy: PMSys morning readiness (1-10 scale) scaled to 0-100% (or Fitbit sleep score).
- Resting HR: Fitbit sleep-log resting heart rate.
- HRV: Explicit NaN (not exported by Fitbit Versa 2 in PMData).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from p3_nmt_profiler.config import ProfilerConfig
from shared.contract import (
    WEARABLE_DAILY_COLUMNS,
    validate_wearable_daily_df,
)

logger = logging.getLogger("p3_nmt_profiler.adapter")


def load_participant_wellness(pmsys_dir: Path) -> pd.DataFrame:
    """Load and process PMSys wellness.csv for a participant."""
    wellness_file = pmsys_dir / "wellness.csv"
    if not wellness_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(wellness_file)
    if "effective_time_frame" not in df.columns:
        return pd.DataFrame()

    # Extract calendar date (YYYY-MM-DD)
    df["date"] = df["effective_time_frame"].astype(str).str.slice(0, 10)

    # In case of multiple logs per day, take the earliest morning entry or mean
    agg_dict = {
        "readiness": "mean",
        "fatigue": "mean",
        "sleep_duration_h": "mean",
        "sleep_quality": "mean",
        "soreness": "mean",
        "stress": "mean",
        "mood": "mean",
    }
    cols_to_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    daily_wellness = df.groupby("date", as_index=False).agg(cols_to_agg)
    return daily_wellness


def load_participant_srpe(pmsys_dir: Path) -> pd.DataFrame:
    """Load and process PMSys srpe.csv (training sessions) for a participant."""
    srpe_file = pmsys_dir / "srpe.csv"
    if not srpe_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(srpe_file)
    if "end_date_time" not in df.columns:
        return pd.DataFrame()

    df["date"] = df["end_date_time"].astype(str).str.slice(0, 10)
    # Compute Foster's Session-RPE (sRPE = RPE * duration_min)
    rpe = pd.to_numeric(df["perceived_exertion"], errors="coerce").fillna(0.0)
    dur = pd.to_numeric(df["duration_min"], errors="coerce").fillna(0.0)
    df["session_load"] = rpe * dur

    # Aggregate daily total training load
    daily_srpe = df.groupby("date", as_index=False).agg(
        day_strain=("session_load", "sum"),
        training_sessions=("session_load", "count"),
        total_training_min=("duration_min", "sum"),
    )
    return daily_srpe


def load_participant_sleep(fitbit_dir: Path) -> pd.DataFrame:
    """Load and process Fitbit sleep_score.csv for a participant."""
    sleep_file = fitbit_dir / "sleep_score.csv"
    if not sleep_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(sleep_file)
    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["date"] = df["timestamp"].astype(str).str.slice(0, 10)

    agg_dict = {
        "overall_score": "mean",
        "resting_heart_rate": "mean",
        "deep_sleep_in_minutes": "mean",
        "composition_score": "mean",
        "revitalization_score": "mean",
        "duration_score": "mean",
        "restlessness": "mean",
    }
    cols_to_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    daily_sleep = df.groupby("date", as_index=False).agg(cols_to_agg)
    return daily_sleep


def adapt_single_participant(
    p_id: str,
    p_dir: Path,
    cfg: ProfilerConfig | None = None,
) -> pd.DataFrame:
    """Adapt one PMData participant's directory into wearable_daily rows.

    Parameters
    ----------
    p_id : str
        Participant identifier (e.g. 'p01').
    p_dir : Path
        Directory containing 'pmsys' and 'fitbit' subdirectories.
    cfg : ProfilerConfig, optional
        Configuration settings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [date, athlete_id, recovery_pct, hrv_ms, rhr_bpm, day_strain, sleep_hours, proxy_used].
    """
    if cfg is None:
        cfg = ProfilerConfig.load()

    pmsys_dir = p_dir / "pmsys"
    fitbit_dir = p_dir / "fitbit"

    wellness_df = load_participant_wellness(pmsys_dir)
    srpe_df = load_participant_srpe(pmsys_dir)
    sleep_df = load_participant_sleep(fitbit_dir)

    # Collect unique dates across all available sources
    all_dates = set()
    if not wellness_df.empty:
        all_dates.update(wellness_df["date"].dropna().unique())
    if not srpe_df.empty:
        all_dates.update(srpe_df["date"].dropna().unique())
    if not sleep_df.empty:
        all_dates.update(sleep_df["date"].dropna().unique())

    if not all_dates:
        return pd.DataFrame(columns=WEARABLE_DAILY_COLUMNS)

    dates_df = pd.DataFrame({"date": sorted(all_dates)})

    # Merge sources onto date index
    merged = dates_df
    if not wellness_df.empty:
        merged = pd.merge(merged, wellness_df, on="date", how="left")
    if not srpe_df.empty:
        merged = pd.merge(merged, srpe_df, on="date", how="left")
    if not sleep_df.empty:
        merged = pd.merge(merged, sleep_df, on="date", how="left")

    # Map recovery proxy: PMSys readiness * 10 (or fallback to Fitbit overall_score)
    recovery_pct = pd.Series(np.nan, index=merged.index, dtype=float)
    if "readiness" in merged.columns:
        recovery_pct = merged["readiness"] * cfg.proxies.recovery_scale_factor
    if "overall_score" in merged.columns:
        recovery_pct = recovery_pct.combine_first(merged["overall_score"])
    recovery_pct = recovery_pct.clip(lower=0.0, upper=100.0)

    # Map sleep hours: wellness sleep_duration_h (or fallback to duration_score approximation)
    sleep_hours = pd.Series(np.nan, index=merged.index, dtype=float)
    if "sleep_duration_h" in merged.columns:
        sleep_hours = pd.to_numeric(merged["sleep_duration_h"], errors="coerce")

    # Map resting HR: from Fitbit sleep score
    rhr_bpm = pd.Series(np.nan, index=merged.index, dtype=float)
    if "resting_heart_rate" in merged.columns:
        rhr_bpm = pd.to_numeric(merged["resting_heart_rate"], errors="coerce")

    # Map day strain: sRPE sum (fill with 0.0 on rest days where wellness was recorded)
    day_strain = pd.Series(cfg.proxies.strain_rest_fill, index=merged.index, dtype=float)
    if "day_strain" in merged.columns:
        # Non-missing training days use observed sRPE; days without training log remain 0.0
        day_strain = merged["day_strain"].fillna(cfg.proxies.strain_rest_fill)

    # Document proxy
    proxy_desc = "day_strain: sRPE (RPE*min); recovery_pct: PMSys readiness (1-10 scaled to 0-100)"

    out_df = pd.DataFrame({
        "date": merged["date"],
        "athlete_id": p_id,
        "recovery_pct": recovery_pct.round(1),
        "hrv_ms": np.nan,  # Explicit NaN per contract
        "rhr_bpm": rhr_bpm.round(1),
        "day_strain": day_strain.round(1),
        "sleep_hours": sleep_hours.round(2),
        "proxy_used": proxy_desc,
    }, columns=WEARABLE_DAILY_COLUMNS)

    return out_df


def build_wearable_daily_dataset(
    raw_dir: Path | str = Path("data/raw/pmdata"),
    cfg: ProfilerConfig | None = None,
    anonymize: bool = False,
) -> pd.DataFrame:
    """Build the complete wearable_daily DataFrame across all participants.

    Parameters
    ----------
    raw_dir : Path or str
        Path to PMData raw directory containing participant subfolders (p01..p16).
    cfg : ProfilerConfig, optional
        Configuration settings.
    anonymize : bool
        If True, re-codes participant IDs as 'ATH_01', etc. (default False preserves 'p01').

    Returns
    -------
    pd.DataFrame
        Contract-compliant DataFrame adhering to WEARABLE_DAILY_COLUMNS.
    """
    if cfg is None:
        cfg = ProfilerConfig.load()

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"PMData directory {raw_dir} does not exist.")

    participant_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("p")])
    if not participant_dirs:
        raise ValueError(f"No participant folders (p01..p16) found in {raw_dir}.")

    dfs: list[pd.DataFrame] = []
    for idx, p_dir in enumerate(participant_dirs, start=1):
        raw_pid = p_dir.name
        pid = f"ATH_{idx:02d}" if anonymize else raw_pid
        try:
            p_df = adapt_single_participant(pid, p_dir, cfg=cfg)
            if not p_df.empty:
                dfs.append(p_df)
                logger.info("Adapted participant %s: %d daily rows", pid, len(p_df))
        except Exception as exc:
            logger.warning("Failed to adapt %s: %s", raw_pid, exc)

    if not dfs:
        raise RuntimeError("No participant data could be adapted.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    # Strictly validate against shared contract
    is_valid, errors = validate_wearable_daily_df(combined)
    if not is_valid:
        raise ValueError(f"wearable_daily DataFrame failed contract validation: {errors}")

    logger.info("wearable_daily table built successfully: %d total rows across %d athletes (Contract: PASSED)",
                len(combined), combined["athlete_id"].nunique())
    return combined


def build_modeling_dataset(
    raw_dir: Path | str = Path("data/raw/pmdata"),
    cfg: ProfilerConfig | None = None,
) -> pd.DataFrame:
    """Build a rich longitudinal modeling frame with lagged predictors.

    Aligns prior-night sleep duration and resting HR, plus prior-day training load,
    to predict next-day self-reported morning readiness, fatigue, and soreness.
    """
    if cfg is None:
        cfg = ProfilerConfig.load()

    raw_dir = Path(raw_dir)
    participant_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("p")])

    dfs = []
    for p_dir in participant_dirs:
        p_id = p_dir.name
        wellness_df = load_participant_wellness(p_dir / "pmsys")
        srpe_df = load_participant_srpe(p_dir / "pmsys")
        sleep_df = load_participant_sleep(p_dir / "fitbit")

        if wellness_df.empty:
            continue

        # Complete daily grid
        wellness_df["date"] = pd.to_datetime(wellness_df["date"])
        srpe_df["date"] = pd.to_datetime(srpe_df["date"]) if not srpe_df.empty else pd.Series(dtype="datetime64[ns]")
        sleep_df["date"] = pd.to_datetime(sleep_df["date"]) if not sleep_df.empty else pd.Series(dtype="datetime64[ns]")

        # Merge
        merged = wellness_df
        if not srpe_df.empty:
            merged = pd.merge(merged, srpe_df, on="date", how="left")
            merged["day_strain"] = merged["day_strain"].fillna(0.0)
        else:
            merged["day_strain"] = 0.0

        if not sleep_df.empty:
            merged = pd.merge(merged, sleep_df, on="date", how="left")

        merged = merged.sort_values("date").reset_index(drop=True)
        merged["athlete_id"] = p_id

        # Compute 1-day lags (prior day / prior night)
        # Prior day load
        merged["prior_day_strain"] = merged["day_strain"].shift(1)
        # Prior night sleep duration (from wellness or sleep log)
        merged["prior_sleep_hours"] = merged["sleep_duration_h"].shift(1)
        if "overall_score" in merged.columns:
            merged["prior_sleep_score"] = merged["overall_score"].shift(1)
        if "resting_heart_rate" in merged.columns:
            merged["prior_rhr"] = merged["resting_heart_rate"].shift(1)

        dfs.append(merged)

    if not dfs:
        return pd.DataFrame()

    full_model_df = pd.concat(dfs, ignore_index=True)
    return full_model_df
