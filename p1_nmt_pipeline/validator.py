
"""Input validation, column normalization, and metadata extraction for force-plate trials.

Governed by specs/01_neuromuscular_pipeline.md and PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.contract import STANDARD_G

logger = logging.getLogger(__name__)


@dataclass
class ValidatedTrialData:
    """Standardized and validated trial input structure."""
    time_s: np.ndarray
    force_n: np.ndarray
    force_left_n: np.ndarray | None
    force_right_n: np.ndarray | None
    fs_hz: float
    body_mass_kg: float
    body_weight_n: float
    body_mass_estimated: bool
    qa_flags: list[str] = field(default_factory=list)
    test_type: str | None = None
    drop_height_cm: float | None = None

    @property
    def flags_str(self) -> str:
        return ";".join(self.qa_flags)


def find_adaptive_quiet_window(
    time_arr: np.ndarray,
    force_arr: np.ndarray,
    fs_hz: float,
    quiet_start_s: float = 0.05,
    max_quiet_window_s: float = 0.50,
    min_quiet_window_s: float = 0.15,
    max_quiet_sd_n: float = 15.0,
    max_quiet_trend_n_per_s: float = 20.0,
) -> tuple[int, int, float, float, float, float]:
    """Find the most stable initial quiet standing segment adaptively.

    Detects the quiet baseline segment by analyzing standard deviation and linear trend
    (slope). If movement onset or unweighting begins before `max_quiet_window_s`
    (e.g., trials with short lead-in < 0.5 s), the window adapts to end before the
    dip so baseline mass and quiet SD are not contaminated.

    Parameters
    ----------
    time_arr : np.ndarray
        Array of timestamps (s).
    force_arr : np.ndarray
        Vertical force array (N).
    fs_hz : float
        Sampling rate in Hz.
    quiet_start_s : float
        Initial settling period to skip (default 0.05 s).
    max_quiet_window_s : float
        Target maximum quiet window duration (default 0.50 s).
    min_quiet_window_s : float
        Minimum window duration required for baseline statistics (default 0.15 s).
    max_quiet_sd_n : float
        Maximum acceptable SD before flagging instability (default 15.0 N).
    max_quiet_trend_n_per_s : float
        Maximum acceptable slope before flagging drift (default 20.0 N/s).

    Returns
    -------
    tuple[int, int, float, float, float, float]
        (start_idx, end_idx, mean_force, std_force, slope, duration_s)
    """
    actual_quiet_start_s = quiet_start_s if quiet_start_s < (max_quiet_window_s * 0.5) else 0.0
    t_start = time_arr[0] + actual_quiet_start_s
    i_start = int(np.searchsorted(time_arr, t_start))

    n_pts = len(time_arr)
    if n_pts - i_start < 2:
        raise ValueError(f"Insufficient samples for quiet window analysis: {n_pts}")

    avail = n_pts - i_start
    min_samples = max(2, min(int(round(fs_hz * min_quiet_window_s)), avail - 1))
    target_end_s = time_arr[0] + max_quiet_window_s
    target_end_idx = min(n_pts - 1, int(np.searchsorted(time_arr, target_end_s)))
    if target_end_idx <= i_start + min_samples:
        target_end_idx = min(n_pts - 1, i_start + min_samples)

    if target_end_idx - i_start <= min_samples:
        best_end = target_end_idx
    else:
        # Seed statistics from the initial segment
        seed_f = force_arr[i_start : i_start + min_samples]
        seed_mean = float(np.mean(seed_f))
        seed_sd = float(np.std(seed_f))
        best_end = i_start + min_samples
        step = max(1, int(round(fs_hz * 0.005)))  # 5 ms evaluation step

        for i_cand in range(i_start + min_samples, target_end_idx + 1, step):
            sub_f = force_arr[i_start:i_cand]
            sub_t = time_arr[i_start:i_cand]
            cur_sd = float(np.std(sub_f))
            dt = sub_t - np.mean(sub_t)
            var_t = float(np.sum(dt**2))
            cur_slope = float(np.sum(dt * (sub_f - np.mean(sub_f))) / var_t) if var_t > 0 else 0.0

            # Local deviation in recent 30 ms window
            recent_win = max(2, int(round(fs_hz * 0.030)))
            recent_f = force_arr[max(i_start, i_cand - recent_win) : i_cand]
            recent_dev = abs(float(np.mean(recent_f)) - seed_mean)

            # Check if entering movement unweighting or excessive drift
            if recent_dev > max(4.0 * seed_sd, 12.0) or (cur_sd > max_quiet_sd_n and cur_sd > seed_sd * 1.8) or abs(cur_slope) > max_quiet_trend_n_per_s:
                break
            best_end = i_cand

    quiet_times = time_arr[i_start : best_end + 1]
    quiet_forces = force_arr[i_start : best_end + 1]
    mean_quiet_force = float(np.mean(quiet_forces))
    std_quiet_force = float(np.std(quiet_forces))
    dt_quiet = quiet_times - np.mean(quiet_times)
    var_t = float(np.sum(dt_quiet**2))
    quiet_slope = float(np.sum(dt_quiet * (quiet_forces - mean_quiet_force)) / var_t) if var_t > 0 else 0.0
    quiet_duration_s = float(time_arr[best_end] - time_arr[0])

    return i_start, best_end, mean_quiet_force, std_quiet_force, quiet_slope, quiet_duration_s


def load_and_validate_trial(
    source: str | Path | pd.DataFrame | dict[str, Any],
    body_mass_kg: float | None = None,
    test_type: str | None = None,
    drop_height_cm: float | None = None,
    quiet_window_s: float = 0.5,
    quiet_start_s: float = 0.05,
) -> ValidatedTrialData:
    """Load, standardize, and validate raw force-time trial telemetry.

    Parameters
    ----------
    source : str | Path | pd.DataFrame | dict
        Input CSV path, JSON path, DataFrame, or dict containing time and force.
    body_mass_kg : float | None
        Athlete body mass in kg. If None, estimated from initial quiet standing window.
    test_type : str | None
        Optional explicit test type metadata (e.g. 'CMJ', 'SJ', 'DJ', 'IMTP').
    drop_height_cm : float | None
        Drop height in cm for Drop Jump tests.
    quiet_window_s : float
        Duration of baseline quiet window used for mass estimation (default 0.5 s).
    quiet_start_s : float
        Initial settling period to skip before mass estimation (default 0.05 s).

    Returns
    -------
    ValidatedTrialData
        Structured and validated trial data.

    Raises
    ------
    ValueError
        If required columns are missing, data contains NaNs, timestamps are non-monotonic,
        or sampling rate cannot be determined.
    """
    qa_flags: list[str] = []

    # 1. Parse source into DataFrame
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Force-plate data file not found: {path}")
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            df = pd.DataFrame(data_dict)
        else:
            df = pd.read_csv(path)
    elif isinstance(source, dict):
        df = pd.DataFrame(source)
    elif isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if df.empty:
        raise ValueError("Force-time dataset is empty.")

    # 2. Normalize column headers
    col_map = {}
    for col in df.columns:
        c_clean = str(col).strip().lower().replace(" ", "_")
        if c_clean in ("time", "time_s", "time_(sec)", "time_(s)", "time(s)"):
            col_map[col] = "time_s"
        elif c_clean in ("force", "force_n", "force_(n)", "force(n)", "fz", "vgrf", "total_force"):
            col_map[col] = "force_n"
        elif c_clean in ("force_left", "force_left_n", "fl", "fz_left"):
            col_map[col] = "force_left_n"
        elif c_clean in ("force_right", "force_right_n", "fr", "fz_right"):
            col_map[col] = "force_right_n"

    df = df.rename(columns=col_map)

    if "time_s" not in df.columns or "force_n" not in df.columns:
        raise ValueError(
            f"Missing required columns ('time' and 'force'). Found columns: {list(df.columns)}"
        )

    # 3. Clean numeric conversion & check NaNs / Infs
    time_series = pd.to_numeric(df["time_s"], errors="coerce")
    force_series = pd.to_numeric(df["force_n"], errors="coerce")

    if time_series.isna().any() or force_series.isna().any():
        raise ValueError("Force-time series contains NaN or unparseable numeric values.")

    time_arr = time_series.to_numpy(dtype=float)
    force_arr = force_series.to_numpy(dtype=float)

    if np.isinf(time_arr).any() or np.isinf(force_arr).any():
        raise ValueError("Force-time series contains infinite values.")

    # 4. Validate Monotonicity and Duplicate Timestamps
    dt = np.diff(time_arr)
    if (dt <= 0.0).any():
        n_duplicates = int(np.sum(dt == 0.0))
        n_backwards = int(np.sum(dt < 0.0))
        if n_duplicates > 0:
            raise ValueError(f"Found {n_duplicates} duplicate timestamps in force-time series.")
        if n_backwards > 0:
            raise ValueError(f"Found {n_backwards} non-monotonic backwards timestamps.")

    # 5. Extract Bilateral Channels if Present
    force_left_arr: np.ndarray | None = None
    force_right_arr: np.ndarray | None = None
    if "force_left_n" in df.columns:
        fl_series = pd.to_numeric(df["force_left_n"], errors="coerce")
        if not fl_series.isna().any():
            force_left_arr = fl_series.to_numpy(dtype=float)
    if "force_right_n" in df.columns:
        fr_series = pd.to_numeric(df["force_right_n"], errors="coerce")
        if not fr_series.isna().any():
            force_right_arr = fr_series.to_numpy(dtype=float)

    # 6. Check for Negative Forces on Input Series
    raw_neg_count = int(np.sum(force_arr < 0.0))
    if raw_neg_count > 0:
        min_force = float(np.min(force_arr))
        qa_flags.append(f"negative_force_{raw_neg_count}_samples_min_{min_force:.1f}N")
        logger.warning("Detected %d negative force samples (minimum %.1f N)", raw_neg_count, min_force)

    # 7. Inquire Sampling Rate & Resample Irregular Timestamps onto Uniform Grid
    median_dt = float(np.median(dt))
    if median_dt <= 0.0:
        raise ValueError("Invalid median time step (<= 0).")
    fs_hz = round(1.0 / median_dt, 1)

    # Detect irregular timestamps deviating by more than 1% from median
    irregular_count = int(np.sum(np.abs(dt - median_dt) > (0.01 * median_dt)))
    if irregular_count > 0:
        pct_irregular = (irregular_count / len(dt)) * 100.0
        qa_flags.append(f"resampled_irregular_dt_{irregular_count}_samples_{pct_irregular:.1f}pct")
        logger.warning(
            "Detected %d irregular timestamp intervals (%.1f%% of samples, median dt=%.6f s). "
            "Resampling onto uniform time grid at fs=%.1f Hz.",
            irregular_count,
            pct_irregular,
            median_dt,
            fs_hz,
        )
        # Construct strictly uniform time grid from t[0] to t[-1]
        n_uniform = int(round((time_arr[-1] - time_arr[0]) / median_dt)) + 1
        uniform_time = np.linspace(time_arr[0], time_arr[-1], n_uniform)
        force_arr = np.interp(uniform_time, time_arr, force_arr)
        if force_left_arr is not None:
            force_left_arr = np.interp(uniform_time, time_arr, force_left_arr)
        if force_right_arr is not None:
            force_right_arr = np.interp(uniform_time, time_arr, force_right_arr)
        time_arr = uniform_time

    # If both bilateral channels present, verify sum consistency
    if force_left_arr is not None and force_right_arr is not None:
        bilateral_diff = np.abs((force_left_arr + force_right_arr) - force_arr)
        if np.max(bilateral_diff) > 1.0:  # > 1 N difference
            qa_flags.append("bilateral_sum_mismatch")
            logger.warning("Left + Right force does not match total force (max diff %.2f N)", np.max(bilateral_diff))

    # 8. Adaptive Quiet Window Quality Assessment (SD and Drift Trend)
    (
        i_quiet_start,
        i_quiet_end,
        mean_quiet_force,
        std_quiet_force,
        quiet_slope,
        quiet_duration_s,
    ) = find_adaptive_quiet_window(
        time_arr=time_arr,
        force_arr=force_arr,
        fs_hz=fs_hz,
        quiet_start_s=quiet_start_s,
        max_quiet_window_s=quiet_window_s,
    )

    # Flag trials with under 0.5 s of quiet
    if quiet_duration_s < 0.50 - 1e-3:
        qa_flags.append(f"quiet_window_short_{quiet_duration_s:.2f}s")
        qa_flags.append("quiet_window_under_0.5s")
        logger.warning(
            "Quiet standing window is under 0.5 s (duration = %.2f s < 0.50 s)",
            quiet_duration_s,
        )

    # QA checks on quiet window stability
    if std_quiet_force > 15.0:  # > 15 N standard deviation
        qa_flags.append(f"quiet_window_noisy_sd_{std_quiet_force:.1f}N")
        logger.warning("Quiet standing window is noisy (SD = %.1f N > 15.0 N)", std_quiet_force)

    if abs(quiet_slope) > 20.0:  # > 20 N/s trend
        qa_flags.append(f"quiet_window_drifting_trend_{quiet_slope:.1f}N_s")
        logger.warning(
            "Quiet standing window exhibits drifting trend (slope = %.1f N/s > 20.0 N/s)",
            quiet_slope,
        )

    # 9. Body Mass Estimation or Validation
    body_mass_estimated = False
    if body_mass_kg is None or body_mass_kg <= 0.0:
        if mean_quiet_force < 100.0 or mean_quiet_force > 3000.0:
            qa_flags.append(f"implausible_estimated_bw_{mean_quiet_force:.1f}N")
            logger.warning(
                "Estimated baseline force %.1f N is outside plausible human range [100, 3000] N",
                mean_quiet_force,
            )

        body_mass_kg = round(mean_quiet_force / STANDARD_G, 2)
        body_weight_n = mean_quiet_force
        body_mass_estimated = True
        qa_flags.append("body_mass_estimated_from_baseline")
    else:
        body_weight_n = body_mass_kg * STANDARD_G

    return ValidatedTrialData(
        time_s=time_arr,
        force_n=force_arr,
        force_left_n=force_left_arr,
        force_right_n=force_right_arr,
        fs_hz=fs_hz,
        body_mass_kg=body_mass_kg,
        body_weight_n=body_weight_n,
        body_mass_estimated=body_mass_estimated,
        qa_flags=qa_flags,
        test_type=test_type,
        drop_height_cm=drop_height_cm,
    )
