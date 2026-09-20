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

    # 5. Inquire Sampling Rate & Detect Irregular Timestamps
    median_dt = float(np.median(dt))
    if median_dt <= 0.0:
        raise ValueError("Invalid median time step (<= 0).")
    fs_hz = round(1.0 / median_dt, 1)

    # Detect irregular timestamps deviating by more than 1% from median
    irregular_count = int(np.sum(np.abs(dt - median_dt) > (0.01 * median_dt)))
    if irregular_count > 0:
        pct_irregular = (irregular_count / len(dt)) * 100.0
        qa_flags.append(f"irregular_dt_{irregular_count}_samples_{pct_irregular:.1f}pct")
        logger.warning(
            "Detected %d irregular timestamp intervals (%.1f%% of samples). Inferred fs=%.1f Hz",
            irregular_count,
            pct_irregular,
            fs_hz,
        )

    # 6. Check for Negative Forces
    neg_count = int(np.sum(force_arr < 0.0))
    if neg_count > 0:
        min_force = float(np.min(force_arr))
        qa_flags.append(f"negative_force_{neg_count}_samples_min_{min_force:.1f}N")
        logger.warning("Detected %d negative force samples (minimum %.1f N)", neg_count, min_force)

    # 7. Bilateral columns handling
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

    # If both present, verify sum consistency
    if force_left_arr is not None and force_right_arr is not None:
        bilateral_diff = np.abs((force_left_arr + force_right_arr) - force_arr)
        if np.max(bilateral_diff) > 1.0:  # > 1 N difference
            qa_flags.append("bilateral_sum_mismatch")
            logger.warning("Left + Right force does not match total force (max diff %.2f N)", np.max(bilateral_diff))

    # 8. Body Mass Estimation or Validation
    body_mass_estimated = False
    if body_mass_kg is None or body_mass_kg <= 0.0:
        # Estimate from quiet window
        quiet_mask = (time_arr >= (time_arr[0] + quiet_start_s)) & (
            time_arr <= (time_arr[0] + quiet_window_s)
        )
        if not np.any(quiet_mask):
            raise ValueError(
                f"Cannot estimate body mass: quiet window [{quiet_start_s}, {quiet_window_s}] s contains no samples."
            )
        quiet_forces = force_arr[quiet_mask]
        mean_quiet_force = float(np.mean(quiet_forces))
        std_quiet_force = float(np.std(quiet_forces))

        if mean_quiet_force < 100.0 or mean_quiet_force > 3000.0:
            qa_flags.append(f"implausible_estimated_bw_{mean_quiet_force:.1f}N")
            logger.warning("Estimated baseline force %.1f N is outside plausible human range [100, 3000] N", mean_quiet_force)

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
