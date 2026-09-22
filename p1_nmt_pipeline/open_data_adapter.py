"""Open data adapter for the White et al. (2022/2025) CMJ dataset.

Processes body-weight normalized vertical ground reaction force (vGRF) recordings
from White et al. (Zenodo DOI: 10.5281/zenodo.19136480; PLOS ONE DOI: 10.1371/journal.pone.0263846).

DATASET CHARACTERISTICS:
    - 663 CMJ trials from 67 unique participants (subject_ids 0..68 excluding {14, 29}).
    - The metadata field `n_subjects=69` reflects the parent study recruitment cohort,
      not the active participant count of `cmj_dataset_both.npz`.
    - vGRF is 1000 Hz, body-weight normalized (BW), and pre-trimmed to end strictly at takeoff.
    - `grf_takeoff` is the index in the raw continuous un-trimmed recording; it is stored
      as `raw_continuous_takeoff_idx` for provenance and NEVER used to index `grf_signals`.

JUMP HEIGHT NAMING CONVENTION:
    - `jump_height_takeoff_vel_m`: Flight-trajectory elevation of COM above takeoff = v_to^2 / (2g).
    - `s_takeoff_m`: COM displacement at takeoff relative to standing quiet baseline = double-integral of a(t).
    - `jump_height_total_com_m`: Total COM excursion above quiet standing = s_to + v_to^2 / (2g).
    - `ref_jump_height_m`: The author's reference jump height (which equals total COM excursion).

PHYSICAL CONSTANT:
    - Standard gravity g = 9.80665 m/s^2 is used across all pipeline calculations.
    - The reference dataset used g = 9.80 m/s^2, resulting in a documented ~0.12% offset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

logger = logging.getLogger(__name__)

# Standard physical constants
STANDARD_G_M_PER_S2: float = 9.80665
DEFAULT_GRF_SAMPLING_RATE_HZ: float = 1000.0
DEFAULT_BASELINE_WINDOW_S: float = 0.150


@dataclass(frozen=True)
class OpenTrialResult:
    """Calculated metrics and QA metadata for a single open-data CMJ trial."""

    trial_idx: int
    subject_id: int
    original_participant_id: int
    condition_code: int  # 1 = CMJNA, 2 = CMJA
    condition_name: str  # 'CMJNA' or 'CMJA'
    sampling_rate_hz: float
    signal_duration_s: float
    n_samples: int
    baseline_mean_bw: float
    baseline_sd_bw: float
    terminal_force_bw: float
    raw_continuous_takeoff_idx: int
    takeoff_sample_idx: int
    s_takeoff_m: float
    v_takeoff_m_per_s: float
    jump_height_takeoff_vel_m: float
    jump_height_total_com_m: float
    ref_jump_height_m: float
    peak_power_w_per_kg: float
    ref_peak_power_w_per_kg: float
    diff_jump_height_total_com_cm: float
    pct_error_jump_height_total_com: float
    diff_peak_power_w_per_kg: float
    pct_error_peak_power: float
    flag_negative_terminal_force: bool
    flag_terminal_force_outlier: bool
    flag_baseline_unstable: bool
    qa_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert result dataclass to a flat dictionary."""
        return {
            "trial_idx": self.trial_idx,
            "subject_id": self.subject_id,
            "original_participant_id": self.original_participant_id,
            "condition_code": self.condition_code,
            "condition_name": self.condition_name,
            "sampling_rate_hz": self.sampling_rate_hz,
            "signal_duration_s": self.signal_duration_s,
            "n_samples": self.n_samples,
            "baseline_mean_bw": self.baseline_mean_bw,
            "baseline_sd_bw": self.baseline_sd_bw,
            "terminal_force_bw": self.terminal_force_bw,
            "raw_continuous_takeoff_idx": self.raw_continuous_takeoff_idx,
            "takeoff_sample_idx": self.takeoff_sample_idx,
            "s_takeoff_m": self.s_takeoff_m,
            "v_takeoff_m_per_s": self.v_takeoff_m_per_s,
            "jump_height_takeoff_vel_m": self.jump_height_takeoff_vel_m,
            "jump_height_total_com_m": self.jump_height_total_com_m,
            "ref_jump_height_m": self.ref_jump_height_m,
            "peak_power_w_per_kg": self.peak_power_w_per_kg,
            "ref_peak_power_w_per_kg": self.ref_peak_power_w_per_kg,
            "diff_jump_height_total_com_cm": self.diff_jump_height_total_com_cm,
            "pct_error_jump_height_total_com": self.pct_error_jump_height_total_com,
            "diff_peak_power_w_per_kg": self.diff_peak_power_w_per_kg,
            "pct_error_peak_power": self.pct_error_peak_power,
            "flag_negative_terminal_force": self.flag_negative_terminal_force,
            "flag_terminal_force_outlier": self.flag_terminal_force_outlier,
            "flag_baseline_unstable": self.flag_baseline_unstable,
            "qa_flags": ";".join(self.qa_flags),
        }


@dataclass
class OpenDataset:
    """Container for loaded raw arrays and verified metadata from cmj_dataset_both.npz."""

    grf_signals: np.ndarray
    acc_signals: np.ndarray
    grf_takeoff: np.ndarray
    acc_takeoff: np.ndarray
    jump_height: np.ndarray
    peak_power: np.ndarray
    subject_ids: np.ndarray
    original_participant_ids: np.ndarray
    condition_labels: np.ndarray
    acc_sampling_rate: float
    grf_sampling_rate: float
    n_subjects_metadata: int
    n_unique_subjects: int
    n_trials: int


def load_open_dataset(npz_path: str | Path = "data/raw/open/cmj_dataset_both.npz") -> OpenDataset:
    """Load and validate the White et al. (2025) NPZ archive.

    Parameters
    ----------
    npz_path : str | Path
        Path to cmj_dataset_both.npz.

    Returns
    -------
    OpenDataset
        Validated container of raw arrays.
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Open dataset file not found at: {path.resolve()}")

    logger.info("Loading open dataset from %s", path)
    # allow_pickle=True is required because grf_signals and acc_signals are ragged arrays of 1D arrays
    data = np.load(path, allow_pickle=True)

    grf_signals = data["grf_signals"]
    acc_signals = data["acc_signals"]
    grf_takeoff = data["grf_takeoff"]
    acc_takeoff = data["acc_takeoff"]
    jump_height = data["jump_height"]
    peak_power = data["peak_power"]
    subject_ids = data["subject_ids"]
    original_participant_ids = data["original_participant_ids"]
    condition_labels = data["condition_labels"]

    n_trials = len(grf_signals)
    unique_subjs = np.unique(subject_ids)
    n_unique = len(unique_subjs)
    n_meta = int(data["n_subjects"])

    # Audit checks
    if n_trials != 663:
        logger.warning("Expected 663 trials in cmj_dataset_both.npz, found %d", n_trials)
    if n_unique != 67:
        logger.warning("Expected 67 unique subject_ids, found %d", n_unique)

    grf_sr = float(data["grf_sampling_rate"]) if "grf_sampling_rate" in data else DEFAULT_GRF_SAMPLING_RATE_HZ
    acc_sr = float(data["acc_sampling_rate"]) if "acc_sampling_rate" in data else 250.0

    return OpenDataset(
        grf_signals=grf_signals,
        acc_signals=acc_signals,
        grf_takeoff=grf_takeoff,
        acc_takeoff=acc_takeoff,
        jump_height=jump_height,
        peak_power=peak_power,
        subject_ids=subject_ids,
        original_participant_ids=original_participant_ids,
        condition_labels=condition_labels,
        acc_sampling_rate=acc_sr,
        grf_sampling_rate=grf_sr,
        n_subjects_metadata=n_meta,
        n_unique_subjects=n_unique,
        n_trials=n_trials,
    )


def analyze_open_trial(
    dataset: OpenDataset,
    trial_idx: int,
    g: float = STANDARD_G_M_PER_S2,
    baseline_duration_s: float = DEFAULT_BASELINE_WINDOW_S,
) -> OpenTrialResult:
    """Analyze a single CMJ trial from the open dataset.

    Calculates takeoff velocity, center of mass position at takeoff,
    flight jump height, total COM excursion, and peak relative power
    using forward trapezoidal integration.

    Parameters
    ----------
    dataset : OpenDataset
        Container loaded via load_open_dataset.
    trial_idx : int
        Index of the trial (0..n_trials-1).
    g : float
        Acceleration due to gravity in m/s^2 (default: 9.80665).
    baseline_duration_s : float
        Duration of initial quiet window for baseline checks (default: 0.150 s).

    Returns
    -------
    OpenTrialResult
        Structured calculation results with QA flags.
    """
    if trial_idx < 0 or trial_idx >= dataset.n_trials:
        raise IndexError(f"Trial index {trial_idx} out of range [0, {dataset.n_trials - 1}]")

    f_bw = np.asarray(dataset.grf_signals[trial_idx], dtype=np.float64)
    n_samples = len(f_bw)
    dt = 1.0 / dataset.grf_sampling_rate
    duration_s = n_samples * dt

    # Baseline quiet standing check (initial 150 ms)
    baseline_samples = max(10, int(round(baseline_duration_s * dataset.grf_sampling_rate)))
    baseline_window = f_bw[:min(baseline_samples, n_samples)]
    base_mean = float(np.mean(baseline_window))
    base_sd = float(np.std(baseline_window))

    # Net vertical acceleration in m/s^2:
    # a(t) = (F_bw(t) - 1.0) * g
    acc = (f_bw - 1.0) * g

    # Forward integration using trapezoidal rule:
    # v(t) = \int_0^t a(\tau) d\tau
    vel = cumulative_trapezoid(acc, dx=dt, initial=0.0)

    # Position displacement:
    # s(t) = \int_0^t v(\tau) d\tau
    pos = cumulative_trapezoid(vel, dx=dt, initial=0.0)

    # Instantaneous relative power (W/kg):
    # P(t) = (F_bw(t) * g) * v(t)
    power = (f_bw * g) * vel
    peak_pwr = float(np.max(power))

    # Takeoff state (movement window terminates right at toe-off)
    v_to = float(vel[-1])
    s_to = float(pos[-1])
    terminal_f = float(f_bw[-1])

    # Distinct jump height definitions
    h_to_vel = float((v_to**2) / (2.0 * g))
    h_total_com = float(s_to + h_to_vel)

    # Reference values
    ref_jh = float(dataset.jump_height[trial_idx])
    ref_pp = float(dataset.peak_power[trial_idx])

    # Residuals
    diff_jh_cm = (h_total_com - ref_jh) * 100.0
    pct_err_jh = ((h_total_com - ref_jh) / ref_jh) * 100.0 if ref_jh != 0.0 else 0.0

    diff_pp = peak_pwr - ref_pp
    pct_err_pp = ((peak_pwr - ref_pp) / ref_pp) * 100.0 if ref_pp != 0.0 else 0.0

    # QA checks
    qa_flags: list[str] = []
    flag_neg = terminal_f < 0.0
    if flag_neg:
        qa_flags.append(f"NEGATIVE_TERMINAL_FORCE_{terminal_f:.5f}_BW")

    flag_outlier = abs(terminal_f) > 0.02
    if flag_outlier:
        qa_flags.append(f"TERMINAL_FORCE_OUTLIER_{terminal_f:.4f}_BW")

    flag_unstable = base_sd > 0.05
    if flag_unstable:
        qa_flags.append(f"BASELINE_UNSTABLE_SD_{base_sd:.4f}_BW")

    cond_code = int(dataset.condition_labels[trial_idx])
    cond_name = "CMJNA" if cond_code == 1 else "CMJA"

    return OpenTrialResult(
        trial_idx=trial_idx,
        subject_id=int(dataset.subject_ids[trial_idx]),
        original_participant_id=int(dataset.original_participant_ids[trial_idx]),
        condition_code=cond_code,
        condition_name=cond_name,
        sampling_rate_hz=dataset.grf_sampling_rate,
        signal_duration_s=duration_s,
        n_samples=n_samples,
        baseline_mean_bw=base_mean,
        baseline_sd_bw=base_sd,
        terminal_force_bw=terminal_f,
        raw_continuous_takeoff_idx=int(dataset.grf_takeoff[trial_idx]),
        takeoff_sample_idx=n_samples - 1,
        s_takeoff_m=s_to,
        v_takeoff_m_per_s=v_to,
        jump_height_takeoff_vel_m=h_to_vel,
        jump_height_total_com_m=h_total_com,
        ref_jump_height_m=ref_jh,
        peak_power_w_per_kg=peak_pwr,
        ref_peak_power_w_per_kg=ref_pp,
        diff_jump_height_total_com_cm=diff_jh_cm,
        pct_error_jump_height_total_com=pct_err_jh,
        diff_peak_power_w_per_kg=diff_pp,
        pct_error_peak_power=pct_err_pp,
        flag_negative_terminal_force=flag_neg,
        flag_terminal_force_outlier=flag_outlier,
        flag_baseline_unstable=flag_unstable,
        qa_flags=tuple(qa_flags),
    )


def process_all_open_trials(
    dataset: OpenDataset,
    g: float = STANDARD_G_M_PER_S2,
) -> tuple[list[OpenTrialResult], pd.DataFrame]:
    """Process all trials in the open dataset into a results list and pandas DataFrame.

    Parameters
    ----------
    dataset : OpenDataset
        Loaded open dataset.
    g : float
        Acceleration due to gravity (default: 9.80665).

    Returns
    -------
    tuple[list[OpenTrialResult], pd.DataFrame]
        List of result dataclasses and comprehensive DataFrame.
    """
    logger.info("Processing %d trials with g=%.5f m/s^2...", dataset.n_trials, g)
    results: list[OpenTrialResult] = []
    records: list[dict[str, Any]] = []

    for i in range(dataset.n_trials):
        res = analyze_open_trial(dataset, i, g=g)
        results.append(res)
        records.append(res.to_dict())

    df = pd.DataFrame.from_records(records)
    logger.info("Finished processing %d open CMJ trials.", len(df))
    return results, df


def build_balanced_trial_matrix(
    df: pd.DataFrame,
    condition: str = "CMJNA",
    k_trials: int = 4,
    metric_col: str = "jump_height_total_com_m",
) -> tuple[np.ndarray, list[int], str]:
    """Extract a balanced (n_subjects x k_trials) matrix for repeated-measures reliability.

    Takes the first k_trials in chronological recording order for each subject
    having at least k_trials in the given condition.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of processed trials.
    condition : str
        'CMJNA' or 'CMJA'.
    k_trials : int
        Number of trials per subject (e.g. 3 or 4).
    metric_col : str
        Name of metric column to extract.

    Returns
    -------
    tuple[np.ndarray, list[int], str]
        - matrix: 2D array of shape (n_subjects, k_trials)
        - subject_list: list of included subject_ids
        - denominator_str: explicit documentation string
    """
    df_cond = df[df["condition_name"] == condition].sort_values("trial_idx")

    # Group trials by subject
    by_subj: dict[int, list[float]] = {}
    for _, row in df_cond.iterrows():
        s = int(row["subject_id"])
        val = float(row[metric_col])
        if s not in by_subj:
            by_subj[s] = []
        by_subj[s].append(val)

    # Filter subjects with at least k_trials and take first k
    qualifying_subjs = [s for s in sorted(by_subj.keys()) if len(by_subj[s]) >= k_trials]
    matrix = np.array([by_subj[s][:k_trials] for s in qualifying_subjs], dtype=np.float64)

    n_qualifying = len(qualifying_subjs)
    n_obs = n_qualifying * k_trials
    denom_str = (
        f"{condition} balanced {k_trials}-trial set, "
        f"n={n_qualifying} subjects x {k_trials} trials = {n_obs} observations"
    )

    return matrix, qualifying_subjs, denom_str
