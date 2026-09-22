"""Bilateral and limb asymmetry profiling module for Project 3 (p3_nmt_profiler).

Implements:
1. Bilateral Asymmetry Index (BAI) and Limb Symmetry Index (LSI):
   - BAI = ((Left - Right) / (Left + Right)) * 100% (Directional: + = Left-dominant, - = Right-dominant)
   - Absolute Asymmetry: |BAI| = (|Left - Right| / (Left + Right)) * 100%
   - LSI = (Dominant / Non-Dominant) * 100%
2. Asymmetry Persistence Tracking:
   - Evaluates whether directional limb bias is systematic/persistent (>75% of trials in same direction)
     vs variable/swapping.
3. Asymmetry-Fatigue Sensitivity Slope:
   - Estimates change in asymmetry per unit acute fatigue/load deviation:
     slope = d(|BAI|) / d(Load_deviation) or d(|BAI|) / d(Trial_ordinal)
4. Comprehensive testing on labeled synthetic & real bilateral trials.
"""
from dataclasses import dataclass
import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsymmetryTrialResult:
    """Bilateral asymmetry metrics for a single trial or observation."""
    trial_id: str
    left_metric: float
    right_metric: float
    bai_pct: float             # ((L - R) / (L + R)) * 100
    abs_bai_pct: float         # |BAI|
    lsi_pct: float             # (Dominant / NonDominant) * 100
    favored_limb: str          # "Left", "Right", or "Symmetric"


@dataclass
class LongitudinalAsymmetryProfile:
    """Longitudinal asymmetry profile across repeated sessions/trials for an athlete."""
    athlete_id: str
    n_trials: int
    mean_bai_pct: float
    mean_abs_bai_pct: float
    dominant_limb: str
    persistence_pct: float     # Percentage of trials matching dominant limb direction
    is_persistent: bool        # True if persistence_pct >= 75.0%
    fatigue_slope: float       # Change in |BAI| per unit load deviation or trial index
    fatigue_p_value: float     # Significance of asymmetry increase under fatigue


def compute_bilateral_asymmetry(
    left_val: float,
    right_val: float,
    trial_id: str = "T01",
    symmetry_tolerance_pct: float = 2.0,
) -> AsymmetryTrialResult:
    """Compute directional BAI, absolute asymmetry, and LSI for a single trial.

    Parameters
    ----------
    left_val : float
        Metric value for left limb (e.g. peak force N, impulse N·s).
    right_val : float
        Metric value for right limb.
    trial_id : str
        Trial or session identifier.
    symmetry_tolerance_pct : float
        Threshold below which the trial is categorized as 'Symmetric' (default 2.0%).

    Returns
    -------
    AsymmetryTrialResult
        Structured asymmetry metrics.
    """
    total = left_val + right_val
    if total <= 0.0 or np.isnan(left_val) or np.isnan(right_val):
        return AsymmetryTrialResult(
            trial_id=trial_id,
            left_metric=left_val,
            right_metric=right_val,
            bai_pct=float("nan"),
            abs_bai_pct=float("nan"),
            lsi_pct=float("nan"),
            favored_limb="Unknown",
        )

    # Directional BAI: + = Left-dominant, - = Right-dominant
    bai = ((left_val - right_val) / total) * 100.0
    abs_bai = abs(bai)

    # Limb Symmetry Index: Dominant / Non-Dominant
    dom_val = max(left_val, right_val)
    nondom_val = min(left_val, right_val)
    lsi = (dom_val / nondom_val) * 100.0 if nondom_val > 0 else float("nan")

    if abs_bai < symmetry_tolerance_pct:
        favored = "Symmetric"
    elif bai > 0:
        favored = "Left"
    else:
        favored = "Right"

    return AsymmetryTrialResult(
        trial_id=trial_id,
        left_metric=round(left_val, 2),
        right_metric=round(right_val, 2),
        bai_pct=round(bai, 2),
        abs_bai_pct=round(abs_bai, 2),
        lsi_pct=round(lsi, 2),
        favored_limb=favored,
    )


def evaluate_asymmetry_profile(
    athlete_id: str,
    trials: Sequence[AsymmetryTrialResult],
    fatigue_covariate: Sequence[float] | None = None,
    persistence_threshold_pct: float = 75.0,
) -> LongitudinalAsymmetryProfile:
    """Evaluate longitudinal asymmetry persistence and fatigue-related slope.

    Parameters
    ----------
    athlete_id : str
        Athlete identifier.
    trials : Sequence[AsymmetryTrialResult]
        Collection of trial asymmetry results.
    fatigue_covariate : Sequence[float], optional
        Concurrent fatigue index (e.g. session-RPE, trial ordinal, or neuromuscular fatigue index).
        If None, trial ordinal sequence (0, 1, 2...) is used.
    persistence_threshold_pct : float
        Percentage of trials required in same direction to classify as persistent (default 75.0%).

    Returns
    -------
    LongitudinalAsymmetryProfile
        Longitudinal summary including dominant limb, persistence flag, and fatigue sensitivity slope.
    """
    valid_trials = [t for t in trials if not np.isnan(t.bai_pct)]
    n_trials = len(valid_trials)
    if n_trials == 0:
        raise ValueError(f"No valid asymmetry trials provided for athlete {athlete_id}")

    bai_vals = np.array([t.bai_pct for t in valid_trials])
    abs_bai_vals = np.array([t.abs_bai_pct for t in valid_trials])

    mean_bai = float(np.mean(bai_vals))
    mean_abs_bai = float(np.mean(abs_bai_vals))

    # Count limb preferences
    left_count = sum(1 for t in valid_trials if t.bai_pct > 0.5)
    right_count = sum(1 for t in valid_trials if t.bai_pct < -0.5)

    if left_count >= right_count and left_count > 0:
        dom_limb = "Left"
        persistence_pct = (left_count / n_trials) * 100.0
    elif right_count > left_count:
        dom_limb = "Right"
        persistence_pct = (right_count / n_trials) * 100.0
    else:
        dom_limb = "Symmetric"
        persistence_pct = 0.0

    is_persistent = bool(persistence_pct >= persistence_threshold_pct)

    # Fatigue sensitivity slope: d(|BAI|) / d(fatigue)
    if fatigue_covariate is not None and len(fatigue_covariate) == n_trials:
        x = np.array(fatigue_covariate)
    else:
        x = np.arange(n_trials, dtype=float)

    if n_trials >= 3 and np.std(x) > 0:
        # Linear regression of absolute asymmetry on fatigue
        from scipy.stats import linregress
        reg = linregress(x, abs_bai_vals)
        slope = float(reg.slope)
        p_val = float(reg.pvalue)
    else:
        slope = 0.0
        p_val = 1.0

    return LongitudinalAsymmetryProfile(
        athlete_id=athlete_id,
        n_trials=n_trials,
        mean_bai_pct=round(mean_bai, 2),
        mean_abs_bai_pct=round(mean_abs_bai, 2),
        dominant_limb=dom_limb,
        persistence_pct=round(persistence_pct, 1),
        is_persistent=is_persistent,
        fatigue_slope=round(slope, 4),
        fatigue_p_value=round(p_val, 4),
    )
