"""Unit tests for Project 3 bilateral asymmetry module (p3_nmt_profiler.asymmetry)."""
import numpy as np
import pytest

from p3_nmt_profiler.asymmetry import (
    compute_bilateral_asymmetry,
    evaluate_asymmetry_profile,
    AsymmetryTrialResult,
)
from shared.synth import make_cmj_trial


def test_compute_bilateral_asymmetry_basic() -> None:
    # 600 N Left, 400 N Right -> BAI = ((600-400)/1000)*100 = +20% Left-dominant
    res = compute_bilateral_asymmetry(600.0, 400.0, trial_id="T01")
    assert res.bai_pct == 20.0
    assert res.abs_bai_pct == 20.0
    assert res.favored_limb == "Left"
    assert res.lsi_pct == 150.0  # 600 / 400 * 100

    # Symmetric case
    res_sym = compute_bilateral_asymmetry(500.0, 505.0, trial_id="T02", symmetry_tolerance_pct=2.0)
    assert res_sym.favored_limb == "Symmetric"
    assert res_sym.abs_bai_pct < 1.0


def test_asymmetry_persistence_and_fatigue_slope() -> None:
    # Simulate 8 trials with persistent Left bias and increasing asymmetry under fatigue
    trials = []
    fatigue_loads = [100, 200, 300, 400, 500, 600, 700, 800]
    for i, load in enumerate(fatigue_loads):
        # Asymmetry increases from +5% to +19% as load increases
        left = 500.0 + (i * 10.0)
        right = 450.0 - (i * 5.0)
        t_res = compute_bilateral_asymmetry(left, right, trial_id=f"T{i+1:02d}")
        trials.append(t_res)

    prof = evaluate_asymmetry_profile("ATH_01", trials, fatigue_covariate=fatigue_loads)
    assert prof.dominant_limb == "Left"
    assert prof.persistence_pct == 100.0
    assert prof.is_persistent is True
    assert prof.fatigue_slope > 0.0  # Positive slope indicates increasing asymmetry with fatigue
    assert prof.fatigue_p_value < 0.05


def test_asymmetry_on_synthetic_force_generator() -> None:
    # Use shared.synth.generate_cmj_trial with injected limb asymmetry
    syn = make_cmj_trial(
        asymmetry_index=0.15,  # 15% right bias
        fs_hz=2000.0,
        random_seed=42,
    )
    assert syn.force_left_n is not None
    assert syn.force_right_n is not None

    peak_left = float(np.max(syn.force_left_n))
    peak_right = float(np.max(syn.force_right_n))

    res = compute_bilateral_asymmetry(peak_left, peak_right, trial_id="SYN_01")
    assert res.favored_limb == "Right"
    assert res.bai_pct < 0.0
    # Formula in synth: ASI = (F_R - F_L) / max(F_R, F_L) = 0.15 -> BAI = (L-R)/(L+R) approx -8.1%
    assert abs(res.bai_pct) > 5.0
