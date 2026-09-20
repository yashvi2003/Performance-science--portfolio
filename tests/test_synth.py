"""Unit tests for shared/synth.py with sampling-rate-dependent tolerances."""
import numpy as np
import pytest

from shared.contract import (
    STANDARD_G,
    TestType,
    validate_sessions_df,
    validate_wearable_daily_df,
)
from shared.synth import (
    make_cmj_trial,
    make_dj_trial,
    make_force_trial,
    make_imtp_trial,
    make_sessions_table,
    make_sj_trial,
    make_wearable_daily,
)


def test_cmj_analytical_ground_truth_2000hz():
    """Verify numerical integration matches analytical true jump height within 1-2 mm at 2000 Hz."""
    target_height = 0.380
    fs = 2000.0
    mass = 80.0

    trial = make_cmj_trial(
        body_mass_kg=mass,
        target_jump_height_m=target_height,
        fs_hz=fs,
        noise_sd_n=0.0,
    )

    true_h = trial.true_metrics["jump_height_imp_m"]
    assert true_h == pytest.approx(target_height, abs=1e-6)

    bw = trial.body_weight_n
    idx_onset = trial.event_indices["onset"]
    idx_to = trial.event_indices["takeoff"]

    net_force = trial.force_n[idx_onset:idx_to + 1] - bw
    acc = net_force / mass
    dt = 1.0 / fs

    v_to_num = np.sum((acc[:-1] + acc[1:]) / 2.0) * dt
    h_num = (v_to_num ** 2) / (2.0 * STANDARD_G)

    abs_error_m = abs(h_num - true_h)
    assert abs_error_m <= 0.002, f"Expected error <= 2 mm at 2000 Hz, got {abs_error_m * 1000:.3f} mm"


def test_cmj_analytical_ground_truth_250hz():
    """Verify looser tolerance (<= 10 mm) at 250 Hz due to discrete sampling integration error."""
    target_height = 0.350
    fs = 250.0
    mass = 75.0

    trial = make_cmj_trial(
        body_mass_kg=mass,
        target_jump_height_m=target_height,
        fs_hz=fs,
        noise_sd_n=0.0,
    )

    true_h = trial.true_metrics["jump_height_imp_m"]
    bw = trial.body_weight_n
    idx_onset = trial.event_indices["onset"]
    idx_to = trial.event_indices["takeoff"]

    net_force = trial.force_n[idx_onset:idx_to + 1] - bw
    acc = net_force / mass
    dt = 1.0 / fs

    v_to_num = np.sum((acc[:-1] + acc[1:]) / 2.0) * dt
    h_num = (v_to_num ** 2) / (2.0 * STANDARD_G)

    abs_error_m = abs(h_num - true_h)
    assert abs_error_m <= 0.010, f"Expected error <= 10 mm at 250 Hz, got {abs_error_m * 1000:.3f} mm"


def test_sj_analytical_ground_truth():
    """Verify Squat Jump generator produces concentric acceleration with known ground truth."""
    target_h = 0.280
    fs = 2000.0
    mass = 70.0

    trial = make_sj_trial(
        body_mass_kg=mass,
        target_jump_height_m=target_h,
        fs_hz=fs,
        inject_countermovement=False,
    )

    assert trial.test == TestType.SJ
    assert trial.true_metrics["countermovement_injected"] == 0.0

    # Ensure force never dips below body weight prior to takeoff
    idx_onset = trial.event_indices["onset"]
    idx_to = trial.event_indices["takeoff"]
    pre_to_force = trial.force_n[idx_onset:idx_to]
    assert np.all(pre_to_force >= trial.body_weight_n - 1e-6)

    # Test countermovement injection flag
    trial_invalid = make_sj_trial(inject_countermovement=True, dip_magnitude_n=150.0)
    assert trial_invalid.true_metrics["countermovement_injected"] == 1.0
    min_pre_force = np.min(trial_invalid.force_n[:trial_invalid.event_indices["takeoff"]])
    assert min_pre_force < trial_invalid.body_weight_n - 50.0


def test_dj_analytical_ground_truth():
    """Verify Drop Jump generator produces initial drop flight, ground contact, and rebound."""
    drop_h = 0.30
    rebound_h = 0.35
    gct = 0.20
    fs = 2000.0

    trial = make_dj_trial(
        drop_height_m=drop_h,
        target_rebound_height_m=rebound_h,
        gct_s=gct,
        fs_hz=fs,
    )

    assert trial.test == TestType.DJ
    assert trial.true_metrics["rsi"] == pytest.approx(rebound_h / gct, rel=1e-4)

    # Verify initial drop flight phase has force near 0 N
    idx_impact = trial.event_indices["impact"]
    assert np.all(trial.force_n[:idx_impact] == 0.0)

    # Verify ground contact phase duration
    idx_to = trial.event_indices["takeoff"]
    actual_gct = (idx_to - idx_impact) / fs
    assert actual_gct == pytest.approx(gct, abs=2.0 / fs)


def test_imtp_analytical_ground_truth():
    """Verify IMTP generator produces baseline, pull rise to plateau, and no flight phase."""
    fs = 2000.0
    mass = 80.0
    net_peak = 2500.0

    trial = make_imtp_trial(
        body_mass_kg=mass,
        peak_force_net_n=net_peak,
        fs_hz=fs,
    )

    assert trial.test == TestType.IMTP
    bw = trial.body_weight_n
    gross_peak = trial.true_metrics["peak_force_gross_n"]
    assert gross_peak == pytest.approx(bw + net_peak, abs=1e-3)

    # Verify force never drops to 0 (no flight phase)
    assert np.all(trial.force_n > 0.5 * bw)

    # Verify force at 100ms > baseline
    f_100ms = trial.true_metrics["force_100ms_n"]
    assert f_100ms > bw


def test_bilateral_asymmetry_sum_and_ratio():
    """Verify left and right limb forces sum exactly to total force and match asymmetry index."""
    asi = 0.12
    trial = make_cmj_trial(asymmetry_index=asi, fs_hz=1000.0)

    assert trial.force_left_n is not None
    assert trial.force_right_n is not None

    summed = trial.force_left_n + trial.force_right_n
    np.testing.assert_allclose(summed, trial.force_n, rtol=1e-12, atol=1e-12)

    idx_quiet = int(trial.event_indices["onset"] * 0.5)
    f_l = trial.force_left_n[idx_quiet]
    f_r = trial.force_right_n[idx_quiet]
    calculated_asi = (f_r - f_l) / max(f_r, f_l)
    assert calculated_asi == pytest.approx(asi, rel=1e-4)


def test_flight_force_offset_injection():
    """Verify synthetic generator accurately injects non-zero flight phase force offsets."""
    offset_val = 4.5
    trial = make_cmj_trial(flight_force_offset_n=offset_val, fs_hz=1000.0, noise_sd_n=0.0)

    idx_to = trial.event_indices["takeoff"]
    idx_land = trial.event_indices["landing"]

    mid_flight = (idx_to + idx_land) // 2
    assert trial.force_n[mid_flight] == pytest.approx(offset_val, abs=1e-6)


def test_make_sessions_table_contract_compliance():
    """Verify make_sessions_table produces 100% contract-compliant output."""
    df = make_sessions_table(n_athletes=3, n_weeks=4)
    is_valid, errors = validate_sessions_df(df)
    assert is_valid is True, f"Validation errors: {errors}"
    assert len(df) > 0


def test_make_wearable_daily_contract_compliance():
    """Verify make_wearable_daily produces 100% contract-compliant output."""
    df = make_wearable_daily(n_athletes=3, n_days=28)
    is_valid, errors = validate_wearable_daily_df(df)
    assert is_valid is True, f"Validation errors: {errors}"
    assert len(df) == 3 * 28
