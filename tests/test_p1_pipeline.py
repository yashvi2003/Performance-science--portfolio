"""Unit tests for Project 1 Milestone M1a: Config, Validator, and Filter."""
import numpy as np
import pandas as pd
import pytest

from p1_nmt_pipeline.config import PipelineConfig
from p1_nmt_pipeline.filter import butterworth_lowpass_filter
from p1_nmt_pipeline.validator import load_and_validate_trial
from shared.synth import make_cmj_trial


def test_pipeline_config_defaults():
    """Verify PipelineConfig loads default YAML values correctly."""
    cfg = PipelineConfig.load()
    assert cfg.filters.order == 4
    assert cfg.filters.get_cutoff("CMJ") == 30.0
    assert cfg.filters.get_cutoff("DJ") == 50.0
    assert cfg.filters.get_cutoff("IMTP") == 20.0
    assert cfg.filters.get_cutoff("unknown") == 30.0

    assert cfg.thresholds.onset_sd_multiplier == 5.0
    assert cfg.thresholds.integration_lead_s == 0.030
    assert cfg.thresholds.correct_flight_phase_offset is False


def test_filter_preserves_dc_baseline():
    """Verify zero-phase filter preserves constant DC force component without drift."""
    fs = 2000.0
    dc_force = 750.0
    data = np.full(1000, dc_force)

    filtered = butterworth_lowpass_filter(data, fs_hz=fs, cutoff_hz=30.0, order=4)
    np.testing.assert_allclose(filtered, dc_force, rtol=1e-5, atol=1e-5)


def test_filter_attenuates_high_frequency():
    """Verify 30 Hz low-pass filter suppresses 120 Hz sinusoidal noise."""
    fs = 2000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    clean_signal = 750.0 + 200.0 * np.sin(2 * np.pi * 3.0 * t)  # 3 Hz physiological motion
    noise = 50.0 * np.sin(2 * np.pi * 120.0 * t)                # 120 Hz high-frequency ripple
    noisy_signal = clean_signal + noise

    filtered = butterworth_lowpass_filter(noisy_signal, fs_hz=fs, cutoff_hz=30.0, order=4)

    # Residual high frequency amplitude should be heavily attenuated (> 20 dB / factor of 10)
    noise_residual = np.std(filtered - clean_signal)
    assert noise_residual < 2.0  # Originally std ~ 35.3


def test_filter_zero_phase_alignment():
    """Verify forward-backward filter exhibits zero phase delay (peak location identical)."""
    fs = 2000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    # Gaussian pulse centered at t = 0.5 s
    pulse = 1000.0 * np.exp(-((t - 0.5) ** 2) / (2 * (0.05 ** 2)))

    filtered = butterworth_lowpass_filter(pulse, fs_hz=fs, cutoff_hz=30.0, order=4)

    idx_peak_raw = int(np.argmax(pulse))
    idx_peak_filt = int(np.argmax(filtered))
    assert idx_peak_raw == idx_peak_filt


def test_filter_nyquist_violation():
    """Verify filter raises ValueError when cutoff violates Nyquist frequency."""
    fs = 250.0
    with pytest.raises(ValueError, match="Nyquist"):
        butterworth_lowpass_filter(np.ones(100), fs_hz=fs, cutoff_hz=150.0)


def test_validator_loads_and_estimates_body_mass():
    """Verify trial validator normalizes columns and estimates body mass from quiet window."""
    trial = make_cmj_trial(body_mass_kg=75.0, fs_hz=1000.0)
    df = trial.to_df()

    # Pass DataFrame without explicit mass
    validated = load_and_validate_trial(df)
    assert validated.body_mass_estimated is True
    assert validated.body_mass_kg == pytest.approx(75.0, abs=0.5)
    assert validated.fs_hz == 1000.0
    assert any("body_mass_estimated" in flag for flag in validated.qa_flags)


def test_validator_detects_duplicate_timestamps():
    """Verify validator flags duplicate timestamps."""
    df = pd.DataFrame({
        "time": [0.0, 0.001, 0.001, 0.002],
        "force": [700, 700, 700, 700],
    })
    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_and_validate_trial(df)


def test_validator_detects_non_monotonic_time():
    """Verify validator rejects backwards timestamps."""
    df = pd.DataFrame({
        "time": [0.0, 0.002, 0.001, 0.003],
        "force": [700, 700, 700, 700],
    })
    with pytest.raises(ValueError, match="non-monotonic"):
        load_and_validate_trial(df)


def test_validator_resamples_irregular_dt_and_flags_negative_force():
    """Verify irregular sampling intervals are resampled onto a uniform grid with a QA warning."""
    t = [0.0, 0.001, 0.002, 0.0035, 0.0045, 0.0055]  # gap at index 3
    f = [700.0, 700.0, 700.0, -15.0, 700.0, 700.0]   # negative force at index 3
    df = pd.DataFrame({"time": t, "force": f})

    validated = load_and_validate_trial(df, body_mass_kg=70.0, quiet_window_s=0.005)
    assert any("resampled_irregular_dt" in flag for flag in validated.qa_flags)
    assert any("negative_force" in flag for flag in validated.qa_flags)
    # Check that resampled time array has strictly uniform step
    diffs = np.diff(validated.time_s)
    assert np.allclose(diffs, diffs[0], atol=1e-8)


def test_validator_quiet_window_qa_flags():
    """Verify quiet window QA flags noisy SD (>15 N) and drifting trend (>20 N/s)."""
    t = np.linspace(0.0, 0.6, 601)
    # Drifting trend: 700 N + 30 N/s * t
    f_drifting = 700.0 + 30.0 * t
    df_drifting = pd.DataFrame({"time": t, "force": f_drifting})
    val_drift = load_and_validate_trial(df_drifting, body_mass_kg=70.0)
    assert any("quiet_window_drifting_trend" in flag for flag in val_drift.qa_flags)

    # High noise: SD ~ 20 N
    np.random.seed(42)
    f_noisy = 700.0 + np.random.normal(0, 20.0, len(t))
    df_noisy = pd.DataFrame({"time": t, "force": f_noisy})
    val_noisy = load_and_validate_trial(df_noisy, body_mass_kg=70.0)
    assert any("quiet_window_noisy_sd" in flag for flag in val_noisy.qa_flags)
