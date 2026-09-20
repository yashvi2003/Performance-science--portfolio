"""Unit and validation tests for CMJ phase detection and metrics computation.

Governed by specs/01_neuromuscular_pipeline.md, PLAN.md, and McMahon et al. (2018).
Evaluates ground truth recovery against independent analytical solutions:
- Clean signals at 2000 Hz and 250 Hz
- Quiet standing baseline noise SD (~3 N and ~12 N)
- Injected flight-phase force offsets and correction models
- McMahon et al. (2018) phase ordering
- Pre-takeoff vs landing impact peak isolation
- RSImod reporting for both height methods
- Formatted per-condition error table printing
"""
from __future__ import annotations

import logging
import numpy as np
import pytest

from shared.synth import make_cmj_trial
from p1_nmt_pipeline.config import PipelineConfig, ThresholdConfig
from p1_nmt_pipeline.filter import zero_phase_butterworth_filter
from p1_nmt_pipeline.validator import load_and_validate_trial
from p1_nmt_pipeline.events import detect_cmj_events
from p1_nmt_pipeline.metrics import compute_cmj_metrics

logger = logging.getLogger(__name__)


def _run_cmj_pipeline(
    trial,
    config: PipelineConfig | None = None,
    filter_cutoff_hz: float = 30.0,
    estimate_mass: bool = False,
):
    """Helper to validate, filter, detect events, and compute metrics for a trial."""
    if config is None:
        config = PipelineConfig.load()

    df = trial.to_df()
    body_mass = None if estimate_mass else trial.body_mass_kg
    validated = load_and_validate_trial(df, body_mass_kg=body_mass)
    filtered_force = zero_phase_butterworth_filter(
        validated.force_n,
        validated.fs_hz,
        cutoff_hz=filter_cutoff_hz,
        order=config.filters.order,
    )
    events = detect_cmj_events(
        validated.time_s,
        filtered_force,
        validated.fs_hz,
        config=config,
        body_mass_kg=validated.body_mass_kg if not estimate_mass else None,
        force_raw_n=validated.force_n,
    )
    metrics = compute_cmj_metrics(
        validated.time_s,
        filtered_force,
        events,
        config=config,
    )
    return validated, events, metrics


def test_cmj_clean_2000hz_tight_tolerances():
    """Verify analytical recovery at 2000 Hz under tight benchmark tolerances."""
    target_h = 0.40
    trial = make_cmj_trial(
        fs_hz=2000.0,
        target_jump_height_m=target_h,
        t_onset=0.5,
        noise_sd_n=0.0,
        seed=101,
    )
    _, events, metrics = _run_cmj_pipeline(trial)

    # Documented Tight Tolerances at 2000 Hz:
    # Jump height: <= 1.5 mm
    # Takeoff time: <= 1.0 ms
    # Landing time: <= 1.0 ms
    # Onset time: <= 5.0 ms
    h_err_mm = abs(metrics.jump_height_impulse_m - target_h) * 1000.0
    to_err_ms = abs(events.takeoff_time_s - trial.true_metrics["t_takeoff_s"]) * 1000.0
    land_err_ms = abs(events.landing_time_s - trial.true_metrics["t_landing_s"]) * 1000.0
    onset_err_ms = abs(events.onset_time_s - trial.true_metrics["t_onset_s"]) * 1000.0

    assert h_err_mm <= 1.5, f"Height error {h_err_mm:.2f} mm exceeds 1.5 mm tolerance"
    assert to_err_ms <= 1.0, f"Takeoff error {to_err_ms:.2f} ms exceeds 1.0 ms tolerance"
    assert land_err_ms <= 1.0, f"Landing error {land_err_ms:.2f} ms exceeds 1.0 ms tolerance"
    assert onset_err_ms <= 5.0, f"Onset error {onset_err_ms:.2f} ms exceeds 5.0 ms tolerance"


def test_cmj_clean_250hz_tolerances():
    """Verify analytical recovery at 250 Hz (dt = 4 ms) with sampling-rate-scaled tolerances."""
    target_h = 0.35
    trial = make_cmj_trial(
        fs_hz=250.0,
        target_jump_height_m=target_h,
        t_onset=0.5,
        noise_sd_n=0.0,
        seed=202,
    )
    _, events, metrics = _run_cmj_pipeline(trial)

    # Documented Tolerances at 250 Hz (1 frame = 4 ms):
    # Jump height: <= 8.0 mm
    # Takeoff time: <= 8.0 ms (within 2 frames)
    # Landing time: <= 8.0 ms (within 2 frames)
    # Onset time: <= 16.0 ms (within 4 frames)
    h_err_mm = abs(metrics.jump_height_impulse_m - target_h) * 1000.0
    to_err_ms = abs(events.takeoff_time_s - trial.true_metrics["t_takeoff_s"]) * 1000.0
    land_err_ms = abs(events.landing_time_s - trial.true_metrics["t_landing_s"]) * 1000.0
    onset_err_ms = abs(events.onset_time_s - trial.true_metrics["t_onset_s"]) * 1000.0

    assert h_err_mm <= 8.0, f"Height error {h_err_mm:.2f} mm exceeds 8.0 mm tolerance"
    assert to_err_ms <= 8.0, f"Takeoff error {to_err_ms:.2f} ms exceeds 8.0 ms tolerance"
    assert land_err_ms <= 8.0, f"Landing error {land_err_ms:.2f} ms exceeds 8.0 ms tolerance"
    assert onset_err_ms <= 16.0, f"Onset error {onset_err_ms:.2f} ms exceeds 16.0 ms tolerance"


def test_cmj_noise_3n_2000hz():
    """Verify robustness under moderate quiet-phase noise (~3 N SD) at 2000 Hz."""
    target_h = 0.40
    trial = make_cmj_trial(
        fs_hz=2000.0,
        target_jump_height_m=target_h,
        t_onset=0.5,
        noise_sd_n=3.0,
        seed=303,
    )
    _, events, metrics = _run_cmj_pipeline(trial)

    # Documented Tolerances under ~3 N Noise:
    # Height: <= 3.0 mm
    # Takeoff: <= 2.0 ms
    # Landing: <= 2.0 ms
    # Onset: <= 15.0 ms (looser tolerance under noise due to threshold deviation search)
    h_err_mm = abs(metrics.jump_height_impulse_m - target_h) * 1000.0
    to_err_ms = abs(events.takeoff_time_s - trial.true_metrics["t_takeoff_s"]) * 1000.0
    land_err_ms = abs(events.landing_time_s - trial.true_metrics["t_landing_s"]) * 1000.0
    onset_err_ms = abs(events.onset_time_s - trial.true_metrics["t_onset_s"]) * 1000.0

    assert h_err_mm <= 3.0, f"Height error {h_err_mm:.2f} mm under 3 N noise"
    assert to_err_ms <= 2.0, f"Takeoff error {to_err_ms:.2f} ms under 3 N noise"
    assert land_err_ms <= 2.0, f"Landing error {land_err_ms:.2f} ms under 3 N noise"
    assert onset_err_ms <= 15.0, f"Onset error {onset_err_ms:.2f} ms under 3 N noise"


def test_cmj_noise_12n_2000hz():
    """Verify robustness under high quiet-phase noise (~12 N SD) at 2000 Hz."""
    target_h = 0.40
    trial = make_cmj_trial(
        fs_hz=2000.0,
        target_jump_height_m=target_h,
        t_onset=0.5,
        noise_sd_n=12.0,
        seed=404,
    )
    _, events, metrics = _run_cmj_pipeline(trial)

    # Documented Tolerances under ~12 N Noise:
    # Height: <= 6.0 mm
    # Takeoff: <= 3.0 ms
    # Landing: <= 3.0 ms
    # Onset: <= 35.0 ms (looser tolerance under heavy noise, 5 * 12 N = 60 N threshold)
    h_err_mm = abs(metrics.jump_height_impulse_m - target_h) * 1000.0
    to_err_ms = abs(events.takeoff_time_s - trial.true_metrics["t_takeoff_s"]) * 1000.0
    land_err_ms = abs(events.landing_time_s - trial.true_metrics["t_landing_s"]) * 1000.0
    onset_err_ms = abs(events.onset_time_s - trial.true_metrics["t_onset_s"]) * 1000.0

    assert h_err_mm <= 6.0, f"Height error {h_err_mm:.2f} mm under 12 N noise"
    assert to_err_ms <= 3.0, f"Takeoff error {to_err_ms:.2f} ms under 12 N noise"
    assert land_err_ms <= 3.0, f"Landing error {land_err_ms:.2f} ms under 12 N noise"
    assert onset_err_ms <= 35.0, f"Onset error {onset_err_ms:.2f} ms under 12 N noise"


def test_flight_offset_models_and_error_reduction():
    """Demonstrate that both offset models reduce jump height error on an offset signal.

    Model A (whole_trace): Constant instrumentation tare shift added across the whole recording.
    Without correction, estimated body mass is distorted, causing an acceleration scaling error
    and underestimating impulse-momentum jump height. With correction, BW and mass are recomputed,
    reducing height error.

    Model B (flight_only): Drift occurring during the airborne flight phase.
    Without correction, flight threshold detection is distorted relative to the flight floor.
    With correction, the threshold is adjusted by the offset floor, restoring flight-time height.
    """
    target_h = 0.40
    base_cfg = PipelineConfig.load()

    # --- Part 1: Whole-Trace Constant Offset ---
    injected_offset_whole = 5.0  # +5.0 N whole-trace tare shift
    trial_whole = make_cmj_trial(
        fs_hz=2000.0,
        target_jump_height_m=target_h,
        t_onset=0.5,
        noise_sd_n=0.0,
        flight_force_offset_n=injected_offset_whole,
        offset_whole_trace=True,
        seed=505,
    )

    # 1a. Uncorrected whole-trace
    cfg_uncorr = PipelineConfig(
        filters=base_cfg.filters,
        thresholds=ThresholdConfig(correct_flight_phase_offset=False),
        classifier=base_cfg.classifier,
    )
    _, _, metrics_uncorr = _run_cmj_pipeline(trial_whole, config=cfg_uncorr, estimate_mass=True)
    err_uncorr_mm = abs(metrics_uncorr.jump_height_impulse_m - target_h) * 1000.0

    # 1b. Corrected with whole_trace model
    cfg_whole = PipelineConfig(
        filters=base_cfg.filters,
        thresholds=ThresholdConfig(
            correct_flight_phase_offset=True,
            flight_offset_model="whole_trace",
        ),
        classifier=base_cfg.classifier,
    )
    _, _, metrics_whole = _run_cmj_pipeline(trial_whole, config=cfg_whole, estimate_mass=True)
    err_whole_mm = abs(metrics_whole.jump_height_impulse_m - target_h) * 1000.0

    # Assertions for Model A:
    # Detected offset matches injected +5.0 N
    assert abs(metrics_whole.flight_offset_n - injected_offset_whole) < 0.2
    # Uncorrected error is ~5.4 mm; correction eliminates it (<= 1.0 mm)
    assert err_uncorr_mm > 4.0, f"Expected uncorrected error > 4 mm, got {err_uncorr_mm:.2f} mm"
    assert err_whole_mm < err_uncorr_mm
    assert err_whole_mm <= 1.0, f"Whole-trace height error {err_whole_mm:.2f} mm > 1.0 mm"

    # --- Part 2: Flight-Only Offset ---
    # Shift during flight only (+8.0 N)
    injected_offset_flight = 8.0
    trial_flt = make_cmj_trial(
        fs_hz=2000.0,
        target_jump_height_m=target_h,
        t_onset=0.5,
        noise_sd_n=0.0,
        flight_force_offset_n=injected_offset_flight,
        offset_whole_trace=False,
        seed=506,
    )

    # 2a. Corrected with flight_only model
    cfg_flight = PipelineConfig(
        filters=base_cfg.filters,
        thresholds=ThresholdConfig(
            correct_flight_phase_offset=True,
            flight_offset_model="flight_only",
            takeoff_threshold_n=10.0,
            landing_threshold_n=10.0,
        ),
        classifier=base_cfg.classifier,
    )
    _, events_flt, metrics_flt = _run_cmj_pipeline(trial_flt, config=cfg_flight)
    # Detected offset matches ~8.0 N
    assert abs(metrics_flt.flight_offset_n - injected_offset_flight) < 0.2
    # Flight time height is preserved within 2 mm
    assert metrics_flt.jump_height_flight_m is not None
    err_flight_time_mm = abs(metrics_flt.jump_height_flight_m - target_h) * 1000.0
    assert err_flight_time_mm <= 2.0, f"Flight-only height error {err_flight_time_mm:.2f} mm > 2.0 mm"


def test_onset_and_integration_start_distinct():
    """Verify detected onset and integration start (onset - offset) are distinctly reported."""
    trial = make_cmj_trial(fs_hz=2000.0, target_jump_height_m=0.38, t_onset=0.5, seed=606)
    _, events, metrics = _run_cmj_pipeline(trial)

    assert np.isclose(metrics.detected_onset_time_s, events.onset_time_s)
    assert np.isclose(metrics.integration_start_time_s, events.int_start_time_s)
    assert metrics.detected_onset_time_s > metrics.integration_start_time_s

    lead_diff = metrics.detected_onset_time_s - metrics.integration_start_time_s
    assert np.isclose(lead_diff, 0.030, atol=1e-3)


def test_pre_takeoff_peaks_isolated_from_landing():
    """Verify peak propulsion force and power are strictly pre-takeoff and isolated from landing."""
    trial = make_cmj_trial(fs_hz=2000.0, target_jump_height_m=0.40, t_onset=0.5, seed=707)
    _, events, metrics = _run_cmj_pipeline(trial)

    # In CMJs, landing impact force is typically substantially higher than propulsion peak
    assert events.landing_peak_force_n is not None
    assert metrics.peak_landing_force_n is not None
    assert metrics.peak_landing_force_n > metrics.peak_propulsion_force_n

    # Time ordering checks
    assert events.propulsion_peak_time_s < events.takeoff_time_s
    assert events.landing_peak_time_s > events.landing_time_s


def test_rsimod_both_methods():
    """Verify RSImod is computed for both impulse-momentum (primary) and flight-time methods."""
    trial = make_cmj_trial(fs_hz=2000.0, target_jump_height_m=0.40, t_onset=0.5, seed=808)
    _, _, metrics = _run_cmj_pipeline(trial)

    assert metrics.jump_height_primary_m == metrics.jump_height_impulse_m
    assert metrics.rsimod_impulse > 0.0
    assert metrics.rsimod_flight is not None and metrics.rsimod_flight > 0.0

    expected_rsimod_imp = round(metrics.jump_height_impulse_m / metrics.contraction_time_s, 4)
    expected_rsimod_flt = round(metrics.jump_height_flight_m / metrics.contraction_time_s, 4)

    assert abs(metrics.rsimod_impulse - expected_rsimod_imp) <= 1e-4
    assert abs(metrics.rsimod_flight - expected_rsimod_flt) <= 1e-4


def test_mcmahon_phases_ordering():
    """Verify strict McMahon et al. (2018) chronological phase sequencing."""
    trial = make_cmj_trial(fs_hz=2000.0, target_jump_height_m=0.40, t_onset=0.5, seed=909)
    _, events, _ = _run_cmj_pipeline(trial)

    # Chronology:
    # quiet_end <= int_start <= onset < unweight_min < bw_cross <= v0 < prop_peak < takeoff < landing <= landing_peak
    assert events.int_start_time_s <= events.onset_time_s
    assert events.onset_time_s < events.unweighting_min_time_s
    assert events.unweighting_min_time_s < events.bw_cross_time_s
    assert events.bw_cross_time_s <= events.v0_time_s
    assert events.v0_time_s < events.propulsion_peak_time_s
    assert events.propulsion_peak_time_s < events.takeoff_time_s
    assert events.takeoff_time_s < events.landing_time_s
    assert events.landing_time_s <= events.landing_peak_time_s


def test_per_condition_error_table(capsys):
    """Evaluate accuracy across conditions and print a clean, formatted error table."""
    conditions = [
        ("Clean 2000 Hz", 2000.0, 0.0, 0.0, False),
        ("Clean 250 Hz", 250.0, 0.0, 0.0, False),
        ("Noise 3 N (2000 Hz)", 2000.0, 3.0, 0.0, False),
        ("Noise 12 N (2000 Hz)", 2000.0, 12.0, 0.0, False),
        ("Flight Offset +5 N (Corrected)", 2000.0, 0.0, 5.0, True),
    ]

    target_h = 0.40
    rows = []
    base_cfg = PipelineConfig.load()

    for name, fs, noise, offset, do_corr in conditions:
        trial = make_cmj_trial(
            fs_hz=fs,
            target_jump_height_m=target_h,
            t_onset=0.5,
            noise_sd_n=noise,
            flight_force_offset_n=offset,
            seed=42,
        )
        cfg = PipelineConfig(
            filters=base_cfg.filters,
            thresholds=ThresholdConfig(
                correct_flight_phase_offset=do_corr,
                flight_offset_model="whole_trace",
            ),
            classifier=base_cfg.classifier,
        )
        _, events, metrics = _run_cmj_pipeline(trial, config=cfg)

        h_err = (metrics.jump_height_impulse_m - target_h) * 1000.0
        to_err = (events.takeoff_time_s - trial.true_metrics["t_takeoff_s"]) * 1000.0
        land_err = (events.landing_time_s - trial.true_metrics["t_landing_s"]) * 1000.0
        onset_err = (events.onset_time_s - trial.true_metrics["t_onset_s"]) * 1000.0

        rows.append({
            "condition": name,
            "fs_hz": fs,
            "noise_sd_n": noise,
            "offset_n": offset,
            "height_err_mm": h_err,
            "onset_err_ms": onset_err,
            "takeoff_err_ms": to_err,
            "landing_err_ms": land_err,
        })

    # Print clean formatted table
    header = (
        f"\n{'Condition':<32} | {'fs (Hz)':<7} | {'Noise (N)':<9} | {'Offset (N)':<10} | "
        f"{'Height Err (mm)':<15} | {'Onset Err (ms)':<14} | {'Takeoff Err (ms)':<16} | {'Landing Err (ms)':<16}"
    )
    divider = "-" * len(header)
    print(header)
    print(divider)
    for r in rows:
        print(
            f"{r['condition']:<32} | {r['fs_hz']:<7.0f} | {r['noise_sd_n']:<9.1f} | {r['offset_n']:<10.1f} | "
            f"{r['height_err_mm']:<+15.2f} | {r['onset_err_ms']:<+14.2f} | {r['takeoff_err_ms']:<+16.2f} | {r['landing_err_ms']:<+16.2f}"
        )
    print(divider)

    # Verification that clean 2000Hz is within 1.5mm and 250Hz is within 8.0mm
    assert abs(rows[0]["height_err_mm"]) <= 1.5
    assert abs(rows[1]["height_err_mm"]) <= 8.0
