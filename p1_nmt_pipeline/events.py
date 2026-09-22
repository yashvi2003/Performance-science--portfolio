"""CMJ phase and event detection engine.

Implements countermovement jump event and phase boundary detection following
the biomechanical definitions established by McMahon et al. (2018):
- Unweighting Phase: Movement onset (t_onset) to minimum vertical force (t_unweight_min).
- Braking Phase (Eccentric Deceleration): Minimum vertical force (t_unweight_min)
  to zero vertical velocity (t_v0). Upward BW crossing (t_bw_cross) is recorded as a landmark.
- Propulsion Phase (Concentric Acceleration): Zero vertical velocity (t_v0) to takeoff (t_takeoff).
- Flight Phase: Takeoff (t_takeoff) to landing impact (t_landing).
- Landing Phase: Landing impact (t_landing) onwards.

Reference:
    McMahon, J. J., Suchomel, T. J., Lake, J. P., & Comfort, P. (2018).
    Understanding the Key Phases of the Countermovement Jump Cycle with Refinement
    to Movement Analysis Methodologies. Sports, 6(1), 13.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.contract import STANDARD_G
from p1_nmt_pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CMJEvents:
    """Detected event indices, timestamps, and landmarks for a CMJ trial.

    Attributes
    ----------
    quiet_start_idx : int
        Start sample index of the quiet standing baseline window.
    quiet_end_idx : int
        End sample index of the quiet standing baseline window.
    quiet_bw_mean_n : float
        Mean vertical ground reaction force during quiet standing (N).
        Formula: BW = (1 / N_quiet) * sum_{i=1}^{N_quiet} F(t_i)
    quiet_bw_std_n : float
        Standard deviation of vertical ground reaction force during quiet standing (N).
        Formula: s_BW = sqrt( (1 / (N_quiet - 1)) * sum_{i=1}^{N_quiet} (F(t_i) - BW)^2 )
    body_mass_kg : float
        Calculated body mass (kg).
        Formula: m = BW / g, where g = 9.80665 m/s^2
    onset_idx : int
        Sample index of detected movement onset.
    onset_time_s : float
        Timestamp of detected movement onset (s).
        Criterion: Earliest instant where |F(t) - BW| > k * s_BW prior to the unweighting trough.
    int_start_idx : int
        Sample index of integration start.
    int_start_time_s : float
        Timestamp of integration start (s).
        Formula: t_int_start = max(0, t_onset - delta_t_lead)
    unweighting_min_idx : int
        Sample index of minimum force during countermovement dip (t_unweight_min).
        Formula: F_min = min_{t in [t_onset, t_bw_cross]} F(t)
    unweighting_min_time_s : float
        Timestamp of unweighting minimum force (s).
    unweighting_min_force_n : float
        Magnitude of minimum force during unweighting (N).
    bw_cross_idx : int
        Sample index where force crosses back upward through body weight (BW).
    bw_cross_time_s : float
        Timestamp of upward body weight crossing (s).
    v0_idx : int
        Sample index of zero vertical velocity (lowest center-of-mass position).
        Transition from eccentric braking to concentric propulsion.
        Formula: v(t_v0) = 0, where v(t) = int_{t_int_start}^t ((F(tau) - BW)/m) dtau
    v0_time_s : float
        Timestamp of zero vertical velocity (s).
    propulsion_peak_idx : int
        Sample index of peak vertical force during concentric propulsion (prior to takeoff).
        Formula: F_peak_prop = max_{t in [t_v0, t_takeoff]} F(t)
    propulsion_peak_time_s : float
        Timestamp of peak propulsion force (s).
    propulsion_peak_force_n : float
        Peak propulsion force magnitude (N).
    takeoff_idx : int | None
        Sample index of takeoff, or None if takeoff detection failed.
        Criterion: First sample after propulsion peak where force drops below F_thresh
        for N consecutive samples.
    takeoff_time_s : float | None
        Timestamp of takeoff (s), or None if takeoff detection failed.
    landing_idx : int | None
        Sample index of landing impact, or None if no landing occurred.
        Criterion: First sample after takeoff where force rises above F_thresh
        for N consecutive samples.
    landing_time_s : float | None
        Timestamp of landing impact (s).
    flight_time_s : float | None
        Flight duration (s).
        Formula: t_flight = t_landing - t_takeoff
    landing_peak_idx : int | None
        Sample index of maximum landing impact force.
        Formula: F_peak_land = max_{t >= t_landing} F(t)
    landing_peak_time_s : float | None
        Timestamp of peak landing impact force (s).
    landing_peak_force_n : float | None
        Peak landing impact force magnitude (N).
    flight_offset_n : float
        Estimated residual force during flight phase (N).
        Formula: F_offset = mean_{t in [t_TO + delta, t_land - delta]} F(t)
    flight_offset_applied : bool
        Whether flight-phase force-offset correction was applied.
    flight_offset_model : str
        The offset model used: 'whole_trace' or 'flight_only'.
    qa_warnings : list[str]
        List of detection warning flags.
    """
    quiet_start_idx: int
    quiet_end_idx: int
    quiet_bw_mean_n: float
    quiet_bw_std_n: float
    body_mass_kg: float
    onset_idx: int
    onset_time_s: float
    int_start_idx: int
    int_start_time_s: float
    unweighting_min_idx: int
    unweighting_min_time_s: float
    unweighting_min_force_n: float
    bw_cross_idx: int
    bw_cross_time_s: float
    v0_idx: int
    v0_time_s: float
    propulsion_peak_idx: int
    propulsion_peak_time_s: float
    propulsion_peak_force_n: float
    takeoff_idx: int | None
    takeoff_time_s: float | None
    landing_idx: int | None
    landing_time_s: float | None
    flight_time_s: float | None
    landing_peak_idx: int | None
    landing_peak_time_s: float | None
    landing_peak_force_n: float | None
    flight_offset_n: float
    flight_offset_applied: bool
    flight_offset_model: str
    qa_warnings: list[str]


def estimate_flight_phase_offset(
    time_s: np.ndarray,
    force_n: np.ndarray,
    prelim_takeoff_idx: int,
    prelim_landing_idx: int | None,
    trim_pct: float = 0.20,
) -> float:
    """Estimate residual force-plate offset during the airborne flight phase.

    During true ballistic flight, vertical ground reaction force is physically zero:
        F_true(t) = 0 N for t in (t_takeoff, t_landing)
    Any measured force during this interval represents residual instrumentation tare offset:
        F_offset = (1 / N_trim) * sum_{i in flight_core} F(t_i)

    Parameters
    ----------
    time_s : np.ndarray
        Array of timestamps (s).
    force_n : np.ndarray
        Array of vertical force values (N).
    prelim_takeoff_idx : int
        Preliminary sample index of takeoff.
    prelim_landing_idx : int | None
        Preliminary sample index of landing impact.
    trim_pct : float
        Fraction of flight interval to trim from each edge to avoid impact/release transients.

    Returns
    -------
    float
        Estimated flight-phase force offset (N).
    """
    if prelim_landing_idx is None or prelim_landing_idx <= (prelim_takeoff_idx + 10):
        # Fallback: estimate from lowest 5% of samples after takeoff
        post_to_forces = force_n[prelim_takeoff_idx:]
        if len(post_to_forces) > 10:
            k = max(5, int(len(post_to_forces) * 0.1))
            return float(np.mean(np.sort(post_to_forces)[:k]))
        return 0.0

    flight_len = prelim_landing_idx - prelim_takeoff_idx
    trim = max(2, int(flight_len * trim_pct))
    core_start = prelim_takeoff_idx + trim
    core_end = prelim_landing_idx - trim

    if core_start >= core_end:
        core_start = prelim_takeoff_idx + 1
        core_end = prelim_landing_idx - 1

    flight_forces = force_n[core_start:core_end]
    if len(flight_forces) == 0:
        return 0.0
    return float(np.mean(flight_forces))


def detect_cmj_events(
    time_s: np.ndarray,
    force_n: np.ndarray,
    fs_hz: float,
    config: PipelineConfig | None = None,
    body_mass_kg: float | None = None,
    force_raw_n: np.ndarray | None = None,
) -> CMJEvents:
    """Detect all CMJ phase transitions and discrete landmarks per McMahon et al. (2018).

    Formulas and Step-by-Step Processing:
    ------------------------------------
    1. Baseline Quiet Standing Window:
       Window: [t_0 + quiet_start_s, t_0 + quiet_window_s]
       Mean BW: BW = (1 / N_q) * sum_{i=1}^{N_q} F(t_i)
       SD: s_BW = sqrt((1 / (N_q - 1)) * sum (F(t_i) - BW)^2)
       Mass: m = BW / g (if body_mass_kg is not provided)

    2. Movement Onset:
       Threshold: F_thresh_onset = BW - k * s_BW, where k = onset_sd_multiplier (default 5.0).
       Searched backward from the unweighting dip to find the first sample crossing below threshold.

    3. Integration Start:
       t_int_start = max(0, t_onset - delta_t_lead), where delta_t_lead = integration_lead_s (default 30 ms).

    4. Unweighting Minimum (t_unweight_min):
       F_min = min_{t in [t_onset, t_bw_cross]} F(t)
       Strictly bounded between onset and the first upward body weight crossing (or propulsion peak).

    5. Velocity & Braking/Propulsion Transition (t_v0):
       a(t) = (F(t) - BW) / m
       v(t) = int_{t_int_start}^t a(tau) dtau (trapezoidal integration)
       t_v0 is the instant after t_unweight_min where v(t) crosses from negative to positive (v(t) >= 0).
       Phases:
       - Unweighting: [t_onset, t_unweight_min]
       - Braking (eccentric): [t_unweight_min, t_v0]
       - Propulsion (concentric): [t_v0, t_takeoff]

    6. Takeoff (t_takeoff):
       Searched strictly AFTER the concentric propulsion peak (max force before takeoff).
       Condition: F(t) <= F_takeoff_threshold for N consecutive samples, where
       N = max(1, round(fs * takeoff_consecutive_ms / 1000)).

    7. Landing (t_landing):
       Condition: F(t) >= F_landing_threshold for N consecutive samples after takeoff.
       Landing Peak: F_peak_landing = max_{t >= t_landing} F(t).

    8. Flight-Phase Force Offset:
       Estimated from mid-flight. If correct_flight_phase_offset is True:
       - 'whole_trace': F_corr(t) = F(t) - F_offset. Recomputes BW and m.
       - 'flight_only': adjusts threshold relative to flight floor.

    Parameters
    ----------
    time_s : np.ndarray
        Monotonic time array (s).
    force_n : np.ndarray
        Filtered vertical ground reaction force array (N).
    fs_hz : float
        Sampling frequency (Hz).
    config : PipelineConfig | None
        Configuration container. If None, default PipelineConfig is used.
    body_mass_kg : float | None
        Known athlete body mass in kg. If None, estimated from quiet baseline.

    Returns
    -------
    CMJEvents
        Dataclass containing all detected landmarks, indices, timestamps, and QA warnings.
    """
    if config is None:
        config = PipelineConfig.load()

    qa_warnings: list[str] = []
    thresh = config.thresholds

    n_samples = len(force_n)
    if n_samples < 50:
        raise ValueError(f"Insufficient samples for CMJ analysis: {n_samples}")

    # 1. Adaptive Baseline Quiet Standing Assessment
    q_start_s = thresh.quiet_start_s if thresh.quiet_start_s < (thresh.quiet_window_s * 0.5) else 0.0
    from p1_nmt_pipeline.validator import find_adaptive_quiet_window

    (
        q_start_idx,
        q_end_idx,
        quiet_bw,
        quiet_sd,
        trend_slope,
        quiet_duration_s,
    ) = find_adaptive_quiet_window(
        time_arr=time_s,
        force_arr=force_n,
        fs_hz=fs_hz,
        quiet_start_s=q_start_s,
        max_quiet_window_s=thresh.quiet_window_s,
        max_quiet_sd_n=thresh.max_quiet_sd_n,
        max_quiet_trend_n_per_s=thresh.max_quiet_trend_n_per_s,
    )

    # Flag trials with under 0.5 s of quiet
    if quiet_duration_s < 0.50 - 1e-3:
        qa_warnings.append(f"quiet_window_short_{quiet_duration_s:.2f}s")
        qa_warnings.append("quiet_window_under_0.5s")
        logger.warning(
            "Quiet standing window is under 0.5 s (duration = %.2f s < 0.50 s)",
            quiet_duration_s,
        )

    # Quiet window stability QA checks
    if quiet_sd > thresh.max_quiet_sd_n:
        qa_warnings.append(f"quiet_window_noisy_sd_{quiet_sd:.1f}N")

    if abs(trend_slope) > thresh.max_quiet_trend_n_per_s:
        qa_warnings.append(f"quiet_window_drifting_trend_{trend_slope:.1f}N_s")

    if body_mass_kg is None or body_mass_kg <= 0.0:
        body_mass = quiet_bw / STANDARD_G
    else:
        body_mass = body_mass_kg

    # 2. Preliminary Takeoff & Landing Search (to allow flight offset estimation)
    # Scaled consecutive samples: N = max(1, round(fs * ms / 1000))
    to_consec = max(1, int(round(fs_hz * (thresh.takeoff_consecutive_ms / 1000.0))))
    land_consec = max(1, int(round(fs_hz * (thresh.landing_consecutive_ms / 1000.0))))
    # Minimum flight phase duration prevents filter edge transients and post-takeoff plate ringing from tripping landing
    min_flight_samples = max(to_consec * 2, int(round(fs_hz * thresh.min_flight_duration_s)))

    # Use raw force if available for initial thresholding to avoid filter ringing
    init_detect_force = force_raw_n if force_raw_n is not None else force_n

    # Drop threshold: during flight, force drops to near zero (< 5% BW or < to_thresh)
    prelim_drop_thresh = max(thresh.takeoff_threshold_n, 0.05 * quiet_bw)

    prelim_to_idx = None
    for i in range(q_end_idx, n_samples - to_consec):
        if np.all(init_detect_force[i : i + to_consec] < prelim_drop_thresh):
            if np.max(init_detect_force[q_end_idx:i]) > quiet_bw * 1.05:
                prelim_to_idx = i
                break

    # Preliminary landing: search after minimum flight duration
    prelim_land_idx = None
    if prelim_to_idx is not None:
        for j in range(prelim_to_idx + min_flight_samples, n_samples - land_consec):
            if np.all(init_detect_force[j : j + land_consec] >= prelim_drop_thresh):
                prelim_land_idx = j
                break

    # 3. Flight Phase Offset Handling
    flight_offset_n = 0.0
    if prelim_to_idx is not None:
        flight_offset_n = estimate_flight_phase_offset(
            time_s, init_detect_force, prelim_to_idx, prelim_land_idx
        )

    working_force = force_n.copy()
    working_bw = quiet_bw
    working_mass = body_mass
    offset_applied = False
    to_thresh = thresh.takeoff_threshold_n
    land_thresh = thresh.landing_threshold_n

    if thresh.correct_flight_phase_offset and abs(flight_offset_n) > 0.05:
        offset_applied = True
        if thresh.flight_offset_model == "whole_trace":
            # Model A: Instrumentation/tare shift across whole trace
            working_force = force_n - flight_offset_n
            working_bw = float(np.mean(working_force[q_start_idx : q_end_idx + 1]))
            working_mass = working_bw / STANDARD_G
            logger.info(
                "Applied whole-trace flight offset correction: offset=%.2f N, adjusted BW=%.2f N",
                flight_offset_n,
                working_bw,
            )
        elif thresh.flight_offset_model == "flight_only":
            # Model B: Adjust takeoff/landing detection threshold relative to flight floor
            to_thresh = max(1.0, thresh.takeoff_threshold_n + flight_offset_n)
            land_thresh = max(1.0, thresh.landing_threshold_n + flight_offset_n)
            logger.info(
                "Applied flight-only offset adjustment: offset=%.2f N, adjusted to_thresh=%.2f N, land_thresh=%.2f N",
                flight_offset_n,
                to_thresh,
                land_thresh,
            )

    # Item B: Robust landing threshold based on flight-phase noise level.
    # If flight-phase samples are noisy, a fixed landing_threshold_n may
    # trigger prematurely. We compute the flight-floor noise SD and raise
    # the threshold to: max(landing_threshold_n, flight_floor_mean + k * flight_floor_sd)
    # where k = thresh.landing_noise_k (default 3.0).
    if prelim_to_idx is not None:
        # Collect mid-flight samples (trimmed to avoid takeoff/landing transients)
        if prelim_land_idx is not None and (prelim_land_idx - prelim_to_idx) > 10:
            _flight_trim = max(2, int((prelim_land_idx - prelim_to_idx) * 0.15))
            _fc = prelim_to_idx + _flight_trim
            _fe = prelim_land_idx - _flight_trim
        else:
            _fc = prelim_to_idx + 1
            _fe = min(prelim_to_idx + max(5, int(fs_hz * 0.08)), n_samples)

        _flight_sig = init_detect_force[_fc : _fe]
        if len(_flight_sig) >= 4:
            _flight_mean = float(np.mean(_flight_sig))
            _flight_sd = float(np.std(_flight_sig))
            _robust_land = _flight_mean + thresh.landing_noise_k * _flight_sd
            land_thresh = max(land_thresh, _robust_land)
            logger.debug(
                "Robust landing threshold: floor_mean=%.2f N, floor_sd=%.2f N, k=%.1f, robust=%.2f N",
                _flight_mean, _flight_sd, thresh.landing_noise_k, _robust_land,
            )

    # 4. Movement Onset Detection
    # Threshold: F_thresh_onset = BW - k * s_BW
    onset_delta = thresh.onset_sd_multiplier * max(quiet_sd, 1.0)
    onset_thresh_low = working_bw - onset_delta

    # Find the unweighting dip region: search between quiet window end and the first peak
    above_bw_indices = np.where(
        (np.arange(n_samples) > q_end_idx) & (working_force > working_bw * 1.05)
    )[0]
    if len(above_bw_indices) > 0:
        first_above_bw = above_bw_indices[0]
    else:
        first_above_bw = n_samples // 2

    # Unweighting minimum search strictly bounded between quiet window and first_above_bw
    unweight_search_end = min(first_above_bw, n_samples - 1)
    unweight_min_idx = int(q_end_idx + np.argmin(working_force[q_end_idx:unweight_search_end]))
    unweight_min_force = float(working_force[unweight_min_idx])
    unweight_min_time = float(time_s[unweight_min_idx])

    # Search backward from unweighting minimum to find movement onset
    onset_idx = q_start_idx
    for k in range(unweight_min_idx, q_start_idx - 1, -1):
        if working_force[k] >= onset_thresh_low:
            onset_idx = k
            break

    onset_time = float(time_s[onset_idx])

    # 5. Integration Start
    lead_s = thresh.integration_lead_s
    int_start_time = max(float(time_s[0]), onset_time - lead_s)
    int_start_idx = int(np.searchsorted(time_s, int_start_time))
    int_start_idx = max(0, min(int_start_idx, onset_idx))
    int_start_time = float(time_s[int_start_idx])

    # 6. Upward BW Crossing (t_bw_cross)
    cross_indices = np.where(
        (np.arange(n_samples) >= unweight_min_idx) & (working_force >= working_bw)
    )[0]
    if len(cross_indices) > 0:
        bw_cross_idx = int(cross_indices[0])
    else:
        bw_cross_idx = unweight_min_idx + 1
    bw_cross_time = float(time_s[bw_cross_idx])

    # 7. Velocity Profile & Concentric Propulsion Start (t_v0)
    accel = (working_force - working_bw) / working_mass
    vel = np.zeros(n_samples, dtype=float)
    dt_arr = np.diff(time_s)

    for i in range(int_start_idx, n_samples - 1):
        vel[i + 1] = vel[i] + 0.5 * (accel[i] + accel[i + 1]) * dt_arr[i]

    # Concentric start (t_v0): zero velocity point after unweighting minimum
    v0_candidates = np.where(
        (np.arange(n_samples) >= unweight_min_idx) & (vel >= 0.0)
    )[0]
    if len(v0_candidates) > 0:
        v0_idx = int(v0_candidates[0])
    else:
        v0_idx = bw_cross_idx
    v0_time = float(time_s[v0_idx])

    # 8. Concentric Propulsion Peak Force (strictly before takeoff)
    search_takeoff_limit = prelim_to_idx if prelim_to_idx is not None else (n_samples - 10)
    if search_takeoff_limit <= v0_idx:
        search_takeoff_limit = n_samples - 5

    prop_region = working_force[v0_idx:search_takeoff_limit]
    if len(prop_region) > 0:
        prop_peak_offset = int(np.argmax(prop_region))
        prop_peak_idx = v0_idx + prop_peak_offset
    else:
        prop_peak_idx = v0_idx

    prop_peak_force = float(working_force[prop_peak_idx])
    prop_peak_time = float(time_s[prop_peak_idx])

    # Select force series for threshold event detection:
    # High-frequency/raw force preserves steep takeoff/landing transitions without filter rounding
    if force_raw_n is not None:
        detect_force = (
            (force_raw_n - flight_offset_n)
            if (offset_applied and thresh.flight_offset_model == "whole_trace")
            else force_raw_n
        )
    else:
        detect_force = working_force

    # 9. Definitive Takeoff Detection (strictly AFTER propulsion peak)
    takeoff_idx = None
    for i in range(prop_peak_idx, n_samples - to_consec):
        if np.all(detect_force[i : i + to_consec] <= to_thresh):
            takeoff_idx = i
            break

    if takeoff_idx is None:
        qa_warnings.append("takeoff_detection_failed")
        takeoff_time = None
        landing_idx = None
        landing_time = None
        flight_time = None
    else:
        takeoff_time = float(time_s[takeoff_idx])
        if thresh.subsample_takeoff and takeoff_idx > 1:
            F_floor = flight_offset_n
            f_curr = float(detect_force[takeoff_idx])
            t_curr = float(time_s[takeoff_idx])
            t_prev = float(time_s[takeoff_idx - 1])
            dt = t_curr - t_prev
            if f_curr <= F_floor:
                f_prev = float(detect_force[takeoff_idx - 1])
                f_prev2 = float(detect_force[takeoff_idx - 2])
                dF_dt = (f_prev - f_prev2) / dt if dt > 0 else -1.0
                if dF_dt < 0:
                    dt_tail = (F_floor - f_prev) / dF_dt
                    dt_tail = max(0.0, min(dt_tail, dt))
                    takeoff_time = t_prev + dt_tail
            else:
                f_prev = float(detect_force[takeoff_idx - 1])
                dF_dt = (f_curr - f_prev) / dt if dt > 0 else -1.0
                if dF_dt < 0:
                    dt_tail = (F_floor - f_curr) / dF_dt
                    dt_tail = max(0.0, min(dt_tail, dt))
                    takeoff_time = t_curr + dt_tail

        # 10. Definitive Landing Detection (after takeoff and minimum flight duration)
        landing_idx = None
        for j in range(takeoff_idx + min_flight_samples, n_samples - land_consec):
            if np.all(detect_force[j : j + land_consec] >= land_thresh):
                landing_idx = j
                break

        if landing_idx is None:
            qa_warnings.append("landing_detection_failed")
            landing_time = None
            flight_time = None
        else:
            landing_time = float(time_s[landing_idx])
            flight_time = (landing_time - takeoff_time)


    # 11. Landing Peak Force (strictly AFTER landing)
    # IMPORTANT: landing peak uses detect_force (raw/unfiltered signal) because
    # the landing impact is a fast transient (~5 ms rise time) that the 30 Hz
    # Butterworth filter severely attenuates (up to 16% error). Per Chavda et al.
    # (2018), peak landing force is measured from the unfiltered signal.
    # Peak propulsion force (a slower transient) uses the filtered signal as usual.
    landing_peak_idx = None
    landing_peak_time = None
    landing_peak_force = None

    if landing_idx is not None and landing_idx < n_samples - 1:
        # Use raw force for landing peak; fall back to working_force if raw unavailable
        land_force_series = detect_force if force_raw_n is not None else working_force
        post_land_region = land_force_series[landing_idx:]
        if len(post_land_region) > 0:
            rel_peak = int(np.argmax(post_land_region))
            landing_peak_idx = landing_idx + rel_peak
            landing_peak_time = float(time_s[landing_peak_idx])
            landing_peak_force = float(land_force_series[landing_peak_idx])

    return CMJEvents(
        quiet_start_idx=q_start_idx,
        quiet_end_idx=q_end_idx,
        quiet_bw_mean_n=working_bw,
        quiet_bw_std_n=quiet_sd,
        body_mass_kg=working_mass,
        onset_idx=onset_idx,
        onset_time_s=onset_time,
        int_start_idx=int_start_idx,
        int_start_time_s=int_start_time,
        unweighting_min_idx=unweight_min_idx,
        unweighting_min_time_s=unweight_min_time,
        unweighting_min_force_n=unweight_min_force,
        bw_cross_idx=bw_cross_idx,
        bw_cross_time_s=bw_cross_time,
        v0_idx=v0_idx,
        v0_time_s=v0_time,
        propulsion_peak_idx=prop_peak_idx,
        propulsion_peak_time_s=prop_peak_time,
        propulsion_peak_force_n=prop_peak_force,
        takeoff_idx=takeoff_idx,
        takeoff_time_s=takeoff_time,
        landing_idx=landing_idx,
        landing_time_s=landing_time,
        flight_time_s=flight_time,
        landing_peak_idx=landing_peak_idx,
        landing_peak_time_s=landing_peak_time,
        landing_peak_force_n=landing_peak_force,
        flight_offset_n=flight_offset_n,
        flight_offset_applied=offset_applied,
        flight_offset_model=thresh.flight_offset_model,
        qa_warnings=qa_warnings,
    )
