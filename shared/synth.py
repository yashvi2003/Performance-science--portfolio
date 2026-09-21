"""Synthetic Data Generator for Force-Plate Trials and Longitudinal Monitoring.

Governed by specs/00_shared_contract.md, specs/01_neuromuscular_pipeline.md, and PLAN.md.
Provides physically consistent force-time curves with analytical ground truth for
Countermovement Jump (CMJ), Squat Jump (SJ), Drop Jump (DJ), and Isometric Mid-Thigh Pull (IMTP),
as well as contract-compliant sessions.csv and wearable_daily.csv tables.

All outputs are SYNTHETIC DATA. Do not interpret them as research findings.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

from shared.contract import (
    LimbSide,
    METRIC_REGISTRY,
    STANDARD_G,
    SESSIONS_COLUMNS,
    TestType,
    WEARABLE_DAILY_COLUMNS,
)

# ---------------------------------------------------------------------------
# ForceTrialData container
# ---------------------------------------------------------------------------

@dataclass
class ForceTrialData:
    """Represents a simulated force-plate trial with analytical ground truth."""
    time_s: np.ndarray
    force_n: np.ndarray
    force_left_n: np.ndarray | None
    force_right_n: np.ndarray | None
    fs_hz: float
    test: TestType
    body_mass_kg: float
    body_weight_n: float
    true_metrics: dict[str, float]
    event_indices: dict[str, int]
    is_synthetic: bool = True

    def to_df(self) -> pd.DataFrame:
        """Export raw trace as a pandas DataFrame."""
        data = {
            "time_s": self.time_s,
            "force_n": self.force_n,
        }
        if self.force_left_n is not None:
            data["force_left_n"] = self.force_left_n
        if self.force_right_n is not None:
            data["force_right_n"] = self.force_right_n
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_asymmetry_split(
    force_n: np.ndarray,
    asymmetry_index: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split total force into left/right limbs according to asymmetry index (ASI).

    Formula
    -------
    ASI = (F_R - F_L) / max(F_R, F_L)
    If ASI > 0 (right dominant):  frac_R = 1 / (2 - ASI), frac_L = 1 - frac_R
    If ASI < 0 (left dominant):   frac_L = 1 / (2 - |ASI|), frac_R = 1 - frac_L
    If ASI = 0:                   frac_L = frac_R = 0.5
    """
    if abs(asymmetry_index) > 1e-6:
        if asymmetry_index > 0:  # Right dominant
            frac_r = 1.0 / (2.0 - asymmetry_index)
            frac_l = 1.0 - frac_r
        else:  # Left dominant
            frac_l = 1.0 / (2.0 - abs(asymmetry_index))
            frac_r = 1.0 - frac_l
        f_right = force_n * frac_r
        f_left = force_n * frac_l
    else:
        f_left = force_n * 0.5
        f_right = force_n * 0.5
    return f_left, f_right


def _build_cmj_acceleration(
    time_s: np.ndarray,
    t_onset: float,
    t_unweight_end: float,
    t_braking_end: float,
    t_takeoff: float,
    t_landing: float,
    t_unweight_s: float,
    t_braking_s: float,
    t_propulsion_s: float,
    v_takeoff_analytical: float,
    A_unw: float,
    A_brake: float,
    A_prop: float,
    t_landing_duration: float,
    v_impact: float,
    body_mass_kg: float,
    bw_n: float,
    template: str,
) -> np.ndarray:
    """Build the piecewise acceleration array for a CMJ trial.

    The base model (single_peak) uses sinusoidal segments:
    - Unweighting: a(t) = -A_unw * sin(π*τ)              (downward, dip)
    - Braking:     a(t) = +A_brake * sin(π*τ)             (upward, deceleration)
    - Propulsion:  a(t) = A_prop*sin(π*τ) - g*sin(π*τ/2) (upward, propulsion + gravity)
    - Flight:      a(t) = -g                              (free fall)
    - Landing:     half-sine decelerating impact with impulse = m * v_impact

    Templates modify the propulsion/braking segments while maintaining
    the same v_takeoff (closed-form ground truth unchanged).

    Parameters
    ----------
    template : str
        One of: 'single_peak', 'double_peak', 'deep_unweight', 'high_braking'.
    """
    n_pts = len(time_s)
    acc = np.zeros(n_pts)

    # Landing: half-sine impulse model
    # Net impulse over [t_landing, t_landing + t_landing_duration] = m * v_impact
    # int(A_land * sin(pi*tau)) dt from 0..t_ld = A_land * 2*t_ld/pi = v_impact
    # -> A_land = v_impact * pi / (2 * t_ld)
    A_land = (v_impact * np.pi) / (2.0 * t_landing_duration)

    for i, t in enumerate(time_s):
        if t < t_onset:
            acc[i] = 0.0
        elif t < t_unweight_end:
            tau = (t - t_onset) / t_unweight_s
            if template == "deep_unweight":
                # Deeper dip: amplitude 1.6x, same closed-form impulse
                acc[i] = -1.6 * A_unw * np.sin(np.pi * tau)
            else:
                acc[i] = -A_unw * np.sin(np.pi * tau)
        elif t < t_braking_end:
            tau = (t - t_unweight_end) / t_braking_s
            if template in ("high_braking",):
                # Braking force well above BW at v=0 transition
                # Double the amplitude but half the time for the first half, normal second half
                if tau < 0.5:
                    acc[i] = 1.8 * A_brake * np.sin(np.pi * tau * 2.0)
                else:
                    acc[i] = A_brake * np.sin(np.pi * tau)
            else:
                acc[i] = A_brake * np.sin(np.pi * tau)
        elif t < t_takeoff:
            tau = (t - t_braking_end) / t_propulsion_s
            base_acc = A_prop * np.sin(np.pi * tau) - STANDARD_G * np.sin(0.5 * np.pi * tau)
            if template == "double_peak":
                # Add secondary propulsion micro-peak (plateau then second push)
                # Small extra sine burst in the second half of propulsion, scaled to keep impulse identical
                if tau > 0.5:
                    tau2 = (tau - 0.5) / 0.5
                    # Perturbation integrates to 0 (positive + negative equal area)
                    acc[i] = base_acc + 0.15 * A_prop * np.sin(2.0 * np.pi * tau2)
                else:
                    acc[i] = base_acc
            else:
                acc[i] = base_acc
        elif t < t_landing:
            acc[i] = -STANDARD_G
        else:
            tau_land = t - t_landing
            if tau_land < t_landing_duration:
                # Net acceleration half-sine: F_net = m * A_land * sin(pi*tau/T)
                # Integral over [0,T] = A_land * 2T/pi = v_impact (momentum-consistent)
                # Force = BW + m * A_land * sin(...) → acc (in our convention) = A_land * sin(...)
                acc[i] = A_land * np.sin(np.pi * tau_land / t_landing_duration)
            else:
                acc[i] = 0.0

    return acc


# ---------------------------------------------------------------------------
# CMJ generator
# ---------------------------------------------------------------------------

CMJTemplate = Literal["single_peak", "double_peak", "deep_unweight", "high_braking"]


def make_cmj_trial(
    body_mass_kg: float = 75.0,
    target_jump_height_m: float = 0.35,
    fs_hz: float = 2000.0,
    t_quiet_s: float = 0.5,
    t_unweight_s: float = 0.25,
    t_braking_s: float = 0.20,
    t_propulsion_s: float = 0.25,
    asymmetry_index: float = 0.0,
    noise_sd_n: float = 0.0,
    noise_flight_sd_n: float = 0.0,
    flight_force_offset_n: float = 0.0,
    offset_whole_trace: bool = False,
    random_seed: int | None = 42,
    t_onset: float | None = None,
    seed: int | None = None,
    jitter_s: float = 0.0,
    onset_ramp_s: float = 0.0,
    t_landing_duration: float = 0.25,
    template: CMJTemplate = "single_peak",
) -> ForceTrialData:
    """Generate a physically consistent Countermovement Jump (CMJ) force-time trial.

    This generator produces SYNTHETIC DATA. It demonstrates that the pipeline works
    correctly on known ground truth. It is NOT a research finding.

    Parameters
    ----------
    body_mass_kg : float
        Athlete body mass (kg).
    target_jump_height_m : float
        Prescribed jump height (m). Determines takeoff velocity analytically:
        v_to = sqrt(2 * g * h). Ground truth stored in true_metrics.
    fs_hz : float
        Sampling frequency (Hz).
    t_quiet_s : float
        Duration of quiet standing baseline before movement onset (s).
    t_unweight_s : float
        Duration of unweighting (countermovement dip) phase (s).
    t_braking_s : float
        Duration of eccentric braking phase (s).
    t_propulsion_s : float
        Duration of concentric propulsion phase (s).
    asymmetry_index : float
        Asymmetry index (ASI) for left/right limb split. ASI = (F_R - F_L) / max(F_R, F_L).
    noise_sd_n : float
        Gaussian noise standard deviation (N) applied to all non-flight samples.
    noise_flight_sd_n : float
        Gaussian noise standard deviation (N) applied to flight-phase samples only.
        Set to 0 to keep flight phase clean (default). Separate from noise_sd_n to
        avoid creating artificial zero-force notches when noise_sd_n > 0 and offset = 0.
    flight_force_offset_n : float
        Residual force offset (N) applied during the flight phase (or whole trace if
        offset_whole_trace=True). Models instrumentation tare drift.
    offset_whole_trace : bool
        If True, offset is applied to the entire trace (constant tare shift).
        If False (default), offset is applied only during the flight phase.
    random_seed : int | None
        NumPy random seed for noise reproducibility.
    t_onset : float | None
        Alias for t_quiet_s (backward compatibility). If provided, overrides t_quiet_s.
    seed : int | None
        Alias for random_seed (backward compatibility). If provided, overrides random_seed.
    jitter_s : float
        Maximum uniform jitter (s) added to each of the four event durations
        (t_quiet_s, t_unweight_s, t_braking_s, t_propulsion_s). Default 0 = deterministic.
        Each duration is perturbed by Uniform(-jitter_s, +jitter_s).
    onset_ramp_s : float
        Duration of a smooth cosine onset ramp (s). When > 0, the force transition at
        t_onset is ramped from BW to the unweighting curve over this window.
        Default 0 = step onset (backward compatible).
    t_landing_duration : float
        Duration of the landing impact half-sine pulse (s). Default 0.25 s.
        The landing impulse satisfies: integral(F_net) = m * v_impact.
    template : CMJTemplate
        Force-curve template to use. Options:
        - 'single_peak': standard single propulsion peak (default, backward compatible)
        - 'double_peak': secondary propulsion micro-peak in second half of propulsion phase
        - 'deep_unweight': deeper countermovement dip (amplitude 1.6x)
        - 'high_braking': braking force well above BW at v=0 transition

    Returns
    -------
    ForceTrialData
        Trial with true_metrics containing ground-truth jump height, takeoff time,
        landing time, flight time, contraction time, and RSImod.
    """
    # --- Backward-compatible aliases ---
    if seed is not None:
        random_seed = seed
    if t_onset is not None:
        t_quiet_s = t_onset

    rng = np.random.default_rng(random_seed)
    bw_n = body_mass_kg * STANDARD_G

    # --- Optional event-time jitter (item C) ---
    if jitter_s > 0.0:
        t_quiet_s = max(0.2, t_quiet_s + rng.uniform(-jitter_s, jitter_s))
        t_unweight_s = max(0.08, t_unweight_s + rng.uniform(-jitter_s, jitter_s))
        t_braking_s = max(0.06, t_braking_s + rng.uniform(-jitter_s, jitter_s))
        t_propulsion_s = max(0.08, t_propulsion_s + rng.uniform(-jitter_s, jitter_s))

    # --- Analytical ground truth ---
    v_takeoff_analytical = np.sqrt(2.0 * STANDARD_G * target_jump_height_m)
    t_flight_analytical = 2.0 * v_takeoff_analytical / STANDARD_G
    h_flight_analytical = (STANDARD_G * (t_flight_analytical ** 2)) / 8.0
    # Landing impact velocity (conservation of energy, symmetric)
    v_impact = v_takeoff_analytical

    # --- Time markers ---
    t_onset_abs = t_quiet_s
    t_unweight_end = t_onset_abs + t_unweight_s
    t_braking_end = t_unweight_end + t_braking_s
    t_takeoff = t_braking_end + t_propulsion_s
    t_landing = t_takeoff + t_flight_analytical
    t_total = t_landing + t_landing_duration + 0.35

    dt = 1.0 / fs_hz
    time_s = np.arange(0.0, t_total, dt)
    n_pts = len(time_s)

    # --- Acceleration amplitudes ---
    A_unw = 4.0  # unweighting amplitude (m/s^2)
    A_brake = A_unw * (t_unweight_s / t_braking_s)
    # Propulsion amplitude: integral gives exactly v_takeoff_analytical
    # int_0^T [A_prop * sin(pi*tau) - g * sin(pi*tau/2)] dtau = A_prop * 2*T/pi - g * 2*T/pi * 2
    # Setting = v_to:  A_prop = (v_to * pi)/(2*T) + g
    A_prop = (v_takeoff_analytical * np.pi) / (2.0 * t_propulsion_s) + STANDARD_G

    # --- Build acceleration profile ---
    acc = _build_cmj_acceleration(
        time_s=time_s,
        t_onset=t_onset_abs,
        t_unweight_end=t_unweight_end,
        t_braking_end=t_braking_end,
        t_takeoff=t_takeoff,
        t_landing=t_landing,
        t_unweight_s=t_unweight_s,
        t_braking_s=t_braking_s,
        t_propulsion_s=t_propulsion_s,
        v_takeoff_analytical=v_takeoff_analytical,
        A_unw=A_unw,
        A_brake=A_brake,
        A_prop=A_prop,
        t_landing_duration=t_landing_duration,
        v_impact=v_impact,
        body_mass_kg=body_mass_kg,
        bw_n=bw_n,
        template=template,
    )

    # --- Convert acceleration -> force ---
    force_n = body_mass_kg * (acc + STANDARD_G)

    # --- Gradual onset ramp (item C) ---
    # Replace the step at t_onset_abs with a smooth cosine ramp from BW
    # to whatever the piecewise model prescribes, over [t_onset_abs, t_onset_abs + onset_ramp_s].
    if onset_ramp_s > 0.0:
        ramp_end = t_onset_abs + onset_ramp_s
        ramp_mask = (time_s >= t_onset_abs) & (time_s < ramp_end)
        ramp_indices = np.where(ramp_mask)[0]
        if len(ramp_indices) > 0:
            tau_ramp = (time_s[ramp_indices] - t_onset_abs) / onset_ramp_s
            # Smoothstep blend: 0 at onset, 1 at onset+ramp_s
            blend = 0.5 * (1.0 - np.cos(np.pi * tau_ramp))
            force_n[ramp_indices] = bw_n + blend * (force_n[ramp_indices] - bw_n)

    # --- Flight-phase handling (items C + F fix) ---
    # Apply flight force BEFORE adding noise, so noise is added uniformly.
    # This eliminates the ~0 N notch that appeared when noise was added first
    # then zeroed out by the flight mask.
    flight_mask = (time_s >= t_takeoff) & (time_s < t_landing)

    if offset_whole_trace:
        # Constant tare shift across entire trace
        force_n += flight_force_offset_n
    else:
        # Flight-only offset: set flight phase force to the specified offset value
        # (0.0 = true zero, realistic for a perfectly calibrated plate in flight)
        force_n[flight_mask] = flight_force_offset_n

    # --- Noise (applied AFTER flight mask, item C + F fix) ---
    # Quiet/active phase noise
    if noise_sd_n > 0.0:
        non_flight_mask = ~flight_mask
        noise_vec = np.zeros(n_pts)
        noise_vec[non_flight_mask] = rng.normal(0.0, noise_sd_n, size=int(np.sum(non_flight_mask)))
        force_n = force_n + noise_vec

    # Flight-phase noise (separate, default 0 to keep flight clean)
    if noise_flight_sd_n > 0.0:
        flight_noise = np.zeros(n_pts)
        n_flight = int(np.sum(flight_mask))
        if n_flight > 0:
            flight_noise[flight_mask] = rng.normal(0.0, noise_flight_sd_n, size=n_flight)
        force_n = force_n + flight_noise

    # --- Per-limb split ---
    f_left, f_right = _apply_asymmetry_split(force_n, asymmetry_index)

    # --- Event indices ---
    idx_onset = int(round(t_onset_abs * fs_hz))
    idx_takeoff = int(round(t_takeoff * fs_hz))
    idx_landing = int(round(t_landing * fs_hz))

    t_contraction = t_takeoff - t_onset_abs
    rsimod_analytical = target_jump_height_m / t_contraction

    true_metrics: dict[str, float] = {
        "jump_height_imp_m": float(target_jump_height_m),
        "jump_height_flight_m": float(h_flight_analytical),
        "v_takeoff_m_s": float(v_takeoff_analytical),
        "v_impact_m_s": float(v_impact),
        "t_onset_s": float(t_onset_abs),
        "t_takeoff_s": float(t_takeoff),
        "t_landing_s": float(t_landing),
        "t_flight_s": float(t_flight_analytical),
        "contraction_time_s": float(t_contraction),
        "rsimod": float(rsimod_analytical),
        "body_mass_kg": float(body_mass_kg),
        "body_weight_n": float(bw_n),
        "flight_force_offset_n": float(flight_force_offset_n),
        "t_landing_duration_s": float(t_landing_duration),
    }

    return ForceTrialData(
        time_s=time_s,
        force_n=force_n,
        force_left_n=f_left,
        force_right_n=f_right,
        fs_hz=fs_hz,
        test=TestType.CMJ,
        body_mass_kg=body_mass_kg,
        body_weight_n=bw_n,
        true_metrics=true_metrics,
        event_indices={"onset": idx_onset, "takeoff": idx_takeoff, "landing": idx_landing},
        is_synthetic=True,
    )


# ---------------------------------------------------------------------------
# SJ generator (unchanged except import path)
# ---------------------------------------------------------------------------

def make_sj_trial(
    body_mass_kg: float = 75.0,
    target_jump_height_m: float = 0.30,
    fs_hz: float = 2000.0,
    t_quiet_s: float = 0.6,
    t_propulsion_s: float = 0.30,
    inject_countermovement: bool = False,
    dip_magnitude_n: float = 120.0,
    asymmetry_index: float = 0.0,
    noise_sd_n: float = 0.0,
    random_seed: int | None = 42,
) -> ForceTrialData:
    """Generate a Squat Jump (SJ) force-time trial with known concentric ground truth.

    Unlike CMJ, pure SJ has no countermovement dip (strictly positive acceleration).
    If inject_countermovement=True, a pre-propulsion dip is injected to test QA invalidation.
    """
    rng = np.random.default_rng(random_seed)
    bw_n = body_mass_kg * STANDARD_G

    v_takeoff_analytical = np.sqrt(2.0 * STANDARD_G * target_jump_height_m)
    t_flight_analytical = 2.0 * v_takeoff_analytical / STANDARD_G
    h_flight_analytical = (STANDARD_G * (t_flight_analytical ** 2)) / 8.0

    t_onset = t_quiet_s
    t_takeoff = t_onset + t_propulsion_s
    t_landing = t_takeoff + t_flight_analytical
    t_total = t_landing + 0.6

    dt = 1.0 / fs_hz
    time_s = np.arange(0.0, t_total, dt)
    n_pts = len(time_s)

    # Pure sinusoidal propulsion: a(t) = A_prop * sin(pi * tau), int(a) = v_to
    A_prop = (v_takeoff_analytical * np.pi) / (2.0 * t_propulsion_s)
    peak_force_analytical = bw_n + body_mass_kg * A_prop

    acc = np.zeros(n_pts)
    for i, t in enumerate(time_s):
        if t < t_onset:
            acc[i] = 0.0
        elif t < t_takeoff:
            tau = (t - t_onset) / t_propulsion_s
            acc[i] = A_prop * np.sin(np.pi * tau)
        elif t < t_landing:
            acc[i] = -STANDARD_G
        else:
            tau_land = t - t_landing
            if tau_land < 0.15:
                acc[i] = 25.0 * np.sin(np.pi * tau_land / 0.15) * np.exp(-15.0 * tau_land)
            else:
                acc[i] = 0.0

    force_n = body_mass_kg * (acc + STANDARD_G)

    # If testing SJ invalidation, inject dip right before onset
    if inject_countermovement:
        dip_start_idx = max(0, int((t_onset - 0.15) * fs_hz))
        dip_end_idx = int(t_onset * fs_hz)
        dip_len = dip_end_idx - dip_start_idx
        if dip_len > 0:
            dip_curve = dip_magnitude_n * np.sin(np.linspace(0, np.pi, dip_len))
            force_n[dip_start_idx:dip_end_idx] -= dip_curve

    flight_mask = (time_s >= t_takeoff) & (time_s < t_landing)
    force_n[flight_mask] = 0.0

    if noise_sd_n > 0.0:
        force_n += rng.normal(0.0, noise_sd_n, size=n_pts)

    f_left, f_right = _apply_asymmetry_split(force_n, asymmetry_index)

    idx_onset = int(round(t_onset * fs_hz))
    idx_takeoff = int(round(t_takeoff * fs_hz))
    idx_landing = int(round(t_landing * fs_hz))

    # Analytical RFD from onset
    def _force_at(dt_s: float) -> float:
        if dt_s >= t_propulsion_s:
            return float(bw_n)
        tau = dt_s / t_propulsion_s
        return float(bw_n + body_mass_kg * A_prop * np.sin(np.pi * tau))

    rfd_50 = (_force_at(0.05) - bw_n) / 0.05
    rfd_100 = (_force_at(0.10) - bw_n) / 0.10
    rfd_200 = (_force_at(0.20) - bw_n) / 0.20
    peak_rfd = body_mass_kg * A_prop * (np.pi / t_propulsion_s)

    true_metrics: dict[str, float] = {
        "jump_height_imp_m": float(target_jump_height_m),
        "jump_height_flight_m": float(h_flight_analytical),
        "v_takeoff_m_s": float(v_takeoff_analytical),
        "peak_force_n": float(peak_force_analytical),
        "t_onset_s": float(t_onset),
        "t_takeoff_s": float(t_takeoff),
        "t_landing_s": float(t_landing),
        "rfd_0_50ms_n_s": float(rfd_50),
        "rfd_0_100ms_n_s": float(rfd_100),
        "rfd_0_200ms_n_s": float(rfd_200),
        "peak_rfd_n_s": float(peak_rfd),
        "countermovement_injected": float(1.0 if inject_countermovement else 0.0),
    }

    return ForceTrialData(
        time_s=time_s,
        force_n=force_n,
        force_left_n=f_left,
        force_right_n=f_right,
        fs_hz=fs_hz,
        test=TestType.SJ,
        body_mass_kg=body_mass_kg,
        body_weight_n=bw_n,
        true_metrics=true_metrics,
        event_indices={"onset": idx_onset, "takeoff": idx_takeoff, "landing": idx_landing},
        is_synthetic=True,
    )


# ---------------------------------------------------------------------------
# DJ generator (unchanged)
# ---------------------------------------------------------------------------

def make_dj_trial(
    body_mass_kg: float = 75.0,
    drop_height_m: float = 0.30,
    target_rebound_height_m: float = 0.32,
    gct_s: float = 0.22,
    fs_hz: float = 2000.0,
    asymmetry_index: float = 0.0,
    noise_sd_n: float = 0.0,
    random_seed: int | None = 42,
) -> ForceTrialData:
    """Generate a Drop Jump (DJ) force-time trial starting in airborne drop."""
    rng = np.random.default_rng(random_seed)
    bw_n = body_mass_kg * STANDARD_G

    t_drop = np.sqrt(2.0 * drop_height_m / STANDARD_G)
    v_impact = np.sqrt(2.0 * STANDARD_G * drop_height_m)  # downward velocity
    v_takeoff = np.sqrt(2.0 * STANDARD_G * target_rebound_height_m)
    t_flight_rebound = 2.0 * v_takeoff / STANDARD_G

    t_contact_start = t_drop
    t_takeoff = t_contact_start + gct_s
    t_landing = t_takeoff + t_flight_rebound
    t_total = t_landing + 0.5

    dt = 1.0 / fs_hz
    time_s = np.arange(0.0, t_total, dt)
    n_pts = len(time_s)

    # Impulse required over GCT: delta_v = v_takeoff - (-v_impact) = v_takeoff + v_impact
    delta_v = v_takeoff + v_impact
    # Model contact acceleration as half-sine pulse: integral = delta_v
    # int(A * sin(pi * tau)) = A * 2 * gct / pi = delta_v -> A = delta_v * pi / (2 * gct)
    A_gct = (delta_v * np.pi) / (2.0 * gct_s)

    force_n = np.zeros(n_pts)
    for i, t in enumerate(time_s):
        if t < t_contact_start:
            force_n[i] = 0.0  # Initial flight from drop
        elif t < t_takeoff:
            tau = (t - t_contact_start) / gct_s
            acc_i = A_gct * np.sin(np.pi * tau)
            force_n[i] = body_mass_kg * (acc_i + STANDARD_G)
        elif t < t_landing:
            force_n[i] = 0.0  # Rebound flight
        else:
            tau_land = t - t_landing
            if tau_land < 0.15:
                force_n[i] = bw_n + body_mass_kg * 25.0 * np.sin(np.pi * tau_land / 0.15) * np.exp(-15.0 * tau_land)
            else:
                force_n[i] = bw_n

    if noise_sd_n > 0.0:
        force_n += rng.normal(0.0, noise_sd_n, size=n_pts)

    f_left, f_right = _apply_asymmetry_split(force_n, asymmetry_index)

    rsi_analytical = target_rebound_height_m / gct_s

    true_metrics: dict[str, float] = {
        "drop_height_m": float(drop_height_m),
        "jump_height_flight_m": float(target_rebound_height_m),
        "gct_s": float(gct_s),
        "rsi": float(rsi_analytical),
        "t_impact_s": float(t_contact_start),
        "t_takeoff_s": float(t_takeoff),
        "t_landing_s": float(t_landing),
        "flight_time_s": float(t_flight_rebound),
    }

    return ForceTrialData(
        time_s=time_s,
        force_n=force_n,
        force_left_n=f_left,
        force_right_n=f_right,
        fs_hz=fs_hz,
        test=TestType.DJ,
        body_mass_kg=body_mass_kg,
        body_weight_n=bw_n,
        true_metrics=true_metrics,
        event_indices={
            "impact": int(round(t_contact_start * fs_hz)),
            "takeoff": int(round(t_takeoff * fs_hz)),
            "landing": int(round(t_landing * fs_hz)),
        },
        is_synthetic=True,
    )


# ---------------------------------------------------------------------------
# IMTP generator (unchanged)
# ---------------------------------------------------------------------------

def make_imtp_trial(
    body_mass_kg: float = 75.0,
    peak_force_net_n: float = 2400.0,
    pull_duration_s: float = 3.0,
    fs_hz: float = 2000.0,
    t_quiet_s: float = 0.8,
    t_rise_s: float = 0.15,
    inject_pretension: bool = False,
    asymmetry_index: float = 0.0,
    noise_sd_n: float = 0.0,
    random_seed: int | None = 42,
) -> ForceTrialData:
    """Generate an Isometric Mid-Thigh Pull (IMTP) force-time trial."""
    rng = np.random.default_rng(random_seed)
    bw_n = body_mass_kg * STANDARD_G
    gross_peak_force = bw_n + peak_force_net_n

    t_onset = t_quiet_s
    t_peak = t_onset + t_rise_s
    t_release = t_onset + pull_duration_s
    t_total = t_release + 0.6

    dt = 1.0 / fs_hz
    time_s = np.arange(0.0, t_total, dt)
    n_pts = len(time_s)

    force_n = np.zeros(n_pts)
    for i, t in enumerate(time_s):
        if t < t_onset:
            force_n[i] = bw_n
        elif t < t_peak:
            # S-shaped smooth rise to peak
            tau = (t - t_onset) / t_rise_s
            force_n[i] = bw_n + peak_force_net_n * (0.5 - 0.5 * np.cos(np.pi * tau))
        elif t < t_release:
            # Isometric plateau with slight organic decay (fatigue)
            tau_plat = (t - t_peak) / (t_release - t_peak)
            decay = 0.96 + 0.04 * np.cos(np.pi * tau_plat * 0.5)
            force_n[i] = bw_n + peak_force_net_n * decay
        else:
            # Release back to quiet standing
            tau_rel = t - t_release
            if tau_rel < 0.2:
                force_n[i] = bw_n + peak_force_net_n * 0.96 * np.exp(-15.0 * tau_rel)
            else:
                force_n[i] = bw_n

    if inject_pretension:
        # Pre-tension: force elevated by 80 N before command
        pre_idx_start = int((t_onset - 0.3) * fs_hz)
        pre_idx_end = int(t_onset * fs_hz)
        force_n[pre_idx_start:pre_idx_end] += 80.0

    if noise_sd_n > 0.0:
        force_n += rng.normal(0.0, noise_sd_n, size=n_pts)

    f_left, f_right = _apply_asymmetry_split(force_n, asymmetry_index)

    # Exact forces at 50, 100, 150, 200, 250 ms from onset
    def _force_at(dt_s: float) -> float:
        t_target = t_onset + dt_s
        idx = int(round(t_target * fs_hz))
        return float(f_left[min(idx, len(f_left) - 1)] + f_right[min(idx, len(f_right) - 1)])

    true_metrics: dict[str, float] = {
        "peak_force_gross_n": float(gross_peak_force),
        "peak_force_net_n": float(peak_force_net_n),
        "t_onset_s": float(t_onset),
        "t_to_peak_s": float(t_rise_s),
        "force_50ms_n": _force_at(0.05),
        "force_100ms_n": _force_at(0.10),
        "force_150ms_n": _force_at(0.15),
        "force_200ms_n": _force_at(0.20),
        "force_250ms_n": _force_at(0.25),
        "pretension_injected": float(1.0 if inject_pretension else 0.0),
    }

    return ForceTrialData(
        time_s=time_s,
        force_n=force_n,
        force_left_n=f_left,
        force_right_n=f_right,
        fs_hz=fs_hz,
        test=TestType.IMTP,
        body_mass_kg=body_mass_kg,
        body_weight_n=bw_n,
        true_metrics=true_metrics,
        event_indices={
            "onset": int(round(t_onset * fs_hz)),
            "peak": int(round(t_peak * fs_hz)),
        },
        is_synthetic=True,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_force_trial(test: TestType | str, **kwargs) -> ForceTrialData:
    """Factory creating a synthetic force trial for any valid test type."""
    test_enum = TestType(test) if isinstance(test, str) else test
    if test_enum == TestType.CMJ:
        return make_cmj_trial(**kwargs)
    elif test_enum == TestType.SJ:
        return make_sj_trial(**kwargs)
    elif test_enum == TestType.DJ:
        return make_dj_trial(**kwargs)
    elif test_enum == TestType.IMTP:
        return make_imtp_trial(**kwargs)
    raise ValueError(f"Unsupported test type: {test}")


# ---------------------------------------------------------------------------
# Contract table generators (unchanged)
# ---------------------------------------------------------------------------

def make_sessions_table(
    n_athletes: int = 5,
    n_weeks: int = 6,
    start_date: str = "2026-03-02",
    fatigue_athlete: str = "A01",
    fatigue_week: int = 4,
    fatigue_drop_pct: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a contract-compliant sessions.csv DataFrame with known fatigue trends."""
    rng = np.random.default_rng(random_seed)
    base_date = datetime.date.fromisoformat(start_date)

    rows: list[dict[str, Any]] = []

    for a_idx in range(1, n_athletes + 1):
        athlete_id = f"A{a_idx:02d}"
        base_h = rng.uniform(0.30, 0.42)
        base_rsimod = rng.uniform(0.40, 0.60)
        base_power = rng.uniform(3500, 4800)

        for week in range(1, n_weeks + 1):
            session_date = (base_date + datetime.timedelta(days=(week - 1) * 7)).isoformat()
            session_id = f"{athlete_id}_{session_date}"

            is_fatigued = (athlete_id == fatigue_athlete) and (week == fatigue_week)
            drop_factor = (1.0 - fatigue_drop_pct) if is_fatigued else 1.0

            for trial in (1, 2):
                var_h = rng.normal(0.0, 0.01)
                var_rsi = rng.normal(0.0, 0.02)
                var_pow = rng.normal(0.0, 50.0)

                h_val = max(0.15, (base_h + var_h) * drop_factor)
                rsi_val = max(0.20, (base_rsimod + var_rsi) * drop_factor)
                pow_val = max(2000.0, (base_power + var_pow) * drop_factor)

                flag_str = "fatigue_drop" if is_fatigued else ""

                rows.append({
                    "session_id": session_id,
                    "date": session_date,
                    "athlete_id": athlete_id,
                    "test": TestType.CMJ.value,
                    "trial": trial,
                    "metric": "cmj_jump_height_imp",
                    "value": round(float(h_val), 4),
                    "unit": "m",
                    "side": LimbSide.BOTH.value,
                    "valid": True,
                    "flags": flag_str,
                    "block": f"week_{week}",
                })
                rows.append({
                    "session_id": session_id,
                    "date": session_date,
                    "athlete_id": athlete_id,
                    "test": TestType.CMJ.value,
                    "trial": trial,
                    "metric": "cmj_rsimod",
                    "value": round(float(rsi_val), 4),
                    "unit": "ratio",
                    "side": LimbSide.BOTH.value,
                    "valid": True,
                    "flags": flag_str,
                    "block": f"week_{week}",
                })
                rows.append({
                    "session_id": session_id,
                    "date": session_date,
                    "athlete_id": athlete_id,
                    "test": TestType.CMJ.value,
                    "trial": trial,
                    "metric": "cmj_peak_power",
                    "value": round(float(pow_val), 1),
                    "unit": "W",
                    "side": LimbSide.BOTH.value,
                    "valid": True,
                    "flags": flag_str,
                    "block": f"week_{week}",
                })

    df = pd.DataFrame(rows)
    return df[SESSIONS_COLUMNS]


def make_wearable_daily(
    n_athletes: int = 5,
    n_days: int = 42,
    start_date: str = "2026-03-02",
    fatigue_athlete: str = "A01",
    fatigue_start_day: int = 21,
    fatigue_duration_days: int = 4,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a contract-compliant wearable_daily.csv DataFrame with matching recovery dips."""
    rng = np.random.default_rng(random_seed)
    base_date = datetime.date.fromisoformat(start_date)

    rows: list[dict[str, Any]] = []

    for a_idx in range(1, n_athletes + 1):
        athlete_id = f"A{a_idx:02d}"
        base_hrv = rng.uniform(65.0, 95.0)
        base_rhr = rng.uniform(48.0, 56.0)

        for day in range(n_days):
            current_date = (base_date + datetime.timedelta(days=day)).isoformat()
            is_fatigued = (
                (athlete_id == fatigue_athlete)
                and (fatigue_start_day <= day < fatigue_start_day + fatigue_duration_days)
            )

            if is_fatigued:
                recovery = float(rng.uniform(18.0, 31.0))
                hrv = float(base_hrv * rng.uniform(0.65, 0.78))
                rhr = float(base_rhr * rng.uniform(1.12, 1.25))
                strain = float(rng.uniform(14.0, 18.5))
                sleep = float(rng.uniform(4.5, 6.0))
            else:
                recovery = float(rng.uniform(55.0, 92.0))
                hrv = float(base_hrv + rng.normal(0.0, 4.0))
                rhr = float(base_rhr + rng.normal(0.0, 1.5))
                strain = float(rng.uniform(9.0, 15.0))
                sleep = float(rng.uniform(7.0, 8.8))

            rows.append({
                "date": current_date,
                "athlete_id": athlete_id,
                "recovery_pct": round(recovery, 1),
                "hrv_ms": round(hrv, 1),
                "rhr_bpm": round(rhr, 1),
                "day_strain": round(strain, 1),
                "sleep_hours": round(sleep, 2),
                "proxy_used": "",
            })

    df = pd.DataFrame(rows)
    return df[WEARABLE_DAILY_COLUMNS]
