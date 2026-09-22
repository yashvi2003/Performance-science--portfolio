"""Kinematic and kinetic metrics computation engine for CMJ.

Extracts performance and neuromuscular fatigue metrics strictly following
the biomechanical principles defined in specs/01_neuromuscular_pipeline.md,
PLAN.md, and McMahon et al. (2018).

All mathematical formulas are explicitly documented in each function and class docstring.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from shared.contract import STANDARD_G
from p1_nmt_pipeline.config import PipelineConfig
from p1_nmt_pipeline.events import CMJEvents

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CMJMetrics:
    """Biomechanical metrics extracted from a Countermovement Jump (CMJ) trial.

    All formulas adhere to standard sports biomechanics conventions.

    Attributes
    ----------
    jump_height_impulse_m : float
        Primary jump height derived from takeoff velocity via the impulse-momentum theorem (m).
        Formula: h_impulse = v_takeoff^2 / (2 * g), where g = 9.80665 m/s^2.
    jump_height_flight_m : float | None
        Secondary jump height derived from flight time under ballistic assumptions (m).
        Formula: h_flight = (g * t_flight^2) / 8, where t_flight = t_landing - t_takeoff.
    jump_height_primary_m : float
        Explicit alias for primary jump height (impulse-momentum theorem).
    rsimod_impulse : float
        Primary Modified Reactive Strength Index (m/s).
        Formula: RSImod_impulse = h_impulse / (t_takeoff - t_onset).
    rsimod_flight : float | None
        Secondary Modified Reactive Strength Index calculated using flight-time height (m/s).
        Formula: RSImod_flight = h_flight / (t_takeoff - t_onset).
    contraction_time_s : float
        Total movement contraction duration from detected onset to takeoff (s).
        Formula: t_contraction = t_takeoff - t_onset.
    flight_time_s : float | None
        Airborne flight duration from takeoff to landing (s).
        Formula: t_flight = t_landing - t_takeoff.
    peak_velocity_m_per_s : float
        Maximum vertical velocity achieved prior to takeoff (m/s).
        Formula: v_peak = max_{t in [t_int_start, t_takeoff]} v(t).
    peak_power_w : float
        Maximum instantaneous mechanical power achieved during concentric propulsion (W).
        Formula: P_peak = max_{t in [t_v0, t_takeoff]} (F_z(t) * v(t)).
    peak_relative_power_w_per_kg : float
        Peak mechanical power normalized by athlete body mass (W/kg).
        Formula: P_rel_peak = P_peak / m.
    peak_propulsion_force_n : float
        Maximum vertical ground reaction force achieved during concentric propulsion prior to takeoff (N).
        Formula: F_peak_prop = max_{t in [t_v0, t_takeoff]} F_z(t).
    peak_braking_force_n : float
        Maximum vertical force observed during the eccentric braking phase (N).
        Formula: F_peak_braking = max_{t in [t_unweight_min, t_v0]} F_z(t).
    peak_landing_force_n : float | None
        Maximum vertical force observed during landing impact (N), reported separately from propulsion.
        Formula: F_peak_landing = max_{t >= t_landing} F_z(t).
    body_weight_n : float
        Athlete body weight during quiet baseline standing (N).
        Formula: BW = (1 / N_quiet) * sum_{i=1}^{N_quiet} F(t_i).
    body_mass_kg : float
        Athlete body mass in kg.
        Formula: m = BW / g.
    detected_onset_time_s : float
        Timestamp of movement onset detection (s).
    integration_start_time_s : float
        Timestamp where trapezoidal velocity integration began (s).
        Formula: t_int_start = max(0, t_onset - delta_t_lead).
    takeoff_time_s : float
        Timestamp of takeoff (s).
    landing_time_s : float | None
        Timestamp of landing impact (s).
    flight_offset_n : float
        Residual force detected during flight phase (N).
    flight_offset_applied : bool
        Whether flight-phase offset correction was applied to force data.
    flight_offset_model : str
        The offset correction model used ('whole_trace' or 'flight_only').
    """
    jump_height_impulse_m: float
    jump_height_flight_m: float | None
    jump_height_primary_m: float
    rsimod_impulse: float
    rsimod_flight: float | None
    contraction_time_s: float
    flight_time_s: float | None
    peak_velocity_m_per_s: float
    peak_power_w: float
    peak_relative_power_w_per_kg: float
    peak_propulsion_force_n: float
    peak_braking_force_n: float
    peak_landing_force_n: float | None
    body_weight_n: float
    body_mass_kg: float
    detected_onset_time_s: float
    integration_start_time_s: float
    takeoff_time_s: float
    landing_time_s: float | None
    flight_offset_n: float
    flight_offset_applied: bool
    flight_offset_model: str


def compute_cmj_metrics(
    time_s: np.ndarray,
    force_n: np.ndarray,
    events: CMJEvents,
    config: PipelineConfig | None = None,
) -> CMJMetrics:
    """Compute comprehensive CMJ metrics from force telemetry and detected event landmarks.

    Mathematical Formulations:
    --------------------------
    1. Acceleration Profile:
       a(t) = (F_z(t) - BW) / m
       where BW = events.quiet_bw_mean_n and m = events.body_mass_kg.

    2. Velocity Profile via Trapezoidal Integration:
       v(t_0) = 0 m/s at t_0 = t_int_start
       v(t_{i+1}) = v(t_i) + 0.5 * (a(t_i) + a(t_{i+1})) * (t_{i+1} - t_i)

    3. Takeoff Velocity:
       v_takeoff = v(t_takeoff)

    4. Jump Height by Impulse-Momentum Theorem (Primary):
       h_impulse = v_takeoff^2 / (2 * g)

    5. Jump Height by Flight Time (Secondary):
       t_flight = t_landing - t_takeoff
       h_flight = (g * t_flight^2) / 8

    6. Contraction Time:
       t_contraction = t_takeoff - t_onset

    7. Modified Reactive Strength Index (RSImod):
       RSImod_impulse (Primary) = h_impulse / t_contraction
       RSImod_flight (Secondary) = h_flight / t_contraction

    8. Instantaneous Power:
       P(t) = F_z(t) * v(t)

    9. Pre-Takeoff Peak Values (strictly prior to takeoff):
       v_peak = max_{t in [t_int_start, t_takeoff]} v(t)
       P_peak = max_{t in [t_v0, t_takeoff]} P(t)
       P_rel_peak = P_peak / m
       F_peak_prop = max_{t in [t_v0, t_takeoff]} F_z(t)

    10. Peak Eccentric Braking Force:
        F_peak_braking = max_{t in [t_unweight_min, t_v0]} F_z(t)

    11. Isolated Peak Landing Impact Force:
        F_peak_landing = max_{t >= t_landing} F_z(t)

    Parameters
    ----------
    time_s : np.ndarray
        Array of timestamps (s).
    force_n : np.ndarray
        Filtered vertical ground reaction force array (N).
    events : CMJEvents
        Detected event landmarks from detect_cmj_events().
    config : PipelineConfig | None
        Configuration container.

    Returns
    -------
    CMJMetrics
        Dataclass populated with all verified CMJ kinetic and kinematic metrics.
    """
    if config is None:
        config = PipelineConfig.load()

    # Determine the force trace to integrate:
    # If whole-trace offset correction was applied, subtract offset from force
    if events.flight_offset_applied and events.flight_offset_model == "whole_trace":
        corr_force = force_n - events.flight_offset_n
    else:
        corr_force = force_n.copy()

    bw = events.quiet_bw_mean_n
    mass = events.body_mass_kg

    # 1. Acceleration Profile
    # a(t) = (F(t) - BW) / m
    accel = (corr_force - bw) / mass

    # 2. Velocity Profile via Trapezoidal Rule from t_int_start
    # v(t_{i+1}) = v(t_i) + 0.5 * (a(t_i) + a(t_{i+1})) * dt_i
    n_samples = len(time_s)
    vel = np.zeros(n_samples, dtype=float)
    dt_arr = np.diff(time_s)

    for i in range(events.int_start_idx, n_samples - 1):
        vel[i + 1] = vel[i] + 0.5 * (accel[i] + accel[i + 1]) * dt_arr[i]

    # Handle Takeoff Detection Failure
    if events.takeoff_idx is None or events.takeoff_time_s is None:
        braking_forces = corr_force[events.unweighting_min_idx : events.v0_idx + 1]
        peak_braking_force = float(np.max(braking_forces)) if len(braking_forces) > 0 else bw
        return CMJMetrics(
            jump_height_impulse_m=float("nan"),
            jump_height_flight_m=float("nan"),
            jump_height_primary_m=float("nan"),
            rsimod_impulse=float("nan"),
            rsimod_flight=float("nan"),
            contraction_time_s=float("nan"),
            flight_time_s=float("nan"),
            peak_velocity_m_per_s=float("nan"),
            peak_power_w=float("nan"),
            peak_relative_power_w_per_kg=float("nan"),
            peak_propulsion_force_n=float("nan"),
            peak_braking_force_n=round(peak_braking_force, 2),
            peak_landing_force_n=float("nan"),
            body_weight_n=round(bw, 2),
            body_mass_kg=round(mass, 2),
            detected_onset_time_s=round(events.onset_time_s, 4),
            integration_start_time_s=round(events.int_start_time_s, 4),
            takeoff_time_s=float("nan"),
            landing_time_s=float("nan"),
            flight_offset_n=round(events.flight_offset_n, 3),
            flight_offset_applied=events.flight_offset_applied,
            flight_offset_model=events.flight_offset_model,
        )

    # 3. Takeoff Velocity
    # v_takeoff = v(t_takeoff)
    v_to = float(vel[events.takeoff_idx])
    if config.thresholds.subsample_takeoff and events.takeoff_idx > 0 and events.takeoff_time_s is not None:
        dt_offset = float(events.takeoff_time_s) - float(time_s[events.takeoff_idx])
        a_floor = (events.flight_offset_n - bw) / mass
        v_to = float(v_to + a_floor * dt_offset)


    # 4. Jump Height by Impulse-Momentum Theorem (Primary)
    # h_impulse = v_takeoff^2 / (2 * g)
    if v_to > 0.0:
        h_impulse = float((v_to**2) / (2.0 * STANDARD_G))
    else:
        h_impulse = 0.0

    # 5. Jump Height by Flight Time (Secondary)
    # t_flight = t_landing - t_takeoff
    # h_flight = (g * t_flight^2) / 8
    h_flight = float("nan")
    flight_time = events.flight_time_s
    if events.landing_idx is not None and flight_time is not None and flight_time > 0.0:
        h_flight = float((STANDARD_G * (flight_time**2)) / 8.0)

    # 6. Contraction Time
    # t_contraction = t_takeoff - t_onset
    contraction_time = float(events.takeoff_time_s - events.onset_time_s)
    if contraction_time <= 0.0:
        contraction_time = float(events.takeoff_time_s - events.int_start_time_s)
    contraction_time = max(0.001, contraction_time)

    # 7. Modified Reactive Strength Index (RSImod)
    # RSImod_impulse = h_impulse / t_contraction
    # RSImod_flight = h_flight / t_contraction
    rsimod_impulse = float(h_impulse / contraction_time)
    rsimod_flight = float(h_flight / contraction_time) if not np.isnan(h_flight) else float("nan")

    # 8. Instantaneous Power
    # P(t) = F_z(t) * v(t)
    power = corr_force * vel

    # 9. Pre-Takeoff Peak Velocity (strictly pre-takeoff)
    # v_peak = max_{t in [t_int_start, t_takeoff]} v(t)
    pre_to_vel = vel[events.int_start_idx : events.takeoff_idx + 1]
    peak_vel = float(np.max(pre_to_vel)) if len(pre_to_vel) > 0 else v_to

    # 10. Pre-Takeoff Peak Power (during concentric propulsion [v0, takeoff])
    # P_peak = max_{t in [t_v0, t_takeoff]} P(t)
    prop_power = power[events.v0_idx : events.takeoff_idx + 1]
    peak_power = float(np.max(prop_power)) if len(prop_power) > 0 else 0.0
    peak_rel_power = float(peak_power / mass)

    # 11. Pre-Takeoff Peak Propulsion Force
    # F_peak_prop = max_{t in [t_v0, t_takeoff]} F_z(t)
    prop_forces = corr_force[events.v0_idx : events.takeoff_idx + 1]
    peak_prop_force = float(np.max(prop_forces)) if len(prop_forces) > 0 else bw

    # 12. Peak Eccentric Braking Force
    # F_peak_braking = max_{t in [t_unweight_min, t_v0]} F_z(t)
    braking_forces = corr_force[events.unweighting_min_idx : events.v0_idx + 1]
    peak_braking_force = float(np.max(braking_forces)) if len(braking_forces) > 0 else bw

    # 13. Isolated Peak Landing Force
    # F_peak_landing = max_{t >= t_landing} F_z(t)
    peak_landing_force = events.landing_peak_force_n if events.landing_idx is not None else float("nan")

    return CMJMetrics(
        jump_height_impulse_m=round(h_impulse, 4),
        jump_height_flight_m=round(h_flight, 4) if not np.isnan(h_flight) else float("nan"),
        jump_height_primary_m=round(h_impulse, 4),
        rsimod_impulse=round(rsimod_impulse, 4),
        rsimod_flight=round(rsimod_flight, 4) if not np.isnan(rsimod_flight) else float("nan"),
        contraction_time_s=round(contraction_time, 4),
        flight_time_s=round(flight_time, 4) if flight_time is not None and not np.isnan(flight_time) else float("nan"),
        peak_velocity_m_per_s=round(peak_vel, 4),
        peak_power_w=round(peak_power, 2),
        peak_relative_power_w_per_kg=round(peak_rel_power, 2),
        peak_propulsion_force_n=round(peak_prop_force, 2),
        peak_braking_force_n=round(peak_braking_force, 2),
        peak_landing_force_n=round(peak_landing_force, 2) if peak_landing_force is not None and not np.isnan(peak_landing_force) else float("nan"),
        body_weight_n=round(bw, 2),
        body_mass_kg=round(mass, 2),
        detected_onset_time_s=round(events.onset_time_s, 4),
        integration_start_time_s=round(events.int_start_time_s, 4),
        takeoff_time_s=round(events.takeoff_time_s, 4) if events.takeoff_time_s is not None else float("nan"),
        landing_time_s=round(events.landing_time_s, 4) if events.landing_time_s is not None else float("nan"),
        flight_offset_n=round(events.flight_offset_n, 3),
        flight_offset_applied=events.flight_offset_applied,
        flight_offset_model=events.flight_offset_model,
    )

