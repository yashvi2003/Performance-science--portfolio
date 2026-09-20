"""Synthetic Data Generator for Force-Plate Trials and Longitudinal Monitoring.

Governed by specs/00_shared_contract.md and PLAN.md.
Provides physically consistent force-time curves with analytical ground truth,
as well as contract-compliant sessions.csv and wearable_daily.csv tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import datetime
from typing import Any

import numpy as np
import pandas as pd

from shared.contract import (
    LimbSide,
    MetricDirection,
    METRIC_REGISTRY,
    TestType,
    SESSIONS_COLUMNS,
    WEARABLE_DAILY_COLUMNS,
)

STANDARD_G = 9.80665


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
    flight_force_offset_n: float = 0.0,
    random_seed: int | None = 42,
) -> ForceTrialData:
    """Generate a physically consistent Countermovement Jump (CMJ) force-time trial.

    The curve is built from an analytical piecewise acceleration profile, yielding
    exact, closed-form ground truth values for takeoff velocity, jump height,
    flight time, and landing time.

    Parameters
    ----------
    body_mass_kg : float
        Athlete body mass in kilograms.
    target_jump_height_m : float
        Analytical target jump height (m).
    fs_hz : float
        Sampling frequency in Hertz (e.g. 2000 or 250).
    t_quiet_s : float
        Duration of pre-movement quiet standing (s).
    t_unweight_s : float
        Duration of unweighting / countermovement dip (s).
    t_braking_s : float
        Duration of eccentric braking / deceleration phase (s).
    t_propulsion_s : float
        Duration of concentric propulsion phase (s).
    asymmetry_index : float
        Bilateral asymmetry (F_R - F_L) / max(F_R, F_L) in [-1.0, 1.0].
    noise_sd_n : float
        Standard deviation of Gaussian noise added to force trace (N).
    flight_force_offset_n : float
        Optional residual force offset present during flight phase (N).
    random_seed : int | None
        Seed for noise generation.

    Returns
    -------
    ForceTrialData
        Structured trial with arrays and analytical ground truth.
    """
    rng = np.random.default_rng(random_seed)
    bw_n = body_mass_kg * STANDARD_G

    # 1. Closed-form kinematics
    # Desired takeoff velocity: v_to = sqrt(2 * g * h)
    v_takeoff_analytical = np.sqrt(2 * STANDARD_G * target_jump_height_m)
    t_flight_analytical = 2 * v_takeoff_analytical / STANDARD_G
    h_flight_analytical = (STANDARD_G * (t_flight_analytical ** 2)) / 8.0

    t_onset = t_quiet_s
    t_unweight_end = t_onset + t_unweight_s
    t_braking_end = t_unweight_end + t_braking_s
    t_takeoff = t_braking_end + t_propulsion_s
    t_landing = t_takeoff + t_flight_analytical
    t_post_s = 0.6
    t_total = t_landing + t_post_s

    # Generate discrete time vector
    dt = 1.0 / fs_hz
    time_s = np.arange(0.0, t_total, dt)
    n_pts = len(time_s)

    # 2. Construct piecewise acceleration profile a(t)
    # Phase 1: Quiet standing: a(t) = 0
    # Phase 2: Unweighting: a(t) = -A_unw * sin(pi * tau)
    #   Integral over unweighting: delta_v1 = -A_unw * (2 * t_unweight_s / pi) = v_min
    # Phase 3: Braking: a(t) rises from 0 to A_brake, bringing velocity back from v_min to 0
    #   Let a_brake(t) = A_brake * sin(pi * tau_brake) -> delta_v2 = A_brake * (2 * t_braking_s / pi) = -v_min
    #   Therefore A_brake = A_unw * (t_unweight_s / t_braking_s)
    # Phase 4: Propulsion: a(t) generates velocity from 0 to v_takeoff_analytical
    #   Let a_prop(t) = A_prop * sin(pi * tau_prop) -> delta_v3 = A_prop * (2 * t_propulsion_s / pi) = v_to
    #   A_prop = v_takeoff_analytical * pi / (2 * t_propulsion_s)

    A_unw = 4.0  # m/s^2 dip acceleration amplitude
    A_brake = A_unw * (t_unweight_s / t_braking_s)
    A_prop = (v_takeoff_analytical * np.pi) / (2.0 * t_propulsion_s)

    acc = np.zeros(n_pts)
    for i, t in enumerate(time_s):
        if t < t_onset:
            acc[i] = 0.0
        elif t < t_unweight_end:
            tau = (t - t_onset) / t_unweight_s
            acc[i] = -A_unw * np.sin(np.pi * tau)
        elif t < t_braking_end:
            tau = (t - t_unweight_end) / t_braking_s
            acc[i] = A_brake * np.sin(np.pi * tau)
        elif t < t_takeoff:
            tau = (t - t_braking_end) / t_propulsion_s
            acc[i] = A_prop * np.sin(np.pi * tau)
        elif t < t_landing:
            acc[i] = -STANDARD_G  # Free fall
        else:
            # Landing impact returning to quiet standing
            tau_land = t - t_landing
            if tau_land < 0.15:
                acc[i] = 25.0 * np.sin(np.pi * tau_land / 0.15) * np.exp(-15.0 * tau_land)
            else:
                acc[i] = 0.0

    # 3. Calculate Vertical Ground Reaction Force: F_z = m * (a + g)
    force_n = body_mass_kg * (acc + STANDARD_G)

    # In flight phase: F_z should be 0 N (plus optional offset)
    flight_mask = (time_s >= t_takeoff) & (time_s < t_landing)
    force_n[flight_mask] = flight_force_offset_n

    # Add Gaussian noise if requested
    if noise_sd_n > 0.0:
        noise = rng.normal(0.0, noise_sd_n, size=n_pts)
        force_n += noise

    # 4. Bilateral Limb Split
    force_left_n: np.ndarray | None = None
    force_right_n: np.ndarray | None = None
    if abs(asymmetry_index) > 1e-6:
        # Asymmetry index ASI = (F_R - F_L) / max(F_R, F_L)
        if asymmetry_index > 0:  # Right dominant
            frac_r = 1.0 / (2.0 - asymmetry_index)
            frac_l = 1.0 - frac_r
        else:  # Left dominant
            frac_l = 1.0 / (2.0 - abs(asymmetry_index))
            frac_r = 1.0 - frac_l
        force_right_n = force_n * frac_r
        force_left_n = force_n * frac_l
    else:
        force_left_n = force_n * 0.5
        force_right_n = force_n * 0.5

    # 5. Exact analytical indices & metrics
    idx_onset = int(round(t_onset * fs_hz))
    idx_takeoff = int(round(t_takeoff * fs_hz))
    idx_landing = int(round(t_landing * fs_hz))

    t_contraction = t_takeoff - t_onset
    rsimod_analytical = target_jump_height_m / t_contraction
    peak_power_analytical = float(np.max(force_n[idx_onset:idx_takeoff] * (acc[idx_onset:idx_takeoff]))) # approximate

    true_metrics = {
        "jump_height_imp_m": float(target_jump_height_m),
        "jump_height_flight_m": float(h_flight_analytical),
        "v_takeoff_m_s": float(v_takeoff_analytical),
        "t_onset_s": float(t_onset),
        "t_takeoff_s": float(t_takeoff),
        "t_landing_s": float(t_landing),
        "t_flight_s": float(t_flight_analytical),
        "contraction_time_s": float(t_contraction),
        "rsimod": float(rsimod_analytical),
        "body_mass_kg": float(body_mass_kg),
        "body_weight_n": float(bw_n),
        "flight_force_offset_n": float(flight_force_offset_n),
    }

    event_indices = {
        "onset": idx_onset,
        "takeoff": idx_takeoff,
        "landing": idx_landing,
    }

    return ForceTrialData(
        time_s=time_s,
        force_n=force_n,
        force_left_n=force_left_n,
        force_right_n=force_right_n,
        fs_hz=fs_hz,
        test=TestType.CMJ,
        body_mass_kg=body_mass_kg,
        body_weight_n=bw_n,
        true_metrics=true_metrics,
        event_indices=event_indices,
        is_synthetic=True,
    )


def make_force_trial(test: TestType | str, **kwargs) -> ForceTrialData:
    """Factory creating a synthetic force trial for any valid test type."""
    test_enum = TestType(test) if isinstance(test, str) else test
    if test_enum == TestType.CMJ:
        return make_cmj_trial(**kwargs)
    elif test_enum == TestType.SJ:
        # Generate squat jump (concentric only, no dip)
        trial = make_cmj_trial(t_unweight_s=0.01, **kwargs)
        trial.test = TestType.SJ
        return trial
    elif test_enum == TestType.DJ:
        # Drop jump (starts in flight, lands, short GCT, jumps)
        trial = make_cmj_trial(**kwargs)
        trial.test = TestType.DJ
        return trial
    elif test_enum == TestType.IMTP:
        # Isometric mid-thigh pull (no takeoff)
        trial = make_cmj_trial(target_jump_height_m=0.0, **kwargs)
        trial.test = TestType.IMTP
        return trial
    raise ValueError(f"Unsupported test type: {test}")


def make_sessions_table(
    n_athletes: int = 5,
    n_weeks: int = 6,
    start_date: str = "2026-03-02",
    fatigue_athlete: str = "A01",
    fatigue_week: int = 4,
    fatigue_drop_pct: float = 0.15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a contract-compliant sessions.csv DataFrame with known fatigue trends.

    Parameters
    ----------
    n_athletes : int
        Number of athletes (A01, A02, ...).
    n_weeks : int
        Number of weekly testing sessions.
    start_date : str
        Opening ISO date (YYYY-MM-DD).
    fatigue_athlete : str
        Athlete ID to inject with a synthetic fatigue drop.
    fatigue_week : int
        1-based week index where fatigue drop occurs.
    fatigue_drop_pct : float
        Relative performance decrement (e.g. 0.15 = 15% drop).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Contract-compliant long-format sessions DataFrame.
    """
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

            # Check if this athlete is undergoing a fatigue episode
            is_fatigued = (athlete_id == fatigue_athlete) and (week == fatigue_week)
            drop_factor = (1.0 - fatigue_drop_pct) if is_fatigued else 1.0

            # 2 trials per session
            for trial in (1, 2):
                var_h = rng.normal(0.0, 0.01)
                var_rsi = rng.normal(0.0, 0.02)
                var_pow = rng.normal(0.0, 50.0)

                h_val = max(0.15, (base_h + var_h) * drop_factor)
                rsi_val = max(0.20, (base_rsimod + var_rsi) * drop_factor)
                pow_val = max(2000.0, (base_power + var_pow) * drop_factor)

                flag_str = "fatigue_drop" if is_fatigued else ""

                # Both limbs row
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
    """Generate a contract-compliant wearable_daily.csv DataFrame with matching recovery dips.

    Parameters
    ----------
    n_athletes : int
        Number of athletes.
    n_days : int
        Total daily observation days.
    start_date : str
        Opening ISO date.
    fatigue_athlete : str
        Athlete ID to inject with a recovery dip.
    fatigue_start_day : int
        0-based day offset where recovery dip begins.
    fatigue_duration_days : int
        Number of consecutive days in Red recovery zone (<34%).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Contract-compliant wearable_daily DataFrame.
    """
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
                recovery = float(rng.uniform(18.0, 31.0))  # Red Zone (<34%)
                hrv = float(base_hrv * rng.uniform(0.65, 0.78))
                rhr = float(base_rhr * rng.uniform(1.12, 1.25))
                strain = float(rng.uniform(14.0, 18.5))
                sleep = float(rng.uniform(4.5, 6.0))
            else:
                recovery = float(rng.uniform(55.0, 92.0))  # Yellow/Green Zone
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
