"""Shared Data Contract: Schemas, Validators, and Metric Registry.

Governed by specs/00_shared_contract.md.
Projects import only from shared/.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
from typing import Any, Sequence

import pandas as pd


class TestType(str, Enum):
    __test__ = False
    CMJ = "CMJ"
    SJ = "SJ"
    IMTP = "IMTP"
    DJ = "DJ"


class LimbSide(str, Enum):
    BOTH = "both"
    LEFT = "left"
    RIGHT = "right"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class MetricDefinition:
    """Registered metric definition with standardized unit and optimization direction."""
    name: str
    test: TestType
    unit: str
    direction: MetricDirection
    description: str


# Canonical registry of all supported metrics across projects
METRIC_REGISTRY: dict[str, MetricDefinition] = {
    # CMJ Metrics
    "cmj_jump_height_imp": MetricDefinition(
        name="cmj_jump_height_imp",
        test=TestType.CMJ,
        unit="m",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Jump height calculated via impulse-momentum integration (takeoff velocity squared over 2g)",
    ),
    "cmj_jump_height_flight": MetricDefinition(
        name="cmj_jump_height_flight",
        test=TestType.CMJ,
        unit="m",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Jump height calculated from flight time (g * t_flight^2 / 8)",
    ),
    "cmj_rsimod": MetricDefinition(
        name="cmj_rsimod",
        test=TestType.CMJ,
        unit="ratio",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Modified Reactive Strength Index (jump height / time from onset to takeoff)",
    ),
    "cmj_peak_power": MetricDefinition(
        name="cmj_peak_power",
        test=TestType.CMJ,
        unit="W",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Peak instantaneous mechanical power prior to takeoff (F_z * v)",
    ),
    "cmj_peak_power_rel": MetricDefinition(
        name="cmj_peak_power_rel",
        test=TestType.CMJ,
        unit="W/kg",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Peak propulsion power normalized to athlete body mass",
    ),
    "cmj_peak_velocity": MetricDefinition(
        name="cmj_peak_velocity",
        test=TestType.CMJ,
        unit="m/s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Peak vertical velocity achieved prior to takeoff",
    ),
    "cmj_peak_braking_force": MetricDefinition(
        name="cmj_peak_braking_force",
        test=TestType.CMJ,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Maximum vertical force during the eccentric deceleration phase",
    ),
    "cmj_time_to_takeoff": MetricDefinition(
        name="cmj_time_to_takeoff",
        test=TestType.CMJ,
        unit="s",
        direction=MetricDirection.LOWER_IS_BETTER,
        description="Total contraction time from movement onset to takeoff",
    ),
    "cmj_flight_time": MetricDefinition(
        name="cmj_flight_time",
        test=TestType.CMJ,
        unit="s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Duration of flight phase between takeoff and landing",
    ),
    "cmj_body_mass": MetricDefinition(
        name="cmj_body_mass",
        test=TestType.CMJ,
        unit="kg",
        direction=MetricDirection.NEUTRAL,
        description="Estimated or measured body mass from quiet standing window",
    ),
    # SJ Metrics
    "sj_jump_height": MetricDefinition(
        name="sj_jump_height",
        test=TestType.SJ,
        unit="m",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Squat jump height from impulse-momentum or flight time",
    ),
    "sj_peak_force": MetricDefinition(
        name="sj_peak_force",
        test=TestType.SJ,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Peak concentric force during squat jump",
    ),
    "sj_peak_velocity": MetricDefinition(
        name="sj_peak_velocity",
        test=TestType.SJ,
        unit="m/s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Peak vertical velocity during squat jump",
    ),
    "sj_rfd_0_50ms": MetricDefinition(
        name="sj_rfd_0_50ms",
        test=TestType.SJ,
        unit="N/s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Rate of force development from 0 to 50 ms after onset",
    ),
    "sj_rfd_0_100ms": MetricDefinition(
        name="sj_rfd_0_100ms",
        test=TestType.SJ,
        unit="N/s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Rate of force development from 0 to 100 ms after onset",
    ),
    "sj_rfd_0_200ms": MetricDefinition(
        name="sj_rfd_0_200ms",
        test=TestType.SJ,
        unit="N/s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Rate of force development from 0 to 200 ms after onset",
    ),
    "sj_peak_rfd": MetricDefinition(
        name="sj_peak_rfd",
        test=TestType.SJ,
        unit="N/s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Peak instantaneous rate of force development during concentric rise",
    ),
    # DJ Metrics
    "dj_rsi": MetricDefinition(
        name="dj_rsi",
        test=TestType.DJ,
        unit="ratio",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Reactive Strength Index (jump height / ground contact time)",
    ),
    "dj_gct": MetricDefinition(
        name="dj_gct",
        test=TestType.DJ,
        unit="s",
        direction=MetricDirection.LOWER_IS_BETTER,
        description="Ground contact time from initial impact to secondary takeoff",
    ),
    "dj_jump_height": MetricDefinition(
        name="dj_jump_height",
        test=TestType.DJ,
        unit="m",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Jump height achieved from drop landing",
    ),
    "dj_flight_time": MetricDefinition(
        name="dj_flight_time",
        test=TestType.DJ,
        unit="s",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Flight time after drop jump rebound",
    ),
    # IMTP Metrics (Stretch Goal S3)
    "imtp_peak_force_gross": MetricDefinition(
        name="imtp_peak_force_gross",
        test=TestType.IMTP,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Maximum absolute force recorded during isometric pull",
    ),
    "imtp_peak_force_net": MetricDefinition(
        name="imtp_peak_force_net",
        test=TestType.IMTP,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Peak force minus baseline body weight",
    ),
    "imtp_time_to_peak": MetricDefinition(
        name="imtp_time_to_peak",
        test=TestType.IMTP,
        unit="s",
        direction=MetricDirection.LOWER_IS_BETTER,
        description="Time from onset to peak force",
    ),
    "imtp_force_50ms": MetricDefinition(
        name="imtp_force_50ms",
        test=TestType.IMTP,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Force recorded at 50 ms from pull onset",
    ),
    "imtp_force_100ms": MetricDefinition(
        name="imtp_force_100ms",
        test=TestType.IMTP,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Force recorded at 100 ms from pull onset",
    ),
    "imtp_force_150ms": MetricDefinition(
        name="imtp_force_150ms",
        test=TestType.IMTP,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Force recorded at 150 ms from pull onset",
    ),
    "imtp_force_200ms": MetricDefinition(
        name="imtp_force_200ms",
        test=TestType.IMTP,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Force recorded at 200 ms from pull onset",
    ),
    "imtp_force_250ms": MetricDefinition(
        name="imtp_force_250ms",
        test=TestType.IMTP,
        unit="N",
        direction=MetricDirection.HIGHER_IS_BETTER,
        description="Force recorded at 250 ms from pull onset",
    ),
}

# Required columns per specs/00_shared_contract.md
SESSIONS_COLUMNS: list[str] = [
    "session_id",
    "date",
    "athlete_id",
    "test",
    "trial",
    "metric",
    "value",
    "unit",
    "side",
    "valid",
    "flags",
    "block",
]

WEARABLE_DAILY_COLUMNS: list[str] = [
    "date",
    "athlete_id",
    "recovery_pct",
    "hrv_ms",
    "rhr_bpm",
    "day_strain",
    "sleep_hours",
    "proxy_used",
]

ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ContractValidationError(ValueError):
    """Raised when data violates the shared contract schema."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_sessions_df(df: pd.DataFrame, strict: bool = True) -> tuple[bool, list[str]]:
    """Validate a DataFrame against the sessions.csv contract schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    strict : bool
        If True, unrecognised metrics or extra columns produce errors.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error messages)
    """
    errors: list[str] = []

    # Check required columns
    missing_cols = set(SESSIONS_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        return False, errors

    if len(df) == 0:
        return True, []

    # Validate ISO date
    date_series = df["date"].astype(str)
    invalid_dates = date_series[~date_series.str.match(ISO_DATE_REGEX)]
    if not invalid_dates.empty:
        errors.append(f"Invalid ISO dates in 'date' column: {invalid_dates.iloc[:3].tolist()}")

    # Validate test types
    valid_tests = {t.value for t in TestType}
    invalid_tests = df[~df["test"].isin(valid_tests)]["test"].unique()
    if len(invalid_tests) > 0:
        errors.append(f"Invalid test types: {invalid_tests.tolist()}. Must be one of {valid_tests}")

    # Validate trial numbers (positive integers)
    if not pd.api.types.is_integer_dtype(df["trial"]):
        try:
            non_int = df[df["trial"] != df["trial"].astype(int)]
            if not non_int.empty:
                errors.append("Column 'trial' contains non-integer values.")
        except Exception:
            errors.append("Column 'trial' must be integer.")
    if (df["trial"] < 1).any():
        errors.append("Column 'trial' must contain positive 1-based integers.")

    # Validate metric registry
    if strict:
        invalid_metrics = set(df["metric"]) - set(METRIC_REGISTRY.keys())
        if invalid_metrics:
            errors.append(f"Unregistered metrics: {sorted(invalid_metrics)}")

    # Validate metric-unit consistency
    for metric_name, group in df.groupby("metric"):
        if metric_name in METRIC_REGISTRY:
            expected_unit = METRIC_REGISTRY[metric_name].unit
            wrong_units = group[group["unit"] != expected_unit]["unit"].unique()
            if len(wrong_units) > 0:
                errors.append(
                    f"Unit mismatch for '{metric_name}': expected '{expected_unit}', found {wrong_units.tolist()}"
                )

    # Validate side column
    valid_sides = {s.value for s in LimbSide}
    invalid_sides = df[~df["side"].isin(valid_sides)]["side"].unique()
    if len(invalid_sides) > 0:
        errors.append(f"Invalid side values: {invalid_sides.tolist()}. Must be one of {valid_sides}")

    # Validate valid column (boolean)
    if not pd.api.types.is_bool_dtype(df["valid"]):
        # Check if values are coerceable to boolean
        unique_vals = set(df["valid"].dropna().unique())
        if not unique_vals.issubset({True, False, 1, 0, "True", "False", "true", "false"}):
            errors.append(f"Column 'valid' must be boolean, found: {unique_vals}")

    # Validate non-null values for essential columns
    for col in ["session_id", "athlete_id", "metric", "value"]:
        if df[col].isna().any():
            errors.append(f"Column '{col}' contains NaN or null values.")

    return len(errors) == 0, errors


def validate_wearable_daily_df(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate a DataFrame against the wearable_daily.csv contract schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list of error messages)
    """
    errors: list[str] = []

    missing_cols = set(WEARABLE_DAILY_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {sorted(missing_cols)}")
        return False, errors

    if len(df) == 0:
        return True, []

    # Validate ISO date
    date_series = df["date"].astype(str)
    invalid_dates = date_series[~date_series.str.match(ISO_DATE_REGEX)]
    if not invalid_dates.empty:
        errors.append(f"Invalid ISO dates in 'date' column: {invalid_dates.iloc[:3].tolist()}")

    # Validate athlete_id non-null
    if df["athlete_id"].isna().any():
        errors.append("Column 'athlete_id' contains NaN values.")

    # Validate recovery_pct (0 to 100 or NaN)
    rec = df["recovery_pct"].dropna()
    if not rec.empty and ((rec < 0.0) | (rec > 100.0)).any():
        errors.append("Column 'recovery_pct' must be within [0.0, 100.0].")

    # Validate positive values where present
    for col in ["hrv_ms", "rhr_bpm", "day_strain", "sleep_hours"]:
        vals = df[col].dropna()
        if not vals.empty and (vals < 0.0).any():
            errors.append(f"Column '{col}' contains negative values.")

    return len(errors) == 0, errors


def validate_sessions_csv(path: str | Path, strict: bool = True) -> tuple[bool, list[str]]:
    """Read and validate a sessions.csv file."""
    path = Path(path)
    if not path.exists():
        return False, [f"File does not exist: {path}"]
    df = pd.read_csv(path)
    return validate_sessions_df(df, strict=strict)


def validate_wearable_daily_csv(path: str | Path) -> tuple[bool, list[str]]:
    """Read and validate a wearable_daily.csv file."""
    path = Path(path)
    if not path.exists():
        return False, [f"File does not exist: {path}"]
    df = pd.read_csv(path)
    return validate_wearable_daily_df(df)
