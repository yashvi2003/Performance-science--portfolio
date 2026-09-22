"""Pipeline adapter for running the CMJ analysis pipeline on a single CSV trial.

This module provides a clean, reusable interface between the file-based import
(from scripts/import_cmj_xlsx.py) and the analysis pipeline
(validator, filter, events, metrics).

SYNTHETIC vs REAL DATA:
    This module operates on CSV files in data/raw/cmj/ (gitignored).
    It does NOT import any .xlsx files directly.

Usage:
    from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_csv
    events, metrics, validated = run_pipeline_on_csv(Path("data/raw/cmj/A01.csv"))
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from p1_nmt_pipeline.config import PipelineConfig
from p1_nmt_pipeline.events import CMJEvents, detect_cmj_events
from p1_nmt_pipeline.filter import zero_phase_butterworth_filter
from p1_nmt_pipeline.metrics import CMJMetrics, compute_cmj_metrics
from p1_nmt_pipeline.validator import ValidatedTrialData, load_and_validate_trial

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Bundled output from a single CMJ trial pipeline run.

    Attributes
    ----------
    athlete_id : str
        Opaque trial identifier (e.g. 'A01'). Never embed in public outputs.
    validated : ValidatedTrialData
        Parsed and quality-checked trial data including QA flags.
    filtered_force : np.ndarray
        Zero-phase filtered force array (N).
    events : CMJEvents
        Detected CMJ event landmarks and QA warnings.
    metrics : CMJMetrics
        Computed CMJ kinematic and kinetic metrics.
    config : PipelineConfig
        Configuration used for this run.
    """

    athlete_id: str
    validated: ValidatedTrialData
    filtered_force: np.ndarray
    events: CMJEvents
    metrics: CMJMetrics
    config: PipelineConfig

    @property
    def h_impulse_m(self) -> float:
        """Jump height by impulse-momentum theorem (m); NaN if detection failed."""
        return self.metrics.jump_height_impulse_m

    @property
    def h_flight_m(self) -> float:
        """Jump height by flight time (m); NaN if landing not detected."""
        return self.metrics.jump_height_flight_m

    @property
    def detection_ok(self) -> bool:
        """True if both takeoff and landing were successfully detected."""
        return not (
            np.isnan(self.metrics.jump_height_impulse_m)
            or np.isnan(self.metrics.jump_height_flight_m)
        )

    @property
    def all_qa_flags(self) -> list[str]:
        """Combined QA flags from validator and event detector."""
        return list(self.validated.qa_flags) + list(self.events.qa_warnings)


def run_pipeline_on_csv(
    csv_path: str | Path,
    config: PipelineConfig | None = None,
    body_mass_kg: float | None = None,
    filter_cutoff_hz: float | None = None,
) -> PipelineResult:
    """Run the full CMJ analysis pipeline on a single CSV trial file.

    Steps:
        1. Load and validate the trial (handles irregular timestamps, negative force, etc.)
        2. Apply zero-phase Butterworth low-pass filter
        3. Detect CMJ event landmarks (onset, unweighting, takeoff, landing)
        4. Compute CMJ metrics (h_impulse, h_flight, RSImod, peak power, etc.)

    Parameters
    ----------
    csv_path : str | Path
        Absolute or relative path to a trial CSV with columns 'time_s', 'force_n'.
        Typically data/raw/cmj/A01.csv (gitignored).
    config : PipelineConfig | None
        Analysis configuration. If None, loads from p1_nmt_pipeline/config.yaml.
    body_mass_kg : float | None
        Known body mass. If None, estimated from the quiet standing window.
    filter_cutoff_hz : float | None
        Override filter cutoff. If None, uses config.filters.get_cutoff('CMJ').

    Returns
    -------
    PipelineResult
        Complete pipeline output including validated data, events, and metrics.
    """
    csv_path = Path(csv_path)
    athlete_id = csv_path.stem  # e.g. 'A01'

    if config is None:
        config = PipelineConfig.load()

    cutoff = filter_cutoff_hz if filter_cutoff_hz is not None else config.filters.get_cutoff("CMJ")

    # 1. Load and validate
    validated = load_and_validate_trial(
        source=csv_path,
        body_mass_kg=body_mass_kg,
        test_type="CMJ",
        quiet_window_s=config.thresholds.quiet_window_s,
        quiet_start_s=config.thresholds.quiet_start_s,
    )
    logger.info(
        "Loaded trial %s: %d samples @ %.0f Hz, mass=%.1f kg, flags=%s",
        athlete_id,
        len(validated.time_s),
        validated.fs_hz,
        validated.body_mass_kg,
        validated.qa_flags,
    )

    # 2. Filter
    raw_force = validated.force_n.copy()
    filtered_force = zero_phase_butterworth_filter(
        raw_force,
        fs_hz=validated.fs_hz,
        cutoff_hz=cutoff,
        order=config.filters.order,
    )

    # 3. Detect events
    events = detect_cmj_events(
        time_s=validated.time_s,
        force_n=filtered_force,
        fs_hz=validated.fs_hz,
        config=config,
        body_mass_kg=validated.body_mass_kg,
        force_raw_n=raw_force,
    )

    if events.takeoff_idx is None:
        logger.warning("Takeoff detection FAILED for trial %s", athlete_id)
    if events.landing_idx is None:
        logger.warning("Landing detection FAILED for trial %s", athlete_id)

    # 4. Compute metrics
    metrics = compute_cmj_metrics(
        time_s=validated.time_s,
        force_n=filtered_force,
        events=events,
        config=config,
    )

    return PipelineResult(
        athlete_id=athlete_id,
        validated=validated,
        filtered_force=filtered_force,
        events=events,
        metrics=metrics,
        config=config,
    )


def run_pipeline_on_all(
    cmj_dir: str | Path,
    config: PipelineConfig | None = None,
    filter_cutoff_hz: float | None = None,
) -> list[PipelineResult]:
    """Run the pipeline on every A*.csv file in the given directory.

    Runs trials in alphabetical order (A01, A02, ..., A10).
    Errors in individual trials are logged but do not stop the batch run.

    Parameters
    ----------
    cmj_dir : str | Path
        Directory containing per-athlete CSV files (e.g. data/raw/cmj/).
    config : PipelineConfig | None
        Shared configuration for all trials.
    filter_cutoff_hz : float | None
        Override filter cutoff for all trials.

    Returns
    -------
    list[PipelineResult]
        One result per successfully processed trial, in alphabetical order.
    """
    cmj_dir = Path(cmj_dir)
    csvs = sorted(cmj_dir.glob("A*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No A*.csv files found in {cmj_dir}")

    results: list[PipelineResult] = []
    for csv in csvs:
        try:
            r = run_pipeline_on_csv(csv, config=config, filter_cutoff_hz=filter_cutoff_hz)
            results.append(r)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", csv.name, exc)
    return results
