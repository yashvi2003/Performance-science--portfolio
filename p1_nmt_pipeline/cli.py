"""Command-Line Interface (CLI) for Project 1 Neuromuscular Analysis Pipeline.

Processes raw or imported force plate trials, runs event detection and metric
computation, and exports a fully contract-compliant sessions.csv per
specs/00_shared_contract.md.

Usage:
    python -m p1_nmt_pipeline.cli --input data/raw/cmj --output data/processed/sessions.csv
    python -m p1_nmt_pipeline.cli --input data/raw/cmj/A01.csv --date 2026-09-22
"""
from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd

from p1_nmt_pipeline.config import PipelineConfig
from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_csv
from shared.contract import (
    METRIC_REGISTRY,
    SESSIONS_COLUMNS,
    LimbSide,
    TestType,
    validate_sessions_df,
)

logger = logging.getLogger("p1_nmt_pipeline.cli")


def build_sessions_dataframe(
    results: list[dict],
    session_date: str,
    anonymize: bool = True,
) -> pd.DataFrame:
    """Build a contract-compliant sessions DataFrame from pipeline trial results.

    Parameters
    ----------
    results : list[dict]
        List of dicts containing 'athlete_id', 'trial_idx', 'events', 'metrics', 'validated'.
    session_date : str
        ISO format date string (YYYY-MM-DD).
    anonymize : bool
        If True, replaces raw athlete IDs with sequential de-identified IDs ('ATH_01', etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame strictly adhering to SESSIONS_COLUMNS and METRIC_REGISTRY.
    """
    rows: list[dict] = []

    # Map raw athlete IDs to anonymized tokens if requested
    athlete_map: dict[str, str] = {}
    for i, r in enumerate(results, start=1):
        raw_id = r["athlete_id"]
        if raw_id not in athlete_map:
            athlete_map[raw_id] = f"ATH_{len(athlete_map) + 1:02d}" if anonymize else raw_id

    for r in results:
        raw_id = r["athlete_id"]
        athlete_id = athlete_map[raw_id]
        trial_num = r.get("trial_num", 1)
        session_id = f"{athlete_id}_{session_date}"
        m = r["metrics"]
        e = r["events"]
        v = r["validated"]

        # Collect QA flags
        all_flags = list(e.qa_warnings) + list(v.qa_flags)
        flag_str = ";".join(all_flags) if all_flags else ""
        is_valid = len(all_flags) == 0 or not any("failed" in f.lower() for f in all_flags)

        # Map CMJ metrics to contract registry names
        metric_mappings = [
            ("cmj_jump_height_imp", round(float(m.jump_height_impulse_m), 4)),
            ("cmj_jump_height_flight", round(float(m.jump_height_flight_m), 4) if m.jump_height_flight_m is not None else None),
            ("cmj_rsimod", round(float(m.rsimod_impulse), 4)),
            ("cmj_peak_power", round(float(m.peak_power_w), 2)),
            ("cmj_peak_power_rel", round(float(m.peak_relative_power_w_per_kg), 2)),
            ("cmj_peak_velocity", round(float(m.peak_velocity_m_per_s), 4)),
            ("cmj_peak_braking_force", round(float(m.peak_braking_force_n), 2)),
            ("cmj_time_to_takeoff", round(float(m.contraction_time_s), 4)),
            ("cmj_flight_time", round(float(m.flight_time_s), 4) if m.flight_time_s is not None else None),
            ("cmj_body_mass", round(float(v.body_mass_kg), 2)),
        ]

        for metric_name, val in metric_mappings:
            if val is None:
                continue
            reg_entry = METRIC_REGISTRY[metric_name]
            rows.append({
                "session_id": session_id,
                "date": session_date,
                "athlete_id": athlete_id,
                "test": TestType.CMJ.value,
                "trial": trial_num,
                "metric": metric_name,
                "value": val,
                "unit": reg_entry.unit,
                "side": LimbSide.BOTH.value,
                "valid": is_valid,
                "flags": flag_str,
                "block": "testing_battery",
            })

    df = pd.DataFrame(rows, columns=SESSIONS_COLUMNS)
    return df


def run_cli(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for running the jump analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="VALD-style Countermovement Jump Analysis Engine & Sessions Exporter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input CSV file or directory containing trial CSV files.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/processed/sessions.csv"),
        help="Path for exported sessions.csv adhering to shared contract.",
    )
    parser.add_argument(
        "--date", "-d",
        type=str,
        default=datetime.date.today().isoformat(),
        help="Session date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Optional path to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Preserve raw athlete IDs instead of de-identifying them.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input.exists():
        logger.error("Input path does not exist: %s", args.input)
        return 1

    # Load configuration
    cfg = PipelineConfig.load(args.config) if args.config else PipelineConfig.load()

    # Collect target files
    if args.input.is_file():
        trial_files = [args.input]
    else:
        trial_files = sorted(args.input.glob("*.csv"))

    if not trial_files:
        logger.error("No CSV trial files found in %s", args.input)
        return 1

    logger.info("Found %d trial file(s) to process.", len(trial_files))

    pipeline_results: list[dict] = []
    for idx, f in enumerate(trial_files, start=1):
        try:
            res = run_pipeline_on_csv(f, config=cfg)
            pipeline_results.append({
                "athlete_id": res.athlete_id,
                "trial_num": 1,  # Single-trial files
                "events": res.events,
                "metrics": res.metrics,
                "validated": res.validated,
            })
            logger.info("  Processed: %s -> Jump Height = %.2f cm, Peak Power = %.1f W/kg",
                        f.name, res.metrics.jump_height_impulse_m * 100, res.metrics.peak_relative_power_w_per_kg)
        except Exception as exc:
            logger.warning("  Failed to process %s: %s", f.name, exc)

    if not pipeline_results:
        logger.error("No trials were successfully processed.")
        return 1

    # Build sessions.csv
    df_sessions = build_sessions_dataframe(
        pipeline_results,
        session_date=args.date,
        anonymize=not args.no_anonymize,
    )

    # Validate against shared contract
    is_valid, errors = validate_sessions_df(df_sessions, strict=True)
    if not is_valid:
        logger.error("Generated sessions DataFrame violates shared contract:")
        for err in errors:
            logger.error("  - %s", err)
        return 2

    # Export to disk
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_sessions.to_csv(args.output, index=False)
    logger.info("Successfully exported %d session records to %s", len(df_sessions), args.output.resolve())
    logger.info("Contract validation: PASSED (strict=True)")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())
