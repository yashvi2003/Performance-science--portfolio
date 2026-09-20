"""Master reproduction script for the Neuromuscular Monitoring Toolkit.

Executes end-to-end generation, contract validation, and tests.
Usage:
    uv run python scripts/run_all.py
"""
from __future__ import annotations

import logging
from pathlib import Path
import subprocess
import sys

from shared.contract import validate_sessions_df, validate_wearable_daily_df
from shared.synth import make_cmj_trial, make_sessions_table, make_wearable_daily

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    data_synth_dir = root_dir / "data" / "synthetic"
    data_synth_dir.mkdir(parents=True, exist_ok=True)

    logger.info("1. Generating synthetic contract datasets...")
    sessions_df = make_sessions_table(n_athletes=10, n_weeks=8, random_seed=42)
    sessions_path = data_synth_dir / "sessions.csv"
    sessions_df.to_csv(sessions_path, index=False)
    logger.info("Saved %d rows to %s", len(sessions_df), sessions_path)

    wearable_df = make_wearable_daily(n_athletes=10, n_days=56, random_seed=42)
    wearable_path = data_synth_dir / "wearable_daily.csv"
    wearable_df.to_csv(wearable_path, index=False)
    logger.info("Saved %d rows to %s", len(wearable_df), wearable_path)

    logger.info("2. Validating contract compliance...")
    valid_sessions, sess_errs = validate_sessions_df(sessions_df)
    if not valid_sessions:
        logger.error("sessions.csv contract violations: %s", sess_errs)
        return 1
    logger.info("sessions.csv passed contract validation.")

    valid_wearable, wear_errs = validate_wearable_daily_df(wearable_df)
    if not valid_wearable:
        logger.error("wearable_daily.csv contract violations: %s", wear_errs)
        return 1
    logger.info("wearable_daily.csv passed contract validation.")

    logger.info("3. Generating sample force-time trials...")
    cmj_2000 = make_cmj_trial(fs_hz=2000.0, target_jump_height_m=0.38, asymmetry_index=0.08)
    cmj_2000.to_df().to_csv(data_synth_dir / "sample_cmj_2000hz.csv", index=False)

    cmj_250 = make_cmj_trial(fs_hz=250.0, target_jump_height_m=0.32)
    cmj_250.to_df().to_csv(data_synth_dir / "sample_cmj_250hz.csv", index=False)
    logger.info("Sample force traces written to %s", data_synth_dir)

    logger.info("4. Running test suite...")
    res = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=str(root_dir))
    if res.returncode != 0:
        logger.error("Test suite failed with exit code %d", res.returncode)
        return res.returncode

    logger.info("Reproduction run complete. All outputs verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
