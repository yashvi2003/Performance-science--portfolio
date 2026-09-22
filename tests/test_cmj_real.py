"""Real-data tests for M2c–M2f: professor-supplied CMJ trials.

IMPORTANT:
    These tests operate on data/raw/cmj/ (gitignored).
    They are automatically SKIPPED if that directory does not exist,
    so CI and fresh checkouts are unaffected.

    Marked @pytest.mark.real_data. Run with:
        uv run pytest tests/test_cmj_real.py -v -m real_data

    DO NOT read .xlsx, data/private, or anything not in data/raw/.
    DO NOT print or assert on individual athlete IDs (A01-A10) in
    ways that could appear in public CI logs.

ANONYMIZATION:
    Assertions are made on aggregate values or per-trial ordinal positions.
    No athlete identifier is embedded in assert messages where it would
    appear in public test output.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

logger = logging.getLogger(__name__)

REAL_DATA_DIR = Path("data/raw/cmj")
REFERENCE_DIR = Path("data/raw/reference")
REAL_DATA_AVAILABLE = REAL_DATA_DIR.exists() and len(list(REAL_DATA_DIR.glob("A*.csv"))) > 0

skip_if_no_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE,
    reason="Real data not available (data/raw/cmj/ missing or empty — run import_cmj_xlsx.py)",
)


@pytest.mark.real_data
@skip_if_no_data
def test_all_real_trials_parse_without_error():
    """All 10 CSVs must load and run through the full pipeline without exception."""
    from p1_nmt_pipeline.config import PipelineConfig
    from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_all

    config = PipelineConfig.load()
    results = run_pipeline_on_all(REAL_DATA_DIR, config=config)
    assert len(results) == 10, f"Expected 10 trial results, got {len(results)}"


@pytest.mark.real_data
@skip_if_no_data
def test_all_real_trials_have_plausible_mass():
    """Every trial's estimated body mass must be in [40, 150] kg."""
    from p1_nmt_pipeline.config import PipelineConfig
    from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_all

    config = PipelineConfig.load()
    results = run_pipeline_on_all(REAL_DATA_DIR, config=config)
    for i, r in enumerate(results):
        mass = r.metrics.body_mass_kg
        assert 40.0 <= mass <= 150.0, (
            f"Trial T{i+1:02d}: implausible estimated body mass {mass:.1f} kg "
            f"(expected 40-150 kg)"
        )


@pytest.mark.real_data
@skip_if_no_data
def test_all_real_trials_height_plausible_or_nan_with_flag():
    """Jump height by impulse must be in [0.10, 0.90] m, or NaN with a QA flag."""
    from p1_nmt_pipeline.config import PipelineConfig
    from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_all

    config = PipelineConfig.load()
    results = run_pipeline_on_all(REAL_DATA_DIR, config=config)
    for i, r in enumerate(results):
        h = r.metrics.jump_height_impulse_m
        if np.isnan(h):
            # Must have a QA flag explaining the failure
            all_flags = r.all_qa_flags
            assert any("failed" in f.lower() or "detection" in f.lower() for f in all_flags), (
                f"Trial T{i+1:02d}: NaN height but no detection-failure QA flag. "
                f"Flags: {all_flags}"
            )
        else:
            assert 0.10 <= h <= 0.90, (
                f"Trial T{i+1:02d}: implausible jump height {h:.4f} m "
                f"(expected 0.10-0.90 m)"
            )


@pytest.mark.real_data
@skip_if_no_data
def test_real_trials_negative_force_flagged():
    """Every trial with negative force samples must have a negative_force QA flag."""
    from p1_nmt_pipeline.config import PipelineConfig
    from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_all

    config = PipelineConfig.load()
    results = run_pipeline_on_all(REAL_DATA_DIR, config=config)
    for i, r in enumerate(results):
        n_neg = int(np.sum(r.validated.force_n < 0.0))
        if n_neg > 0:
            flags = r.validated.qa_flags
            assert any("negative_force" in f for f in flags), (
                f"Trial T{i+1:02d}: has {n_neg} negative force samples but no negative_force QA flag"
            )


@pytest.mark.real_data
@skip_if_no_data
def test_real_trials_irregular_timestamps_flagged():
    """Trials with irregular timestamps must have a resampled_irregular_dt QA flag."""
    from p1_nmt_pipeline.config import PipelineConfig
    from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_all
    import pandas as pd

    config = PipelineConfig.load()
    results = run_pipeline_on_all(REAL_DATA_DIR, config=config)
    manifest_path = REAL_DATA_DIR / "cmj_manifest.csv"
    if not manifest_path.exists():
        pytest.skip("cmj_manifest.csv not found")

    manifest = pd.read_csv(manifest_path)
    for i, r in enumerate(results):
        row = manifest.iloc[i] if i < len(manifest) else None
        if row is not None and row["n_irregular_dt"] > 0:
            flags = r.validated.qa_flags
            assert any("resampled_irregular" in f for f in flags), (
                f"Trial T{i+1:02d}: manifest shows {row['n_irregular_dt']} irregular timestamps "
                f"but no resampled_irregular_dt QA flag. Flags: {flags}"
            )


@pytest.mark.real_data
@skip_if_no_data
def test_golden_test_reference_plausible():
    """Reference trial must produce a jump height in a plausible range.

    This test is NOT a golden-test assertion on exact values.
    Exact golden-test comparisons live in scripts/m2d_golden_test.py,
    which fails loudly on unexplained divergences.
    """
    from p1_nmt_pipeline.config import PipelineConfig
    from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_csv

    ref_csv = REFERENCE_DIR / "reference_trial.csv"
    if not ref_csv.exists():
        pytest.skip("Reference trial CSV not found")

    config = PipelineConfig.load()
    result = run_pipeline_on_csv(ref_csv, config=config)
    h = result.metrics.jump_height_impulse_m

    # Plausibility check only — exact agreement is in m2d_golden_test.py
    assert np.isnan(h) or (0.10 <= h <= 0.60), (
        f"Reference trial jump height implausible: {h:.4f} m"
    )


@pytest.mark.real_data
@skip_if_no_data
def test_reference_expected_json_has_required_keys():
    """The reference JSON must have the key metrics we compare in M2d."""
    ref_json = REFERENCE_DIR / "reference_expected.json"
    if not ref_json.exists():
        pytest.skip("reference_expected.json not found")

    with open(ref_json) as f:
        data = json.load(f)

    required_keys = [
        "Jump Height \u2013 Velocity Method (m)",
        "Jump Height \u2013 Flight-Time Method (m)",
        "Flight Time (s)",
        "Total Contraction Time (s)",
        "Peak Power (W)",
    ]
    results = data.get("results", {})
    for key in required_keys:
        assert key in results, f"Missing required key in reference_expected.json: {key!r}"


@pytest.mark.real_data
@skip_if_no_data
def test_real_data_pipeline_does_not_commit_private_data():
    """Guard: confirm data/raw/ is not tracked by git (i.e. it is gitignored)."""
    import subprocess
    result = subprocess.run(
        ["git", "ls-files", "data/raw"],
        capture_output=True,
        text=True,
    )
    tracked = result.stdout.strip()
    assert tracked == "", (
        f"data/raw/ contains git-tracked files! This must never happen.\n"
        f"Tracked files: {tracked}"
    )
