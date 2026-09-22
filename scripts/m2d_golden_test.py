"""M2d: Golden test — compare Python pipeline against reference workbook values.

IMPORTANT — UNVERIFIED REFERENCE:
    The reference values in data/raw/reference/reference_expected.json are
    CACHED Excel formula outputs from CMJ_Force_Time_Analysis.xlsx, following
    Chavda et al. (2018). Their provenance and correctness are UNVERIFIED.
    Any mismatch is investigated in BOTH directions. A divergence is not
    automatically assumed to be a bug in the Python; it may reflect a genuine
    difference in algorithm (e.g., different landing threshold, integration
    start offset, or baseline window).

FAILURE POLICY:
    This script FAILS with a non-zero exit code if any divergence is found
    that is not classified as "understood and explained" in EXPLAINED_DIVERGENCES
    below. Once a divergence is understood, it is documented here and the script
    converts it to a passing comparison with an explanation note.

DATA PROVENANCE:
    Input:  data/raw/reference/reference_trial.csv   (gitignored)
            data/raw/reference/reference_expected.json (gitignored)
    Output: reports/figures/cmj_golden_test_comparison.png  (PUBLIC; no raw values)
            data/raw/golden_test_results.json               (PRIVATE; gitignored)

Usage:
    uv run python scripts/m2d_golden_test.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p1_nmt_pipeline.config import PipelineConfig, ThresholdConfig
from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_csv

logging.basicConfig(level=logging.WARNING)

REFERENCE_CSV = Path("data/raw/reference/reference_trial.csv")
REFERENCE_JSON = Path("data/raw/reference/reference_expected.json")
FIGURE_DIR = Path("reports/figures")
PRIVATE_OUT = Path("data/raw/golden_test_results.json")
STANDARD_G = 9.80665

# ── EXPLAINED DIVERGENCES ───────────────────────────────────────────────────
# Document every understood divergence here. Format:
#   metric_key: {
#       "ref": reference value,
#       "tolerance_abs": absolute tolerance to pass,
#       "tolerance_pct": relative tolerance to pass (use one or the other),
#       "explanation": plain-language cause of divergence,
#   }
# A metric listed here will be reported as "understood" rather than "unexplained".
# DO NOT add entries without a genuine explanation — the script will fail if
# the divergence is not within the stated tolerance even with an entry here.
EXPLAINED_DIVERGENCES: dict[str, dict] = {
    # ── Contraction time: −14 ms ──────────────────────────────────────────────
    # Reference onset time: row 1250 from start of CSV (absolute t=3.6909 s per JSON).
    # Python onset time: t=3.7049 s (+14 ms later).
    #
    # Root cause: The reference CSV contains 107 irregular timestamp intervals
    # (1.8% of samples). Our validator resamples the signal to a uniform 2000 Hz
    # grid. Resampling redistributes samples and shifts the effective time axis
    # relative to the original row indices used by the Excel workbook. This causes
    # a ~14 ms (28-sample) discrepancy in onset time, propagating to contraction time.
    #
    # Direction: Our contraction time is 14 ms SHORTER than the reference. The reference
    # detects onset earlier in its original (non-resampled) row space. This is a
    # data-handling boundary choice, not a formula error in either direction.
    #
    # We allow ±20 ms (40 samples at 2000 Hz) for this resampling-induced offset.
    "Contraction time (s)": {
        "tolerance_abs": 0.020,
        "explanation": (
            "The reference CSV has 107 irregular timestamp intervals. The reference workbook "
            "uses original raw row indices (no resampling). Our validator resamples to uniform "
            "2000 Hz, shifting the effective time axis by ~14 ms (28 samples), which propagates "
            "directly to onset detection and contraction time. No formula error on either side."
        ),
    },

    # ── Peak propulsion force: −9.3 N (−0.78%) ───────────────────────────────
    # Reference value: 1185.3 N (from raw, unfiltered signal; verified: raw peak = 1185.2 N).
    # Python filtered peak: 1178.6 N. Our metrics.py reports from propulsion region of
    # corr_force (filtered). The 30 Hz Butterworth filter attenuates sharp force transients
    # slightly — here by ~0.6-0.8%. The propulsion peak is a broader peak (~50 ms duration)
    # so attenuation is small but non-zero.
    #
    # Direction: Python is LOWER than reference because we use filtered force.
    # Using raw force for propulsion peak would match reference within noise, but for
    # velocity integration consistency we keep propulsion metrics on filtered force.
    # The 9 N error (0.78%) is within accepted measurement uncertainty for 1-trial data.
    "Peak propulsion force (N)": {
        "tolerance_abs": 12.0,
        "explanation": (
            "Reference uses the unfiltered raw signal (verified: raw peak = 1185.2 N ≈ ref 1185.3 N). "
            "Python uses the 30 Hz zero-phase Butterworth filtered signal, which attenuates the "
            "propulsion peak by ~9 N (0.78%). This is a deliberate algorithm choice: filtered force "
            "is used for all integration-based metrics (velocity, impulse, power) for consistency. "
            "9 N is within single-trial measurement uncertainty."
        ),
    },
}

# ── CONFIG TO MATCH REFERENCE WORKBOOK SETTINGS ─────────────────────────────
# The reference workbook uses:
#   - Baseline window: rows 2-1001 → 1000 samples at 2000 Hz = 0.5 s
#   - Onset threshold: BW - 5*SD (matches onset_sd_multiplier=5.0)
#   - Integration start: 60 ms before onset (not 30 ms)
#   - Takeoff threshold: 10 N (matches our default)
#   - Takeoff sustain: 5 samples (matches our default)
#   - Landing threshold: 50 N (differs from our default of 10 N)
#   - Landing sustain: 10 samples (at 2000 Hz = 5 ms; our default = 2.5 ms)
REFERENCE_CONFIG = PipelineConfig.load()
# Override to match reference workbook
import dataclasses
REFERENCE_CONFIG = dataclasses.replace(
    REFERENCE_CONFIG,
    thresholds=dataclasses.replace(
        REFERENCE_CONFIG.thresholds,
        onset_sd_multiplier=5.0,
        quiet_window_s=0.5,
        integration_lead_s=0.060,     # reference: 60 ms before onset
        takeoff_threshold_n=10.0,     # reference: 10 N
        landing_threshold_n=50.0,     # reference: 50 N (documented in workbook)
        landing_consecutive_ms=5.0,   # reference: 10 samples at 2000 Hz = 5 ms
        takeoff_consecutive_ms=2.5,   # reference: 5 samples at 2000 Hz = 2.5 ms
        subsample_takeoff=False,      # reference uses raw sample, no sub-sample
    ),
)


def run_golden_test() -> tuple[list[dict], bool]:
    """Run the pipeline against the reference and collect comparison rows.

    Returns
    -------
    rows : list[dict]
        One row per compared metric.
    all_pass : bool
        True only if every divergence is understood and within tolerance.
    """
    if not REFERENCE_CSV.exists():
        print("ERROR: Reference trial CSV not found. Run import_cmj_xlsx.py first.")
        sys.exit(1)

    with open(REFERENCE_JSON) as f:
        ref_data = json.load(f)

    ref_results = ref_data.get("results", {})
    ref_params = ref_data.get("parameters", {})

    # Run pipeline
    result = run_pipeline_on_csv(
        REFERENCE_CSV,
        config=REFERENCE_CONFIG,
        body_mass_kg=None,  # estimate from baseline, as workbook does
    )
    m = result.metrics
    ev = result.events

    # Build comparison table
    # Each entry: (metric_label, ref_key, python_value, ref_value, tol_abs, tol_pct)
    comparisons = [
        # Body mechanics
        ("Body weight (N)", "Bodyweight (N)",
         m.body_weight_n, ref_results.get("Bodyweight (N)"), 5.0, None),
        ("Body mass (kg)", "Mass (kg)",
         m.body_mass_kg, ref_results.get("Mass (kg)"), 0.5, None),

        # Jump height — primary metric
        ("h_impulse (m) [velocity method]", "Jump Height – Velocity Method (m)",
         m.jump_height_impulse_m, ref_results.get("Jump Height \u2013 Velocity Method (m)"),
         0.010, None),
        ("h_flight (m) [flight-time method]", "Jump Height – Flight-Time Method (m)",
         m.jump_height_flight_m, ref_results.get("Jump Height \u2013 Flight-Time Method (m)"),
         0.005, None),

        # Temporal
        ("Flight time (s)", "Flight Time (s)",
         m.flight_time_s, ref_results.get("Flight Time (s)"), 0.001, None),
        ("Contraction time (s)", "Total Contraction Time (s)",
         m.contraction_time_s, ref_results.get("Total Contraction Time (s)"), 0.005, None),

        # Force peaks
        ("Peak propulsion force (N)", "Peak Force – absolute (N)",
         m.peak_propulsion_force_n, ref_results.get("Peak Force \u2013 absolute (N)"), 5.0, None),
        ("Peak landing force (N)", "Peak Landing Force (N)",
         m.peak_landing_force_n, ref_results.get("Peak Landing Force (N)"), 10.0, None),

        # Power
        ("Peak power (W)", "Peak Power (W)",
         m.peak_power_w, ref_results.get("Peak Power (W)"), 50.0, None),

        # RSImod
        ("RSImod (m/s)", "RSI-modified (Jump Height / Contraction Time)",
         m.rsimod_impulse, ref_results.get("RSI-modified (Jump Height / Contraction Time)"),
         0.020, None),
    ]

    rows = []
    all_pass = True

    for label, ref_key, py_val, ref_val, tol_abs, tol_pct in comparisons:
        if ref_val is None:
            status = "REF_MISSING"
            diff_abs = float("nan")
            diff_pct = float("nan")
            within_tol = True  # can't fail on absent reference
        elif py_val is None or np.isnan(py_val):
            status = "PYTHON_NAN"
            diff_abs = float("nan")
            diff_pct = float("nan")
            within_tol = False
            all_pass = False
        else:
            diff_abs = py_val - ref_val
            diff_pct = (diff_abs / ref_val * 100.0) if ref_val != 0 else float("nan")
            if tol_abs is not None:
                within_tol = abs(diff_abs) <= tol_abs
            else:
                within_tol = abs(diff_pct) <= (tol_pct or 5.0)

            if not within_tol:
                if label in EXPLAINED_DIVERGENCES:
                    ed = EXPLAINED_DIVERGENCES[label]
                    # Check if within explained tolerance
                    exp_tol = ed.get("tolerance_abs", float("inf"))
                    if abs(diff_abs) <= exp_tol:
                        status = f"UNDERSTOOD: {ed['explanation']}"
                        within_tol = True
                    else:
                        status = "UNEXPLAINED_DIVERGENCE"
                        all_pass = False
                else:
                    status = "UNEXPLAINED_DIVERGENCE"
                    all_pass = False
            else:
                status = "PASS"

        rows.append(
            {
                "metric": label,
                "python_value": round(py_val, 5) if py_val is not None and not np.isnan(py_val) else py_val,
                "reference_value": round(ref_val, 5) if ref_val is not None else None,
                "diff_abs": round(diff_abs, 5) if not np.isnan(diff_abs) else diff_abs,
                "diff_pct": round(diff_pct, 2) if not np.isnan(diff_pct) else diff_pct,
                "tolerance_abs": tol_abs,
                "status": status,
            }
        )

    return rows, all_pass


def print_golden_table(rows: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("M2d GOLDEN TEST  —  Python Pipeline vs Reference Workbook (UNVERIFIED)")
    print("Reference source: CMJ_Force_Time_Analysis.xlsx (Chavda et al. 2018 approach)")
    print("NOTE: Reference values are UNVERIFIED cached Excel outputs. Divergences")
    print("      investigated in BOTH directions. See EXPLAINED_DIVERGENCES in script.")
    print("=" * 100)
    header = f"{'Metric':<50} {'Python':>10} {'Reference':>12} {'Diff':>10} {'Diff%':>7} {'Status'}"
    print(header)
    print("-" * 100)
    for r in rows:
        py = r["python_value"]
        ref = r["reference_value"]
        diff = r["diff_abs"]
        pct = r["diff_pct"]
        py_s = f"{py:.5f}" if py is not None and not (isinstance(py, float) and np.isnan(py)) else "NaN"
        ref_s = f"{ref:.5f}" if ref is not None else "N/A"
        diff_s = f"{diff:+.5f}" if not (isinstance(diff, float) and np.isnan(diff)) else "NaN"
        pct_s = f"{pct:+.2f}%" if not (isinstance(pct, float) and np.isnan(pct)) else "NaN"
        print(f"{r['metric']:<50} {py_s:>10} {ref_s:>12} {diff_s:>10} {pct_s:>7}  {r['status']}")
    print("=" * 100)


def save_figure(rows: list[dict]) -> None:
    """Save a public comparison bar chart (no raw values from data)."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Only show metrics where both values are available and not NaN
    plotable = [
        r for r in rows
        if r["reference_value"] is not None
        and r["python_value"] is not None
        and not (isinstance(r["diff_pct"], float) and np.isnan(r["diff_pct"]))
    ]

    if not plotable:
        return

    labels = [r["metric"] for r in plotable]
    diffs_pct = [r["diff_pct"] for r in plotable]
    colors = [
        "#D65F5F" if "UNEXPLAINED" in str(r["status"])
        else "#59A14F" if r["status"] == "PASS"
        else "#F28E2B"
        for r in plotable
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(labels))
    bars = ax.barh(y, diffs_pct, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("(Python − Reference) / Reference × 100 %")
    ax.set_title(
        "M2d Golden Test: % Difference from Reference Workbook\n"
        "(Reference values are UNVERIFIED cached Excel outputs — Chavda et al. 2018)",
        fontsize=10,
    )
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#59A14F", label="PASS"),
        Patch(facecolor="#F28E2B", label="Understood divergence"),
        Patch(facecolor="#D65F5F", label="UNEXPLAINED DIVERGENCE"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    plt.tight_layout()
    fig_path = FIGURE_DIR / "cmj_golden_test_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


def main() -> None:
    rows, all_pass = run_golden_test()
    print_golden_table(rows)
    save_figure(rows)

    # Save private results (includes actual numeric values)
    PRIVATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PRIVATE_OUT, "w") as f:
        json.dump({"rows": rows, "all_pass": all_pass}, f, indent=2, default=str)

    unexplained = [r for r in rows if "UNEXPLAINED" in str(r["status"])]
    if unexplained:
        print(
            f"\n{'!'*80}\n"
            f"GOLDEN TEST FAILED: {len(unexplained)} unexplained divergence(s) detected.\n"
            f"Each divergence must be investigated, documented in EXPLAINED_DIVERGENCES\n"
            f"in {Path(__file__).name}, and re-run before this script can pass.\n"
            f"{'!'*80}"
        )
        sys.exit(1)
    else:
        print("\nGolden test PASSED (all divergences accounted for).")
        sys.exit(0)


if __name__ == "__main__":
    main()
