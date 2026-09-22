"""M2c: Data-quality report for all 10 professor-supplied CMJ trials.

DATA PROVENANCE:
    Input:  data/raw/cmj/A01.csv ... A10.csv  (gitignored; never committed)
    Output: data/raw/cmj_quality_report.csv   (gitignored; private full table)
            reports/figures/cmj_data_quality_summary.png  (PUBLIC; anonymized)

ANONYMIZATION:
    Internal athlete IDs (A01-A10) appear only in data/raw/ (gitignored).
    All public-facing outputs use anonymous ordinal labels T01-T10 and never
    contain A01-A10 or any information that could identify an athlete.

NOTE ON SYNTHETIC DATA:
    This script operates on REAL data supplied by the professor.
    Results reflect real measurement conditions and are NOT synthetic.
    The pipeline was validated on synthetic data (see tests/test_cmj.py).

Usage:
    uv run python scripts/m2c_data_quality_report.py
"""
from __future__ import annotations

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
import pandas as pd

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_all
from p1_nmt_pipeline.config import PipelineConfig

matplotlib.use("Agg")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

STANDARD_G = 9.80665
CMJ_DIR = Path("data/raw/cmj")
FIGURE_DIR = Path("reports/figures")
PRIVATE_OUT = Path("data/raw/cmj_quality_report.csv")


def main() -> None:
    if not CMJ_DIR.exists():
        print("ERROR: data/raw/cmj/ not found. Run scripts/import_cmj_xlsx.py first.")
        sys.exit(1)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig.load()
    results = run_pipeline_on_all(CMJ_DIR, config=config)

    rows = []
    for i, r in enumerate(results):
        # Public ordinal label — never expose r.athlete_id in public outputs
        trial_label = f"T{i+1:02d}"

        ev = r.events
        m = r.metrics
        v = r.validated

        # Quiet-phase statistics
        q_start = ev.quiet_start_idx
        q_end = ev.quiet_end_idx
        quiet_duration_s = float(v.time_s[q_end] - v.time_s[q_start])
        quiet_sd_n = float(ev.quiet_bw_std_n)
        quiet_mean_n = float(ev.quiet_bw_mean_n)

        # Estimated body mass (always estimated here — no provided mass)
        est_mass_kg = float(ev.body_mass_kg)
        mass_plausible = 40.0 <= est_mass_kg <= 150.0

        # Timestamp irregularity (from validator QA flags)
        n_irregular = 0
        for flag in v.qa_flags:
            if flag.startswith("resampled_irregular_dt_"):
                try:
                    n_irregular = int(flag.split("_")[3])
                except (IndexError, ValueError):
                    pass

        # Negative force count and minimum
        raw_neg = int(np.sum(v.force_n < 0.0))
        raw_min_n = float(np.min(v.force_n))

        # Flight-phase force offset (always estimated, even if correction not applied)
        flight_offset_n = float(ev.flight_offset_n)

        # Detection success
        to_ok = ev.takeoff_idx is not None
        land_ok = ev.landing_idx is not None

        # QA flags (combined)
        all_flags = list(v.qa_flags) + list(ev.qa_warnings)
        qa_str = "; ".join(all_flags) if all_flags else "none"

        rows.append(
            {
                # Private column — only in gitignored CSV
                "_athlete_id": r.athlete_id,
                # Public label
                "trial": trial_label,
                "quiet_duration_s": round(quiet_duration_s, 3),
                "quiet_sd_n": round(quiet_sd_n, 2),
                "quiet_bw_n": round(quiet_mean_n, 1),
                "est_mass_kg": round(est_mass_kg, 2),
                "mass_plausible": mass_plausible,
                "n_irregular_dt": n_irregular,
                "n_negative_force": raw_neg,
                "force_min_n": round(raw_min_n, 1),
                "flight_offset_est_n": round(flight_offset_n, 2),
                "takeoff_detected": to_ok,
                "landing_detected": land_ok,
                "h_impulse_m": round(m.jump_height_impulse_m, 4) if to_ok else float("nan"),
                "h_flight_m": round(m.jump_height_flight_m, 4) if land_ok else float("nan"),
                "rsimod": round(m.rsimod_impulse, 4) if to_ok else float("nan"),
                "peak_power_w": round(m.peak_power_w, 1) if to_ok else float("nan"),
                "contraction_time_s": round(m.contraction_time_s, 4) if to_ok else float("nan"),
                "qa_flags": qa_str,
            }
        )

    df = pd.DataFrame(rows)

    # Save private full table (includes _athlete_id)
    PRIVATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRIVATE_OUT, index=False)
    logger.info("Private quality report saved to %s", PRIVATE_OUT)

    # ── Print anonymized summary (no athlete IDs) ────────────────────────────
    pub_cols = [
        "trial", "quiet_duration_s", "quiet_sd_n", "est_mass_kg",
        "n_irregular_dt", "n_negative_force", "force_min_n",
        "flight_offset_est_n", "takeoff_detected", "landing_detected",
        "h_impulse_m", "h_flight_m", "rsimod", "peak_power_w",
        "contraction_time_s",
    ]
    pub_df = df[pub_cols]

    print("\n" + "=" * 80)
    print("M2c DATA-QUALITY REPORT -- Real CMJ Trials (n=10)")
    print("NOTE: Results from professor-supplied force-plate data (not synthetic).")
    print("      Athlete labels T01-T10 are anonymized ordinals.")
    print("=" * 80)
    print(pub_df.to_string(index=False))

    print("\n-- QA FLAGS --")
    for _, row in df.iterrows():
        flags = row["qa_flags"]
        if flags and flags != "none":
            print(f"  {row['trial']}: {flags}")

    print("\n-- PLAUSIBILITY CHECKS --")
    n_implausible = int((~df["mass_plausible"]).sum())
    n_no_takeoff = int((~df["takeoff_detected"]).sum())
    n_no_landing = int((~df["landing_detected"]).sum())
    print(f"  Implausible body mass (<40 or >150 kg): {n_implausible} trials")
    print(f"  Takeoff detection failed: {n_no_takeoff} trials")
    print(f"  Landing detection failed: {n_no_landing} trials")
    print(f"  Trials with negative force samples: {int((df['n_negative_force'] > 0).sum())}/10")

    # ── Figure: anonymized quality summary ──────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "CMJ Data-Quality Summary — Real Trials (n=10)\n"
        "(SYNTHETIC data used for method validation; these are real force-plate measurements)",
        fontsize=12,
    )
    trial_labels = df["trial"].tolist()
    x = np.arange(len(trial_labels))

    # Panel 1: quiet-window SD
    ax = axes[0, 0]
    bars = ax.bar(x, df["quiet_sd_n"], color="#4878CF")
    ax.axhline(15.0, color="red", ls="--", lw=1.2, label="Threshold (15 N)")
    ax.set_title("Quiet-Phase SD (N)")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=45, fontsize=8)
    ax.set_ylabel("SD (N)")
    ax.legend(fontsize=8)

    # Panel 2: Negative force sample count
    ax = axes[0, 1]
    ax.bar(x, df["n_negative_force"], color="#D65F5F")
    ax.set_title("Negative Force Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=45, fontsize=8)
    ax.set_ylabel("Count")

    # Panel 3: Flight-phase offset
    ax = axes[0, 2]
    ax.bar(x, df["flight_offset_est_n"], color="#59A14F")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Flight-Phase Offset Estimate (N)")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=45, fontsize=8)
    ax.set_ylabel("Offset (N)")

    # Panel 4: Jump height (impulse)
    ax = axes[1, 0]
    h_vals = df["h_impulse_m"].tolist()
    bar_colors = ["#4878CF" if not np.isnan(v) else "#AAAAAA" for v in h_vals]
    ax.bar(x, [v if not np.isnan(v) else 0 for v in h_vals], color=bar_colors)
    ax.set_title("Jump Height — Impulse (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=45, fontsize=8)
    ax.set_ylabel("h (m)")
    # Annotate failed detections
    for xi, v in zip(x, h_vals):
        if np.isnan(v):
            ax.text(xi, 0.01, "FAIL", ha="center", va="bottom", color="red", fontsize=7)

    # Panel 5: Estimated body mass
    ax = axes[1, 1]
    ax.bar(x, df["est_mass_kg"], color="#B07AA1")
    ax.axhline(40, color="red", ls="--", lw=1, label="Plausibility bounds")
    ax.axhline(150, color="red", ls="--", lw=1)
    ax.set_title("Estimated Body Mass (kg)")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=45, fontsize=8)
    ax.set_ylabel("Mass (kg)")
    ax.legend(fontsize=8)

    # Panel 6: Timestamp irregularities
    ax = axes[1, 2]
    ax.bar(x, df["n_irregular_dt"], color="#F28E2B")
    ax.set_title("Irregular Timestamp Count")
    ax.set_xticks(x)
    ax.set_xticklabels(trial_labels, rotation=45, fontsize=8)
    ax.set_ylabel("Count")

    plt.tight_layout()
    fig_path = FIGURE_DIR / "cmj_data_quality_summary.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {fig_path}")
    print("(Anonymous T01-T10 labels; no raw force traces or athlete IDs in figure)")


if __name__ == "__main__":
    main()
