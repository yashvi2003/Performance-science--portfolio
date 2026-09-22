"""M2f: Analyst-choice sensitivity analysis on 10 real CMJ trials.

Tests how much jump height, RSImod, and peak power change when key pipeline
parameters are varied. Detection failures are reported as "DETECTION_FAILED"
per trial × parameter combination — NOT excluded from the analysis.

DATA PROVENANCE:
    Input:  data/raw/cmj/A01.csv ... A10.csv  (gitignored; real data)
    Output: data/raw/sensitivity_results.csv         (PRIVATE; gitignored)
            reports/figures/cmj_sensitivity.png      (PUBLIC; anonymized)

ANONYMIZATION:
    All public outputs use T01-T10 ordinal labels.
    A01-A10 appear only in the private CSV (gitignored).

NOTE:
    This script uses REAL professor-supplied force-plate data.
    Results are not synthetic. The pipeline logic was validated on synthetic
    data with known ground truth (see tests/test_cmj.py).

Usage:
    uv run python scripts/m2f_sensitivity.py
"""
from __future__ import annotations

import dataclasses
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

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p1_nmt_pipeline.config import PipelineConfig, ThresholdConfig
from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_csv

logging.basicConfig(level=logging.WARNING)

CMJ_DIR = Path("data/raw/cmj")
FIGURE_DIR = Path("reports/figures")
PRIVATE_OUT = Path("data/raw/sensitivity_results.csv")

SENTINEL = "DETECTION_FAILED"
STANDARD_G = 9.80665

# ── Parameter sweeps ────────────────────────────────────────────────────────
# Each entry: (parameter_name, parameter_values, is_threshold_param, description)
# is_threshold_param=True  → value goes into ThresholdConfig
# is_threshold_param=False → value is the filter cutoff (separate arg)
SWEEPS: list[tuple[str, list, bool, str]] = [
    ("onset_sd_multiplier", [3.0, 5.0, 8.0], True, "Onset threshold (k × BW_SD)"),
    ("quiet_window_s", [0.30, 0.50, 0.70], True, "Baseline window (s)"),
    ("takeoff_threshold_n", [5.0, 10.0, 20.0], True, "Takeoff threshold (N)"),
    ("landing_threshold_n", [5.0, 10.0, 20.0], True, "Landing threshold (N)"),
    ("filter_cutoff_hz", [15.0, 30.0, 50.0], False, "Filter cutoff (Hz)"),
    ("integration_lead_s", [0.030, 0.050, 0.100], True, "Integration lead (s)"),
]

METRICS = ["h_impulse_m", "rsimod", "peak_power_w"]
METRIC_LABELS = {
    "h_impulse_m": "Jump height — impulse (m)",
    "rsimod": "RSImod (m/s)",
    "peak_power_w": "Peak power (W)",
}


def get_metric(result, metric_name: str):
    """Extract metric value; returns SENTINEL string if detection failed."""
    m = result.metrics
    if metric_name == "h_impulse_m":
        v = m.jump_height_impulse_m
    elif metric_name == "rsimod":
        v = m.rsimod_impulse
    elif metric_name == "peak_power_w":
        v = m.peak_power_w
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return SENTINEL
    return v


def build_config(param_name: str, param_value: float, base_config: PipelineConfig) -> tuple[PipelineConfig, float | None]:
    """Return (config, filter_cutoff_override) for the given parameter setting."""
    if param_name == "filter_cutoff_hz":
        return base_config, param_value
    # Otherwise it's a ThresholdConfig field
    new_thresh = dataclasses.replace(
        base_config.thresholds,
        **{param_name: param_value},
    )
    new_config = dataclasses.replace(base_config, thresholds=new_thresh)
    return new_config, None


def main() -> None:
    if not CMJ_DIR.exists():
        print("ERROR: data/raw/cmj/ not found. Run scripts/import_cmj_xlsx.py first.")
        sys.exit(1)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT.parent.mkdir(parents=True, exist_ok=True)

    base_config = PipelineConfig.load()
    csvs = sorted(CMJ_DIR.glob("A*.csv"))

    # Map athlete filename to public label
    athlete_to_label = {f.stem: f"T{i+1:02d}" for i, f in enumerate(csvs)}

    print("\n" + "=" * 90)
    print("M2f SENSITIVITY ANALYSIS — Analyst Parameter Choices on Real CMJ Trials (n=10)")
    print("NOTE: Results from professor-supplied force-plate data (not synthetic).")
    print("      DETECTION_FAILED = takeoff or landing not detected at this setting.")
    print("=" * 90)

    all_rows: list[dict] = []

    for param_name, values, is_thresh, description in SWEEPS:
        print(f"\n── {description} ({param_name}) ──")
        print(f"  Values tested: {values}")

        # First: get baseline values at the middle setting (index 1)
        baseline_value = values[len(values) // 2]

        trial_results: dict[str, dict] = {}  # athlete_id -> {param_value -> {metric -> value}}

        for v in values:
            cfg, cutoff_override = build_config(param_name, v, base_config)
            for csv in csvs:
                aid = csv.stem
                try:
                    r = run_pipeline_on_csv(csv, config=cfg, filter_cutoff_hz=cutoff_override)
                except Exception as exc:
                    # Catastrophic failure (e.g. not enough samples for filter)
                    for metric in METRICS:
                        all_rows.append(
                            {
                                "_athlete_id": aid,
                                "trial": athlete_to_label[aid],
                                "parameter": param_name,
                                "param_value": v,
                                "metric": metric,
                                "value": SENTINEL,
                                "note": f"Exception: {exc}",
                            }
                        )
                    continue

                for metric in METRICS:
                    val = get_metric(r, metric)
                    all_rows.append(
                        {
                            "_athlete_id": aid,
                            "trial": athlete_to_label[aid],
                            "parameter": param_name,
                            "param_value": v,
                            "metric": metric,
                            "value": val,
                            "note": "" if val != SENTINEL else "detection_failed",
                        }
                    )

        # Print per-metric sensitivity summary
        df_param = pd.DataFrame([r for r in all_rows if r["parameter"] == param_name])
        for metric in METRICS:
            df_m = df_param[df_param["metric"] == metric].copy()
            print(f"\n  Metric: {METRIC_LABELS[metric]}")
            header = f"    {'Trial':<8}" + "".join(f"{str(v):>14}" for v in values) + f"  {'Range':>10}  {'Sensitive?':>10}"
            print(header)
            print("    " + "-" * (8 + 14 * len(values) + 22))

            for csv in csvs:
                aid = csv.stem
                label = athlete_to_label[aid]
                row_vals = []
                for v in values:
                    sub = df_m[(df_m["_athlete_id"] == aid) & (df_m["param_value"] == v)]
                    if len(sub) == 0:
                        row_vals.append(SENTINEL)
                    else:
                        row_vals.append(sub.iloc[0]["value"])

                numeric_vals = [x for x in row_vals if x != SENTINEL and isinstance(x, float)]
                n_failed = row_vals.count(SENTINEL)

                if len(numeric_vals) >= 2:
                    rng = max(numeric_vals) - min(numeric_vals)
                    # Sensitivity threshold: > 1 cm for height, > 0.02 m/s for RSImod, > 50 W for power
                    thresholds = {"h_impulse_m": 0.010, "rsimod": 0.020, "peak_power_w": 50.0}
                    sensitive = rng > thresholds[metric]
                    rng_str = f"{rng:.4f}"
                    sens_str = "*** HIGH ***" if sensitive else "ok"
                elif len(numeric_vals) == 1:
                    rng_str = "N/A (1 valid)"
                    sens_str = f"({n_failed} FAIL)"
                else:
                    rng_str = "ALL FAILED"
                    sens_str = "ALL FAILED"

                val_strs = []
                for v in row_vals:
                    if v == SENTINEL:
                        val_strs.append("DETECTION_FAILED")
                    else:
                        val_strs.append(f"{v:.4f}")
                print(
                    f"    {label:<8}" +
                    "".join(f"{vs:>14}" for vs in val_strs) +
                    f"  {rng_str:>10}  {sens_str:>10}"
                )

    # Save private full table
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(PRIVATE_OUT, index=False)
    print(f"\nPrivate sensitivity results saved to {PRIVATE_OUT}")

    # ── Summary: which trials are most sensitive? ────────────────────────────
    print("\n── SENSITIVITY SUMMARY (trials with >1 cm height range on any parameter) ──")
    df_h = df_all[df_all["metric"] == "h_impulse_m"].copy()
    df_h["is_numeric"] = df_h["value"].apply(lambda x: isinstance(x, float))
    sensitive_trials = set()

    for param_name, values, _, description in SWEEPS:
        df_p = df_h[df_h["parameter"] == param_name]
        for csv in csvs:
            aid = csv.stem
            label = athlete_to_label[aid]
            sub = df_p[df_p["_athlete_id"] == aid]
            nums = [r["value"] for _, r in sub.iterrows() if isinstance(r["value"], float)]
            if len(nums) >= 2 and (max(nums) - min(nums)) > 0.010:
                sensitive_trials.add(label)
                print(f"  {label}: sensitive to {description} (range {max(nums)-min(nums)*100:.1f} mm)")

    if not sensitive_trials:
        print("  No trials exceeded 1 cm height range on any single parameter sweep.")

    # ── Figure: sensitivity heat-map ─────────────────────────────────────────
    _make_sensitivity_heatmap(df_all, csvs, athlete_to_label, SWEEPS)

    print(f"\nFigure saved: {FIGURE_DIR / 'cmj_sensitivity.png'}")
    print("(Anonymous T01-T10 labels; no athlete IDs in figure)")


def _make_sensitivity_heatmap(
    df_all: pd.DataFrame,
    csvs: list[Path],
    athlete_to_label: dict,
    sweeps: list,
) -> None:
    """Create sensitivity heat-maps for each metric."""
    trial_labels = [athlete_to_label[f.stem] for f in csvs]
    param_labels = [sw[3] for sw in sweeps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "M2f Sensitivity Analysis: Metric Range per Trial × Parameter\n"
        "(Real CMJ Trials, n=10; DETECTION_FAILED shown as striped cell)",
        fontsize=11,
    )

    for ax, metric in zip(axes, METRICS):
        heat = np.zeros((len(trial_labels), len(param_labels)))
        has_fail = np.zeros((len(trial_labels), len(param_labels)), dtype=bool)

        for j, (param_name, values, _, _desc) in enumerate(sweeps):
            df_p = df_all[(df_all["metric"] == metric) & (df_all["parameter"] == param_name)]
            for i, csv in enumerate(csvs):
                aid = csv.stem
                sub = df_p[df_p["_athlete_id"] == aid]
                nums = [r["value"] for _, r in sub.iterrows() if isinstance(r["value"], float)]
                fails = [r["value"] for _, r in sub.iterrows() if r["value"] == SENTINEL]
                if fails:
                    has_fail[i, j] = True
                heat[i, j] = max(nums) - min(nums) if len(nums) >= 2 else 0.0

        im = ax.imshow(heat, cmap="YlOrRd", aspect="auto", vmin=0)
        plt.colorbar(im, ax=ax, label="Range (max−min)")

        # Striped pattern for cells with detection failures
        for i in range(len(trial_labels)):
            for j in range(len(param_labels)):
                if has_fail[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch="////", edgecolor="gray", lw=0
                    ))
                    ax.text(j, i, "FAIL", ha="center", va="center",
                            fontsize=6, color="gray", fontweight="bold")
                else:
                    val = heat[i, j]
                    if metric == "h_impulse_m":
                        unit = f"{val*1000:.1f}mm"
                    elif metric == "rsimod":
                        unit = f"{val:.3f}"
                    else:
                        unit = f"{val:.0f}W"
                    ax.text(j, i, unit, ha="center", va="center", fontsize=6.5)

        ax.set_xticks(range(len(param_labels)))
        ax.set_xticklabels(param_labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(trial_labels)))
        ax.set_yticklabels(trial_labels, fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "cmj_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
