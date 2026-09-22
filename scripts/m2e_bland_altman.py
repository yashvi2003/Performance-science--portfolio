"""M2e: Bland-Altman method comparison — flight-time vs impulse-momentum jump height.

WHAT THIS ANALYSIS IS AND IS NOT:
    This is a BETWEEN-ATHLETE method comparison (n=10, one trial per athlete).
    It shows whether the two height estimation methods agree across different people.
    It is NOT a within-athlete reliability estimate (which would require repeated
    trials from the same athlete under the same conditions). For within-athlete
    reliability (CV, ICC, SEM, MDC), see the open-data validation (M2g, D4).

    With n=10, confidence intervals for bias and limits of agreement are wide.
    These results should not be extrapolated to other populations or instruments.

DATA PROVENANCE:
    Input:  data/raw/cmj/A01.csv ... A10.csv  (gitignored; real data)
    Output: reports/figures/cmj_bland_altman.png  (PUBLIC; anonymized)

ANONYMIZATION:
    No athlete IDs appear in the figure or printed statistics.
    Point labels use T01–T10 ordinals.

NOTE ON SYNTHETIC DATA:
    This script uses REAL force-plate data. Results are NOT synthetic.
    The pipeline was validated on synthetic data (see tests/test_cmj.py).

Usage:
    uv run python scripts/m2e_bland_altman.py
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
from scipy import stats

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p1_nmt_pipeline.config import PipelineConfig
from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_all

logging.basicConfig(level=logging.WARNING)

CMJ_DIR = Path("data/raw/cmj")
FIGURE_DIR = Path("reports/figures")


def bland_altman_stats(
    method_a: np.ndarray,
    method_b: np.ndarray,
) -> dict:
    """Compute Bland-Altman agreement statistics.

    Formula (Bland & Altman, 1999):
        mean_i   = (A_i + B_i) / 2          (average of the two methods)
        diff_i   = A_i - B_i                 (A minus B)
        bias     = mean(diff_i)
        SD_d     = std(diff_i, ddof=1)
        LoA_lo   = bias - 1.96 * SD_d
        LoA_hi   = bias + 1.96 * SD_d

    95% CI for bias (t-distribution with n-1 df):
        bias ± t_{0.025, n-1} × SD_d / sqrt(n)

    95% CI for limits of agreement (Bland & Altman 1999, Eq. 5):
        LoA ± t_{0.025, n-1} × sqrt(3 * SD_d^2 / n)

    Parameters
    ----------
    method_a, method_b : np.ndarray
        Paired measurements from the two methods.

    Returns
    -------
    dict with keys: bias, sd_diff, loa_lo, loa_hi, ci_bias_lo, ci_bias_hi,
                    ci_loa_lo_lo, ci_loa_lo_hi, ci_loa_hi_lo, ci_loa_hi_hi,
                    means, diffs, n
    """
    n = len(method_a)
    diffs = method_a - method_b
    means = (method_a + method_b) / 2.0
    bias = float(np.mean(diffs))
    sd_d = float(np.std(diffs, ddof=1))

    t_crit = stats.t.ppf(0.975, df=n - 1)

    loa_lo = bias - 1.96 * sd_d
    loa_hi = bias + 1.96 * sd_d

    se_bias = sd_d / np.sqrt(n)
    se_loa = np.sqrt(3.0 * sd_d**2 / n)

    return {
        "n": n,
        "bias": bias,
        "sd_diff": sd_d,
        "loa_lo": loa_lo,
        "loa_hi": loa_hi,
        "ci_bias_lo": bias - t_crit * se_bias,
        "ci_bias_hi": bias + t_crit * se_bias,
        "ci_loa_lo_lo": loa_lo - t_crit * se_loa,
        "ci_loa_lo_hi": loa_lo + t_crit * se_loa,
        "ci_loa_hi_lo": loa_hi - t_crit * se_loa,
        "ci_loa_hi_hi": loa_hi + t_crit * se_loa,
        "means": means,
        "diffs": diffs,
    }


def main() -> None:
    if not CMJ_DIR.exists():
        print("ERROR: data/raw/cmj/ not found. Run scripts/import_cmj_xlsx.py first.")
        sys.exit(1)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig.load()
    results = run_pipeline_on_all(CMJ_DIR, config=config)

    # Collect paired heights — only include trials where both methods succeeded
    h_imp_all: list[float] = []
    h_flt_all: list[float] = []
    trial_labels: list[str] = []
    excluded: list[str] = []

    for i, r in enumerate(results):
        label = f"T{i+1:02d}"
        h_imp = r.metrics.jump_height_impulse_m
        h_flt = r.metrics.jump_height_flight_m

        if np.isnan(h_imp) or np.isnan(h_flt):
            excluded.append(f"{label} (detection failed: h_imp={'NaN' if np.isnan(h_imp) else h_imp:.3f}, "
                            f"h_flt={'NaN' if np.isnan(h_flt) else h_flt:.3f})")
            continue

        h_imp_all.append(h_imp)
        h_flt_all.append(h_flt)
        trial_labels.append(label)

    n_included = len(h_imp_all)

    print("\n" + "=" * 75)
    print("M2e BLAND-ALTMAN METHOD COMPARISON")
    print("Methods compared: Impulse-momentum height vs Flight-time height")
    print()
    print("IMPORTANT — SCOPE OF THIS ANALYSIS:")
    print("  This is a BETWEEN-ATHLETE comparison (n=10, one trial per athlete).")
    print("  It addresses: Do the two height methods agree across individuals?")
    print("  It does NOT estimate within-athlete reliability (which requires")
    print("  repeated trials per athlete — see M2g for open-data reliability).")
    print("  With n=10, confidence intervals are wide and should not be")
    print("  extrapolated to other populations or measurement conditions.")
    print("=" * 75)

    if excluded:
        print(f"\nExcluded trials (detection failure): {len(excluded)}")
        for e in excluded:
            print(f"  {e}")
        print()

    if n_included < 2:
        print("ERROR: Fewer than 2 trials had valid paired heights. Cannot compute BA.")
        sys.exit(1)

    h_imp = np.array(h_imp_all)
    h_flt = np.array(h_flt_all)

    ba = bland_altman_stats(h_imp, h_flt)

    print(f"\nSample size for BA analysis: n = {n_included}")
    if n_included < 10:
        print(f"  NOTE: {10 - n_included} trial(s) excluded due to detection failure.")

    print("\n── Paired height values (impulse | flight-time | difference) ──")
    for lbl, hi, hf in zip(trial_labels, h_imp_all, h_flt_all):
        print(f"  {lbl}: h_imp={hi:.4f} m  h_flt={hf:.4f} m  diff={hi-hf:+.4f} m")

    print(f"\n── Bland-Altman Statistics (h_impulse − h_flight) ──")
    print(f"  Bias (mean difference):     {ba['bias']:+.4f} m  (95% CI: {ba['ci_bias_lo']:+.4f} to {ba['ci_bias_hi']:+.4f} m)")
    print(f"  SD of differences:          {ba['sd_diff']:.4f} m")
    print(f"  95% Limits of Agreement:")
    print(f"    Lower LoA:                {ba['loa_lo']:+.4f} m  (95% CI: {ba['ci_loa_lo_lo']:+.4f} to {ba['ci_loa_lo_hi']:+.4f} m)")
    print(f"    Upper LoA:                {ba['loa_hi']:+.4f} m  (95% CI: {ba['ci_loa_hi_lo']:+.4f} to {ba['ci_loa_hi_hi']:+.4f} m)")
    print()
    print("  INTERPRETATION:")
    if ba["bias"] > 0:
        print(f"  Impulse-momentum consistently estimates height as higher than flight-time")
        print(f"  by {ba['bias']*1000:.1f} mm on average. This is consistent with impulse integration")
        print(f"  including some residual non-flight force (flight-floor offset).")
    elif ba["bias"] < 0:
        print(f"  Impulse-momentum consistently estimates height as lower than flight-time")
        print(f"  by {abs(ba['bias'])*1000:.1f} mm on average.")
    else:
        print("  Bias is negligible.")
    loa_range_m = ba["loa_hi"] - ba["loa_lo"]
    print(f"  LoA range: {loa_range_m*1000:.1f} mm ({loa_range_m*100:.1f} cm). With n={n_included}, these limits have")
    print(f"  wide CIs and are indicative only.")

    # ── Bland-Altman Plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    means = ba["means"]
    diffs = ba["diffs"]

    ax.scatter(means, diffs, color="#4878CF", zorder=5, s=60, label="Trial pair")
    for lbl, mx, dy in zip(trial_labels, means, diffs):
        ax.annotate(lbl, (mx, dy), textcoords="offset points", xytext=(5, 3), fontsize=7, color="#444444")

    # Bias line
    ax.axhline(ba["bias"], color="#E15759", lw=1.5, label=f"Bias = {ba['bias']*1000:.1f} mm")
    ax.fill_between(
        [means.min() - 0.01, means.max() + 0.01],
        ba["ci_bias_lo"], ba["ci_bias_hi"],
        color="#E15759", alpha=0.12,
    )

    # LoA lines
    ax.axhline(ba["loa_lo"], color="#59A14F", lw=1.2, ls="--",
               label=f"95% LoA: {ba['loa_lo']*1000:.1f} to {ba['loa_hi']*1000:.1f} mm")
    ax.axhline(ba["loa_hi"], color="#59A14F", lw=1.2, ls="--")
    ax.fill_between(
        [means.min() - 0.01, means.max() + 0.01],
        ba["ci_loa_lo_lo"], ba["ci_loa_lo_hi"],
        color="#59A14F", alpha=0.10,
    )
    ax.fill_between(
        [means.min() - 0.01, means.max() + 0.01],
        ba["ci_loa_hi_lo"], ba["ci_loa_hi_hi"],
        color="#59A14F", alpha=0.10,
    )
    ax.axhline(0, color="k", lw=0.6, ls=":")

    ax.set_xlabel("Mean of impulse-momentum and flight-time height (m)", fontsize=10)
    ax.set_ylabel("Difference: impulse − flight-time (m)", fontsize=10)
    ax.set_title(
        f"Bland-Altman Plot: Jump Height Methods (n={n_included})\n"
        "BETWEEN-ATHLETE comparison only — not a reliability estimate",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")

    # Annotation box
    txt = (
        f"Bias: {ba['bias']*1000:.1f} mm (95% CI {ba['ci_bias_lo']*1000:.1f} to {ba['ci_bias_hi']*1000:.1f} mm)\n"
        f"LoA: {ba['loa_lo']*1000:.1f} to {ba['loa_hi']*1000:.1f} mm\n"
        f"n={n_included}; one trial per athlete; wide CIs"
    )
    ax.text(
        0.02, 0.03, txt, transform=ax.transAxes,
        fontsize=8, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#AAAAAA", alpha=0.9),
    )

    plt.tight_layout()
    fig_path = FIGURE_DIR / "cmj_bland_altman.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {fig_path}")


if __name__ == "__main__":
    main()
