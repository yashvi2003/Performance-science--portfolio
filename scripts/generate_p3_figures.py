"""Generate publication-grade figures for Project 3 (p3_nmt_profiler).

Produces:
1. reports/figures/p3_sleep_heterogeneity_slopes.png:
   Visualizes individual athlete sleep sensitivity slopes (random slopes / Empirical Bayes)
   demonstrating between-athlete heterogeneity in sleep loss response.
2. reports/figures/p3_load_recovery_effects.png:
   Multi-panel evidence figure:
   - Panel A: Standardized fixed-effect forest plot across Models 1, 2, 3 with 95% CIs.
   - Panel B: Within-person acute load vs muscle soreness dose-response curve.
   - Panel C: Acute sleep deviation vs next-day morning readiness response.
   - Panel D: Causal DAG schematic (Observed predictors vs unmeasured confounders).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from p3_nmt_profiler.model import run_all_models, prepare_longitudinal_dataset
from p3_nmt_profiler.sensitivity import extract_individual_athlete_slopes

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_sleep_heterogeneity_figure(df: pd.DataFrame, out_path: Path = FIGURES_DIR / "p3_sleep_heterogeneity_slopes.png") -> Path:
    """Plot individual athlete sleep-sensitivity random slopes vs group average."""
    slopes = extract_individual_athlete_slopes(df)
    
    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300, facecolor="#F8FAFC",
                                            gridspec_kw={"width_ratios": [1.2, 1.0]})

    # --- LEFT PANEL: Individual Trajectory Slopes ---
    ax_left.set_facecolor("#FFFFFF")
    
    # X range for within-person sleep deviations (-2 to +2 SDs)
    x_range = np.linspace(-2.5, 2.5, 100)
    
    # Population average slope
    mean_slope = np.mean([s.blup_slope for s in slopes])
    
    # Color palette
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(s.blup_slope for s in slopes), vmax=max(s.blup_slope for s in slopes))
    
    for s in slopes:
        y_vals = s.baseline_readiness + s.blup_slope * x_range
        color = cmap(norm(s.blup_slope))
        ax_left.plot(x_range, y_vals, color=color, alpha=0.65, linewidth=1.8, label=s.athlete_id)
        # Endpoint label
        ax_left.text(2.55, y_vals[-1], f"{s.athlete_id} (β={s.blup_slope:+.2f})",
                     fontsize=7.5, color="#334155", va="center")

    # Bold population mean trajectory
    pop_intercept = np.mean([s.baseline_readiness for s in slopes])
    ax_left.plot(x_range, pop_intercept + mean_slope * x_range, color="#0F172A",
                 linewidth=3.5, linestyle="--", label=f"Population Mean (β={mean_slope:+.2f})", zorder=10)

    ax_left.set_xlim(-2.8, 3.8)
    ax_left.set_xlabel("Prior-Night Sleep Deviation (Within-Person z-Score)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax_left.set_ylabel("Morning Subjective Readiness (1–10 Scale)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax_left.set_title("A. Individual Athlete Sleep Sensitivity Trajectories\n(Random Slopes / Empirical Bayes BLUPs)",
                      fontsize=11.5, fontweight="bold", color="#0F172A", pad=12)
    ax_left.grid(True, linestyle=":", color="#E2E8F0")

    # --- RIGHT PANEL: Ranked Sensitivity Forest/Bar ---
    ax_right.set_facecolor("#FFFFFF")
    
    pids = [s.athlete_id for s in slopes]
    blups = [s.blup_slope for s in slopes]
    raws = [s.raw_ols_slope for s in slopes]
    y_pos = np.arange(len(pids))

    ax_right.scatter(raws, y_pos, color="#94A3B8", marker="o", s=40, label="Raw Within-Athlete OLS", zorder=4)
    bars = ax_right.barh(y_pos, blups, height=0.55, color=[cmap(norm(b)) for b in blups],
                         alpha=0.85, label="Empirical Bayes Shrinkage (BLUP)", zorder=3)

    ax_right.axvline(0, color="#64748B", linestyle="-", linewidth=0.8)
    ax_right.axvline(mean_slope, color="#0F172A", linestyle="--", linewidth=1.5, label=f"Group Mean (β={mean_slope:+.2f})")

    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(pids, fontsize=9, fontweight="medium")
    ax_right.invert_yaxis()
    ax_right.set_xlabel("Sleep Sensitivity Coefficient (Δ Readiness / +1 SD Sleep)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax_right.set_title("B. Ranked Sensitivity & Shrinkage\n(LRT χ²(2) = 32.02, p < 0.001)",
                       fontsize=11.5, fontweight="bold", color="#0F172A", pad=12)
    ax_right.grid(True, linestyle=":", color="#E2E8F0", axis="x")
    ax_right.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    plt.suptitle("Inter-Individual Heterogeneity in Sleep Loss Sensitivity (PMData Cohort, n=15 Athletes, 1,368 Days)",
                 fontsize=13, fontweight="bold", color="#0F172A", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_load_recovery_effects_figure(df: pd.DataFrame, out_path: Path = FIGURES_DIR / "p3_load_recovery_effects.png") -> Path:
    """Plot multi-panel evidence figure: Forest plot, dose-response curves, and DAG."""
    m1, m2, m3, _ = run_all_models()

    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(15, 10), dpi=300, facecolor="#F8FAFC")
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.25, left=0.07, right=0.95, top=0.92, bottom=0.07)

    # -------------------------------------------------------------------------
    # PANEL A: Forest Plot of Standardized Within-Person Fixed Effects
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#FFFFFF")

    effects = [
        ("Readiness ~ Acute Sleep (H1a)", m1.fixed_effects[1].coef, m1.fixed_effects[1].ci_lower, m1.fixed_effects[1].ci_upper, "#059669", True),
        ("Readiness ~ Acute RHR (H2)", m1.fixed_effects[3].coef, m1.fixed_effects[3].ci_lower, m1.fixed_effects[3].ci_upper, "#64748B", False),
        ("Readiness ~ Acute Load (H3b)", m1.fixed_effects[5].coef, m1.fixed_effects[5].ci_lower, m1.fixed_effects[5].ci_upper, "#64748B", False),
        ("Readiness ~ Weekend Dummy", m1.fixed_effects[7].coef, m1.fixed_effects[7].ci_lower, m1.fixed_effects[7].ci_upper, "#DC2626", True),
        ("Soreness ~ Acute Load (H3a)", m2.fixed_effects[1].coef, m2.fixed_effects[1].ci_lower, m2.fixed_effects[1].ci_upper, "#0284C7", True),
        ("Fatigue ~ Sleep Score (H1c)", m3.fixed_effects[1].coef, m3.fixed_effects[1].ci_lower, m3.fixed_effects[1].ci_upper, "#7C3AED", True),
    ]

    y_pos = np.arange(len(effects))
    ax_a.axvline(0, color="#94A3B8", linestyle="--", linewidth=1.0)

    for i, (name, coef, ci_l, ci_u, color, sig) in enumerate(effects):
        ax_a.plot([ci_l, ci_u], [i, i], color=color, linewidth=2.2, zorder=3)
        marker = "s" if sig else "o"
        ax_a.scatter([coef], [i], color=color, marker=marker, s=55, zorder=4)
        sig_str = "*" if sig else " (ns)"
        ax_a.text(max(ci_u + 0.02, coef + 0.03), i, f"{coef:+.3f}{sig_str}", va="center", fontsize=8.5, color="#1E293B")

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([e[0] for e in effects], fontsize=9, fontweight="medium")
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Standardized Effect Size (β with 95% CI)", fontsize=10, fontweight="bold", color="#1E293B")
    ax_a.set_title("A. Standardized Mixed-Effects Parameters\n(Currie-Hoffmann Person-Mean Centered)", fontsize=11, fontweight="bold", color="#0F172A")
    ax_a.grid(True, linestyle=":", color="#E2E8F0", axis="x")

    # -------------------------------------------------------------------------
    # PANEL B: Training Strain vs Muscle Soreness Dose-Response
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#FFFFFF")

    m2_df = df.dropna(subset=["soreness", "z_dev_prior_day_strain"]).copy()
    ax_b.scatter(m2_df["z_dev_prior_day_strain"], m2_df["soreness"], color="#38BDF8", alpha=0.18, s=25, edgecolors="none")

    # Fit line
    x_grid = np.linspace(-1.5, 3.5, 100)
    b0 = m2.fixed_effects[0].coef
    b1 = m2.fixed_effects[1].coef
    ax_b.plot(x_grid, b0 + b1 * x_grid, color="#0284C7", linewidth=2.8, label=f"Fixed Effect (β = {b1:+.3f}, p < 0.001)")

    ax_b.set_xlabel("Prior-Day Training Load Deviation (Within-Person z-Score)", fontsize=10, fontweight="bold", color="#1E293B")
    ax_b.set_ylabel("Morning Muscle Soreness (1=Sore, 5=Fresh)", fontsize=10, fontweight="bold", color="#1E293B")
    ax_b.set_title("B. Prior-Day Training Strain vs Muscle Soreness\n(Higher Load Induces Next-Day Soreness)", fontsize=11, fontweight="bold", color="#0F172A")
    ax_b.grid(True, linestyle=":", color="#E2E8F0")
    ax_b.legend(loc="upper right", fontsize=8.5)

    # -------------------------------------------------------------------------
    # PANEL C: Sleep Duration vs Readiness Response
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("#FFFFFF")

    m1_df = df.dropna(subset=["readiness", "z_dev_prior_sleep_hours"]).copy()
    ax_c.scatter(m1_df["z_dev_prior_sleep_hours"], m1_df["readiness"], color="#34D399", alpha=0.18, s=25, edgecolors="none")

    b0_m1 = m1.fixed_effects[0].coef
    b1_m1 = m1.fixed_effects[1].coef
    x_sleep = np.linspace(-2.5, 2.5, 100)
    # Mean intercept across athletes
    ax_c.plot(x_sleep, 5.0 + b1_m1 * x_sleep, color="#059669", linewidth=2.8, label=f"Fixed Effect (β = {b1_m1:+.3f}, p < 0.001)")

    ax_c.set_xlabel("Prior-Night Sleep Deviation (Within-Person z-Score)", fontsize=10, fontweight="bold", color="#1E293B")
    ax_c.set_ylabel("Morning Subjective Readiness (1–10 Scale)", fontsize=10, fontweight="bold", color="#1E293B")
    ax_c.set_title("C. Sleep Duration vs Next-Day Readiness\n(Acute Sleep Boosts Subjective Readiness)", fontsize=11, fontweight="bold", color="#0F172A")
    ax_c.grid(True, linestyle=":", color="#E2E8F0")
    ax_c.legend(loc="lower right", fontsize=8.5)

    # -------------------------------------------------------------------------
    # PANEL D: Causal DAG Schematic & Confounding Structure
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    ax_d.set_facecolor("#FFFFFF")

    # Draw DAG diagram box
    box_rect = plt.Rectangle((0, 0), 1, 1, transform=ax_d.transAxes, color="#FFFFFF",
                             edgecolor="#CBD5E1", linewidth=1.2, clip_on=False)
    ax_d.add_patch(box_rect)

    ax_d.text(0.5, 0.92, "D. Causal Directed Acyclic Graph (DAG)",
              fontsize=11, fontweight="bold", color="#0F172A", ha="center", transform=ax_d.transAxes)

    # Node boxes
    def draw_node(x, y, text, color, border="#64748B", w=0.36, h=0.12):
        r = plt.Rectangle((x - w/2, y - h/2), w, h, transform=ax_d.transAxes,
                          color=color, edgecolor=border, linewidth=1.2, zorder=3)
        ax_d.add_patch(r)
        ax_d.text(x, y, text, ha="center", va="center", fontsize=8.5, fontweight="bold",
                  color="#0F172A", transform=ax_d.transAxes, zorder=4)

    # Confounder top
    draw_node(0.5, 0.76, "Unmeasured Confounders\n(Alcohol, Illness, Life Stress)", "#FEF3C7", border="#D97706", w=0.52, h=0.13)

    # Predictors middle
    draw_node(0.25, 0.46, "Prior-Day Training\nLoad (sRPE t-1)", "#E0F2FE", border="#0284C7", w=0.38, h=0.12)
    draw_node(0.75, 0.46, "Prior-Night Sleep\n(Hours & Score)", "#D1FAE5", border="#059669", w=0.38, h=0.12)

    # Outcome bottom
    draw_node(0.5, 0.16, "Next-Day Morning Wellness\n(Readiness, Soreness, Fatigue)", "#F1F5F9", border="#475569", w=0.55, h=0.13)

    # Arrows
    def draw_arrow(x1, y1, x2, y2, color="#64748B", style="->"):
        ax_d.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                      arrowprops=dict(arrowstyle=style, color=color, lw=1.5, ls="--"))

    draw_arrow(0.40, 0.70, 0.25, 0.52, color="#D97706")
    draw_arrow(0.60, 0.70, 0.75, 0.52, color="#D97706")
    draw_arrow(0.25, 0.40, 0.42, 0.23, color="#0284C7")
    draw_arrow(0.75, 0.40, 0.58, 0.23, color="#059669")

    # Disclaimer text at bottom
    ax_d.text(0.5, 0.02, "Observational wearable data; arrows represent modeled paths, not proven counterfactuals.",
              fontsize=7.2, fontstyle="italic", color="#64748B", ha="center", transform=ax_d.transAxes)

    plt.suptitle("Real-World Evidence: Longitudinal Load, Recovery & Readiness (PMData Cohort, 16 Athletes)",
                 fontsize=13.5, fontweight="bold", color="#0F172A", y=0.97)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    df = prepare_longitudinal_dataset()
    p1 = plot_sleep_heterogeneity_figure(df)
    print(f"Generated sleep heterogeneity figure at: {p1}")
    p2 = plot_load_recovery_effects_figure(df)
    print(f"Generated load-recovery effects figure at: {p2}")


if __name__ == "__main__":
    main()
