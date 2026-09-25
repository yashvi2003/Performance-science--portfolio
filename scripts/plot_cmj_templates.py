"""Generate publication-grade phase detection plots for all five synthetic CMJ templates.

Templates evaluated:
1. single_peak: Standard unimodal concentric propulsion profile
2. double_peak: Mid-propulsion dip and secondary push (0.3 < tau < 0.7)
3. double_peak_late: Original late-propulsion perturbation (tau > 0.5, stress template)
4. deep_unweight: Deep countermovement dip (1.6x unweighting depth)
5. high_braking: Steep eccentric deceleration spike at the v=0 transition

All phases defined strictly per McMahon et al. (2018):
- Unweighting: Onset to Unweighting Minimum Force
- Braking (Eccentric): Unweighting Minimum Force to Zero Velocity (v=0)
- Propulsion (Concentric): Zero Velocity to Take-off
- Flight: Take-off to Landing
- Landing: Landing contact to Landing Peak Force

Rule 7 Honesty: All figures prominently labeled '[SYNTHETIC DATA]'.
"""

import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from shared.synth import make_cmj_trial
from p1_nmt_pipeline.validator import load_and_validate_trial
from p1_nmt_pipeline.filter import zero_phase_butterworth_filter
from p1_nmt_pipeline.events import detect_cmj_events
from p1_nmt_pipeline.metrics import compute_cmj_metrics
from p1_nmt_pipeline.config import PipelineConfig


def main():
    templates_spec = [
        {
            "template": "single_peak",
            "title": "Template 1: Standard Single-Peak CMJ (Unimodal Propulsion)",
            "target_h": 0.40,
            "desc": "Canonical CMJ with smooth sinusoidal unweighting, braking, and unimodal propulsion.",
            "seed": 101,
        },
        {
            "template": "double_peak",
            "title": "Template 2: Mid-Propulsion Double-Peak CMJ (0.3 < tau < 0.7)",
            "target_h": 0.40,
            "desc": "Concentric propulsion features mid-phase hesitation and re-acceleration.",
            "seed": 202,
        },
        {
            "template": "double_peak_late",
            "title": "Template 3: Late-Propulsion Stress Double-Peak CMJ (tau > 0.5)",
            "target_h": 0.40,
            "desc": "Late push truncates negative net impulse tail; documents early takeoff trigger & +2.7 mm bias.",
            "seed": 303,
        },
        {
            "template": "deep_unweight",
            "title": "Template 4: Deep Unweighting CMJ (1.6x Countermovement Depth)",
            "target_h": 0.38,
            "desc": "Exaggerated unweighting trough testing robust zero-velocity and braking onset recovery.",
            "seed": 404,
        },
        {
            "template": "high_braking",
            "title": "Template 5: High Braking CMJ (Steep Eccentric Transition)",
            "target_h": 0.42,
            "desc": "High eccentric rate of force development at lowest point of countermovement dip.",
            "seed": 505,
        },
    ]

    phase_colors = {
        "unweighting": "#E8F4F8",  # soft blue
        "braking":     "#FFF2E6",  # soft orange
        "propulsion":  "#EAF7EA",  # soft green
        "flight":      "#F3EBF7",  # soft purple
        "landing":     "#FDE8E8",  # soft red
    }

    cfg = PipelineConfig.load()

    fig, axes = plt.subplots(len(templates_spec), 1, figsize=(14, 18), sharex=False)
    fig.suptitle(
        "[SYNTHETIC DATA] CMJ Force-Time Profiles Across All 5 Biomechanical Templates\n"
        "Phase Detection strictly per McMahon et al. (2018) | Sampling Rate: 2000 Hz",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    for idx, spec in enumerate(templates_spec):
        ax = axes[idx]
        trial = make_cmj_trial(
            fs_hz=2000.0,
            target_jump_height_m=spec["target_h"],
            template=spec["template"],
            noise_sd_n=0.0,
            seed=spec["seed"],
        )

        val = load_and_validate_trial(trial.to_df(), body_mass_kg=trial.body_mass_kg)
        filt_force = zero_phase_butterworth_filter(val.force_n, val.fs_hz, cutoff_hz=30.0, order=cfg.filters.order)
        events = detect_cmj_events(val.time_s, filt_force, val.fs_hz, config=cfg, body_mass_kg=val.body_mass_kg, force_raw_n=val.force_n)
        metrics = compute_cmj_metrics(val.time_s, filt_force, events, config=cfg)

        t = val.time_s
        f = filt_force

        # Plot Force trace
        ax.plot(t, f, color="#1A252F", linewidth=1.8, label="Force (30 Hz LP)")
        ax.axhline(events.quiet_bw_mean_n, color="#7F8C8D", linestyle="--", linewidth=1.0, label=f"BW ({events.quiet_bw_mean_n:.0f} N)")

        # Shade McMahon (2018) phases
        ax.axvspan(events.onset_time_s, events.unweighting_min_time_s, color=phase_colors["unweighting"], alpha=0.85, label="Unweighting")
        ax.axvspan(events.unweighting_min_time_s, events.v0_time_s, color=phase_colors["braking"], alpha=0.85, label="Braking (v < 0)")
        ax.axvspan(events.v0_time_s, events.takeoff_time_s, color=phase_colors["propulsion"], alpha=0.85, label="Propulsion (v > 0)")
        if events.landing_time_s is not None:
            ax.axvspan(events.takeoff_time_s, events.landing_time_s, color=phase_colors["flight"], alpha=0.85, label="Flight")
            if events.landing_peak_time_s is not None:
                ax.axvspan(events.landing_time_s, events.landing_peak_time_s, color=phase_colors["landing"], alpha=0.85, label="Landing")

        # Landmark points
        ax.axvline(events.onset_time_s, color="#8E44AD", linestyle="--", linewidth=1.1, label=f"Onset ({events.onset_time_s:.3f}s)")
        ax.scatter([events.unweighting_min_time_s], [events.unweighting_min_force_n], color="#D35400", marker="v", s=65, zorder=5, label="Unweight Min")
        ax.scatter([events.bw_cross_time_s], [events.quiet_bw_mean_n], color="#16A085", marker="o", s=45, zorder=5, label="BW Cross")
        ax.axvline(events.v0_time_s, color="#C0392B", linestyle="-.", linewidth=1.1, label=f"v = 0 ({events.v0_time_s:.3f}s)")
        ax.scatter([events.propulsion_peak_time_s], [events.propulsion_peak_force_n], color="#E74C3C", marker="^", s=75, zorder=5, label="Peak Force")
        ax.axvline(events.takeoff_time_s, color="#27AE60", linestyle="-", linewidth=1.4, label=f"Take-off ({events.takeoff_time_s:.3f}s)")
        if events.landing_time_s is not None:
            ax.axvline(events.landing_time_s, color="#962D3E", linestyle="-", linewidth=1.4, label=f"Landing ({events.landing_time_s:.3f}s)")
        if events.landing_peak_time_s is not None:
            ax.scatter([events.landing_peak_time_s], [events.landing_peak_force_n], color="#7D1935", marker="*", s=95, zorder=5, label="Landing Peak")

        # Title and labels
        ax.set_title(spec["title"], fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("Force (N)", fontsize=10)
        ax.set_xlim(max(0.0, events.onset_time_s - 0.25), min(t[-1], (events.landing_time_s or 2.0) + 0.35))
        ax.grid(True, linestyle=":", alpha=0.5)

        # Performance summary metrics box
        h_imp_mm = metrics.jump_height_impulse_m * 1000.0
        h_flt_mm = (metrics.jump_height_flight_m or 0.0) * 1000.0
        tgt_mm = spec["target_h"] * 1000.0
        flt_s = events.flight_time_s or 0.0
        rsimod = metrics.rsimod_impulse or 0.0
        f_max = events.propulsion_peak_force_n
        f_land = events.landing_peak_force_n or 0.0

        stats_text = (
            f"Target: {tgt_mm:.1f} mm\n"
            f"h_imp: {h_imp_mm:.1f} mm (Δ={h_imp_mm - tgt_mm:+.1f} mm)\n"
            f"h_flt: {h_flt_mm:.1f} mm (Δ={h_flt_mm - tgt_mm:+.1f} mm)\n"
            f"t_flight: {flt_s*1000:.1f} ms\n"
            f"RSImod: {rsimod:.3f}\n"
            f"F_prop_max: {f_max:.0f} N\n"
            f"F_land_peak: {f_land:.0f} N"
        )
        ax.text(
            0.985, 0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FAFAFA", edgecolor="#BDC3C7", alpha=0.92),
            family="monospace",
        )

        if idx == 0:
            # Legend on top row
            handles, labels = ax.get_legend_handles_labels()
            # De-duplicate legend
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(), by_label.keys(),
                loc="upper left",
                bbox_to_anchor=(0.01, 0.95),
                fontsize=7.5,
                ncol=2,
                framealpha=0.90,
            )

        if idx == len(templates_spec) - 1:
            ax.set_xlabel("Time (s)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cmj_all_templates_detection.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
