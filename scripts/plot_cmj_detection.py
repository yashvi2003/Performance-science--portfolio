"""Generate publication-grade CMJ phase detection and metrics plots across five synthetic trials.

Conditions evaluated:
1. Clean 2000 Hz (h = 0.40 m)
2. Clean 250 Hz (h = 0.35 m)
3. Low Noise 3 N (2000 Hz, h = 0.38 m)
4. High Noise 12 N (2000 Hz, h = 0.42 m)
5. Flight Force Offset +5 N (whole-trace corrected, 2000 Hz, h = 0.40 m)

All phases defined strictly per McMahon et al. (2018):
- Unweighting: Onset to Unweighting Minimum Force
- Braking (Eccentric): Unweighting Minimum Force to Zero Velocity (v=0)
- Propulsion (Concentric): Zero Velocity to Take-off
- Flight: Take-off to Landing
- Landing: Landing contact to Landing Peak Force

Item G refinements applied:
- Velocity/power panels: x-axis clipped to [onset - 0.05 s, takeoff + 0.05 s],
  landing window shaded in grey.
- Legend moved outside axes (bbox_to_anchor) on row 0 only; no legend on other rows.
- Power panel: y-limit clips at 0 from below (no negative power shown).
- Velocity panel: y-limits set symmetrically around 0 with 10% margin.

Rule 7 Honesty: All figures prominently labeled '[SYNTHETIC DATA]'.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from shared.synth import make_cmj_trial
from p1_nmt_pipeline.validator import load_and_validate_trial
from p1_nmt_pipeline.filter import zero_phase_butterworth_filter
from p1_nmt_pipeline.events import detect_cmj_events
from p1_nmt_pipeline.metrics import compute_cmj_metrics
from p1_nmt_pipeline.config import PipelineConfig, ThresholdConfig


def main():
    # 5 representative trial configurations
    trials_spec = [
        {
            "name": "Trial 1: Clean 2000 Hz (h = 0.40 m)",
            "fs_hz": 2000.0,
            "target_h": 0.40,
            "noise": 0.0,
            "offset": 0.0,
            "offset_whole": False,
            "do_corr": False,
            "offset_model": "whole_trace",
            "seed": 101,
        },
        {
            "name": "Trial 2: Clean 250 Hz (h = 0.35 m)",
            "fs_hz": 250.0,
            "target_h": 0.35,
            "noise": 0.0,
            "offset": 0.0,
            "offset_whole": False,
            "do_corr": False,
            "offset_model": "whole_trace",
            "seed": 202,
        },
        {
            "name": "Trial 3: Low Noise (SD = 3 N, 2000 Hz, h = 0.38 m)",
            "fs_hz": 2000.0,
            "target_h": 0.38,
            "noise": 3.0,
            "offset": 0.0,
            "offset_whole": False,
            "do_corr": False,
            "offset_model": "whole_trace",
            "seed": 303,
        },
        {
            "name": "Trial 4: High Noise (SD = 12 N, 2000 Hz, h = 0.42 m)",
            "fs_hz": 2000.0,
            "target_h": 0.42,
            "noise": 12.0,
            "offset": 0.0,
            "offset_whole": False,
            "do_corr": False,
            "offset_model": "whole_trace",
            "seed": 404,
        },
        {
            "name": "Trial 5: Flight Offset +5 N (Corrected whole-trace, 2000 Hz)",
            "fs_hz": 2000.0,
            "target_h": 0.40,
            "noise": 0.0,
            "offset": 5.0,
            "offset_whole": True,
            "do_corr": True,
            "offset_model": "whole_trace",
            "seed": 505,
        },
    ]

    base_cfg = PipelineConfig.load()

    # Create figure with 5 rows (one per trial) and 2 columns (Force left, Velocity/Power right)
    fig, axes = plt.subplots(5, 2, figsize=(18, 22), sharex=False)
    fig.suptitle(
        "[SYNTHETIC DATA] Project 1: CMJ Phase Detection & Biomechanical Metrics (McMahon et al. 2018)",
        fontsize=16,
        fontweight="bold",
        y=0.998,
    )

    phase_colors = {
        "unweighting": "#E8F4F8",  # soft light blue
        "braking": "#FDEBD0",      # soft amber
        "propulsion": "#E8F8F5",   # soft light green
        "flight": "#F5EEF8",       # soft light purple
        "landing": "#FDEDEC",      # soft light red
        "landing_window": "#EBEBEB",  # grey for vp panel landing window (item G)
    }

    for idx, spec in enumerate(trials_spec):
        trial = make_cmj_trial(
            fs_hz=spec["fs_hz"],
            target_jump_height_m=spec["target_h"],
            t_onset=0.5,
            noise_sd_n=spec["noise"],
            flight_force_offset_n=spec["offset"],
            offset_whole_trace=spec["offset_whole"],
            random_seed=spec["seed"],
        )

        cfg = PipelineConfig(
            filters=base_cfg.filters,
            thresholds=ThresholdConfig(
                correct_flight_phase_offset=spec["do_corr"],
                flight_offset_model=spec["offset_model"],
            ),
            classifier=base_cfg.classifier,
        )

        val = load_and_validate_trial(trial.to_df(), body_mass_kg=trial.body_mass_kg)
        filt_force = zero_phase_butterworth_filter(
            val.force_n,
            val.fs_hz,
            cutoff_hz=cfg.filters.get_cutoff("CMJ"),
            order=cfg.filters.order,
        )
        events = detect_cmj_events(
            val.time_s,
            filt_force,
            val.fs_hz,
            config=cfg,
            body_mass_kg=val.body_mass_kg,
            force_raw_n=val.force_n,
        )
        metrics = compute_cmj_metrics(
            val.time_s,
            filt_force,
            events,
            config=cfg,
        )

        # -----------------------------------------------------------------------
        # LEFT PANEL: Force-Time with McMahon Phases
        # -----------------------------------------------------------------------
        ax_f = axes[idx, 0]
        t = val.time_s
        f_raw = val.force_n
        f_filt = filt_force

        # Plot raw and filtered force
        if spec["noise"] > 0:
            ax_f.plot(t, f_raw, color="#BDC3C7", alpha=0.65, label="Raw Force", linewidth=0.8)
        ax_f.plot(t, f_filt, color="#2C3E50", label="Filtered Force (30 Hz LP)", linewidth=1.6)

        # Body weight line
        ax_f.axhline(
            events.quiet_bw_mean_n, color="#7F8C8D", linestyle="--", linewidth=1.1,
            label=f"BW ({events.quiet_bw_mean_n:.1f} N)",
        )

        # Shade McMahon (2018) phases
        ax_f.axvspan(events.onset_time_s, events.unweighting_min_time_s,
                     color=phase_colors["unweighting"], alpha=0.7, label="Unweighting")
        ax_f.axvspan(events.unweighting_min_time_s, events.v0_time_s,
                     color=phase_colors["braking"], alpha=0.7, label="Braking (v < 0)")
        ax_f.axvspan(events.v0_time_s, events.takeoff_time_s,
                     color=phase_colors["propulsion"], alpha=0.7, label="Propulsion (v > 0)")
        if events.landing_time_s is not None:
            ax_f.axvspan(events.takeoff_time_s, events.landing_time_s,
                         color=phase_colors["flight"], alpha=0.7, label="Flight")
            if events.landing_peak_time_s is not None:
                ax_f.axvspan(events.landing_time_s, events.landing_peak_time_s,
                             color=phase_colors["landing"], alpha=0.7, label="Landing")

        # Landmark markers
        ax_f.axvline(events.int_start_time_s, color="#2980B9", linestyle=":", linewidth=1.2,
                     label=f"Int Start ({events.int_start_time_s:.3f}s)")
        ax_f.axvline(events.onset_time_s, color="#8E44AD", linestyle="--", linewidth=1.2,
                     label=f"Onset ({events.onset_time_s:.3f}s)")
        ax_f.scatter([events.unweighting_min_time_s], [events.unweighting_min_force_n],
                     color="#D35400", marker="v", s=60, zorder=5, label="Unweight Min")
        ax_f.scatter([events.bw_cross_time_s], [events.quiet_bw_mean_n],
                     color="#16A085", marker="o", s=40, zorder=5, label="BW Cross")
        ax_f.axvline(events.v0_time_s, color="#C0392B", linestyle="-.", linewidth=1.2,
                     label=f"v = 0 ({events.v0_time_s:.3f}s)")
        ax_f.scatter([events.propulsion_peak_time_s], [events.propulsion_peak_force_n],
                     color="#E74C3C", marker="^", s=70, zorder=5,
                     label=f"Peak Force ({events.propulsion_peak_force_n:.0f}N)")
        ax_f.axvline(events.takeoff_time_s, color="#27AE60", linestyle="-", linewidth=1.5,
                     label=f"Take-off ({events.takeoff_time_s:.3f}s)")
        if events.landing_time_s is not None:
            ax_f.axvline(events.landing_time_s, color="#962D3E", linestyle="-", linewidth=1.5,
                         label=f"Landing ({events.landing_time_s:.3f}s)")
        if events.landing_peak_time_s is not None:
            ax_f.scatter([events.landing_peak_time_s], [events.landing_peak_force_n],
                         color="#7D1935", marker="*", s=90, zorder=5,
                         label=f"Landing Peak ({events.landing_peak_force_n:.0f}N)")

        ax_f.set_title(f"{spec['name']}", fontsize=11, fontweight="bold", loc="left")
        ax_f.set_ylabel("Force (N)", fontsize=10)
        ax_f.set_xlim(0.2, min(t[-1], 2.5))
        ax_f.grid(True, linestyle=":", alpha=0.5)

        # Item G: legend outside axes on row 0 only
        if idx == 0:
            ax_f.legend(
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                fontsize=7.5,
                ncol=1,
                framealpha=0.90,
                borderaxespad=0,
            )

        # -----------------------------------------------------------------------
        # RIGHT PANEL: Velocity & Power (item G: x-axis = onset to takeoff + margin)
        # -----------------------------------------------------------------------
        ax_vp = axes[idx, 1]
        ax_p = ax_vp.twinx()

        g = 9.80665
        mass = events.body_mass_kg
        bw = events.quiet_bw_mean_n

        offset_corr = (
            events.flight_offset_n
            if (events.flight_offset_applied and events.flight_offset_model == "whole_trace")
            else 0.0
        )
        work_f = filt_force - offset_corr
        net_f = work_f - bw

        # Trapezoidal velocity integration from int_start_idx
        dt_scalar = 1.0 / val.fs_hz
        dt_arr = np.diff(val.time_s)
        vel = np.zeros_like(work_f)
        accel_arr = net_f / mass
        for i in range(events.int_start_idx, len(vel) - 1):
            vel[i + 1] = vel[i] + 0.5 * (accel_arr[i] + accel_arr[i + 1]) * dt_arr[i]

        # Overwrite flight phase with ballistic velocity
        to_idx = events.takeoff_idx
        if events.landing_idx is not None:
            land_idx = events.landing_idx
            for k in range(to_idx, land_idx):
                vel[k] = vel[to_idx] - g * (t[k] - t[to_idx])

        # Power = F * v (zero during flight where F = 0)
        power_w = work_f * vel
        if events.landing_idx is not None:
            power_w[to_idx : events.landing_idx] = 0.0

        # Item G: restrict x-axis of vp panel to onset − 0.05 s → takeoff + 0.05 s
        x_left_vp = max(0.0, events.onset_time_s - 0.05)
        x_right_vp = events.takeoff_time_s + 0.05
        ax_vp.set_xlim(x_left_vp, x_right_vp)
        ax_p.set_xlim(x_left_vp, x_right_vp)

        # Item G: shade landing window in grey (mark where landing would be)
        if events.landing_time_s is not None:
            land_window_right = min(events.landing_time_s + 0.05, t[-1])
            ax_vp.axvspan(
                events.takeoff_time_s, x_right_vp,
                color=phase_colors["landing_window"], alpha=0.45,
                label="Post-takeoff\n(flight/landing)",
            )

        l_v, = ax_vp.plot(t, vel, color="#2980B9", linewidth=1.6, label="Velocity (m/s)")
        l_p, = ax_p.plot(t, power_w, color="#E67E22", linewidth=1.4, linestyle="--", label="Power (W)")

        ax_vp.axhline(0.0, color="#7F8C8D", linestyle=":", linewidth=0.9)
        ax_vp.axvline(events.v0_time_s, color="#C0392B", linestyle="-.", linewidth=1.1, alpha=0.7)
        ax_vp.axvline(events.takeoff_time_s, color="#27AE60", linestyle="-", linewidth=1.1, alpha=0.7)

        # Item G: y-limits — velocity symmetric, power clips at 0 from below
        vel_in_window = vel[
            (t >= x_left_vp) & (t <= x_right_vp)
        ]
        if len(vel_in_window) > 0:
            v_abs_max = max(abs(np.min(vel_in_window)), abs(np.max(vel_in_window))) * 1.10
            ax_vp.set_ylim(-v_abs_max, v_abs_max)

        power_in_window = power_w[
            (t >= x_left_vp) & (t <= x_right_vp)
        ]
        if len(power_in_window) > 0:
            p_max = np.max(power_in_window) * 1.12
            ax_p.set_ylim(0, max(p_max, 1.0))  # clips at 0 from below

        ax_vp.set_ylabel("Velocity (m/s)", color="#2980B9", fontsize=10)
        ax_p.set_ylabel("Power (W)", color="#E67E22", fontsize=10)
        ax_vp.grid(True, linestyle=":", alpha=0.5)

        # Performance metrics annotation
        h_flt_str = (
            f"{metrics.jump_height_flight_m * 100:.1f} cm"
            if metrics.jump_height_flight_m is not None
            else "N/A"
        )
        rsi_flt_str = (
            f"{metrics.rsimod_flight:.3f} m/s"
            if metrics.rsimod_flight is not None
            else "N/A"
        )
        land_f_str = (
            f"{metrics.peak_landing_force_n:.0f} N"
            if metrics.peak_landing_force_n is not None
            else "N/A"
        )
        metrics_box = (
            f"PRIMARY METRICS:\n"
            f"• Jump Height (Impulse): {metrics.jump_height_impulse_m * 100:.1f} cm  (True: {spec['target_h'] * 100:.1f} cm)\n"
            f"• Jump Height (Flight):  {h_flt_str}\n"
            f"• RSImod (Impulse, Pri): {metrics.rsimod_impulse:.3f} m/s\n"
            f"• RSImod (Flight Time):  {rsi_flt_str}\n"
            f"• Contraction Time:      {metrics.contraction_time_s * 1000:.0f} ms\n"
            f"• Peak Velocity:         {metrics.peak_velocity_m_per_s:.2f} m/s\n"
            f"• Peak Power:            {metrics.peak_power_w:.0f} W ({metrics.peak_relative_power_w_per_kg:.1f} W/kg)\n"
            f"• Peak Propulsion Force: {metrics.peak_propulsion_force_n:.0f} N\n"
            f"• Peak Landing Force:    {land_f_str}"
        )
        ax_vp.text(
            0.02, 0.97, metrics_box,
            transform=ax_vp.transAxes,
            fontsize=7.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9F9", edgecolor="#BDC3C7", alpha=0.92),
        )

        # Item G: legend only on row 0, placed outside axes (right side)
        if idx == 0:
            lines = [l_v, l_p]
            labels = [l.get_label() for l in lines]
            ax_vp.legend(
                lines, labels,
                loc="upper left",
                bbox_to_anchor=(1.08, 1.0),
                fontsize=8,
                framealpha=0.90,
            )

        ax_vp.set_title(f"Velocity & Power (onset → take-off)", fontsize=10, loc="left", color="#555555")

        if idx == 4:
            axes[idx, 0].set_xlabel("Time (s)", fontsize=11, fontweight="bold")
            axes[idx, 1].set_xlabel("Time (s)", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 0.88, 0.998])

    # Save figure
    out_local = Path("reports/figures/cmj_five_trials_detection.png")
    out_local.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_local, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_local}")


if __name__ == "__main__":
    main()
