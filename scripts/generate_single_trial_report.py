"""Generate a publication-grade, VALD-style single-trial CMJ diagnostic report figure.

Visualizes a representative real-world 2000 Hz CMJ trial:
1. Header bar with test classification and trial metadata (anonymized).
2. Vertical Force-Time curve with McMahon et al. (2018) movement phases shaded:
   - Unweighting (onset -> min force)
   - Braking / Eccentric (min force -> v=0)
   - Propulsion / Concentric (v=0 -> takeoff)
   - Flight (takeoff -> landing)
   - Landing Impact (landing -> impact peak -> stabilization)
3. Coupled Velocity-Time panel showing COM velocity trajectory.
4. Coupled Power-Time panel showing instantaneous mechanical power.
5. High-density metrics summary card formatted in VALD Performance style.

ANONYMIZATION GUARDRAIL:
No athlete IDs, private file paths, or individual-identifying metadata are embedded.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from p1_nmt_pipeline.config import PipelineConfig
from p1_nmt_pipeline.real_data_adapter import run_pipeline_on_all
from shared.contract import STANDARD_G

CMJ_DIR = Path("data/raw/cmj")
OUTPUT_PATH = Path("reports/figures/cmj_single_trial_report.png")


def generate_single_trial_report() -> Path:
    if not CMJ_DIR.exists():
        raise FileNotFoundError(f"Directory {CMJ_DIR} does not exist.")

    cfg = PipelineConfig.load()
    results = run_pipeline_on_all(CMJ_DIR, config=cfg)
    if len(results) < 2:
        raise ValueError("At least two trials are required to select Trial 2.")

    # Select representative clean trial (Trial index 1 = second real trial)
    r = results[1]
    v = r.validated
    e = r.events
    m = r.metrics

    t = v.time_s
    f_raw = v.force_n
    f_filt = r.filtered_force
    fs = v.fs_hz
    bw = e.quiet_bw_mean_n
    mass = v.body_mass_kg

    # Focus visualization window: 0.35 s before onset to 0.45 s after landing peak
    t_start = max(0.0, e.onset_time_s - 0.35)
    t_end = min(t[-1], (e.landing_peak_time_s or (e.takeoff_time_s + 0.8)) + 0.50)

    # Compute Velocity array via trapezoidal integration from int_start_idx
    acc = (f_filt - bw) / mass
    vel = np.zeros_like(f_filt)
    for i in range(e.int_start_idx, len(vel) - 1):
        dt = t[i + 1] - t[i]
        vel[i + 1] = vel[i] + 0.5 * (acc[i] + acc[i + 1]) * dt

    # Overwrite flight phase with exact ballistic COM velocity: v(t) = v_to - g*(t - t_to)
    to_idx = e.takeoff_idx
    if e.landing_idx is not None:
        land_idx = e.landing_idx
        for k in range(to_idx, land_idx):
            vel[k] = vel[to_idx] - STANDARD_G * (t[k] - t[to_idx])

    # Power = F_filt * v (isolated strictly to ground contact propulsion prior to takeoff)
    power_w = np.zeros_like(f_filt)
    power_w[e.v0_idx:to_idx] = f_filt[e.v0_idx:to_idx] * vel[e.v0_idx:to_idx]

    # Set up publication figure with high DPI
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(15, 10.5), dpi=300, facecolor="#F8FAFC")

    # Grid layout: Header (top), Force-Time (middle row), Velocity & Power & Metrics (bottom row)
    gs = gridspec.GridSpec(
        3, 3,
        height_ratios=[0.08, 0.48, 0.44],
        width_ratios=[1.0, 1.0, 0.95],
        hspace=0.28,
        wspace=0.25,
        left=0.06, right=0.95, top=0.96, bottom=0.06
    )

    # -------------------------------------------------------------------------
    # 1. HEADER BANNER
    # -------------------------------------------------------------------------
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_hdr.axis("off")
    ax_hdr.set_facecolor("#F8FAFC")

    # Dark banner card
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax_hdr.transAxes,
                         color="#0F172A", clip_on=False)
    ax_hdr.add_patch(rect)

    ax_hdr.text(0.02, 0.65, "COUNTERMOVEMENT JUMP (CMJ) SINGLE-TRIAL REPORT",
                color="#FFFFFF", fontsize=15, fontweight="bold", transform=ax_hdr.transAxes, va="center")
    ax_hdr.text(0.02, 0.28, "Neuromuscular Monitoring Engine | Dual Force Plate Analysis (2000 Hz) | McMahon et al. (2018) Phase Model",
                color="#94A3B8", fontsize=9.5, transform=ax_hdr.transAxes, va="center")

    # Status Badge
    badge = plt.Rectangle((0.80, 0.25), 0.18, 0.50, transform=ax_hdr.transAxes,
                          color="#059669", clip_on=False)
    ax_hdr.add_patch(badge)
    ax_hdr.text(0.89, 0.50, "VALID TRIAL - CMJ",
                color="#FFFFFF", fontsize=10, fontweight="bold", ha="center", va="center", transform=ax_hdr.transAxes)

    # -------------------------------------------------------------------------
    # 2. MAIN FORCE-TIME CURVE (Spanning Top Row)
    # -------------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, :])
    ax_f.set_facecolor("#FFFFFF")

    # Color palette for movement phases (harmonious, subtle pastels)
    c_unweight = "#BAE6FD"  # Sky blue
    c_brake = "#FEF08A"     # Amber
    c_prop = "#A7F3D0"      # Emerald
    c_flight = "#E9D5FF"    # Purple
    c_land = "#FECDD3"      # Rose

    # Phase shading
    ax_f.axvspan(e.onset_time_s, e.unweighting_min_time_s, color=c_unweight, alpha=0.55, label="Unweighting")
    ax_f.axvspan(e.unweighting_min_time_s, e.v0_time_s, color=c_brake, alpha=0.55, label="Braking (Eccentric)")
    ax_f.axvspan(e.v0_time_s, e.takeoff_time_s, color=c_prop, alpha=0.55, label="Propulsion (Concentric)")
    if e.landing_time_s is not None:
        ax_f.axvspan(e.takeoff_time_s, e.landing_time_s, color=c_flight, alpha=0.50, label="Flight")
        if e.landing_peak_time_s is not None:
            t_land_end = min(e.landing_peak_time_s + 0.20, t_end)
            ax_f.axvspan(e.landing_time_s, t_land_end, color=c_land, alpha=0.50, label="Landing Impact")

    # Traces: Raw Force (subtle grey) + Filtered Force (deep navy)
    ax_f.plot(t, f_raw, color="#94A3B8", linewidth=0.75, alpha=0.7, label="Raw Force (2000 Hz)")
    ax_f.plot(t, f_filt, color="#0F172A", linewidth=2.0, label="Filtered Force (30 Hz Zero-Phase LP)")

    # Body Weight Line
    ax_f.axhline(bw, color="#475569", linestyle="--", linewidth=1.2,
                 label=f"Body Weight (BW = {bw:.0f} N | {mass:.1f} kg)")

    # Key landmark markers
    ax_f.scatter([e.onset_time_s], [f_filt[e.onset_idx]], color="#7C3AED", s=55, zorder=6,
                 label=f"Movement Onset (t={e.onset_time_s:.3f} s)")
    ax_f.scatter([e.unweighting_min_time_s], [e.unweighting_min_force_n], color="#0284C7", marker="v", s=65, zorder=6,
                 label=f"Min Unweighting ({e.unweighting_min_force_n:.0f} N)")
    ax_f.scatter([e.v0_time_s], [f_filt[e.v0_idx]], color="#D97706", marker="s", s=55, zorder=6,
                 label=f"Zero Velocity Transition (v=0 at t={e.v0_time_s:.3f} s)")
    ax_f.scatter([e.propulsion_peak_time_s], [e.propulsion_peak_force_n], color="#059669", marker="^", s=75, zorder=6,
                 label=f"Peak Propulsion Force ({e.propulsion_peak_force_n:.0f} N | {e.propulsion_peak_force_n/bw:.2f} BW)")
    ax_f.scatter([e.takeoff_time_s], [f_filt[e.takeoff_idx]], color="#DC2626", marker="x", s=75, linewidths=2.2, zorder=6,
                 label=f"Takeoff (t={e.takeoff_time_s:.3f} s)")
    if e.landing_time_s is not None:
        ax_f.scatter([e.landing_time_s], [f_filt[e.landing_idx]], color="#BE185D", marker="o", s=60, zorder=6,
                     label=f"Landing Contact (t={e.landing_time_s:.3f} s)")
    if e.landing_peak_time_s is not None:
        ax_f.scatter([e.landing_peak_time_s], [e.landing_peak_force_n], color="#991B1B", marker="*", s=110, zorder=6,
                     label=f"Landing Peak Force ({e.landing_peak_force_n:.0f} N | {e.landing_peak_force_n/bw:.2f} BW)")

    ax_f.set_xlim(t_start, t_end)
    ax_f.set_ylim(-50, max(e.landing_peak_force_n * 1.12, 3000))
    ax_f.set_ylabel("Vertical Force (N)", fontsize=11, fontweight="bold", color="#1E293B")
    ax_f.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold", color="#1E293B")
    ax_f.grid(True, linestyle=":", color="#E2E8F0", linewidth=0.8)
    ax_f.tick_params(labelsize=9.5)

    # Clean multi-column legend
    ax_f.legend(loc="upper right", fontsize=8.2, ncol=3, framealpha=0.92, facecolor="#FFFFFF", edgecolor="#CBD5E1")

    # -------------------------------------------------------------------------
    # 3. LOWER-LEFT PANEL: COM Velocity (m/s)
    # -------------------------------------------------------------------------
    ax_v = fig.add_subplot(gs[2, 0])
    ax_v.set_facecolor("#FFFFFF")

    ax_v.plot(t, vel, color="#2563EB", linewidth=1.8, label="COM Velocity")
    ax_v.axhline(0.0, color="#64748B", linestyle="--", linewidth=1.0)

    # Shaded contraction & flight zones
    ax_v.axvspan(e.onset_time_s, e.takeoff_time_s, color="#F1F5F9", alpha=0.8, label="Contraction Phase")
    if e.landing_time_s is not None:
        ax_v.axvspan(e.takeoff_time_s, e.landing_time_s, color=c_flight, alpha=0.35, label="Flight Phase")

    # Mark peak takeoff velocity
    v_to = vel[e.takeoff_idx]
    ax_v.scatter([e.takeoff_time_s], [v_to], color="#2563EB", marker="o", s=50, zorder=5)
    ax_v.annotate(f"v_to = {v_to:.2f} m/s",
                  xy=(e.takeoff_time_s, v_to), xytext=(e.takeoff_time_s - 0.25, v_to + 0.35),
                  arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.0),
                  fontsize=8.5, fontweight="bold", color="#1E3A8A")

    ax_v.set_xlim(t_start, t_end)
    ax_v.set_ylim(min(np.min(vel[int(t_start*fs):int(t_end*fs)]) * 1.25, -2.5), 3.5)
    ax_v.set_ylabel("COM Velocity (m/s)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax_v.set_xlabel("Time (seconds)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax_v.set_title("Center of Mass Velocity Profile", fontsize=10.5, fontweight="bold", loc="left", color="#0F172A")
    ax_v.grid(True, linestyle=":", color="#E2E8F0", linewidth=0.8)
    ax_v.tick_params(labelsize=9)

    # -------------------------------------------------------------------------
    # 4. LOWER-CENTER PANEL: Mechanical Power (W)
    # -------------------------------------------------------------------------
    ax_p = fig.add_subplot(gs[2, 1])
    ax_p.set_facecolor("#FFFFFF")

    # Plot positive power during movement
    power_plot = np.clip(power_w, a_min=0.0, a_max=None)
    ax_p.plot(t, power_plot, color="#EA580C", linewidth=1.8, label="Instantaneous Power")

    # Shaded propulsion phase
    ax_p.axvspan(e.v0_time_s, e.takeoff_time_s, color=c_prop, alpha=0.45, label="Propulsion Window")

    # Mark peak propulsion power
    ax_p.scatter([e.takeoff_time_s - 0.05], [m.peak_power_w], color="#C2410C", marker="^", s=60, zorder=5)
    ax_p.annotate(f"P_peak = {m.peak_power_w:.0f} W\n({m.peak_relative_power_w_per_kg:.1f} W/kg)",
                  xy=(e.takeoff_time_s - 0.05, m.peak_power_w),
                  xytext=(e.v0_time_s - 0.35, m.peak_power_w * 0.85),
                  arrowprops=dict(arrowstyle="->", color="#C2410C", lw=1.0),
                  fontsize=8.5, fontweight="bold", color="#7C2D12")

    ax_p.set_xlim(t_start, t_end)
    ax_p.set_ylim(0, max(m.peak_power_w * 1.18, 4000))
    ax_p.set_ylabel("Mechanical Power (W)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax_p.set_xlabel("Time (seconds)", fontsize=10.5, fontweight="bold", color="#1E293B")
    ax_p.set_title("Concentric Power Output", fontsize=10.5, fontweight="bold", loc="left", color="#0F172A")
    ax_p.grid(True, linestyle=":", color="#E2E8F0", linewidth=0.8)
    ax_p.tick_params(labelsize=9)

    # -------------------------------------------------------------------------
    # 5. LOWER-RIGHT PANEL: VALD-Style Metrics Summary Card
    # -------------------------------------------------------------------------
    ax_card = fig.add_subplot(gs[2, 2])
    ax_card.axis("off")
    ax_card.set_facecolor("#FFFFFF")

    # Outer rounded container box
    card_bg = plt.Rectangle((0, 0), 1, 1, transform=ax_card.transAxes,
                            facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5, clip_on=False)
    ax_card.add_patch(card_bg)

    # Card Title
    title_bar = plt.Rectangle((0, 0.88), 1, 0.12, transform=ax_card.transAxes,
                              facecolor="#1E293B", clip_on=False)
    ax_card.add_patch(title_bar)
    ax_card.text(0.50, 0.94, "KEY PERFORMANCE METRICS",
                 color="#FFFFFF", fontsize=10.5, fontweight="bold", ha="center", va="center", transform=ax_card.transAxes)

    # Metrics rows (formatted key-value table)
    h_flt_str = f"{m.jump_height_flight_m*100:.1f} cm" if m.jump_height_flight_m else "N/A"
    metrics_data = [
        ("Jump Height (Impulse-Mom.)", f"{m.jump_height_impulse_m*100:.1f} cm", True),
        ("Jump Height (Flight Time)", h_flt_str, False),
        ("Contraction Time (CT)", f"{m.contraction_time_s*1000:.0f} ms", False),
        ("Flight Time (FT)", f"{(m.flight_time_s or 0.0)*1000:.0f} ms", False),
        ("RSI-modified (RSImod)", f"{m.rsimod_impulse:.3f} m/s", True),
        ("Peak Power (Relative)", f"{m.peak_relative_power_w_per_kg:.1f} W/kg", True),
        ("Peak Power (Absolute)", f"{m.peak_power_w:.0f} W", False),
        ("Peak Propulsion Force", f"{e.propulsion_peak_force_n:.0f} N ({e.propulsion_peak_force_n/bw:.2f} BW)", False),
        ("Landing Peak Force", f"{e.landing_peak_force_n:.0f} N ({e.landing_peak_force_n/bw:.2f} BW)", False),
        ("Athlete Body Mass", f"{mass:.1f} kg ({bw:.0f} N)", False),
    ]

    y_pos = 0.81
    row_height = 0.075
    for idx, (label, val, highlight) in enumerate(metrics_data):
        # Alternating subtle row fill
        if idx % 2 == 1:
            row_rect = plt.Rectangle((0.02, y_pos - 0.015), 0.96, row_height,
                                     facecolor="#F8FAFC", clip_on=False, transform=ax_card.transAxes)
            ax_card.add_patch(row_rect)

        txt_color = "#0F172A" if not highlight else "#0369A1"
        font_weight = "bold" if highlight else "normal"
        val_color = "#0F172A" if not highlight else "#0284C7"

        ax_card.text(0.05, y_pos + 0.015, label, fontsize=9.0, color=txt_color,
                     fontweight=font_weight, va="center", transform=ax_card.transAxes)
        ax_card.text(0.95, y_pos + 0.015, val, fontsize=9.2, color=val_color,
                     fontweight="bold", ha="right", va="center", transform=ax_card.transAxes)

        y_pos -= row_height

    # Footer note on anonymization & method
    ax_card.text(0.50, 0.03, "Validated 2000 Hz Force Plate Output | De-identified Trial",
                 fontsize=7.5, color="#64748B", style="italic", ha="center", va="center", transform=ax_card.transAxes)

    # Save figure
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"VALD-style single-trial report generated successfully at: {OUTPUT_PATH.resolve()}")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate_single_trial_report()
