"""Generate publication-quality plots comparing three synthetic CMJ trials.

1. High-resolution 2000 Hz CMJ
2. Downsampled 250 Hz CMJ (matching White et al. Zenodo format)
3. 2000 Hz CMJ with injected +5.0 N flight-phase force offset
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from shared.synth import make_cmj_trial

# Output path in artifacts directory
artifact_dir = Path(r"C:\Users\user\.gemini\antigravity-ide\brain\6e3673ef-b875-445b-8c76-f4561f0f03df")
artifact_dir.mkdir(parents=True, exist_ok=True)
plot_path = artifact_dir / "synthetic_cmj_comparison.png"

# Generate the three trials
cmj_2000 = make_cmj_trial(body_mass_kg=75.0, target_jump_height_m=0.35, fs_hz=2000.0, noise_sd_n=0.0)
cmj_250 = make_cmj_trial(body_mass_kg=75.0, target_jump_height_m=0.35, fs_hz=250.0, noise_sd_n=0.0)
cmj_offset = make_cmj_trial(body_mass_kg=75.0, target_jump_height_m=0.35, fs_hz=2000.0, flight_force_offset_n=5.0, noise_sd_n=0.0)

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

bw = cmj_2000.body_weight_n

# Panel 1: 2000 Hz
ax1 = axes[0]
ax1.plot(cmj_2000.time_s, cmj_2000.force_n, color="#1f77b4", lw=1.8, label="Vertical GRF (2000 Hz)")
ax1.axhline(bw, color="#7f7f7f", ls="--", lw=1.2, label=f"Body Weight ({bw:.1f} N)")
ax1.axhline(0, color="#d62728", ls=":", lw=1.0, alpha=0.7)
t_to = cmj_2000.true_metrics["t_takeoff_s"]
t_land = cmj_2000.true_metrics["t_landing_s"]
ax1.axvspan(t_to, t_land, color="#2ca02c", alpha=0.15, label="Flight Phase (F = 0 N)")
ax1.set_title("Panel A: Synthetic Countermovement Jump (2000 Hz, Clean) [Synthetic]", fontsize=12, fontweight="bold")
ax1.set_ylabel("Force (N)", fontsize=11)
ax1.legend(loc="upper right", frameon=True)
ax1.set_ylim(-50, max(cmj_2000.force_n) * 1.08)

# Panel 2: 250 Hz
ax2 = axes[1]
ax2.plot(cmj_250.time_s, cmj_250.force_n, color="#ff7f0e", lw=1.5, marker="o", markersize=3, alpha=0.85, label="Vertical GRF (250 Hz, dt = 4 ms)")
ax2.axhline(bw, color="#7f7f7f", ls="--", lw=1.2, label=f"Body Weight ({bw:.1f} N)")
ax2.axhline(0, color="#d62728", ls=":", lw=1.0, alpha=0.7)
ax2.axvspan(cmj_250.true_metrics["t_takeoff_s"], cmj_250.true_metrics["t_landing_s"], color="#2ca02c", alpha=0.15, label="Flight Phase")
ax2.set_title("Panel B: Synthetic Countermovement Jump (250 Hz, Discrete Markers) [Synthetic]", fontsize=12, fontweight="bold")
ax2.set_ylabel("Force (N)", fontsize=11)
ax2.legend(loc="upper right", frameon=True)
ax2.set_ylim(-50, max(cmj_250.force_n) * 1.08)

# Panel 3: Flight Offset (+5.0 N)
ax3 = axes[2]
ax3.plot(cmj_offset.time_s, cmj_offset.force_n, color="#9467bd", lw=1.8, label="Vertical GRF (+5.0 N Flight Offset)")
ax3.axhline(bw, color="#7f7f7f", ls="--", lw=1.2, label=f"Body Weight ({bw:.1f} N)")
ax3.axhline(0, color="#d62728", ls=":", lw=1.0, alpha=0.7, label="True Zero Reference (0 N)")
ax3.axhline(5.0, color="#e377c2", ls="-.", lw=1.2, label="Offset Level (+5.0 N)")
ax3.axvspan(cmj_offset.true_metrics["t_takeoff_s"], cmj_offset.true_metrics["t_landing_s"], color="#e377c2", alpha=0.2, label="Flight Phase (+5 N Offset)")
ax3.set_title("Panel C: Synthetic Countermovement Jump with Injected Flight Offset (+5.0 N) [Synthetic]", fontsize=12, fontweight="bold")
ax3.set_xlabel("Time (s)", fontsize=11)
ax3.set_ylabel("Force (N)", fontsize=11)
ax3.legend(loc="upper right", frameon=True)
ax3.set_ylim(-50, max(cmj_offset.force_n) * 1.08)

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Plot successfully saved to: {plot_path}")
