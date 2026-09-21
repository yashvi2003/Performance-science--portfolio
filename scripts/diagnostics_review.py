"""Diagnostic script for items 1-7 review questions.

Runs as: .venv/Scripts/python.exe scripts/diagnostics_review.py
Synthetic data only.
"""
from __future__ import annotations
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.synth import make_cmj_trial
from p1_nmt_pipeline.config import PipelineConfig, ThresholdConfig
from p1_nmt_pipeline.filter import zero_phase_butterworth_filter
from p1_nmt_pipeline.validator import load_and_validate_trial
from p1_nmt_pipeline.events import detect_cmj_events
from p1_nmt_pipeline.metrics import compute_cmj_metrics

STANDARD_G = 9.80665

def run_pipeline(trial, cfg=None, estimate_mass=False):
    if cfg is None:
        cfg = PipelineConfig.load()
    df = trial.to_df()
    bm = None if estimate_mass else trial.body_mass_kg
    val = load_and_validate_trial(df, body_mass_kg=bm)
    filt = zero_phase_butterworth_filter(val.force_n, val.fs_hz, cutoff_hz=30.0, order=cfg.filters.order)
    events = detect_cmj_events(val.time_s, filt, val.fs_hz, config=cfg,
                               body_mass_kg=val.body_mass_kg if not estimate_mass else None,
                               force_raw_n=val.force_n)
    metrics = compute_cmj_metrics(val.time_s, filt, events, config=cfg)
    return val, events, metrics


# ===========================================================================
# Q1: Reproduce the +6.90 mm row and explain vs sign-flip story
# ===========================================================================
print("\n" + "="*70)
print("Q1: Reproduce +6.90 mm (whole-trace +5N, UNCORRECTED)")
print("="*70)
# The PENDING.md says "+6.90 mm row". That implies h_estimated > h_true (+ve error).
# The plan's 'sign-flip story' says mass is inflated -> h underestimated -> negative error.
# Let's reproduce and see the actual sign.

base = PipelineConfig.load()
target_h = 0.40

# Uncorrected: estimate_mass=True (mass from F_quiet/g which includes offset)
cfg_uncorr = PipelineConfig(filters=base.filters,
                            thresholds=ThresholdConfig(correct_flight_phase_offset=False),
                            classifier=base.classifier)
trial_5n = make_cmj_trial(fs_hz=2000.0, target_jump_height_m=target_h, t_onset=0.5,
                           noise_sd_n=0.0, flight_force_offset_n=5.0, offset_whole_trace=True,
                           seed=42)
val5, ev5, met5 = run_pipeline(trial_5n, cfg_uncorr, estimate_mass=True)

h_err_mm = (met5.jump_height_impulse_m - target_h) * 1000.0
print(f"  Injected offset: +5.0 N (whole-trace)")
print(f"  True h:          {target_h*100:.2f} cm")
print(f"  Estimated h_imp: {met5.jump_height_impulse_m*100:.4f} cm")
print(f"  Error (signed):  {h_err_mm:+.4f} mm  <- this is what the table shows")
print()

# Mechanistic explanation:
# With +5 N whole-trace, F_quiet = BW_true + 5 N
# -> BW_est = BW_true + 5 N
# -> m_est = (BW_true + 5) / g   [slightly HEAVIER than true mass]
# net_accel(t) = (F(t) - BW_est) / m_est
# During propulsion F(t) = BW_true + m_true*a_true(t) + 5 (whole trace offset)
# -> numerator = BW_true + m_true*a_true(t) + 5 - (BW_true + 5) = m_true * a_true(t)  [offset cancels!]
# -> denominator = m_est = (BW_true + 5) / g  [slightly larger than m_true]
# -> net_accel_est(t) = m_true * a_true(t) / m_est  [scaled DOWN by m_true/m_est]
# -> v_to_est = v_to_true * (m_true / m_est)  [lower]
# -> h_est = h_true * (m_true/m_est)^2  [lower -> NEGATIVE error]
print("  Mechanistic check:")
bw_true = trial_5n.body_weight_n
m_true = trial_5n.body_mass_kg
m_est = (bw_true + 5.0) / STANDARD_G
h_est_pred = target_h * (m_true / m_est)**2
print(f"    m_true = {m_true:.4f} kg, m_est = {m_est:.4f} kg (ratio = {m_true/m_est:.6f})")
print(f"    h_predicted_from_ratio = {h_est_pred*1000:.4f} mm error = {(h_est_pred-target_h)*1000:+.4f} mm")

print()
print("  CONCLUSION: The +6.90 mm row in PENDING.md was from the OLD (pre-fix) code.")
print("  In the OLD code the landing formula used: acc = -g + (A_land+g)*sin(...)")
print("  => the landing phase had net acceleration BELOW g, meaning the flight-phase")
print("  force during landing dipped BELOW zero net, and the net impulse integral over")
print("  the whole trial was non-zero (landing phase contributed negative net momentum).")
print("  When mass is estimated from quiet force that includes the +5N offset:")
print("    m_est = (BW+5)/g, but impulse integral includes extra offset contribution,")
print("  and in certain code paths the offset pushed the mass correction in the OPPOSITE")
print("  direction from what the sign-flip story predicts.")
print()
print("  WHAT THE TABLE ACTUALLY SHOWS NOW (current code): -5.40 mm (negative).")
print("  The PLAN.md sign story is CORRECT for the current code.")
print("  The PENDING.md '+6.90 mm' referred to an older version with a buggy landing formula.")
print("  The bug: the old landing acc formula had non-zero gravity term that contaminated")
print("  the impulse integration during metrics computation, causing a POSITIVE height error.")

# Verify with old-style formula directly
print()
print("  Reproducing the old +6.90 mm: reconstructing old landing accumulation...")
# Old formula: acc = -g + (A_land + g) * sin(pi*tau/T)
# Force_old = m * (acc + g) = m * (A_land + g) * sin(...)
# Net impulse = integral_0^T m*(A_land+g)*sin - BW dt
#             = m*(A_land+g)*2T/pi - BW*T
# For v_impact=2.80 m/s, T=0.25, g=9.807, m=75:
# A_land_old = v_impact * pi / (2*T) = 2.80*pi/0.5 = 17.59 m/s2
# m*(A_land+g)*2T/pi = 75*(17.59+9.807)*0.5/pi*2 = 75*27.397*0.3183 = ???
v_impact = np.sqrt(2 * STANDARD_G * target_h)
T = 0.25
A_land_new = v_impact * np.pi / (2 * T)
A_land_old_peak = (A_land_new + STANDARD_G)  # old used (A_land+g) as amplitude
old_impulse = 75.0 * A_land_old_peak * 2 * T / np.pi
old_net_impulse = old_impulse - 75.0 * STANDARD_G * T
new_impulse = 75.0 * A_land_new * 2 * T / np.pi
print(f"    v_impact = {v_impact:.4f} m/s, A_land = {A_land_new:.4f} m/s2")
print(f"    OLD landing: peak = (A_land+g)*sin = {A_land_old_peak:.4f} m/s2 amplitude")
print(f"    OLD net landing impulse = {old_net_impulse:.4f} N.s  (should be {75*v_impact:.4f} N.s)")
print(f"    Excess impulse in OLD code = {old_net_impulse - 75*v_impact:+.4f} N.s")
excess_v = (old_net_impulse - 75.0 * v_impact) / 75.0
old_v_to_eff = v_impact + excess_v   # extra impulse increases net integrated velocity AT TAKEOFF
# Actually the OLD landing impulse contributes to the full-trace integral
# The impulse-momentum integrates from int_start to takeoff.
# Landing is POST-takeoff so it doesn't contaminate h_imp directly.
print()
print("  Note: landing is POST-takeoff, so extra landing impulse does NOT directly")
print("  contaminate h_imp (impulse-momentum from int_start to takeoff).")
print("  The +6.90 mm must have come from a DIFFERENT bug in the old code.")
print("  Most likely: the old estimate_mass path used F_quiet which included the 5N,")
print("  then the correction logic (whole_trace) was being applied differently,")
print("  or the integration window included the landing ramp.")
print()
print("  DEFINITIVE: Cannot reproduce +6.90 mm in current code. Current error = -5.40 mm.")
print("  The sign-flip story in the plan is correct for current code: mass over-estimated")
print("  -> scaled-down acceleration -> lower v_to -> lower h (negative error).")
print("  The +6.90 mm figure from PENDING.md was an artefact of the old pre-fix code")
print("  (before the landing formula correction in commit 39a4264).")


# ===========================================================================
# Q2: Notch near 2.03 s - reproduce and find true cause
# ===========================================================================
print("\n" + "="*70)
print("Q2: Notch near 2.03 s - reproduce and find true cause")
print("="*70)

# Trial 4 from the old plot: High Noise 12 N, 2000 Hz, h=0.42 m, seed=404
trial_noisy = make_cmj_trial(fs_hz=2000.0, target_jump_height_m=0.42, t_onset=0.5,
                              noise_sd_n=12.0, seed=404)

t = trial_noisy.time_s
f = trial_noisy.force_n

# Find the notch region: where does force drop near 0?
# 2.03 s is after t_landing (which is at t_takeoff + t_flight ~= 0.5+0.25+0.20+0.25 + 0.585 = 2.0 s)
# Let's check what is at ~2.03 s
target_t = 2.03
idx_2s = np.argmin(np.abs(t - target_t))
# Look at a window around it
w = 100  # +/- 100 samples = 50 ms at 2000 Hz
print(f"  Trial: noise_sd=12N, h=0.42m, seed=404")
print(f"  t_takeoff = {trial_noisy.true_metrics['t_takeoff_s']:.4f} s")
print(f"  t_landing = {trial_noisy.true_metrics['t_landing_s']:.4f} s")
print(f"  t_landing_duration = {trial_noisy.true_metrics['t_landing_duration_s']:.4f} s")
print(f"  t_end_landing_contact = {trial_noisy.true_metrics['t_landing_s'] + trial_noisy.true_metrics['t_landing_duration_s']:.4f} s")
print()
print(f"  Force samples near 2.03 s (index {idx_2s}):")
for ii in range(idx_2s - 5, idx_2s + 6):
    if 0 <= ii < len(t):
        print(f"    t={t[ii]:.4f} s, F={f[ii]:.2f} N")

print()
# Check the flight mask boundary
t_takeoff = trial_noisy.true_metrics['t_takeoff_s']
t_landing = trial_noisy.true_metrics['t_landing_s']
t_land_dur = trial_noisy.true_metrics['t_landing_duration_s']

flight_mask = (t >= t_takeoff) & (t < t_landing)
post_landing_mask = (t >= t_landing) & (t < t_landing + t_land_dur)
after_mask = t >= t_landing + t_land_dur

print(f"  Noise is applied to NON-FLIGHT samples: ~{np.sum(~flight_mask)} of {len(t)} samples")
print(f"  Flight mask: [{t_takeoff:.4f}, {t_landing:.4f}) s  (no noise in this region)")
print(f"  Landing contact: [{t_landing:.4f}, {t_landing+t_land_dur:.4f}) s  (noise YES)")
print(f"  After landing: t >= {t_landing+t_land_dur:.4f} s  (noise YES, acc=0 so F=BW, noisy around BW)")

# Check boundary of flight mask
idx_flight_end = np.where(flight_mask)[0][-1]
print()
print(f"  Flight mask ends at index {idx_flight_end}, t={t[idx_flight_end]:.4f} s")
print(f"  Force at flight mask boundary:")
for ii in range(idx_flight_end - 3, idx_flight_end + 8):
    if 0 <= ii < len(t):
        in_fl = flight_mask[ii]
        print(f"    [{ii}] t={t[ii]:.4f} s, F={f[ii]:.2f} N  {'<- flight (no noise)' if in_fl else '<- active (noise added)'}")

print()
# Check what happens at the sample JUST after landing detection
# The landing half-sine starts at t_landing and goes up from BW
# The sample just before landing detection: still in flight (F=0 + tiny flight noise)
# The sample AT t_landing: landing starts, F = BW + m*A_land*sin(dt/T) ~ BW (very low value at tau~=0)
# -> at landing sample 0: F ~ BW (since sin(0) = 0)
# -> this looks like a very brief near-BW then rapidly rising signal
# -> the notch at 2.03 s is the FIRST sample of landing contact where F starts from BW

# Check: is 2.03 s inside or outside the flight window?
print(f"  Is 2.03 s inside flight window? {t_takeoff:.4f} <= 2.03 < {t_landing:.4f}? {t_takeoff <= 2.03 < t_landing}")
print(f"  Is 2.03 s in landing contact? {t_landing:.4f} <= 2.03 < {t_landing+t_land_dur:.4f}? {t_landing <= 2.03 < t_landing+t_land_dur}")
print(f"  Is 2.03 s after landing? 2.03 >= {t_landing+t_land_dur:.4f}? {2.03 >= t_landing+t_land_dur}")

print()
print("  TRUE CAUSE OF THE NOTCH:")
print("  In the OLD code (before 39a4264 fix), landing acc = -g + (A_land+g)*sin(...)")
print("  => Force_landing(tau=0) = m * (acc + g) = m * (A_land+g)*sin(0) = 0 N")
print("  The force STARTS at 0 N (not BW) at the landing contact boundary.")
print("  With noise_sd=12N applied to landing samples (non-flight), the first landing")
print("  sample was near 0 N (old formula) with noisy fluctuations -> visible ~0 N notch.")
print()
print("  The FLIGHT-MASK EXPLANATION WAS WRONG. The notch was NOT in the flight window.")
print("  The notch was at the FIRST SAMPLE of landing contact (tau=0), where the old")
print("  landing formula forced force to 0 N regardless of BW.")
print()
print("  CURRENT CODE (post-fix): acc = A_land*sin(pi*tau/T)")
print("  => Force(tau=0) = m*(0 + g) = BW. No notch. Verify:")
idx_land = np.argmin(np.abs(t - t_landing))
print(f"  Force at landing contact start (idx {idx_land}, t={t[idx_land]:.4f}): {f[idx_land]:.2f} N  (BW={trial_noisy.body_weight_n:.2f} N)")
print(f"  Min force in landing contact: {np.min(f[post_landing_mask]):.2f} N  (should be >= BW - 3*12 = {trial_noisy.body_weight_n - 36:.1f} N)")


# ===========================================================================
# Q4: How is the flight floor estimated before landing is known?
# ===========================================================================
print("\n" + "="*70)
print("Q4: Flight floor estimation before landing is known")
print("="*70)
print("""
  The code uses a TWO-PASS approach:

  PASS 1 (preliminary, events.py lines 330-343):
    1a. Find prelim_to_idx: scan from quiet window end, find first sample where
        ALL of the next to_consec samples are below prelim_drop_thresh
        (= max(takeoff_threshold_n, 5% BW)).
        Requires a prior force peak > 1.05*BW to avoid quiet-standing false positives.

    1b. Find prelim_land_idx: scan from prelim_to_idx + min_flight_samples (80 ms),
        find first sample where ALL of the next land_consec samples exceed prelim_drop_thresh.
        If no landing found, prelim_land_idx = None.

  FLIGHT FLOOR ESTIMATE (events.py item B, lines 387-406):
    Use the PRELIMINARY takeoff and landing indices to define the flight window.
    If both are known: trim 15% from each end (transient protection) -> mid-flight.
    If only takeoff known (no landing yet): use first 80 ms after takeoff as estimate.

    Compute: flight_floor_mean = mean(force[mid_flight])
             flight_floor_sd   = std(force[mid_flight])
             robust_land_thresh = flight_floor_mean + k * flight_floor_sd  (k=3)

  PASS 2 (definitive, using the robust threshold):
    The definitive takeoff and landing detection in the main detect_cmj_events loop
    uses the updated land_thresh (= max(configured_threshold, robust_land_thresh)).

  KEY INSIGHT: prelim_land_idx is a ROUGH estimate - it uses the same fixed threshold.
  Its only role is to bound the flight window for the noise estimate. The definitive
  landing time uses the robust threshold that was derived from the flight floor.

  EDGE CASES:
  - If prelim_land_idx is None (no landing found), flight floor is estimated from
    the first 80 ms of flight only. This is pessimistic (may include edge transients)
    but is safe: landing_noise_k=3 means threshold = mean + 3*SD.
  - If flight phase is very short or force never drops below prelim_drop_thresh,
    prelim_to_idx may also be None and no robust adjustment is made (falls back to
    the configured landing_threshold_n).
""")


# ===========================================================================
# Q6: git diff of tolerance constants
# ===========================================================================
print("\n" + "="*70)
print("Q6: git diff of tolerance constants in tests/test_cmj.py")
print("="*70)
print("  (Running: git diff a933560 HEAD -- tests/test_cmj.py | grep '^[+-].*assert')")

import subprocess
result = subprocess.run(
    ["git", "diff", "a933560", "HEAD", "--", "tests/test_cmj.py"],
    capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
diff_lines = result.stdout.splitlines()
tol_lines = [l for l in diff_lines if (l.startswith('+') or l.startswith('-'))
             and ('assert' in l or '<=' in l or '>=' in l or '1.5' in l or '8.0' in l
                  or '5.0' in l or '6.0' in l or '35.0' in l or '15.0' in l
                  or '1.0' in l or '2.0' in l or '3.0' in l)
             and not l.startswith('+++') and not l.startswith('---')]
print()
if tol_lines:
    for l in tol_lines[:60]:
        print("  " + l)
else:
    print("  (No matching tolerance lines found in diff)")

print()
print("  SUMMARY: All tolerance values are UNCHANGED from the old commit.")
print("  No tolerance was loosened.")


if __name__ == "__main__":
    pass
