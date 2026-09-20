# PENDING (do before M2c and before ticking M2b)
A. Offset results: reconcile the flight-offset results. Show uncorrected vs corrected height error side by side for both offset models and both height methods, and explain the +6.90 mm row.
B. Landing detection: make it robust to noise (threshold above the flight floor, scaled to flight noise, or a longer sustained window). Also report flight-time height error under noise.
C. Generator realism: random sub-sample event times, gradual unweighting onset, separate quiet-phase and flight-phase noise. Add several force-curve templates (single-peak, double-peak, varied unweighting depth, braking-phase force at v=0 well above body weight).
D. Momentum-consistent landing: net impulse over landing = m x landing velocity, peak landing force scaling with impact velocity, post-landing velocity back to ~0. Add a test.
E. Monte Carlo accuracy test: a few hundred random trials per condition and template. Print mean, SD and 95% limits for height (both methods), onset, take-off and landing.
F. Explain or remove the raw-signal notch to ~0 N near 2.03 s in the noisy trials.
G. Plots: restrict velocity/power panels to onset to take-off (or mark the landing window), move the legend off the data, fix y-limits.
H. List any tolerance that was loosened. Do not loosen any without telling me.
Keep event detection on the raw signal. Synthetic data only.
