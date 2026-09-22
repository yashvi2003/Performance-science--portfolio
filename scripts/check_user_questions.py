"""
Verify White et al. (2025) cmj_dataset_both.npz properties across all 663 trials.
Answers the 5 specific pre-M2g verification questions.
"""
import numpy as np

def run_checks():
    data = np.load("data/raw/open/cmj_dataset_both.npz", allow_pickle=True)
    
    print(f"Data files: {data.files}")
    grf_signals = data["grf_signals"]
    acc_signals = data["acc_signals"]
    grf_takeoff = data["grf_takeoff"]
    acc_takeoff = data["acc_takeoff"]
    jump_height = data["jump_height"]
    peak_power = data["peak_power"]
    subject_ids = data["subject_ids"]
    
    n_trials = len(grf_signals)
    print(f"Total trials: {n_trials}")
    
    # 1. grf_takeoff vs len(grf_signals)
    grf_valid = [grf_takeoff[i] <= len(grf_signals[i]) - 1 for i in range(n_trials)]
    n_grf_valid = sum(grf_valid)
    n_grf_invalid = n_trials - n_grf_valid
    print("\n--- 1. GRF TAKEOFF VS SIGNAL LENGTH ---")
    print(f"Satisfying grf_takeoff[i] <= len(grf_signals[i]) - 1: {n_grf_valid} / {n_trials} ({n_grf_valid/n_trials*100:.1f}%)")
    print(f"Violating: {n_grf_invalid} / {n_trials} ({n_grf_invalid/n_trials*100:.1f}%)")
    
    # Examples
    print("Examples where grf_takeoff > len(grf_signals) - 1:")
    for idx in [0, 1, 2, 10, 100, 200]:
        print(f"  Trial {idx}: len(grf_signals)={len(grf_signals[idx])}, grf_takeoff={grf_takeoff[idx]}, diff={grf_takeoff[idx] - len(grf_signals[idx])}")
    
    if n_grf_valid > 0:
        print("Examples where grf_takeoff <= len(grf_signals) - 1:")
        valid_indices = [i for i, v in enumerate(grf_valid) if v]
        for idx in valid_indices[:5]:
            print(f"  Trial {idx}: len(grf_signals)={len(grf_signals[idx])}, grf_takeoff={grf_takeoff[idx]}")
    else:
        print("  None found: ALL 663 trials have grf_takeoff > len(grf_signals) - 1.")

    # 2. Terminal force grf_signals[i][-1]
    terminal_forces = np.array([grf_signals[i][-1] for i in range(n_trials)])
    print("\n--- 2. TERMINAL FORCE grf_signals[i][-1] ---")
    print(f"Mean: {np.mean(terminal_forces):.5f} BW")
    print(f"Std:  {np.std(terminal_forces):.5f} BW")
    print(f"Median: {np.median(terminal_forces):.5f} BW")
    print(f"IQR: [{np.percentile(terminal_forces, 25):.5f}, {np.percentile(terminal_forces, 75):.5f}] BW")
    print(f"Min:  {np.min(terminal_forces):.5f} BW")
    print(f"Max:  {np.max(terminal_forces):.5f} BW")
    print(f"Percentiles (1st, 5th, 95th, 99th): {np.percentile(terminal_forces, [1, 5, 95, 99])}")
    n_under_002 = np.sum(terminal_forces < 0.02)
    n_under_005 = np.sum(terminal_forces < 0.05)
    print(f"Trials with terminal force < 0.02 BW: {n_under_002} / {n_trials} ({n_under_002/n_trials*100:.2f}%)")
    print(f"Trials with terminal force < 0.05 BW: {n_under_005} / {n_trials} ({n_under_005/n_trials*100:.2f}%)")
    
    # Check if grf_signals[i][-1] is the minimum of the last 50 samples
    is_min_at_end = [np.argmin(grf_signals[i][-50:]) == len(grf_signals[i][-50:]) - 1 for i in range(n_trials)]
    print(f"Trials where final sample is min of last 50 samples: {sum(is_min_at_end)} / {n_trials} ({sum(is_min_at_end)/n_trials*100:.1f}%)")

    # 3. acc_takeoff vs len(acc_signals)
    acc_valid = [acc_takeoff[i] <= len(acc_signals[i]) - 1 for i in range(n_trials)]
    n_acc_valid = sum(acc_valid)
    n_acc_invalid = n_trials - n_acc_valid
    print("\n--- 3. ACC TAKEOFF VS SIGNAL LENGTH ---")
    print(f"Satisfying acc_takeoff[i] <= len(acc_signals[i]) - 1: {n_acc_valid} / {n_trials} ({n_acc_valid/n_trials*100:.1f}%)")
    print(f"Violating: {n_acc_invalid} / {n_trials} ({n_acc_invalid/n_trials*100:.1f}%)")
    acc_post_takeoff_samples = [len(acc_signals[i]) - acc_takeoff[i] for i in range(n_trials)]
    print(f"Post-takeoff samples in acc_signals: Mean={np.mean(acc_post_takeoff_samples):.1f} ({np.mean(acc_post_takeoff_samples)/250:.2f}s), Min={np.min(acc_post_takeoff_samples)}, Max={np.max(acc_post_takeoff_samples)}")
    print("Examples:")
    for idx in [0, 1, 2, 10, 100, 200]:
        print(f"  Trial {idx}: len(acc_signals)={len(acc_signals[idx])}, acc_takeoff={acc_takeoff[idx]}, post_takeoff_len={len(acc_signals[idx]) - acc_takeoff[idx]}")

    # 4. Quiet-phase baseline mean and SD across all 663 trials
    print("\n--- 4. QUIET-PHASE BASELINE (ACROSS ALL 663 TRIALS) ---")
    # Let's test a few standard window definitions:
    # A: First 300 ms (300 samples at 1000 Hz)
    # B: First 200 ms (200 samples)
    # C: First 20% of signal
    for win_len in [200, 300, 400]:
        means = np.array([np.mean(grf_signals[i][:win_len]) for i in range(n_trials)])
        sds = np.array([np.std(grf_signals[i][:win_len]) for i in range(n_trials)])
        print(f"Window: First {win_len} samples ({win_len/1000:.2f} s):")
        print(f"  Across-trial Baseline Mean: mean={np.mean(means):.4f} BW, std={np.std(means):.4f} BW, median={np.median(means):.4f} BW")
        print(f"  Percentiles of Mean [5th, 25th, 50th, 75th, 95th]: {np.percentile(means, [5, 25, 50, 75, 95])}")
        print(f"  Within-trial Noise SD: mean={np.mean(sds):.4f} BW, std={np.std(sds):.4f} BW, median={np.median(sds):.4f} BW")
        print(f"  Percentiles of SD [5th, 25th, 50th, 75th, 95th]: {np.percentile(sds, [5, 25, 50, 75, 95])}")

    # 5. Peak power reproduction on 20 random trials + overall
    print("\n--- 5. PEAK POWER REPRODUCTION ---")
    # Set seed for reproducible 20 random trials
    rng = np.random.default_rng(42)
    random_20 = rng.choice(n_trials, size=20, replace=False)
    
    g = 9.80665
    dt = 0.001  # 1000 Hz
    
    # Calculate for all trials
    calc_peak_powers_std_g = []
    calc_peak_powers_980_g = []
    
    for i in range(n_trials):
        f_bw = grf_signals[i]
        # acceleration in m/s^2 = (F_bw - 1.0) * g
        # velocity = cumsum(acc) * dt
        acc_std = (f_bw - 1.0) * g
        vel_std = np.cumsum(acc_std) * dt
        p_bw_std = (f_bw * g) * vel_std
        calc_peak_powers_std_g.append(np.max(p_bw_std))
        
        # Test with g=9.80
        acc_980 = (f_bw - 1.0) * 9.80
        vel_980 = np.cumsum(acc_980) * dt
        p_bw_980 = (f_bw * 9.80) * vel_980
        calc_peak_powers_980_g.append(np.max(p_bw_980))
        
    calc_peak_powers_std_g = np.array(calc_peak_powers_std_g)
    calc_peak_powers_980_g = np.array(calc_peak_powers_980_g)
    ref_peak_powers = np.array(peak_power)
    
    print("Comparison on 20 Random Trials (using standard g = 9.80665 m/s^2):")
    print(f"{'Trial':>6} | {'Ref Pk Pwr (W/kg)':>18} | {'Calc Pk Pwr (W/kg)':>18} | {'Diff (W/kg)':>12} | {'% Error':>10}")
    print("-" * 72)
    for idx in sorted(random_20):
        ref = ref_peak_powers[idx]
        calc = calc_peak_powers_std_g[idx]
        diff = calc - ref
        pct = (diff / ref) * 100
        print(f"{idx:6d} | {ref:18.4f} | {calc:18.4f} | {diff:12.4f} | {pct:9.3f}%")
        
    print("\nSummary across ALL 663 Trials:")
    diffs_std = calc_peak_powers_std_g - ref_peak_powers
    pct_std = (diffs_std / ref_peak_powers) * 100
    print(f"Standard g=9.80665:")
    print(f"  Mean diff: {np.mean(diffs_std):.4f} W/kg (Mean % error: {np.mean(pct_std):.4f}%)")
    print(f"  SD diff:   {np.std(diffs_std):.4f} W/kg (SD % error:   {np.std(pct_std):.4f}%)")
    print(f"  Max abs diff: {np.max(np.abs(diffs_std)):.4f} W/kg (Max abs % error: {np.max(np.abs(pct_std)):.4f}%)")
    corr = np.corrcoef(calc_peak_powers_std_g, ref_peak_powers)[0, 1]
    print(f"  Pearson correlation r: {corr:.8f}")

    diffs_980 = calc_peak_powers_980_g - ref_peak_powers
    pct_980 = (diffs_980 / ref_peak_powers) * 100
    print(f"\nIf using g=9.80000:")
    print(f"  Mean diff: {np.mean(diffs_980):.4f} W/kg (Mean % error: {np.mean(pct_980):.4f}%)")
    print(f"  SD diff:   {np.std(diffs_980):.4f} W/kg (SD % error:   {np.std(pct_980):.4f}%)")
    print(f"  Max abs diff: {np.max(np.abs(diffs_980)):.4f} W/kg")

if __name__ == "__main__":
    run_checks()
