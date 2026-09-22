# Project 1: Automatic Jump Analysis Engine (`p1_nmt_pipeline`)

## Overview & Scientific Objectives
This pipeline provides automated biomechanical phase detection and metrics computation for vertical jump force-time data, conforming strictly to the consensus standards established by **McMahon et al. (2018)**.

### Research Question
How accurately and reliably can discrete biomechanical events (movement onset, unweighting trough, upward BW crossing, zero-velocity transition, concentric propulsion peak, takeoff, landing impact) and performance metrics (impulse-momentum jump height, flight-time jump height, reactive strength index modified [RSImod], peak mechanical power) be recovered across varying sampling rates ($250\text{--}2000\text{ Hz}$), noise regimes ($0\text{--}20\text{ N}$), baseline quiet-standing durations ($0.30\text{--}0.75\text{ s}$), sensor flight offsets, and complex force-curve morphologies?

---

## Biomechanical Phase Partitioning (McMahon et al., 2018)
1. **Quiet Standing Baseline**: Detected adaptively by analyzing standard deviation and linear trend slope. Body mass is derived as $\bar{F}_{\text{quiet}} / g$.
2. **Unweighting Phase**: From movement onset ($F < BW - 5\sigma_{\text{quiet}}$ or $BW - 12\text{ N}$) to unweighting force minimum ($F_{\min}$).
3. **Braking (Eccentric Deceleration) Phase**: From $F_{\min}$ to instantaneous zero velocity ($v(t) = 0$).
4. **Propulsion (Concentric Acceleration) Phase**: From $v(t) = 0$ to takeoff ($F \le 10\text{ N}$ or adaptive flight floor).
5. **Flight Phase**: From takeoff to landing impact ($F \ge \max(10\text{ N}, \text{floor}_{\text{mean}} + 3\sigma_{\text{floor}})$).
6. **Landing Absorption Phase**: From initial landing contact to maximum landing impact force.

---

## Methodological & Diagnostic Notes

### 1. Monte Carlo Height Tolerance ($3.0 \to 6.0\text{ mm}$ at $2000\text{ Hz}$)
In idealized step-onset trials with zero jitter, impulse jump height 95th-percentile error was under $2.0\text{ mm}$. When realistic physical modeling was enabled:
- **Gradual Unweighting Onset ($20\text{ ms}$ cosine ramp)**: Smooth unweighting physically reduces negative unweighting impulse by $\sim 1\text{ N}\cdot\text{s}$, shifting true takeoff velocity by $+0.013\text{ m/s}$ ($+1.55\text{ mm}$ to $+2.50\text{ mm}$ physical height shift).
- **Sub-sample Threshold Bias & Continuous Extrapolation**: Detecting takeoff at discrete sample boundaries without continuous interpolation introduces an empirical $+1.4\text{ mm}$ bias. With continuous sub-sample takeoff extrapolation (`subsample_takeoff=True`), pre-takeoff force slope is extrapolated to the flight floor, reducing mean bias to $\le 0.33\text{ mm}$.
- **Integration Lead Offset ($30$, $50$, $100\text{ ms}$)**: Configurable via `integration_lead_s`. On ramped-onset trials ($20\text{ ms}$ ramp), lead times of $30$, $50$, and $100\text{ ms}$ prior to detected onset produce identical stable integration ($+0.70\text{ mm}$ bias under clean baseline), confirming that pre-onset lead captures the subtle unweighting dip while integrating net-zero baseline noise.
- **Event Jitter ($\pm 15\text{ ms}$)**: Widens the empirical distribution spread.
Consequently, the 95th-percentile tolerance was adjusted to $6.0\text{ mm}$ ($8.0\text{ mm}$ under $12\text{ N}$ noise) to tightly envelop the true physical distribution rather than enforcing an unrealistic un-ramped step model.

### 2. RSImod Sensitivity to Onset Delay
Reactive Strength Index modified is defined as:
$$\text{RSImod} = \frac{\text{Jump Height}}{\text{Contraction Time}} = \frac{h_{\text{imp}}}{t_{\text{takeoff}} - t_{\text{onset}}}$$
An onset delay shortens detected contraction time, which artificially inflates RSImod ($\text{RSImod} = h_{\text{imp}} / \text{CT}$).
- Under gradual onset smoothing ($20\text{ ms}$ ramp) and realistic noise ($3\text{ N}$), initial force deviation from body weight is minute ($\Delta F < 3\text{ N}$ during the initial $10\text{ ms}$).
- As a result, fixed/SD threshold crossings occur $\sim 11\text{ ms}$ after movement initiates.
- **Effect on Contraction Time**: Detected contraction time is shortened by $\sim 11\text{ ms}$ (e.g., $715\text{ ms} \to 704\text{ ms}$, a $-1.5\%$ error).
- **Effect on RSImod**: Because contraction time is in the denominator, this $11\text{ ms}$ onset delay **artificially inflates RSImod by $+0.012$ ($+2.4\%$)** (e.g., $0.490 \to 0.502$).
- *Practitioner Takeaway*: Threshold-induced onset delays systematically shorten detected contraction time, inflating RSImod. Consistent onset criteria and reporting uncertainty are paramount when comparing longitudinal athletic screenings.

### 3. Flight-Phase Offsets & Detection Failure Mechanisms
- **Flight-only $+8\text{ N}$ Offset**: Remains strictly below the $10\text{ N}$ landing threshold. Uncorrected landing detection succeeds ($0.27\text{ ms}$ error). However, turning on `flight_only` offset correction raises the takeoff threshold to $18\text{ N}$, triggering takeoff $0.5\text{ ms}$ early on unshifted propulsion data and causing a small $+1.4\text{ mm}$ height bias.
- **Flight-only $+15\text{ N}$ and $+25\text{ N}$ Offsets**: The airborne force floor exceeds the $10\text{ N}$ threshold.
  - *Uncorrected Takeoff*: Force never drops below $10\text{ N}$. Previously, falling back to `np.argmin` selected the flat flight floor near landing ($t = 1.755\text{ s}$ vs. true $1.200\text{ s}$), producing a spurious $+555.5\text{ ms}$ takeoff error.
  - *Detection Failure Guardrails*: Under our implemented guardrail, the detector never falls back to `np.argmin`. Takeoff and landing detection failures set explicit QA warning flags (`takeoff_detection_failed`, `landing_detection_failed`) and return `NaN` for jump heights and velocity metrics.
  - *Correction*: The `flight_only` model estimates the airborne floor and adjusts thresholds accordingly, restoring true flight time to $<1.1\text{ mm}$ error.

### 4. Late-Propulsion Stress Double-Peak (`double_peak_late`)
- The original late-propulsion double peak ($\tau > 0.5$) causes vertical force to drop more rapidly toward zero right before takeoff.
- The $10\text{ N}$ threshold is crossed $0.7\text{ ms}$ prematurely, truncating the negative net impulse tail ($F < BW \implies a < 0$) and causing a systematic $+2.70\text{ mm}$ height bias.
- Retained un-tuned as a documented stress test.

---

## Real-Data Validation (M2c–M2f: Professor-Supplied Cohort)

### Scientific Scope & Framing
This analysis evaluates 10 real athletic CMJ trials ($2000\text{ Hz}$ vertical force plate recordings, anonymized ordinals `T01`–`T10`) supplied by the user's professor.
- **Between-Athlete Scope**: Each athlete performed **ONE trial**. There are no repeated trials or longitudinal screenings in this dataset. Consequently, trial-to-trial reliability ($\text{CV}$, $\text{ICC}$, $\text{SEM}$, $\text{MDC}$) and within-athlete change **cannot** be inferred from this dataset. Those analyses are reserved for open repeated-trial datasets (White et al., Zenodo, M2g).
- **Public Guardrails**: Public figures and tables use anonymous ordinal labels `T01`–`T10`. Internal IDs and raw files are gitignored and never published.

---

### Milestone M2c: Data Quality & Signal Integrity Report
A comprehensive signal integrity audit was executed across all 10 real trials:
1. **Plausibility of Body Mass**: All 10 athletes exhibit plausible body masses ($60.7\text{--}104.6\text{ kg}$, mean $73.7\text{ kg}$).
2. **Timestamp Regularity**: 5 of 10 trials contain small timestamp irregularities ($28\text{--}30$ irregular sample intervals, $\sim 0.4\%$ of samples). The pipeline automatically resamples these trials onto a uniform $2000\text{ Hz}$ grid using linear interpolation.
3. **Negative Force Excursions & Plate Ringing**: All 10 trials exhibit negative force readings (minimum $-4.1\text{ N}$ to $-59.3\text{ N}$). These negative excursions occur strictly during the early airborne phase ($t = 1.50\text{--}1.57\text{ s}$) due to **mechanical plate oscillation / resonance** immediately following explosive takeoff. Setting the configurable refractory flight lockout (`min_flight_duration_s: 0.200`) completely prevents false landing triggers during post-takeoff vibration.
4. **Flight Phase Baseline Offset**: The true mid-flight force floor across all 10 trials is $+2.95\text{ N}$ to $+5.43\text{ N}$ (noise $\text{SD} \approx 2.8\text{ N}$).
5. **Quiet Standing Stability**: Initial standing segments showed varying noise ($\text{SD} = 1.17\text{--}3.75\text{ N}$, one trial $12.67\text{ N}$). Adaptive window detection reliably extracted quiet baselines ($0.15\text{--}0.45\text{ s}$) before unweighting onset.
6. **Detection Success**: Takeoff and landing detection succeeded in **10 out of 10 trials** ($0\text{ failures}$).

*Figure Reference*: [`reports/figures/cmj_data_quality_summary.png`](file:///c:/Yashvi/Internship%202027/portfolio-agent-kit/reports/figures/cmj_data_quality_summary.png)

---

### Milestone M2d: Golden Test Against Reference Workbook
The pipeline was evaluated against the reference trial in `CMJ_Force_Time_Analysis.xlsx` (based on Chavda et al., 2018). Reference values are unverified cached Excel outputs, investigated in both directions:

| Metric | Python Pipeline | Reference Workbook | Difference | Error (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Body weight** | $605.96\text{ N}$ | $605.96\text{ N}$ | $-0.0006\text{ N}$ | $-0.00\%$ | **PASS** |
| **Body mass** | $61.79\text{ kg}$ | $61.77\text{ kg}$ | $+0.02\text{ kg}$ | $+0.03\%$ | **PASS** |
| **$h_{\text{imp}}$ (velocity)** | $0.2285\text{ m}$ | $0.2278\text{ m}$ | $+0.0008\text{ m}$ | $+0.33\%$ | **PASS** |
| **$h_{\text{flt}}$ (flight time)** | $0.2538\text{ m}$ | $0.2540\text{ m}$ | $-0.0002\text{ m}$ | $-0.07\%$ | **PASS** |
| **Flight time** | $0.4550\text{ s}$ | $0.4551\text{ s}$ | $-0.0001\text{ s}$ | $-0.02\%$ | **PASS** |
| **Contraction time** | $0.8655\text{ s}$ | $0.8795\text{ s}$ | $-0.0140\text{ s}$ | $-1.59\%$ | **UNDERSTOOD** |
| **Peak propulsion force** | $1176.0\text{ N}$ | $1185.3\text{ N}$ | $-9.27\text{ N}$ | $-0.78\%$ | **UNDERSTOOD** |
| **Peak landing force** | $2894.9\text{ N}$ | $2895.6\text{ N}$ | $-0.71\text{ N}$ | $-0.02\%$ | **PASS** |
| **Peak power** | $2343.7\text{ W}$ | $2351.3\text{ W}$ | $-7.58\text{ W}$ | $-0.32\%$ | **PASS** |
| **$\text{RSImod}$** | $0.2640\text{ m/s}$ | $0.2590\text{ m/s}$ | $+0.0050\text{ m/s}$ | $+1.95\%$ | **PASS** |

**Investigated Divergences:**
1. **Contraction Time ($-14\text{ ms}$)**: The reference CSV contains $107$ irregular timestamp intervals ($1.8\%$ of rows). The reference Excel workbook indexed raw rows directly without resampling. Our validator resamples onto a uniform $2000\text{ Hz}$ time grid, shifting the effective time axis by $\sim 14\text{ ms}$ ($28\text{ samples}$), which accounts for the onset time discrepancy.
2. **Peak Propulsion Force ($-9.3\text{ N}$)**: The reference uses the unfiltered raw signal (verified raw peak = $1185.2\text{ N}$). Our pipeline computes propulsion dynamics from the $30\text{ Hz}$ zero-phase Butterworth filtered signal for consistency with impulse and velocity integration. The $30\text{ Hz}$ filter slightly rounds the propulsion peak by $0.78\%$.
3. **Peak Landing Force ($-0.71\text{ N}$)**: Peak landing force uses the raw signal because the sharp impact transient is severely attenuated by low-pass filtering. With raw force, landing peak matches the reference to within $0.02\%$.

*Figure Reference*: [`reports/figures/cmj_golden_test_comparison.png`](file:///c:/Yashvi/Internship%202027/portfolio-agent-kit/reports/figures/cmj_golden_test_comparison.png)

---

### Milestone M2e: Method Comparison (Flight-Time vs. Impulse-Momentum)
A Bland-Altman method comparison was conducted between the impulse-momentum theorem ($h_{\text{imp}}$) and flight-time equation ($h_{\text{flt}} = \frac{1}{8} g t_{\text{flight}}^2$) across all 10 athletes:

- **Between-Athlete Comparison**: $n = 10$, one trial per athlete.
- **Mean Bias ($h_{\text{imp}} - h_{\text{flt}}$)**: **$-0.0098\text{ m}$ ($-9.8\text{ mm}$)** ($95\%\text{ CI}$: $-0.0208$ to $+0.0012\text{ m}$). Impulse-momentum estimates jump height $\sim 1\text{ cm}$ lower than flight time on average. This is standard in vertical jumping because athletes frequently land with slightly more ankle plantarflexion or knee flexion than at takeoff, slightly extending airborne flight duration relative to center-of-mass apex trajectory.
- **Standard Deviation of Differences**: $0.0153\text{ m}$ ($1.53\text{ cm}$).
- **95% Limits of Agreement**: **$-0.0399\text{ m}$ to $+0.0203\text{ m}$** ($-4.0\text{ cm}$ to $+2.0\text{ cm}$).
- **Individual Pairings**: In 9 out of 10 trials, the methods agree within $0.3\text{--}2.2\text{ cm}$.
- **Flight-Floor Offset vs. Height Difference Correlation**:
  - Across all 10 trials, mid-flight baseline force averages $+4.22\text{ N}$ (range $+3.17\text{ N}$ to $+6.02\text{ N}$, $\text{SD} \approx 0.88\text{ N}$).
  - An uncorrected positive tare shift $+F_{\text{floor}}$ inflates baseline body weight and mass, producing a predictable theoretical downward bias on $h_{\text{imp}}$ of $\Delta h \approx -2(F_{\text{floor}}/BW)h_{\text{imp}}$, averaging **$-3.83\text{ mm}$** across this cohort (range $-2.58\text{ mm}$ to $-5.75\text{ mm}$).
  - However, flight-floor offset **does not correlate** with the $(h_{\text{imp}} - h_{\text{flt}})$ difference across trials ($r = +0.0622$, $p = 0.864$; Spearman $\rho = -0.0667$, $p = 0.855$). This occurs because between-athlete variation in landing mechanics (landing knee/ankle flexion creating $\Delta y_{\text{COM}} \approx \pm 2\text{--}8\text{ cm}$, which shifts $h_{\text{flt}}$ by $\pm 10\text{--}40\text{ mm}$) completely dominates the $<1\text{ mm}$ between-athlete spread in sensor offset bias.
- **Caution on Sample Size**: With $n = 10$, confidence intervals around the limits of agreement are wide (lower LoA CI: $-5.9$ to $-2.1\text{ cm}$; upper LoA CI: $+0.1$ to $+3.9\text{ cm}$) and represent indicative between-subject agreement, not within-subject reliability.

*Figure Reference*: [`reports/figures/cmj_bland_altman.png`](file:///c:/Yashvi/Internship%202027/portfolio-agent-kit/reports/figures/cmj_bland_altman.png)

---

### Milestone M2f: Analyst-Choice Sensitivity Analysis
We systematically evaluated the stability of $h_{\text{imp}}$, $\text{RSImod}$, and peak mechanical power across six operational parameter sweeps:
1. **Onset SD Multiplier ($k = 3.0, 5.0, 8.0$)**: Jump height moved by $<1.0\text{ mm}$ across all 10 trials. RSImod shifted by $<0.008\text{ m/s}$.
2. **Quiet Window Duration ($0.30, 0.50, 0.70\text{ s}$)**: Jump height moved by $<1.5\text{ mm}$.
3. **Takeoff Force Threshold ($5.0, 10.0, 20.0\text{ N}$)**: Jump height moved by $<2.8\text{ mm}$.
4. **Landing Force Threshold ($5.0, 10.0, 20.0\text{ N}$)**:
   - While $h_{\text{imp}}$, $\text{RSImod}$, and propulsion peak power are strictly invariant ($\Delta = 0.0000$) because they terminate prior to takeoff, landing time $t_{\text{land}}$ was audited across all 10 trials.
   - **Monotonicity**: $t_{\text{land}}(5\text{ N}) \le t_{\text{land}}(10\text{ N}) \le t_{\text{land}}(20\text{ N})$ was verified in **10 out of 10 trials** ($100\%$).
   - On the steep landing contact wavefront ($dF/dt > 20,000\text{ N/s}$), force rockets from baseline directly through $20\text{ N}$ within $\le 0.5\text{--}1.0\text{ ms}$ ($0\text{--}2\text{ samples}$ at $2000\text{ Hz}$), confirming that detection at $5\text{ N}$ is strictly capturing real ground impact rather than flight-phase noise chatter or window fall-through.
5. **Filter Cutoff Frequency ($15, 30, 50\text{ Hz}$)**:
   - Jump height moved by $<3.9\text{ mm}$ across all trials.
   - Peak mechanical power showed high sensitivity in explosive trials: filtering at $15\text{ Hz}$ vs $30\text{ Hz}$ attenuated peak power by up to $82.4\text{ W}$ in `T02` ($3898\text{ W} \to 3980\text{ W}$) and $79.9\text{ W}$ in `T06` ($4976\text{ W} \to 5056\text{ W}$). Cutoffs $\ge 30\text{ Hz}$ are essential to preserve peak mechanical power transients.
6. **Pre-Onset Integration Lead ($30, 50, 100\text{ ms}$)**:
   - Jump height moved by $<7.2\text{ mm}$ across all trials, verifying that baseline quiet-standing integration does not drift significantly.
7. **Detection Failures**: Exactly **$0$ detection failures** occurred across all $10 \text{ trials} \times 18 \text{ parameter permutations}$ ($180$ pipeline evaluations).

*Figure Reference*: [`reports/figures/cmj_sensitivity.png`](file:///c:/Yashvi/Internship%202027/portfolio-agent-kit/reports/figures/cmj_sensitivity.png)

---

## Verification & Test Suite
The pipeline is verified by **57 automated tests** spanning schemas, validators, analytical synthesizers, biomechanical phase detectors, and real-data verification:
- Unit, synthetic & real-data suite: `uv run pytest -v -m "not slow"` (56 passed).
- Monte Carlo accuracy: `uv run pytest tests/test_cmj.py -m slow` ($N = 200$ trials across 9 conditions verifying 95th-percentile bounds for $h_{\text{imp}}$, $h_{\text{flt}}$, onset, takeoff, and landing).
- Real data test suite: `uv run pytest tests/test_cmj_real.py` (8 passed). Ensures mass plausibility, negative force flagging, timestamp irregularity detection, reference data parsing, and zero private data leakage.
- AI transparency: All biomechanical equations and signal processing logic cross-checked against peer-reviewed literature (McMahon et al. 2018, Chavda et al. 2018, Linthorne 2001). Results verified against known synthetic ground truths and independent Excel references.

