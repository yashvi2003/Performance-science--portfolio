# Project 1: Automatic Jump Analysis Engine (`p1_nmt_pipeline`)

## Overview & Scientific Objectives
This pipeline provides automated biomechanical phase detection and metrics computation for vertical jump force-time data, conforming strictly to the consensus standards established by **McMahon et al. (2018)**.

### Research Question
How accurately and reliably can discrete biomechanical events (movement onset, unweighting trough, upward BW crossing, zero-velocity transition, concentric propulsion peak, takeoff, landing impact) and performance metrics (impulse-momentum jump height, flight-time jump height, reactive strength index modified [RSImod], peak mechanical power) be recovered across varying sampling rates ($250\text{--}2000\text{ Hz}$), noise levels ($0\text{--}12\text{ N}$ baseline noise, flight noise up to $4\text{ N}$, flight offsets up to $25\text{ N}$), baseline quiet-standing durations ($0.30\text{--}0.75\text{ s}$), and complex force-curve morphologies?

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

---

## Milestone M2g: Open Data Validation & Within-Participant Reliability

### Scientific Scope & Dataset Specification
While Milestones M2c–M2f addressed data quality, golden validation, and sensitivity on a between-athlete sample ($n=10$, 1 trial each), Milestone M2g evaluates **within-participant repeated-trial reliability** using an open dataset:
- **Source**: White et al. (2022, *PLOS ONE*, DOI: [10.1371/journal.pone.0263846](https://doi.org/10.1371/journal.pone.0263846); Zenodo DOI: [10.5281/zenodo.19136480](https://doi.org/10.5281/zenodo.19136480), CC BY 4.0).
- **Cohort Reconciliation**: 663 CMJ trials across **67 unique participants** (`subject_ids` 0..68 with indices 14 and 29 omitted due to missing synchronized accelerometer data). The scalar metadata key `n_subjects=69` reflects the parent study recruitment cohort, not the row count of `cmj_dataset_both.npz`.
- **Condition Split**: 334 trials without arm swing (**CMJNA**, Condition 1) and 329 trials with arm swing (**CMJA**, Condition 2). Conditions are evaluated in strictly separate models.
- **Physical Characteristics**: 1000 Hz dual Kistler force platforms, body-weight normalized vertical GRF, pre-trimmed to terminate at takeoff.

---

### Jump-Height Definition Contract & Naming Scheme
To avoid ambiguous aliasing between flight trajectory height and total center of mass excursion:
1. `jump_height_takeoff_vel_m`: Flight-trajectory height of COM above takeoff position:
   $$h_{\text{to}} = \frac{v_{\text{takeoff}}^2}{2g} \approx 31.8\text{ cm}$$
2. `s_takeoff_m`: COM elevation at toe-off relative to standing quiet baseline:
   $$s_{\text{to}} = \iint_0^{t_{\text{to}}} a(\tau)\,d\tau^2 \approx 12.5\text{ cm}$$
3. `jump_height_total_com_m`: Total apex COM elevation relative to standing quiet baseline:
   $$h_{\text{total\_com}} = s_{\text{to}} + \frac{v_{\text{takeoff}}^2}{2g} \approx 44.2\text{ cm}$$
4. `ref_jump_height_m`: The author's reference jump height, verified to equal $h_{\text{total\_com}}$ ($r = 1.000000$).

---

### Golden Reproduction Results (All 663 Trials)

| Metric | Reference Mean | Calculated Mean | Mean Difference | Mean % Error | Pearson $r$ | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Peak Power (W/kg)** | $48.69\text{ W/kg}$ | $48.64\text{ W/kg}$ | $-0.053\text{ W/kg}$ | $-0.109\%$ | **$1.000000$** | **PASS** |
| **Total COM Jump Height (cm)** | $44.16\text{ cm}$ | $44.26\text{ cm}$ | $+0.101\text{ cm}$ | $+0.235\%$ | **$1.000000$** | **PASS** |

*Note on Physical Constants & Local Gravity Calibration*: The pipeline strictly maintains standard gravity $g = 9.80665\text{ m/s}^2$. Parameter sweep and Somigliana theoretical gravity analysis confirmed White et al. (Swansea University, lat $51.62^\circ\text{ N}$) used local UK gravity $g = 9.81200\text{ m/s}^2$. Evaluated at $g = 9.81200\text{ m/s}^2$, calculated peak power matches reference values with mean diff $+0.00000\text{ W/kg}$ (max $|\text{diff}| = 0.00010\text{ W/kg}$). Under standard gravity ($g = 9.80665\text{ m/s}^2$), peak power is physically $(9.80665/9.81200)^2 = -0.109\%$ lower ($-0.0531\text{ W/kg}$), matching physical theory exactly.

---

### Within-Participant Reliability Results (With Raw Denominators)

All reliability calculations report single-measure reliability ($\text{ICC}(\cdot, 1)$), planned prior to analysis, which represents the precision of evaluating an individual jump:

| Condition & Trial Set | Metric | Mean $\pm$ SD | Within-Subject $\text{CV}\%$ | $\text{ICC}(2, 1)$ [95% CI] | $\text{ICC}(3, 1)$ [95% CI] | $\text{SEM}$ | $\text{MDC}_{95}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CMJNA 4-Trial Set**<br>($n=66\text{ subjs} \times 4\text{ trials} = 264\text{ obs}$) | Total COM Height<br>Takeoff-Vel Height<br>Peak Power<br>Takeoff Velocity | $39.96 \pm 8.31\text{ cm}$<br>$29.40 \pm 7.64\text{ cm}$<br>$45.47 \pm 7.30\text{ W/kg}$<br>$2.38 \pm 0.32\text{ m/s}$ | $4.92 \pm 4.14\%$<br>$5.00 \pm 4.02\%$<br>$3.27 \pm 2.65\%$<br>$2.51 \pm 2.02\%$ | **$0.9060$** [$0.867, 0.937$]<br>**$0.9443$** [$0.920, 0.963$]<br>**$0.9308$** [$0.901, 0.954$]<br>**$0.9448$** [$0.921, 0.963$] | **$0.9052$** [$0.866, 0.936$]<br>**$0.9439$** [$0.920, 0.963$]<br>**$0.9317$** [$0.903, 0.954$]<br>**$0.9447$** [$0.921, 0.963$] | $2.55\text{ cm}$<br>$1.80\text{ cm}$<br>$1.92\text{ W/kg}$<br>$0.07\text{ m/s}$ | $7.07\text{ cm}$<br>$5.00\text{ cm}$<br>$5.33\text{ W/kg}$<br>$0.21\text{ m/s}$ |
| **CMJNA 3-Trial Set**<br>($n=67\text{ subjs} \times 3\text{ trials} = 201\text{ obs}$) | Total COM Height<br>Takeoff-Vel Height<br>Peak Power<br>Takeoff Velocity | $39.80 \pm 8.47\text{ cm}$<br>$29.21 \pm 7.83\text{ cm}$<br>$45.42 \pm 7.57\text{ W/kg}$<br>$2.37 \pm 0.33\text{ m/s}$ | $4.62 \pm 3.07\%$<br>$4.69 \pm 3.06\%$<br>$2.72 \pm 1.63\%$<br>$2.35 \pm 1.53\%$ | **$0.9356$** [$0.906, 0.958$]<br>**$0.9606$** [$0.942, 0.974$]<br>**$0.9656$** [$0.949, 0.978$]<br>**$0.9622$** [$0.944, 0.975$] | **$0.9348$** [$0.904, 0.957$]<br>**$0.9601$** [$0.941, 0.974$]<br>**$0.9665$** [$0.950, 0.978$]<br>**$0.9618$** [$0.943, 0.975$] | $2.15\text{ cm}$<br>$1.55\text{ cm}$<br>$1.40\text{ W/kg}$<br>$0.06\text{ m/s}$ | $5.96\text{ cm}$<br>$4.31\text{ cm}$<br>$3.89\text{ W/kg}$<br>$0.18\text{ m/s}$ |
| **CMJA 4-Trial Set**<br>($n=64\text{ subjs} \times 4\text{ trials} = 256\text{ obs}$) | Total COM Height<br>Takeoff-Vel Height<br>Peak Power<br>Takeoff Velocity | $47.33 \pm 10.32\text{ cm}$<br>$33.54 \pm 8.91\text{ cm}$<br>$51.42 \pm 8.96\text{ W/kg}$<br>$2.54 \pm 0.35\text{ m/s}$ | $5.17 \pm 4.60\%$<br>$5.42 \pm 4.41\%$<br>$4.40 \pm 3.36\%$<br>$2.75 \pm 2.33\%$ | **$0.9128$** [$0.876, 0.942$]<br>**$0.9395$** [$0.913, 0.960$]<br>**$0.8945$** [$0.850, 0.929$]<br>**$0.9370$** [$0.909, 0.958$] | **$0.9149$** [$0.879, 0.943$]<br>**$0.9412$** [$0.915, 0.961$]<br>**$0.8966$** [$0.853, 0.931$]<br>**$0.9390$** [$0.912, 0.960$] | $3.05\text{ cm}$<br>$2.19\text{ cm}$<br>$2.91\text{ W/kg}$<br>$0.09\text{ m/s}$ | $8.44\text{ cm}$<br>$6.07\text{ cm}$<br>$8.07\text{ W/kg}$<br>$0.24\text{ m/s}$ |
| **CMJNA Unbalanced LMM**<br>($n=67\text{ subjs}, 334\text{ obs}$) | Total COM Height<br>Takeoff-Vel Height<br>Peak Power<br>Takeoff Velocity | $40.30\text{ cm}$<br>$29.62\text{ cm}$<br>$45.59\text{ W/kg}$<br>$2.39\text{ m/s}$ | —<br>—<br>—<br>— | **$\text{ICC}_{\text{LMM}} = 0.9189$** ($\sigma_b = 8.22, \sigma_w = 2.44\text{ cm}$)<br>**$\text{ICC}_{\text{LMM}} = 0.9467$** ($\sigma_b = 7.71, \sigma_w = 1.83\text{ cm}$)<br>**$\text{ICC}_{\text{LMM}} = 0.9373$** ($\sigma_b = 7.30, \sigma_w = 1.89\text{ W/kg}$)<br>**$\text{ICC}_{\text{LMM}} = 0.9497$** ($\sigma_b = 0.32, \sigma_w = 0.07\text{ m/s}$) | — | — | — |

---

### QA Audit & Sensitivity to Negative Terminal Forces
- **Negative Terminal Forces**: Exactly 36 trials ($5.43\%$) end with $F[-1] < 0.0\text{ BW}$, down to a minimum of $-0.0102\text{ BW}$ ($-7.5\text{ N}$). This is an expected consequence of force plate zero-offset drift ($\pm 10\text{ N}$) around the $10\text{ N}$ takeoff cutoff.
- **Robustness**: Recomputing the primary CMJNA 4-trial set on clean trials without negative terminal force ($n=55$ subjects, $220$ obs) yields $\text{ICC}(2, 1) = 0.8945$ ($\Delta = -0.0115$) and $\text{CV} = 5.07\%$ ($\Delta = +0.15\%$). The terminal zero offset has negligible impact on reported reliability.
- **Outlier Check**: Zero trials exceed $0.02\text{ BW}$ ($2\%$ BW) at takeoff.
- **Scientific Limits**: Signals are pre-trimmed to takeoff ($1000\text{ Hz}$); flight time and landing mechanics cannot be assessed from this force dataset alone.

*Figure References*:
- [`reports/figures/cmj_open_data_validation.png`](file:///c:/Yashvi/Internship%202027/portfolio-agent-kit/reports/figures/cmj_open_data_validation.png)
- [`reports/figures/cmj_open_data_reliability.png`](file:///c:/Yashvi/Internship%202027/portfolio-agent-kit/reports/figures/cmj_open_data_reliability.png)

---

## Milestone M5a: Single-Trial Diagnostic Report (VALD Performance Style)

A publication-grade diagnostic report figure was developed to provide high-density clinical and performance visualization for single CMJ trials, styled after industry-standard commercial force plate reporting (VALD ForceDecks):

- **Banner Header**: Modern dark slate card (`#0F172A`) summarizing test modality, sampling frequency ($2000\text{ Hz}$), movement phase standard (McMahon et al., 2018), and automated trial validation status.
- **Vertical Force-Time Curve**: Full-span primary axes with dual trace visualization ($30\text{ Hz}$ zero-phase low-pass filtered primary trace over subtle raw trace), annotated body weight baseline, discrete landmark markers (onset, unweighting trough, zero-velocity transition, peak propulsion, takeoff, landing impact, and landing peak), and subtle harmonious pastel phase shading:
  - *Unweighting*: Movement onset $\to$ minimum force trough.
  - *Braking (Eccentric)*: Minimum force trough $\to$ zero-velocity transition ($v(t)=0$).
  - *Propulsion (Concentric)*: Zero-velocity transition $\to$ takeoff threshold crossing.
  - *Flight*: Takeoff $\to$ initial ground impact.
  - *Landing Impact*: Initial contact $\to$ landing impact peak.
- **Center of Mass Velocity Profile**: Coupled sub-panel illustrating numerical trapezoidal COM velocity integration from standing quiet baseline through concentric acceleration to peak takeoff velocity ($v_{\text{to}}$), with exact ballistic flight trajectory projection ($v(t) = v_{\text{to}} - g(t - t_{\text{to}})$).
- **Concentric Mechanical Power Output**: Coupled sub-panel displaying instantaneous mechanical power ($P(t) = F_{\text{filt}}(t) \cdot v(t)$), strictly isolated to ground contact propulsion prior to takeoff.
- **Key Performance Metrics Summary Card**: Formatted VALD-style tabular card with alternating row fills reporting both primary impulse-momentum metrics and secondary flight metrics:
  - Jump Height (Impulse-Momentum & Flight Time)
  - Contraction Time & Flight Time
  - RSImod (modified reactive strength index)
  - Peak Relative Power ($\text{W/kg}$) & Absolute Mechanical Power ($\text{W}$)
  - Peak Propulsion Force ($\text{N}$ & $\text{BW}$)
  - Landing Peak Impact Force ($\text{N}$ & $\text{BW}$)
  - Athlete Body Mass ($\text{kg}$ & $\text{N}$)
  - **Signal Processing & Landing Stage Definitions**: The pipeline deliberately differentiates signal conditioning across landing analysis stages: (1) *Initial contact threshold crossing* uses **raw force** ($F_{\text{raw}} \ge 10\text{ N} + F_{\text{offset}}$) to avoid low-pass filter edge-smearing and preserve millisecond-accurate touchdown timing; (2) *Double-integration of acceleration* uses **$30\text{ Hz}$ zero-phase low-pass filtered force** ($F_{\text{filt}}$) to eliminate high-frequency plate vibration artifacts and prevent cumulative drift; (3) *Velocity-trough inflection point* occurs downstream when $F_{\text{filt}} \ge BW$, where net acceleration switches from downward to upward; and (4) *Landing peak impact force* is extracted strictly from **raw force** ($F_{\text{raw}}$ per Chavda et al., 2018) because aggressive filtering severely attenuates fast transient impact spikes (~5 ms rise time).
- **Privacy & Anonymization Compliance**: Fully compliant with repository privacy guardrails; pulls directly from validated pipeline outputs without hardcoding athlete identifiers or exposing unapproved metadata.

*Figure Reference*: [`reports/figures/cmj_single_trial_report.png`](file:///c:/Yashvi/Internship%202027/portfolio-agent-kit/reports/figures/cmj_single_trial_report.png)

*Regeneration Command*:
```bash
python scripts/generate_single_trial_report.py
```

---

## Milestone M6a: Command-Line Interface (CLI)

The pipeline provides an automated CLI entry point (`p1_nmt_pipeline.cli`) that processes individual trial CSV files or directories of trials and exports a schema-compliant `sessions.csv` strictly adhering to [`specs/00_shared_contract.md`](file:///c:/Yashvi/Internship%202027/portfolio-agent-kit/specs/00_shared_contract.md).

### CLI Usage & Options
```bash
# Process a directory of trial CSVs and export to sessions.csv
python -m p1_nmt_pipeline.cli --input data/raw/cmj --output data/processed/sessions.csv

# Process a single trial with explicit session date
python -m p1_nmt_pipeline.cli --input data/raw/cmj/A01.csv --date 2026-09-22 --output data/processed/sessions.csv

# Optional flags:
#   --config PATH      Custom YAML pipeline configuration
#   --no-anonymize     Preserve raw IDs instead of de-identifying to ATH_XX
#   --verbose          Enable debug logging
```

### Shared Contract Compliance
Every exported `sessions.csv` file is validated on export via `shared.contract.validate_sessions_df(strict=True)`:
- Required schema fields: `session_id`, `date`, `athlete_id`, `test`, `trial`, `metric`, `value`, `unit`, `side`, `valid`, `flags`, `block`.
- Metrics mapped directly to registered keys in `METRIC_REGISTRY`: `cmj_jump_height_imp`, `cmj_jump_height_flight`, `cmj_rsimod`, `cmj_peak_power`, `cmj_peak_power_rel`, `cmj_peak_velocity`, `cmj_peak_braking_force`, `cmj_time_to_takeoff`, `cmj_flight_time`, `cmj_body_mass`.
- Anonymization guardrail: Defaults to automated de-identification mapping (`ATH_01`, `ATH_02`, etc.) to prevent accidental exposure of private identifiers.

---

## Reproducibility & Figure Generation

All analyses, tables, and publication-grade figures in this module can be regenerated with single deterministic commands:

```bash
# 1. Run all fast unit, synthetic, real, and open-data tests
pytest -v -m "not slow"

# 2. Run Monte Carlo validation (N=200 per condition, 9 conditions)
pytest tests/test_cmj.py -m slow

# 3. Generate VALD-style single-trial diagnostic report
python scripts/generate_single_trial_report.py

# 4. Generate open-data validation and reliability figures (White et al.)
python p1_nmt_pipeline/open_data_reliability.py

# 5. Process trials and export contract-compliant sessions.csv
python -m p1_nmt_pipeline.cli --input data/raw/cmj --output data/processed/sessions.csv
```

---

## Limitations & Scientific Guardrails

1. **Single-Trial Athlete Cohort (Professor Data)**: The primary 10-athlete dataset contains exactly one trial per athlete ($n=10$). Within-athlete longitudinal trends or trial-to-trial reliability ($\text{CV}$, $\text{ICC}$, $\text{SEM}$, $\text{MDC}$) cannot be determined from this cohort. Repeated-trial reliability was evaluated separately using open research data (White et al., $n=67$, $663$ trials).
2. **Pre-Trimmed Open Signals**: The White et al. dataset provides vertical GRF pre-trimmed to takeoff at $1000\text{ Hz}$. Consequently, airborne flight-time metrics and landing impact attenuation cannot be validated on that open dataset.
3. **Sensor Tare Drift vs. Landing Kinematics**: Sensor zero-offset drift ($\sim 3\text{--}6\text{ N}$) introduces a small systematic downward bias on impulse-momentum jump height ($\approx -3.8\text{ mm}$), but does not correlate with $(h_{\text{imp}} - h_{\text{flt}})$ differences ($r = 0.062$) because between-athlete variation in landing ankle/knee flexion dominates airborne flight duration.
4. **Synthetic Data Disclaimer**: Synthetic data generated by `shared/synth.py` is used solely to verify pipeline numerical correctness, edge-case handling, and algorithmic bounds against known closed-form mathematical solutions. **Synthetic data is not a research finding.** All figures utilizing synthetic data are explicitly labelled with "synthetic".

---

## AI-Use Transparency & Verification

In accordance with portfolio research conventions:
- **AI Assistance**: AI coding assistants were utilized for scaffolding modular pipeline components, unit test authoring, and test regression checks.
- **Verification of Mathematical & Biomechanical Formulas**:
  - All mathematical equations (trapezoidal impulse integration, McMahon phase definitions, Somigliana normal gravity formula, Shrout & Fleiss $\text{ICC}(2, 1)$ / $\text{ICC}(3, 1)$, Weir $\text{SEM}$ and $\text{MDC}_{95}$) were independently checked against peer-reviewed literature.
  - Closed-form synthetic ground truths verified numerical integration precision across varying sampling rates ($250\text{--}2000\text{ Hz}$).
  - Open data outputs were benchmarked against published research findings (White et al., 2022) across all 663 trials ($r = 1.000000$).
  - Code review was conducted on all filter routines (zero-phase Butterworth `sosfiltfilt`), event boundary detections, and schema validators.

---

## Test Suite Summary

The pipeline is verified by **73 automated tests** (72 fast tests in $\sim 20\text{s}$, 1 slow Monte Carlo test):
- `tests/test_p1_pipeline.py`: Config defaults, filter attenuation, zero-phase alignment, body mass estimation, timestamp resampling, quiet-window QA, CLI execution, contract compliance, and single-trial report generation (13 tests).
- `tests/test_cmj.py`: Phase ordering, impulse consistency, landing impact scaling, flight offset corrections, sub-sample extrapolation, and Monte Carlo tolerance envelopes (21 tests).
- `tests/test_cmj_real.py`: Real-data parsing, golden test comparison, timestamp irregularity handling, and privacy guardrails (8 tests).
- `tests/test_cmj_open.py`: White et al. cohort reconciliation, naming contract, golden reproduction, balanced matrix reliability, and terminal force sensitivity (13 tests).
- `tests/test_contract.py`: Registry integrity, `sessions.csv` and `wearable_daily.csv` schema validation (8 tests).
- `tests/test_synth.py`: Analytical ground-truth verification across CMJ, SJ, DJ, IMTP, bilateral asymmetry, and synthetic contract generators (10 tests).



