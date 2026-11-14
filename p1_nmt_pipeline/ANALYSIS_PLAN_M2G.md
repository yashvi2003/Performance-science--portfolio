# Analysis Plan: Milestone M2g Open Data Validation & Reliability

**Author**: Performance Science / Biomechanics Pipeline  
**Date**: 2026-09-22  
**Dataset**: White et al. (2022, *PLOS ONE*, DOI: [10.1371/journal.pone.0263846](https://doi.org/10.1371/journal.pone.0263846); Zenodo DOI: [10.5281/zenodo.19136480](https://doi.org/10.5281/zenodo.19136480))  
**Status**: Analysis plan written before modelling prior to running reliability calculations.

---

## 1. Objectives & Research Questions

1. **Golden Validation**: Does our open data adapter reproduce the reference values from White et al. across all 663 trials?
   - Reference `peak_power`: Paper-confirmed formula $P(t) = (F_{\text{BW}} \cdot g) \cdot v(t)$ in $\text{W/kg}$.
   - Reference `jump_height`: Data-inferred total COM vertical elevation $h_{\text{total\_com}} = s_{\text{takeoff}} + \frac{v_{\text{takeoff}}^2}{2g}$.
2. **Within-Participant Reliability**: What is the test-retest reliability (CV%, ICC, SEM, MDC) of CMJ performance metrics across repeated trials in healthy participants?
3. **Protocol Integrity**: How does movement constraint (CMJNA: hands on hips vs. CMJA: arm swing) affect performance and reliability?

---

## 2. Dataset & Sample Specification

- **File**: `data/raw/open/cmj_dataset_both.npz` (17.2 MB, 663 trials).
- **Physical Sampling Rate**:
  - Vertical Ground Reaction Force (vGRF): $1000\text{ Hz}$ dual Kistler force platforms (9260AA), body-weight normalized.
  - Accelerometer: $250\text{ Hz}$ Delsys Trigno lower-back sensor, synchronised via Vicon Nexus.
- **Cohort Size**:
  - `data["n_subjects"]` $= 69$ is preserved only as a study-level metadata note recording the parent recruitment cohort.
  - The actual physical participants present in `cmj_dataset_both.npz` is **$N = 67$ unique participants** (`subject_ids` $0\dots68$ with indices $14$ and $29$ omitted due to lack of synchronized accelerometer data).
  - Bijective 1-to-1 mapping with `original_participant_ids` (67 unique IDs from the Swansea university athlete cohort).

---

## 3. Strict Condition Separation

CMJNA and CMJA represent fundamentally different movement mechanics (arm swing adds ~19% jump height and ~14% peak power). Pooling them into a single reliability calculation would conflate protocol manipulation with trial-to-trial measurement error.
- **CMJNA (Condition 1: No Arm Swing, Hands on Hips)**: $n = 334\text{ trials}$ across 67 participants.
  - Primary condition for athlete monitoring benchmarks.
- **CMJA (Condition 2: With Arm Swing)**: $n = 329\text{ trials}$ across 66 participants (Subject 2 did not perform CMJA).
  - Evaluated separately as a secondary comparison.

---

## 4. Metric Definitions & Naming Contract

To prevent silent aliasing, the adapter outputs explicit, unambiguous metric names:

| Metric Key | Formula | Physical Meaning |
| :--- | :--- | :--- |
| `jump_height_takeoff_vel_m` | $\frac{v_{\text{takeoff}}^2}{2g}$ | Flight-trajectory elevation of COM above takeoff position ($~31.8\text{ cm}$) |
| `s_takeoff_m` | $\iint_0^{t_{\text{to}}} a(\tau)\,d\tau^2$ | COM upward elevation at toe-off relative to quiet standing ($~12.5\text{ cm}$) |
| `jump_height_total_com_m` | $s_{\text{takeoff}} + \frac{v_{\text{takeoff}}^2}{2g}$ | Total peak COM elevation above initial quiet standing ($~44.2\text{ cm}$) |
| `ref_jump_height_m` | Raw file value | Author's reference value |
| `peak_power_w_per_kg` | $\max\left((F_{\text{BW}}(t) \cdot g) \cdot v(t)\right)$ | Peak relative power during concentric propulsion |
| `ref_peak_power_w_per_kg` | Raw file value | Author's reference peak power |
| `raw_continuous_takeoff_idx` | Raw `grf_takeoff[i]` | Index in raw un-trimmed continuous recording (provenance only) |

**Physical Constant**: Strict use of standard gravity $g = 9.80665\text{ m/s}^2$ across all pipeline calculations. The ~0.12% difference against the reference's $g = 9.80\text{ m/s}^2$ is logged as an explicit note on reference data provenance.

---

## 5. Planned Balanced Subsets & Denominators

Every reported reliability statistic must output its raw $N$, trial count, and denominator:

### A. Primary Benchmark: CMJNA Balanced 4-Trial Set
- **Inclusion Criterion**: First 4 trials of CMJNA for all participants with $\ge 4$ CMJNA trials.
- **Sample Size**: **$n = 66\text{ participants} \times 4\text{ trials} = 264\text{ observations}$**.
- **Excluded**: Exactly 1 participant (**Subject ID 40**, who completed only 3 trials).

### B. Secondary Benchmark: CMJNA Balanced 3-Trial Set
- **Inclusion Criterion**: First 3 trials of CMJNA for all participants with $\ge 3$ CMJNA trials.
- **Sample Size**: **$n = 67\text{ participants} \times 3\text{ trials} = 201\text{ observations}$**.
- **Excluded**: **0 participants** (100% of CMJNA cohort included).

### C. Secondary Benchmark: CMJA Balanced 4-Trial Set
- **Inclusion Criterion**: First 4 trials of CMJA for all participants with $\ge 4$ CMJA trials.
- **Sample Size**: **$n = 64\text{ participants} \times 4\text{ trials} = 256\text{ observations}$**.
- **Excluded**: Subject 2 (0 trials), Subject 3 (2 trials), Subject 23 (3 trials).

### D. Secondary Benchmark: CMJA Balanced 3-Trial Set
- **Inclusion Criterion**: First 3 trials of CMJA for all participants with $\ge 3$ CMJA trials.
- **Sample Size**: **$n = 65\text{ participants} \times 3\text{ trials} = 195\text{ observations}$**.
- **Excluded**: Subject 2 (0 trials), Subject 3 (2 trials).

### E. Full Cohort Unbalanced Analysis: Linear Mixed-Effects Model (LMM / REML)
- **Model**: $Y_{ij} = \mu + u_i + \epsilon_{ij}$, where $u_i \sim \mathcal{N}(0, \sigma_{\text{between}}^2)$ and $\epsilon_{ij} \sim \mathcal{N}(0, \sigma_{\text{within}}^2)$.
- **Sample Size**: All available trials ($n=334$ for CMJNA across 67 subjects; $n=329$ for CMJA across 66 subjects).

---

## 6. Statistical Reliability Formulas

All reliability statistics assess **single-trial reliability** (monitoring individual jumps, not trial averages):

1. **Within-Participant Coefficient of Variation ($\text{CV}\%$)**:
   $$\text{CV}_i = \frac{s_i}{\bar{x}_i} \times 100\%, \quad \text{Mean CV} = \frac{1}{N}\sum_{i=1}^N \text{CV}_i$$
2. **Two-Way Random Absolute Agreement ($\text{ICC}(2, 1)$)**:
   $$\text{ICC}(2, 1) = \frac{\text{MS}_R - \text{MS}_E}{\text{MS}_R + (k-1)\text{MS}_E + \frac{k}{n}(\text{MS}_C - \text{MS}_E)}$$
   *(Shrout & Fleiss 1979; McGraw & Wong 1996; Koo & Li 2016)*
3. **Two-Way Mixed Consistency ($\text{ICC}(3, 1)$)**:
   $$\text{ICC}(3, 1) = \frac{\text{MS}_R - \text{MS}_E}{\text{MS}_R + (k-1)\text{MS}_E}$$
4. **Standard Error of Measurement ($\text{SEM}$)**:
   $$\text{SEM} = \text{SD}_{\text{pooled}} \times \sqrt{1 - \text{ICC}(2, 1)}$$
5. **Minimal Detectable Change ($\text{MDC}_{95}$)**:
   $$\text{MDC}_{95} = 1.96 \times \sqrt{2} \times \text{SEM} \approx 2.77 \times \text{SEM}$$

---

## 7. QA Flags & Data Quality Gates

The adapter attaches the following QA flags to each trial:
- `flag_negative_terminal_force: bool`: True if $F[-1] < 0.0\text{ BW}$ (flags 36 trials with zero-offset noise).
- `flag_terminal_force_outlier: bool`: True if $|F[-1]| > 0.02\text{ BW}$ (0 trials in dataset).
- `flag_baseline_unstable: bool`: True if quiet-phase noise $\text{SD} > 0.05\text{ BW}$.

Reliability will be evaluated on the primary set and verified on the clean subset (`flag_negative_terminal_force == False`) to confirm that zero-offset noise does not alter reported reliability.
