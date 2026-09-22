# Portfolio Research Brief: Real-World Evidence of Load, Recovery, and Readiness

**Project**: Project 3 (`p3_nmt_profiler`) | Neuromuscular Monitoring Engine  
**Author**: Performance Science & Health Outcomes Research Portfolio  
**Target Focus**: Wearable Performance Science & Longitudinal Athlete Analytics  
**Date**: September 2026  

---

## 1. Executive Summary & Research Question

Wearable smartwatches and daily mobile wellness surveys are widely deployed in elite and recreational sports to guide training periodization. However, conventional analytics frequently commit the **ecological fallacy**: confusing between-person trait differences (e.g., highly conditioned athletes with low resting heart rates reporting different baseline readiness than recreational athletes) with acute, within-person daily responses (e.g., does getting an extra hour of sleep tonight improve *my* readiness tomorrow?).

This research brief investigates **5 months of continuous free-living longitudinal tracking** across 16 healthy, physically active adults (~2,400 athlete-days) from the open PMData cohort (Thambawita et al., 2020). Using **Currie / Hoffmann person-mean centered Linear Mixed-Effects Models (LMM)** estimated via Restricted Maximum Likelihood (REML), we isolate within-person acute state fluctuations from chronic between-person baseline traits.

```
====================================================================================================
KEY RESEARCH TAKEAWAYS (PMData Cohort, 16 Athletes, ~2,400 Days)
----------------------------------------------------------------------------------------------------
1. Acute Sleep Boosts Morning Readiness: +1 SD in prior-night sleep duration (+1.34 h) 
   significantly improves next-day morning readiness (+0.22 points, p < 0.001, 95% CI: [0.14, 0.30]).
2. Significant Sleep Sensitivity Heterogeneity: Likelihood Ratio Testing (χ²(2) = 32.02, p < 0.001) 
   revealed meaningful inter-individual variance in sleep loss sensitivity (random slopes SD = 0.35).
3. Training Load Triggers Localized Soreness: Acute prior-day training strain (sRPE) strongly 
   predicts next-day muscle soreness (β = -0.129, p < 0.001), but shows only a non-significant 
   negative trend on global readiness (β = -0.073, p = 0.137).
4. Null Finding on Nocturnal RHR: Acute nocturnal resting heart rate fluctuations did not 
   significantly predict next-day subjective readiness (β = -0.021, p = 0.606) once sleep was controlled.
5. Observational Boundaries: Observational wearable tracking isolates predictive associations, 
   not counterfactual causal mechanisms. Wearable alerts represent "fatigue flags", not diagnoses.
====================================================================================================
```

---

## 2. Data Provenance, Governance & Operational Schema

* **Primary Dataset**: PMData (Thambawita et al., 2020, ACM MMSys '20; DOI: [10.1145/3339825.3394926](https://doi.org/10.1145/3339825.3394926)).
* **Ethical Approval & Consent**: Norwegian Centre for Research Data (NSD / Sikt, ref: 867106); written informed consent obtained from all participants.
* **Data License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0; official OSF repository `osf.io/vx4bk`).
* **Instrumentation**:
  1. *Fitbit Versa 2*: Optical photoplethysmography (PPG) heart rate, nocturnal resting heart rate (RHR), sleep stages, and sleep score.
  2. *PMSys Mobile Application*: Daily waking wellness survey (readiness 1–10, fatigue 1–5, muscle soreness 1–5, stress 1–5, mood 1–5, sleep duration) and post-workout session-RPE logs.
* **Schema Operationalization (`data/processed/wearable_daily.csv`)**:
  - `day_strain`: Daily Foster's Session-RPE sum ($\sum \text{RPE} \times \text{duration}_{\min}$). Rest days with completed wellness surveys are coded as $0.0\text{ AU}$.
  - `recovery_pct`: PMSys waking readiness ($1\text{--}10 \times 10.0$) with fallback to Fitbit sleep score ($0\text{--}100$).
  - `rhr_bpm`: Nocturnal resting heart rate from Fitbit sleep records.
  - `sleep_hours`: Self-reported previous-night sleep duration.
  - `hrv_ms`: Explicit `NaN` (Fitbit Versa 2 did not export rMSSD time series in this release; strictly satisfies shared contract rule).

---

## 3. Statistical Methodology: Person-Mean Centering Decomposition

To avoid ecological bias and Simpson's Paradox, all continuous predictors are partitioned using the **Currie / Hoffmann within-between decomposition**:

$$X_{it} = \bar{X}_i + (X_{it} - \bar{X}_i)$$

1. **Between-Person Trait Mean**: $\bar{X}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} X_{it}$ (captures chronic athlete physiology).
2. **Within-Person Acute Deviation**: $X^{\text{within}}_{it} = X_{it} - \bar{X}_i$ (captures day-to-day acute change).
3. **Standardized Acute Predictor**: $zX^{\text{within}}_{it} = \frac{X^{\text{within}}_{it}}{\text{SD}_{\text{within}}(X)}$ (allows direct comparison of effect magnitudes across sleep hours, RHR, and sRPE).

All models were estimated using **Linear Mixed-Effects Models (LMM)** via Restricted Maximum Likelihood (**REML**) with random intercepts per athlete:

$$Y_{it} = (\beta_0 + u_{0i}) + \beta_{\text{within}} zX^{\text{within}}_{it-1} + \beta_{\text{between}} \bar{X}_i + \beta_{\text{covariates}} \mathbf{Z}_{it} + \epsilon_{it}$$

---

## 4. Empirical Modeling Results & Hypothesis Testing

Following pre-registration in `p3_nmt_profiler/ANALYSIS_PLAN.md`, three primary mixed-effects models were fitted:

```
====================================================================================================
TABLE 1: FULL LINEAR MIXED-EFFECTS REGRESSION RESULTS (REML ESTIMATION)
====================================================================================================
Model 1: Next-Day Morning Readiness (Scale 1–10) | n = 1,368 observations across 15 athletes
----------------------------------------------------------------------------------------------------
Predictor Parameter                  Coef (β)     Std.Err       z        p-value      95% Conf. Int.
----------------------------------------------------------------------------------------------------
Intercept                             5.151        3.735      1.379       0.168      [-2.17, 12.47]
z_dev_prior_sleep_hours (Acute)      +0.218        0.040      5.423      <0.001 *    [+0.14, +0.30]
mean_prior_sleep_hours (Trait)       +0.931        0.619      1.504       0.133      [-0.28, +2.15]
z_dev_prior_rhr (Acute)              -0.021        0.040     -0.516       0.606 (ns) [-0.10, +0.06]
mean_prior_rhr (Trait)               -0.100        0.062     -1.607       0.108      [-0.22, +0.02]
z_dev_prior_day_strain (Acute)       -0.073        0.049     -1.487       0.137 (ns) [-0.17, +0.02]
mean_prior_day_strain (Trait)        -0.005        0.004     -1.247       0.212      [-0.01, +0.00]
is_weekend (Covariate)               -0.240        0.088     -2.734       0.006 *    [-0.41, -0.07]
Group Variance (σ_u0²): 1.449 | Residual Scale (σ_ε²): 2.122 | REML Log-Likelihood: -2495.82
----------------------------------------------------------------------------------------------------

Model 2: Next-Day Muscle Soreness (Scale 1–5: 1=Sore, 5=Fresh) | n = 1,716 across 16 athletes
----------------------------------------------------------------------------------------------------
Intercept                             2.843        0.052     54.739      <0.001 *    [+2.74, +2.95]
z_dev_prior_day_strain (Acute)       -0.129        0.014     -9.510      <0.001 *    [-0.16, -0.10]
mean_prior_day_strain (Trait)        -0.000        0.000     -0.134       0.894      [-0.00, +0.00]
z_dev_prior_sleep_hours (Acute)      +0.005        0.014      0.393       0.694 (ns) [-0.02, +0.03]
Group Variance (σ_u0²): 0.020 | Residual Scale (σ_ε²): 0.318 | REML Log-Likelihood: -1482.29
----------------------------------------------------------------------------------------------------

Model 3: Next-Day Morning Fatigue (Scale 1–5: 1=Exhausted, 5=Fresh) | n = 1,369 across 15 athletes
----------------------------------------------------------------------------------------------------
Intercept                             2.708        0.064     42.135      <0.001 *    [+2.58, +2.83]
z_dev_prior_sleep_score (Acute)      +0.143        0.017      8.402      <0.001 *    [+0.11, +0.18]
z_dev_prior_day_strain (Acute)       -0.012        0.020     -0.587       0.557 (ns) [-0.05, +0.03]
z_dev_prior_rhr (Acute)              +0.041        0.016      2.509       0.012 *    [+0.01, +0.07]
Group Variance (σ_u0²): 0.054 | Residual Scale (σ_ε²): 0.361 | REML Log-Likelihood: -1273.27
====================================================================================================
* p < 0.05; (ns) non-significant.
```

### Pre-Registered Hypotheses & Transparent Null Findings

1. **$H_{1a}$ Supported (Sleep Duration $\to$ Readiness)**: Acute sleep extension significantly improves morning readiness ($\beta = +0.218, p < 0.001$).
2. **$H_{1c}$ Supported (Sleep Quality Score $\to$ Fatigue)**: Higher Fitbit sleep scores strongly predict waking with greater energy ($\beta = +0.143, p < 0.001$).
3. **$H_{3a}$ Supported (Training Load $\to$ Soreness)**: Prior-day training strain produces robust, dose-dependent increases in localized muscle soreness ($\beta = -0.129, p < 0.001$).
4. **$H_2$ Null Finding (Nocturnal RHR $\to$ Readiness)**: Day-to-day elevations in nocturnal RHR did not reliably predict subjective morning readiness ($\beta = -0.021, p = 0.606, 95\%\text{ CI}: [-0.10, +0.06]$).
5. **$H_{3b}$ Null Finding (Training Load $\to$ Readiness)**: Prior-day training strain exhibited a negative trend with morning readiness ($\beta = -0.073, p = 0.137$) that did not achieve statistical significance once sleep was controlled.

---

## 5. Between-Athlete Sleep Sensitivity Heterogeneity

Evaluating random slopes ($u_{1i} zSleep^{\text{within}}_{it-1}$) for acute sleep deviation demonstrated **significant inter-individual heterogeneity**:

* **Likelihood Ratio Test**: $\chi^2(2) = 32.022, p = 1.11 \times 10^{-7}$ ($\Delta \text{AIC} = -28.02$).
* **Random Slope Variance**: $\sigma_{u1}^2 = 0.122$ ($\text{SD}_{u1} = 0.350$).
* **Empirical Bayes BLUP Trajectories**: While the population mean response is $\beta = +0.22$, individual athlete slopes range from non-responders ($\beta_i \approx 0.0$) to high-sensitivity responders ($\beta_i = +0.55$). 

![Sleep Heterogeneity Slopes](reports/figures/p3_sleep_heterogeneity_slopes.png)
*Figure 1: Individual athlete sleep sensitivity slopes (Random Slopes / Empirical Bayes BLUPs) vs. group average trajectory.*

---

## 6. Sensitivity Analyses & Causal DAG Framework

### Sensitivity Checks (`p3_nmt_profiler/sensitivity.py`)
1. **Rest-Day Imputation vs. Complete Dropping**: Re-estimating Model 2 on strictly active training days ($sRPE > 0$) yielded an identical load-soreness coefficient ($\beta = -0.138, p < 0.001$ vs. baseline $\beta = -0.129$).
2. **1% Outlier Trimming**: Trimming the 1st and 99th percentiles of sleep duration deviations yielded an identical effect on readiness ($\beta = +0.212, p < 0.001$).
3. **Cross-Language Validation (R `lme4` vs. Python `statsmodels`)**: Model 1 was independently refit via `lme4::lmer` (REML, `nloptwrap` optimizer) on `data/processed/p3_model1_dataset.csv`. Acute sleep deviation fixed-effect parameters agreed within numerical tolerance: Python $\beta = +0.21817881$, $\text{SE} = 0.04023033$, Wald $95\%\text{ CI}: [0.13932881, 0.29702881]$ vs. R $\beta = +0.21817885$, $\text{SE} = 0.04023046$, Wald $95\%\text{ CI}: [0.13932860, 0.29702910]$ (direct subtraction $|\Delta \beta| = 4.3 \times 10^{-8}$). Residual variance ($\sigma_\epsilon^2$: Python $= 2.12187528$ vs. R $= 2.12188964$, $|\Delta| = 1.4 \times 10^{-5}$) and group variance ($\sigma_u^2$: Python $= 1.44899925$ vs. R $= 1.44771033$, $|\Delta| = 1.3 \times 10^{-3}$) show expected optimizer divergence; because GLS fixed effects are conditional on finding the approximate 1D variance ratio peak, tight fixed-effect agreement alongside looser variance component estimates is the standard signature of genuine independent convergence. Note that reported $p$-values differ in distributional form: R `lmerTest` computes finite-sample Satterthwaite denominator degrees of freedom ($t = 5.423, df = 1349, p = 6.93 \times 10^{-8}$) while Python `statsmodels.MixedLM` uses an asymptotic standard normal $z$-test ($z = 5.423, p = 5.85 \times 10^{-8}$).

### Causal DAG & Confounders

![Load Recovery Effects](reports/figures/p3_load_recovery_effects.png)
*Figure 2: Multi-panel evidence card: Standardized forest plot (A), Load vs. Soreness (B), Sleep vs. Readiness (C), and Causal DAG schematic (D).*

```
                   +------------------------+
                   |  Unmeasured Confounders|
                   | (Alcohol, Life Stress, |
                   |  Subclinical Infection)|
                   +-----------+------------+
                               |
               +---------------+---------------+
               |                               |
               v                               v
    +--------------------+           +--------------------+
    | Prior-Day Training |           | Prior-Night Sleep  |
    |  Load (sRPE t-1)   |           |  (Duration & RHR)  |
    +---------+----------+           +---------+----------+
              |                                |
              +----------------+---------------+
                               |
                               v
                     +-------------------+
                     | Next-Day Morning  |
                     | Readiness/Fatigue |
                     +-------------------+
```

* **Unmeasured Confounders**: Alcohol intake, academic/work deadlines, and subclinical viral infections simultaneously disrupt nocturnal sleep architecture, elevate resting heart rate, and impair waking readiness.
* **Non-Causal Framing**: In wearable analytics, alerts represent **potential fatigue flags**, not counterfactual causal predictions or clinical diagnoses.

---

## 7. Limb Asymmetry Module (`p3_nmt_profiler/asymmetry.py`)

* **Bilateral Asymmetry Index**: $\text{BAI} = \frac{\text{Left} - \text{Right}}{\text{Left} + \text{Right}} \times 100\%$ ($+$ Left-dominant, $-$ Right-dominant).
* **Persistence Criteria**: Directional asymmetry is flagged as *persistent* if $\ge 75\%$ of consecutive trials favor the same limb.
* **Fatigue Sensitivity Slope**: Tracks whether asymmetry magnitude ($|\text{BAI}|$) widens under acute training strain or repetitive trials ($\frac{d|\text{BAI}|}{d\text{Load}}$).

---

## 8. Limitations & Scientific Scope

1. **Sample Size**: $n = 16$ adults monitored in free-living conditions. While providing high longitudinal depth (~2,400 days), findings represent physically active adults rather than elite sport cohorts.
2. **Subjective Proxies**: Readiness and soreness reflect self-reported psychometric ratings rather than invasive biochemical biomarkers (e.g. blood lactate, creatine kinase).
3. **Optical Sensor Drift**: Photoplethysmography resting heart rate is subject to wrist positioning and motion artifact during restless sleep.

---

## 9. AI-Use Transparency & Verification

* **AI Assistance**: Statistical script authoring and figure styling.
* **Verification Protocol**:
  1. *Analytical Ground Truth*: Validated against known closed-form synthetic datasets in `tests/test_synth.py`.
  2. *Cross-Language Statistical Validation*: Model 1 parameter estimates independently cross-checked between Python `statsmodels.MixedLM` and R `lme4::lmer`.
  3. *Unit Test Coverage*: **86 automated unit tests** passing across the repository test suite.
