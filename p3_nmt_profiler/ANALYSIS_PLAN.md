# Pre-Registered Analysis Plan: Real-World Evidence of Load, Recovery, and Readiness

**Project**: Project 3 (`p3_nmt_profiler`)  
**Date**: September 2026  
**Status**: Pre-Registration Prior to Statistical Modeling  
**Primary Dataset**: PMData (Thambawita et al., 2020, ACM MMSys '20; DOI: [10.1145/3339825.3394926](https://doi.org/10.1145/3339825.3394926))  
**Data Governance**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0; OSF `license.txt`) / CC BY 4.0 (Simula portal); Participant written informed consent; Norwegian Centre for Research Data (NSD / Sikt) approval.

---

## 1. Plain-Language Summary (Executive Summary for Coaches & Practitioners)

Athletes and practitioners regularly wear smartwatches to track their sleep, resting heart rate, and training strain in the hope of anticipating readiness to perform or catching early signs of fatigue. However, naive analyses often commit the "ecological fallacy": comparing Athlete A to Athlete B (e.g., noting that fitter athletes generally sleep more and have lower heart rates) tells us very little about what happens *within* an individual athlete on a day-to-day basis. 

In this study, we evaluate 5 months of continuous real-world longitudinal tracking across 16 participants (~2,400 athlete-days) using Fitbit Versa 2 smartwatches and daily PMSys sports wellness logging. Using within-person centered linear mixed-effects models, we isolate true daily fluctuations from chronic between-person baseline traits. Specifically, we test whether getting one extra hour of sleep or an elevated sleep score predicts higher next-day readiness and reduced fatigue, and whether a spike in prior-day training load triggers next-day muscle soreness and readiness dips. Finally, we provide explicit causal directed acyclic graphs (DAGs) and honest scientific boundaries explaining what observational wearable data can and cannot prove.

---

## 2. Research Questions & Formal Pre-Registered Hypotheses

### Research Question 1 (Sleep & Next-Day Readiness/Fatigue)
To what extent do acute within-person deviations in prior-night sleep duration and sleep quality predict next-day subjective readiness and fatigue, independent of an athlete's chronic average sleep habits?

- **Hypothesis 1a ($H_{1a}$ - Sleep Duration & Readiness)**: On days when an athlete sleeps longer than their personal average ($Sleep_{it-1} - \bar{Sleep}_i > 0$), next-day morning readiness will be higher ($\beta_{\text{within\_sleep}} > 0$).
- **Hypothesis 1b ($H_{1b}$ - Sleep Duration & Fatigue)**: On days when an athlete sleeps longer than their personal average, next-day morning fatigue will be lower ($\beta_{\text{within\_sleep}} < 0$).
- **Hypothesis 1c ($H_{1c}$ - Sleep Quality / Score)**: On days when an athlete achieves an above-average Fitbit sleep score ($SleepScore_{it-1} - \bar{SleepScore}_i > 0$), next-day readiness will be higher and next-day fatigue lower, over and above duration alone.

### Research Question 2 (Prior-Night Autonomic State & Readiness)
Does an acute elevation in nocturnal resting heart rate (RHR) signal impaired next-day readiness within an individual?

- **Hypothesis 2 ($H_2$ - Resting Heart Rate)**: Acute elevations in prior-night resting heart rate above an athlete's personal baseline ($RHR_{it-1} - \bar{RHR}_i > 0$) negatively predict next-day morning readiness ($\beta_{\text{within\_rhr}} < 0$) and positively predict fatigue ($\beta_{\text{within\_rhr}} > 0$).

### Research Question 3 (Prior-Day Training Load & Next-Day Soreness/Readiness)
How does acute prior-day session training load impact next-day muscle soreness and readiness?

- **Hypothesis 3a ($H_{3a}$ - Training Load & Soreness)**: Higher prior-day training load ($sRPE_{it-1} - \bar{sRPE}_i > 0$) positively predicts next-day muscle soreness ($\beta_{\text{within\_load}} > 0$).
- **Hypothesis 3b ($H_{3b}$ - Training Load & Readiness)**: Higher prior-day training load negatively predicts next-day readiness ($\beta_{\text{within\_load}} < 0$).

---

## 3. Data Provenance, Operational Definitions & Schema Mapping

### Cohort Specification
- **Participants**: 16 healthy, physically active adults ($p01$–$p16$), monitored continuously for 5 months (November 2019 to March 2020).
- **Total Observations**: ~2,400 participant-days.
- **Hardware & Software**:
  1. *Fitbit Versa 2*: Smartwatch tracking optical photoplethysmography (PPG) heart rate, resting heart rate, sleep duration and sleep stages (deep, light, REM, awake), daily steps, and calories.
  2. *PMSys Smartphone Application*: Daily self-reported wellness logs administered each morning upon waking (readiness, fatigue, mood, soreness, stress, sleep duration, sleep quality) and session-RPE training logs completed post-workout.

### Metric Operationalization & Schema Mapping (`wearable_daily.csv`)

| Variable in Contract | Source in PMData | Operational Definition / Formula | Unit | Proxy Documentation |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `effective_time_frame` / `timestamp` | Calendar date in ISO format (`YYYY-MM-DD`) | Date | Primary temporal alignment key |
| `athlete_id` | Directory identifier | Participant token (`p01`–$p16$) or de-identified (`ATH_01`–`ATH_16`) | String | Categorical clustering unit |
| `recovery_pct` | `pmsys/wellness.csv` (`readiness`) | Primary: PMSys readiness ($1\text{--}10$) scaled linearly: $\text{Readiness} \times 10.0$. Fallback: Fitbit `overall_score` ($0\text{--}100$). | % ($0\text{--}100$) | Documented proxy; mirrors recovery score heuristic |
| `rhr_bpm` | `fitbit/sleep_score.csv` (`resting_heart_rate`) | Mean nocturnal resting heart rate recorded during sleep | beats/min | Native optical sensor measurement |
| `sleep_hours` | `pmsys/wellness.csv` (`sleep_duration_h`) | Self-reported sleep duration; cross-validated against Fitbit sleep duration | hours | Native self-report + wearable duration |
| `day_strain` | `pmsys/srpe.csv` (`perceived_exertion` $\times$ `duration_min`) | Foster's Session-RPE training load: $\text{Load} = \sum (\text{RPE} \times \text{duration}_{\min})$ | Arbitrary Units (AU) | Documented load proxy; $0.0$ on non-training rest days |
| `hrv_ms` | None available in PMData | Fitbit Versa 2 does not provide exported rMSSD / HRV time series in this release | ms | Explicit `NaN` per shared contract rule |
| `proxy_used` | Metadata string | `"day_strain: sRPE (RPE*min); recovery_pct: PMSys readiness (1-10 scaled to 0-100)"` | String | Mandated transparency field |

---

## 4. Data Cleaning, Filtering & Missing-Data Handling

### Biological Plausibility Truncation & Exclusions
To prevent sensor glitches or survey entry errors from distorting models, observations outside physiological plausibility are flagged and filtered prior to modeling:
- **Sleep Duration**: Must satisfy $2.0 \le \text{sleep\_hours} \le 16.0\text{ h}$. Values $< 2.0\text{ h}$ or $> 16.0\text{ h}$ are treated as missing survey artifacts.
- **Resting Heart Rate**: Must satisfy $35.0 \le \text{rhr\_bpm} \le 110.0\text{ bpm}$.
- **Readiness**: Valid bounded range $[1.0, 10.0]$.
- **Fatigue, Soreness, Stress, Mood**: Valid bounded ordinal ranges $[1.0, 5.0]$.
- **Daily Strain (sRPE)**: Daily sum must satisfy $0.0 \le \text{day\_strain} \le 3000.0\text{ AU}$. An sRPE $> 3000$ represents $> 5\text{ hours}$ at maximal exhaustion (RPE 10) and will be audited for duplicate entry.

### Missing-Data Strategy
- **Rest Days vs. Missing Load**: In athlete monitoring, the absence of an sRPE training entry on a day where morning wellness was completed represents a scheduled **Rest Day** ($\text{Load} = 0.0\text{ AU}$), not missing data. We explicitly impute $0.0\text{ AU}$ for training load on confirmed active days.
- **Lag Alignment & Complete-Case Modeling**: Evaluating prior-night and prior-day effects requires consecutive-day pairs ($t-1$ and $t$). Isolated missing survey days break the temporal chain. Primary models will utilize complete-case analysis on valid consecutive $(t-1, t)$ pairs. The total number of observations, excluded uncoupled days, and retention percentage will be reported explicitly.

---

## 5. Statistical Methodology: Within-Person vs. Between-Person Decomposition

### The Ecological Fallacy in Wearable Analytics
Pooling data from multiple individuals without separating within-person from between-person variance leads to biased and misleading estimates (Simpson's Paradox):
- *Between-Person Trait*: Elite, highly fit athletes often have lower resting heart rates ($45\text{ bpm}$) and may report systematically different subjective readiness than recreational athletes ($65\text{ bpm}$). A pooled regression might show that lower RHR correlates with higher readiness across people.
- *Within-Person State*: Does an acute $+5\text{ bpm}$ elevation above *your own* baseline signal that *you* are fatigued tomorrow?
Only within-person centered models answer the practitioner's actual question.

### Person-Mean Centering Formulation (Currie / Hoffmann Decomposition)
For every time-varying predictor $X_{it}$ (where $i$ indexes the participant and $t$ indexes the day):
1. Compute the participant's longitudinal trait mean:
   $$\bar{X}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} X_{it}$$
2. Compute the acute daily within-person deviation:
   $$X^{\text{within}}_{it} = X_{it} - \bar{X}_i$$
3. Standardize within-person deviations ($z$-scoring) using the pooled within-person standard deviation:
   $$zX^{\text{within}}_{it} = \frac{X_{it} - \bar{X}_i}{\text{SD}_{\text{within}}(X)}$$
   This yields standardized effect sizes ($\beta$), permitting direct comparison of effect magnitudes between sleep duration, resting heart rate, and training load.

---

## 6. Mixed-Effects Model Specification

All models will be estimated using **Linear Mixed-Effects Models (LMM)** via Restricted Maximum Likelihood (**REML**) in `statsmodels.regression.mixed_linear_model.MixedLM`.

### Model 1: Determinants of Next-Day Subjective Readiness
Outcome: Morning Readiness ($Readiness_{it}$, scale 1–10).
Predictors:
- Within-person prior-night sleep duration ($zSleep^{\text{within}}_{it-1}$)
- Between-person mean sleep duration ($\bar{Sleep}_i$)
- Within-person prior-night resting heart rate ($zRHR^{\text{within}}_{it-1}$)
- Between-person mean resting heart rate ($\bar{RHR}_i$)
- Within-person prior-day training load ($zLoad^{\text{within}}_{it-1}$)
- Between-person mean training load ($\bar{Load}_i$)
- Fixed covariate: Day of week / Weekend dummy ($Weekend_{it} \in \{0, 1\}$)

Mathematical Model:
$$Readiness_{it} = (\beta_0 + u_{0i}) + \beta_{1} zSleep^{\text{within}}_{it-1} + \beta_{2} \bar{Sleep}_i + \beta_{3} zRHR^{\text{within}}_{it-1} + \beta_{4} \bar{RHR}_i + \beta_{5} zLoad^{\text{within}}_{it-1} + \beta_{6} \bar{Load}_i + \beta_{7} Weekend_{it} + \epsilon_{it}$$
where:
- $u_{0i} \sim \mathcal{N}(0, \sigma_u^2)$ is the random intercept for participant $i$.
- $\epsilon_{it} \sim \mathcal{N}(0, \sigma_\epsilon^2)$ is the level-1 residual error.

### Model 2: Determinants of Next-Day Muscle Soreness
Outcome: Morning Muscle Soreness ($Soreness_{it}$, scale 1–5; higher = more sore).
Predictors:
- Within-person prior-day training load ($zLoad^{\text{within}}_{it-1}$)
- Between-person mean training load ($\bar{Load}_i$)
- Within-person prior-night sleep duration ($zSleep^{\text{within}}_{it-1}$)

### Model 3: Determinants of Next-Day Morning Fatigue
Outcome: Morning Fatigue ($Fatigue_{it}$, scale 1–5; higher = more fatigued).
Predictors:
- Within-person prior-night sleep score ($zSleepScore^{\text{within}}_{it-1}$)
- Within-person prior-day training load ($zLoad^{\text{within}}_{it-1}$)
- Within-person prior-night resting heart rate ($zRHR^{\text{within}}_{it-1}$)

### Random Slopes Evaluation
We begin with a random-intercept model. To test whether athletes differ significantly in their individual sensitivity to sleep deprivation or training strain, we will fit random-slope models ($u_{1i} zSleep^{\text{within}}_{it-1}$) and evaluate improvement via:
1. Likelihood Ratio Test (LRT, $\chi^2$ distribution with degrees of freedom equal to added variance/covariance components).
2. Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).
If random slopes fail to converge or do not improve model fit ($p > 0.05$), the parsimonious random-intercept model is retained.

---

## 7. Model Diagnostics & Assumption Verification

The following diagnostics will be evaluated and reported in the results:
1. **Normality of Residuals**: Evaluated via visual Q-Q plots of standardized residuals and Shapiro-Wilk / Jarque-Bera tests.
2. **Homoscedasticity**: Standardized residuals plotted against fitted values to detect heteroscedastic fan patterns.
3. **Multicollinearity**: Variance Inflation Factors (VIF) computed across all fixed predictors. Any predictor with $\text{VIF} > 5.0$ will be investigated and reported.
4. **Autocorrelation of Residuals**: Consecutive daily monitoring data may exhibit autoregressive error structures. We compute the lag-1 autocorrelation ($\rho_1$) of residuals within each participant.
5. **Influential Points**: Assessed via Cook's distance / leverage thresholds ($D_i > 4/N$).

---

## 8. Causal Thinking: Directed Acyclic Graph (DAG) & Confounders

```
                   +------------------------+
                   |  Season / Weather /    |
                   |  Academic-Work Stress  |
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
              |          +-----------+         |
              +--------->|  Day of   |<--------+
              |          |   Week    |         |
              |          +-----------+         |
              |                                |
              +----------------+---------------+
                               |
                               v
                     +-------------------+
                     | Next-Day Morning  |
                     | Readiness/Fatigue |
                     +-------------------+
```

### What Can and Cannot Be Concluded
- **Observational Limitations**: This dataset is observational, not an experimental randomized controlled trial. We cannot force athletes into randomized sleep-deprivation or exhaustive exercise protocols.
- **Unmeasured Confounders**:
  - *Dietary intake & alcohol consumption*: Alcohol severely elevates nocturnal resting heart rate and degrades sleep architecture while depressing next-day readiness, but is unrecorded.
  - *Psychological stress*: External life stress impacts both sleep quality and perceived morning readiness simultaneously.
  - *Illness & subclinical infection*: Early immune responses elevate resting heart rate and induce malaise before overt symptoms appear.
- **Scientific Framing**: We report **predictive associations**, never causal claims. We explicitly document that a wearable alert represents a *potential fatigue flag*, never a medical or diagnostic conclusion.

---

## 9. Pre-Registered Sensitivity Analyses

To ensure findings are robust and not artifacts of operational choices:
1. **Rest-Day Imputation Check**: Compare models where rest days are coded as $\text{Load} = 0.0\text{ AU}$ vs. models dropping unlogged training days.
2. **Alternative Recovery Metric**: Re-run Model 1 using Fitbit's proprietary sleep `overall_score` ($0\text{--}100$) in place of PMSys readiness.
3. **Outlier Sensitivity**: Re-estimate coefficients after trimming the top and bottom $1\%$ of within-person sleep duration and sRPE values.

---

## 10. Verification & Sign-Off

This analysis plan is committed to version control **prior** to running statistical mixed-effects models on the PMData cohort. Per user instructions, no mixed-effects model will be fit until this analysis plan has been reviewed and approved.
