# Project 3: Real-World Evidence of Load, Recovery, and Readiness (`p3_nmt_profiler`)

## Scientific Question
To what extent do acute day-to-day fluctuations in prior-night sleep duration, nocturnal resting heart rate (RHR), and prior-day training strain predict next-day subjective readiness, fatigue, and muscle soreness, independent of chronic between-athlete baseline traits?

---

## Analysis Plan & Methodology

1. **Pre-Registration**: Documented in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) prior to statistical modeling.
2. **Dataset**: PMData (Thambawita et al., 2020, ACM MMSys '20; DOI: [10.1145/3339825.3394926](https://doi.org/10.1145/3339825.3394926)). 16 physically active adults tracked continuously for 5 months (~2,400 participant-days).
3. **Currie / Hoffmann Person-Mean Centering Decomposition**:
   - Between-person trait means ($\bar{X}_i$)
   - Within-person acute deviations ($X_{it} - \bar{X}_i$)
   - Standardized acute $z$-scores ($zX^{\text{within}}_{it}$)
4. **Linear Mixed-Effects Models (LMM)** via REML in `p3_nmt_profiler.model`:
   - Model 1: Morning Readiness ~ Acute Sleep + Trait Sleep + Acute RHR + Trait RHR + Acute Load + Trait Load + Weekend + (1|Athlete)
   - Model 2: Morning Muscle Soreness ~ Acute Load + Trait Load + Acute Sleep + (1|Athlete)
   - Model 3: Morning Fatigue ~ Acute Sleep Score + Acute Load + Acute RHR + (1|Athlete)
5. **R Cross-Check Validation**: `p3_nmt_profiler/r_crosscheck.R` independently refits Model 1 via `lme4::lmer` (REML, `nloptwrap`) on `data/processed/p3_model1_dataset.csv`, programmatically comparing results against Python `statsmodels`.
6. **Bilateral Asymmetry Module**: `p3_nmt_profiler.asymmetry` computes Bilateral Asymmetry Index ($\text{BAI}$), Limb Symmetry Index ($\text{LSI}$), persistence flags ($\ge 75\%$), and fatigue sensitivity slopes.

---

## Quickstart & Reproducibility

```bash
# 1. Download summary datasets from PMData OSF repository
python scripts/fetch_pmdata_summaries.py

# 2. Build wearable_daily.csv adhering to shared contract
python -c "from p3_nmt_profiler.pmdata_adapter import build_wearable_daily_dataset; df = build_wearable_daily_dataset(); df.to_csv('data/processed/wearable_daily.csv', index=False)"

# 3. Fit all linear mixed-effects models and print summaries
python -c "from p3_nmt_profiler.model import run_all_models; m1, m2, m3, rs = run_all_models(); print(m1.model_name); print(m2.model_name); print(m3.model_name)"

# 4. Generate publication figures
python scripts/generate_p3_figures.py

# 5. Run full test suite
pytest -v tests/test_p3_pmdata.py tests/test_p3_model.py tests/test_p3_asymmetry.py
```

---

## Key Findings

| Hypothesis | Parameter | Estimate ($\beta$) | $p$-value | 95% Confidence Interval | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$H_{1a}$: Sleep Duration $\to$ Readiness** | $zSleep^{\text{within}}_{it-1}$ | **$+0.218$** | **$<0.001$** | $[+0.139, +0.297]$ | **Supported** |
| **$H_{1c}$: Sleep Score $\to$ Energy** | $zSleepScore^{\text{within}}_{it-1}$ | **$+0.143$** | **$<0.001$** | $[+0.110, +0.177]$ | **Supported** |
| **$H_{3a}$: Training Load $\to$ Soreness** | $zLoad^{\text{within}}_{it-1}$ | **$-0.129$** | **$<0.001$** | $[-0.156, -0.103]$ | **Supported** |
| **$H_2$: Nocturnal RHR $\to$ Readiness** | $zRHR^{\text{within}}_{it-1}$ | $-0.021$ | $0.606$ | $[-0.099, +0.058]$ | **Null Finding** |
| **$H_{3b}$: Training Load $\to$ Readiness** | $zLoad^{\text{within}}_{it-1}$ | $-0.073$ | $0.137$ | $[-0.169, +0.023]$ | **Null Trend** |

### Sleep Sensitivity Heterogeneity
Likelihood Ratio Testing ($\chi^2(2) = 32.02, p < 0.001$) confirmed significant inter-individual variation in sleep loss sensitivity ($\sigma_{u1}^2 = 0.122$). Individual athlete response slopes range from $\beta_i \approx 0.0$ to $\beta_i = +0.55$.

### R Cross-Check (`lme4` vs `statsmodels`)
Model 1 was independently refit in R 4.6.1 using `lme4::lmer` with REML and the `nloptwrap` optimizer on the identical Currie-Hoffmann centered dataset (`p3_model1_dataset.csv`, $n=1,368, 15\text{ athletes}$).
- **Acute Sleep Deviation ($zSleep^{\text{within}}$)**:
  - Python `statsmodels`: $\beta = +0.21817881$, $\text{SE} = 0.04023033$, Wald $95\%\text{ CI}: [0.13932881, 0.29702881]$ (rounded: $\beta = +0.2182$, $\text{SE} = 0.0402$, $95\%\text{ CI}: [0.1393, 0.2970]$)
  - R `lme4`: $\beta = +0.21817885$, $\text{SE} = 0.04023046$, Wald $95\%\text{ CI}: [0.13932860, 0.29702910]$ (rounded: $\beta = +0.2182$, $\text{SE} = 0.0402$, $95\%\text{ CI}: [0.1393, 0.2970]$)
  - Direct Absolute Difference: $|\Delta \beta| = 4.3 \times 10^{-8}$ ($0.0000000428$), agreeing to ~1e-7.
- **Variance Components & Numerical Independence**:
  - Residual Variance ($\sigma_\epsilon^2$): Python $= 2.12187528$ vs. R $= 2.12188964$ ($|\Delta| = 1.4 \times 10^{-5}$).
  - Group Variance ($\sigma_u^2$): Python $= 1.44899925$ vs. R $= 1.44771033$ ($|\Delta| = 1.3 \times 10^{-3}$).
  - *Methodological Note*: While fixed-effect estimates agree to $\approx 10^{-7}$, variance components differ more substantially ($1.4 \times 10^{-5}$ residual, $1.3 \times 10^{-3}$ group). This pattern (extremely tight fixed-effect agreement alongside looser variance-component agreement) is the expected signature of distinct REML optimization routines (Powell/L-BFGS vs. `nloptwrap`/BOBYQA) on a well-conditioned model: conditional on finding the approximate 1D variance ratio peak, fixed effects are calculated via closed-form Generalized Least Squares (GLS), confirming the two fits were genuinely independent.
- **Hypothesis Testing**: $p$-values differ in reporting form because R `lmerTest` computes finite-sample Satterthwaite denominator degrees of freedom ($t = 5.423, df = 1349, p = 6.93 \times 10^{-8}$) whereas Python `statsmodels.MixedLM` reports an asymptotic standard normal $z$-test ($z = 5.423, p = 5.85 \times 10^{-8}$).

---

## Scientific Limitations & Ethical Transparency

1. **Observational Framing**: This analysis evaluates free-living observational data, not randomized experimental interventions. Models capture predictive associations; correlations are never causal.
2. **Unmeasured Confounders**: Alcohol consumption, academic/work deadlines, and subclinical viral infections simultaneously affect sleep quality and waking readiness.
3. **Heuristic Alerting**: In accordance with portfolio research conventions, wearable alerts represent *possible fatigue flags*, never medical or clinical diagnoses.
4. **AI-Use Transparency**: Code was developed with AI assistance and verified against known closed-form analytical models in `tests/test_synth.py`, automated test suites, and independently cross-checked in R `lme4`.
