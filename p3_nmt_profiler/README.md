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
5. **Cross-Platform Scripting**: `p3_nmt_profiler/r_crosscheck.R` provides an external supplementary R script to refit Model 1 via `lme4::lmer` on `data/processed/p3_model1_dataset.csv`.
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

---

## Scientific Limitations & Ethical Transparency

1. **Observational Framing**: This analysis evaluates free-living observational data, not randomized experimental interventions. Models capture predictive associations; correlations are never causal.
2. **Unmeasured Confounders**: Alcohol consumption, academic/work deadlines, and subclinical viral infections simultaneously affect sleep quality and waking readiness.
3. **Heuristic Alerting**: In accordance with portfolio research conventions, wearable alerts represent *possible fatigue flags*, never medical or clinical diagnoses.
4. **AI-Use Transparency**: Code was developed with AI assistance and verified against known closed-form analytical models in `tests/test_synth.py` and extensive automated unit test suites.
