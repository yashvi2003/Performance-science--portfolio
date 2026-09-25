# Project 3: Load, recovery and readiness: real-world-evidence analysis (`p3_nmt_profiler`)

Goal: a hypothesis-driven analysis of load, recovery and readiness on public wearable data, plus an asymmetry
and fatigue module for jump data. Only the shared contract couples it to
other projects.

## Data plan (important)
- The wearable analytics utilize open-access datasets (e.g. PMData) and contract-compliant synthetic benchmarks.
- Mechanical side (per-limb): first check whether the user's 10-athlete CMJ set has left/right force
  columns. If yes, use it (de-identified) for real asymmetry. If it only has total force, asymmetry
  cannot be computed from it; say so and do not impute.
- For more real bilateral data, log candidates in DATA_SOURCES.md before using them. Leads to check
  (availability and licence NOT yet verified): the ForcePose dataset (8 subjects, CMJ/squat jump with
  separate left/right force plates, from an arXiv paper); the directory of public biomechanics datasets at
  github.com/mkjung99/biomechanics_dataset; data-availability statements of force-plate CMJ papers; and
  repositories such as Figshare, Zenodo, OSF, Mendeley Data and Dryad. Contacting corresponding authors for
  data is normal in research. Symbiosis library access mostly gives journal articles and their supplementary
  files, not raw datasets, so use it to find papers, then follow their data links.
- Different sources mean different devices, sampling rates and protocols: one adapter per source, provenance
  recorded per row (`source` column in sessions.csv metadata), and no pooling that hides device differences.
- A search of Kaggle found no real bilateral force-plate CMJ dataset. Otherwise use `shared/synth.py` with a
  documented asymmetry and fatigue slope, labelled synthetic.
- Wearable side: candidates to inspect on Kaggle are PMData (16 people, 5 months, Fitbit Versa 2 plus PMSys
  wellness/readiness/fatigue/sleep/soreness/stress and training logs), "Athlete Training & Recovery Tracker" and
  "Multimodal Sports Injury Dataset" (check whether synthetic first). The adapter maps to wearable_daily.csv.
  There is no native strain, so build a documented proxy (e.g. session-RPE x duration) and set `proxy_used`.
- Pairing real wearable data with real or synthetic jump data from different people is a DEMO of the method.
  Never present the resulting correlations as a finding. Never commit third-party data; cite and respect licences.

## Core analysis (real-world evidence on public wearable data)
- Write the hypotheses and analysis plan BEFORE looking at results: question, outcome, predictors, covariates,
  model, missing-data handling, sensitivity checks. Commit the plan first so the history shows it.
- Example question (adjust to what the data supports): within a person, does a night of shorter sleep or a
  higher prior-day training load predict lower next-day self-reported readiness/fatigue?
- Model: mixed-effects regression with a random intercept per person (random slopes if supported), separating
  within-person from between-person effects (person-mean centring). Report effect sizes with 95% CIs, residual
  diagnostics, and a sensitivity analysis (e.g. different missing-data handling, outlier influence).
- Causal thinking: draw a DAG, list confounders (e.g. season, match days, illness), state plainly what can and
  cannot be concluded from observational wearable data. Never claim causation.
- Communicate for two audiences: a technical methods section and a one-paragraph plain-language summary.

## Asymmetry module (secondary)
- One formula, documented in the README: (higher - lower) / higher x 100, report which limb is higher.
- Per metric: asymmetry %, direction, trend across sessions. Persistent-asymmetry flag
  (configurable threshold, N consecutive sessions). Thresholds are practice conventions, not injury
  predictors. Skip trials without per-limb data; never impute.

## Fatigue tracking
- Per block: trend of chosen metrics (DJ RSI, SJ RFD, CMJ RSImod): slope with confidence interval,
  rolling z-score vs the block's opening baseline.
- Strain comparison: weekly load (sum of day_strain or the documented proxy) vs metric change, plus
  lagged Spearman correlation (0-3 day lags) with CIs and n. Warn when n is small; no causal claims.
- Main figure: strain vs mechanical output, stacked or dual-axis, longitudinal.

## Tests
Synthetic data with known asymmetry and known fatigue slope: recover both within tolerance, correct
limb direction, clean skipping of trials without per-limb data, adapter mapping tests.

## Definition of done
All Project 3 tasks ticked; pytest green; main figure produced; README states data provenance for
every dataset used and the synthetic/real split explicitly.
