# Sports Performance & Biomechanics Analytics Toolkit

[![Tests](https://github.com/yashvi2003/Performance-science--portfolio/actions/workflows/tests.yml/badge.svg)](https://github.com/yashvi2003/Performance-science--portfolio/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![SQL Engine](https://img.shields.io/badge/SQL-DuckDB-orange)](p2_nmt_reports/sql/)
[![Stats](https://img.shields.io/badge/models-Linear%20Mixed--Effects%20(Python%20%2B%20R)-purple)](p3_nmt_profiler/)

Portfolio site: [to be added after deployment]

An applied sports performance engineering and data science portfolio containing three modular, tested pipelines for **force-plate jump biomechanics**, **SQL longitudinal readiness baselines**, and **longitudinal wearable recovery modeling**.

---

## Portfolio Architecture & Modules

```
portfolio/
├── p1_nmt_pipeline/      # Project 1: Automated Jump Biomechanics & Signal Processing Engine
├── p2_nmt_reports/       # Project 2: Rolling Baselines, Readiness Index & DuckDB SQL Layer
├── p3_nmt_profiler/      # Project 3: Wearable Recovery Profiling & Linear Mixed-Effects Models
├── shared/               # Strict schema contracts (Pydantic / CSV schemas) and synthetic data generators
├── reports/              # Figures, summary tables, and technical documentation
└── tests/                # 100 comprehensive unit, statistical, and cross-engine parity tests
```

---

### Project 1: Automatic Jump Analysis Engine (`p1_nmt_pipeline`)
*Automated force-time curve partitioning, event detection, and impulse-momentum numerical integration following the phase definitions of McMahon et al. (2018).*

* **Automated Phase Segmentation**: Detection of Quiet-standing baseline, Unweighting trough, Zero-velocity transition ($v(t)=0$), Peak propulsion force, Takeoff, Flight floor, and Landing absorption impact.
* **Core Metrics**: Impulse-momentum jump height, flight-time jump height, Modified Reactive Strength Index ($\text{RSImod} = \text{JH} / \text{Time to Takeoff}$), peak power via trapezoidal integration, and concentric/eccentric duration ratios.
* **Signal Robustness & Flight Sensor Offsets**: Zero-phase low-pass Butterworth filtering, adaptive noise-floor calibration, and sub-sample extrapolation to eliminate threshold truncation biases.
* **Validation**: Monte Carlo accuracy testing across 9 benchmark conditions ($N=200$ trials per condition) evaluating sampling rates ($250\text{--}2000\text{ Hz}$), noise levels ($0\text{--}12\text{ N}$ baseline noise, up to $4\text{ N}$ flight noise), and four force-curve templates on synthetic trials with known analytical ground truth ($20\text{ ms}$ cosine-ramp onset, $\pm 15\text{ ms}$ event jitter). 95th-percentile impulse-momentum jump height error is $\le 6.0\text{ mm}$ across all clean $2000\text{ Hz}$ conditions ($\le 8.0\text{ mm}$ under $12\text{ N}$ noise; worst-case $\le 12.0\text{ mm}$ at $250\text{ Hz}$ discrete sampling; see per-condition tolerance table in [`p1_nmt_pipeline/README.md`](p1_nmt_pipeline/README.md)).
* **Open-Data Consistency Check (White et al.)**:
  * *(a) Reference Value Reproduction*: Evaluated across all 663 CMJ trials (67 participants, 1000 Hz dual Kistler platforms; Zenodo DOI: [10.5281/zenodo.19136480](https://doi.org/10.5281/zenodo.19136480), CC BY 4.0) as a consistency check against published reference values, reproducing author reference peak power with Pearson $r = 1.000000$ (mean diff $+0.00000\text{ W/kg}$) when evaluated at author local UK gravity ($g = 9.81200\text{ m/s}^2$). Under the pipeline's standard gravity ($g = 9.80665\text{ m/s}^2$), peak power exhibits a physical $-0.109\%$ ($-0.0531\text{ W/kg}$) offset, and total COM jump height agrees to $+0.101\text{ cm}$ ($+0.235\%$, $r = 1.000000$). This reproduction is a consistency check against published reference outputs, not an independent criterion validation.
  * *(b) Within-Participant Reliability*: Computed across repeated trials of the pipeline's outputs on the primary CMJNA balanced 4-trial protocol ($n = 66\text{ participants} \times 4\text{ trials} = 264\text{ observations}$), yielding single-measure reliability for total COM jump height of $\text{ICC}(2, 1) = 0.9060$ ($95\%\text{ CI: }[0.867, 0.937]$), $\text{CV} = 4.92\%$, $\text{SEM} = 2.55\text{ cm}$, and $\text{MDC}_{95} = 7.07\text{ cm}$ (peak power: $\text{ICC}(2, 1) = 0.9308$, $\text{CV} = 3.27\%$, $\text{SEM} = 1.92\text{ W/kg}$). This evaluates trial-to-trial repeatability of the pipeline, not agreement with external benchmarks.
* 📄 **Detailed Documentation**: [`p1_nmt_pipeline/README.md`](p1_nmt_pipeline/README.md)

---

### Project 2: Baselines, Readiness & SQL Reporting Engine (`p2_nmt_reports`)
*Longitudinal monitoring engine calculating personal rolling baselines, athlete variability bounds, and composite readiness scores in both Python and standard SQL.*

* **Rolling Personal Baselines**: 28-day backward-looking calendar baseline with a strict minimum guardrail ($N \ge 3$ sessions) to prevent over-reacting to sparse early entries.
* **Smallest Worthwhile Change (SWC)**: Dynamically calculated per athlete ($\text{SWC} = 0.5 \times \text{SD}_{\text{within}}$, [Hopkins, 2004](https://www.sportsci.org/jour/04/wghtests.htm)) alongside Typical Error of measurement ($\text{TE} = \text{SD} / \sqrt{2}$, [Hopkins, 2000](https://pubmed.ncbi.nlm.nih.gov/10966148/)).
* **Direction-Aware Readiness Index**: Standardized $z$-scoring respecting metric directions ($+z$ always represents favorable performance, properly inverting metrics where lower is better like contact times).
* **DuckDB SQL Window Layer**: Complete analytical parity between Python/Pandas and DuckDB SQL window functions (`OVER (PARTITION BY athlete_id ORDER BY session_date RANGE BETWEEN INTERVAL 28 DAYS PRECEDING...)`), verified down to $|\Delta| < 10^{-6}$.
* **Within-Athlete Linear Mixed-Effects Model [synthetic data]**: Estimates session-to-session variance and derives athlete-specific Minimal Detectable Change ($\text{MDC}_{95}$) thresholds.
* 📄 **Detailed Documentation**: [`p2_nmt_reports/README.md`](p2_nmt_reports/README.md)

---

### Project 3: Wearable Recovery & Mixed-Effects Profiler (`p3_nmt_profiler`)
*Longitudinal investigation of wearable sleep duration, nocturnal resting heart rate (RHR), and training strain predicting next-day readiness and fatigue, with hypotheses and model specifications documented in [`ANALYSIS_PLAN.md`](p3_nmt_profiler/ANALYSIS_PLAN.md) before modelling.*

* **Dataset**: PMData open wearable benchmark ([Thambawita et al., 2020](https://doi.org/10.1145/3339825.3394926)), spanning 16 adults tracked continuously over 5 months (2,080 daily records in `data/processed/wearable_daily.csv`; Model 1: $n=1,368$ across 15 participants; Model 2: $n=1,716$ across 16 participants; Model 3: $n=1,369$ across 15 participants; sensor-matched complete cases).
* **Within-Person Centering (Currie / Hoffmann Method)**: Isolates acute day-to-day fluctuations ($X_{it} - \bar{X}_i$) from chronic between-athlete traits ($\bar{X}_i$), separating within-person from between-person variation without removing unmeasured confounding.
* **Linear Mixed-Effects Models (LMM)**: Formulated with random intercepts per participant (`statsmodels.formula.api.mixedlm` via REML), with random slopes evaluated for standardized acute sleep deviation via Likelihood Ratio Test comparing ML fits ($\chi^2(2) = 32.02, p < 0.001$, estimated slope variance $\sigma_{u1}^2 = 0.122$ with random intercept-slope covariance):
  * **Readiness Model**: Acute sleep duration is positively associated with next-day morning readiness ($\beta = +0.218, p < 0.001, 95\%\text{ CI: }[+0.139, +0.297]$; readiness scale 1–10, higher is more ready).
  * **Soreness Model**: Acute training strain is associated with greater next-day muscle soreness ($\beta = -0.129, p < 0.001, 95\%\text{ CI: }[-0.156, -0.103]$; PMSys soreness scale 1–5, where 1=very sore and 5=fresh, so negative $\beta$ indicates greater soreness). Acute sleep duration was not significantly associated with soreness ($\beta = +0.005, p = 0.694$).
* **Cross-Language Validation (Python $\leftrightarrow$ R)**: Independent execution via R `lme4::lmer` with exact unrounded parameter comparisons confirming model convergence and Wald parameter estimates.
* **Bilateral Asymmetry & Fatigue Sensitivity [synthetic data]**: Bilateral Asymmetry Index ($\text{BAI}$), Limb Symmetry Index ($\text{LSI}$), persistence tracking ($\ge 75\%$), and fatigue sensitivity slopes evaluated on synthetic dual-plate bilateral trials (`shared/synth.py`), as PMData does not record bilateral force plate kinetics.
* 📄 **Detailed Documentation**: [`p3_nmt_profiler/README.md`](p3_nmt_profiler/README.md) & [`p3_nmt_profiler/ANALYSIS_PLAN.md`](p3_nmt_profiler/ANALYSIS_PLAN.md)

---

## Summary of Analytical Methods

| Module | Core Statistical / Engineering Method | Primary Reference | Ground-Truth Test |
| :--- | :--- | :--- | :--- |
| **`p1_nmt_pipeline`** | Trapezoidal Impulse-Momentum Integration | McMahon et al. (2018) | `test_cmj_analytical_ground_truth` |
| **`p1_nmt_pipeline`** | Zero-phase 4th-order Butterworth Filter | Winter (2009) | `test_filter_zero_phase_alignment` |
| **`p2_nmt_reports`** | Rolling SWC & Typical Error | Hopkins (2000, 2004) | `test_rolling_baseline_math_and_swc` |
| **`p2_nmt_reports`** | DuckDB SQL Window Functions | ISO/IEC 9075 SQL:2016 | `test_sql_rolling_baselines_matches_pandas_exact` |
| **`p3_nmt_profiler`** | Person-Mean Centered Mixed-Effects Models | Currie (2019), Hoffmann (2019) | `test_model_1_readiness` |
| **`p3_nmt_profiler`** | Multi-Language Validation (Python vs R) | Bates et al. (2015, `lme4`) | `p3_nmt_profiler/r_crosscheck.R` |

---

## Getting Started & Reproducibility

### 1. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/yashvi2003/Performance-science--portfolio.git
cd Performance-science--portfolio
```

**Option A: Using `uv` (Recommended)**
```bash
# Sync dependencies into local virtual environment
uv sync

# Run tests
uv run pytest
```

**Option B: Standard `pip` and `venv`**
```bash
# Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run tests
pytest
```

### 2. Run Test Suite
Run the full 100-test suite verifying analytical ground truths, mathematical formulas, SQL parity, and statistical models:

```bash
pytest
```

### 3. Generate Reports & Visualizations
Regenerate all figures, summary tables, and research cards:

```bash
# Run Project 1 Single Trial CMJ Report
python scripts/generate_single_trial_report.py

# Run Project 2 Longitudinal Baselines & SQL Engine
python p2_nmt_reports/charts.py

# Run Project 3 Wearable Modeling & Evidence Figures
python p3_nmt_profiler/figures.py
```

---

## Data Privacy, Provenance & Open Science

* **Licence**: Code is released under the [MIT License](LICENSE). Third-party datasets retain their respective original licences and terms of use (see [`DATA_SOURCES.md`](DATA_SOURCES.md)). Note that derived files in `data/processed/` generated from PMData (`wearable_daily.csv`, `p3_model1_dataset.csv`, `p3_model1_py_results.csv`) may carry the dataset's non-commercial licensing terms (CC BY-NC 4.0), separate from the MIT licence governing the source code.
* **Data Provenance**: Fully documented in [`DATA_SOURCES.md`](DATA_SOURCES.md). Open datasets include PMData (Fitbit Versa 2 and PMSys logging, licensed under CC BY-NC 4.0 per OSF `license.txt`; portal states CC BY 4.0) and White et al. (Open force plate CMJ dataset, CC BY 4.0).
* **Data Hygiene**: Real human participant raw files and proprietary spreadsheets are strictly excluded from git tracking via [`.gitignore`](.gitignore). Only de-identified aggregate statistics and synthetic test data are tracked.
* **Scientific Transparency**: Synthetic benchmarks are explicitly designated with `[Synthetic Data]` markers on all generated figures. The composite readiness index is documented as an unvalidated heuristic composite. Acute-drop flags and recovery deficit tags (e.g., $z < -1.0$ or $\Delta\% < -10\%$) are characterized as "possible fatigue flags", not diagnostic determinations. Multi-tier alert rules are deferred to future work.

---

## Limitations

* **1D Vertical Kinetics Only**: The force plate engine models vertical ground reaction force ($F_z$) exclusively; anterior-posterior ($F_y$) and medial-lateral ($F_x$) shear forces, center of pressure excursions, and multi-axis torque dynamics are outside current pipeline scope.
* **Modalities Deferred**: The codebase implements countermovement jump (CMJ) phase detection and metrics; squat jump (SJ), drop jump (DJ), isometric mid-thigh pull (IMTP) metrics, jump-type classifiers, and interactive Streamlit UI dashboards were deferred to prioritize core longitudinal modeling.
* **Project 2 Synthetic Demonstration**: Project 2 longitudinal baselines and readiness scoring run on synthetic multi-session data (`shared/synth.py`) as a local batch computation demo, not an active streaming or real-time athlete monitoring system.
* **Cohort Scope & Observational Design**: Project 3 is observational across 16 physically active adults in free-living conditions (2,080 processed days, with 1,368–1,716 participant-days across fitted models) rather than an elite competitive cohort; associations cannot establish causality, and unmeasured confounders (e.g., life stress, alcohol, training specifics) remain unobserved.
* **No Autonomic Biomarkers in Wearable Cohort**: The PMData cohort relies on optical PPG resting heart rate and self-report wellness surveys without vagal heart rate variability time series (rMSSD is unrecorded/NaN), limiting autonomic recovery resolution.
* **Disjoint Datasets (No Single End-to-End Athlete Pipeline)**: Project 1 evaluates jump force-time series (White et al. open data and 10 single-trial recordings), Project 2 demonstrates longitudinal SQL window transformations on synthetic sessions, and Project 3 investigates PMData wearable telemetry; no single athletic cohort spans all three modules end-to-end.
* **Composite Readiness as an Unvalidated Heuristic**: The weighted composite readiness score is an exploratory pedagogical heuristic and must not be interpreted as a validated clinical or diagnostic metric. Alert flags denote "possible fatigue flags", not formal interventions.
