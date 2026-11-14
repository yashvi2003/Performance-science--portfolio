# Sports Performance & Biomechanics Analytics Toolkit: Master Architecture & Implementation Plan

**Standing Instructions & Governance**: `AGENTS.md`, `STATUS.md`, `specs/00_shared_contract.md`.

---

## 1. Executive Summary & Design Principles

This toolkit is designed as a research-grade neuromuscular and physiological monitoring platform. It bridges high-frequency force plate biomechanics (countermovement jumps, squat jumps, drop jumps) with daily wearable telemetry (recovery, HRV, resting HR, strain, sleep).

### Core Principles
1. **Research Rigor Over Feature Bloat**: Every metric has a documented formula, citation, and measurement error framework (CV, ICC, SEM, MDC, Bland-Altman limits of agreement).
2. **Strict Data Contract Isolation**: Projects import **only** from `shared/`. No cross-project imports (`p2` does not import `p1`, `p3` does not import `p1` or `p2`). Inter-project communication happens exclusively through `sessions.csv` and `wearable_daily.csv`.
3. **Transparent Honesty & Provenance**:
   - Synthetic data demonstrates pipeline function and test coverage; it is explicitly labeled `synthetic` in outputs and plots, never reported as an empirical finding.
   - Composite readiness is labeled an unvalidated heuristic.
   - Alerts are labeled "possible fatigue flags", never diagnoses.
   - Observational wearable associations are never framed as causal.
   - Cloud-AI Guardrail: Professor-supplied athlete data (`D1`, `D1b`) remains private and unprocessed by cloud AI until explicit consent is documented. Public demos use synthetic or open CC BY 4.0 data (`D4` White et al.).
4. **Reproducibility**: One-command reproduction (`python scripts/run_all.py`), versioned YAML configurations, fixed random seeds, and no hardcoded thresholds.

---

## 2. Repository File Tree (Implemented vs Deferred)

```
portfolio-agent-kit/                # Single repository root (workspace root)
├── .gitignore                      # Excludes data/raw/, data/private/, *.xlsx, *.npz, .venv/
├── pyproject.toml                  # Python 3.11+, pinned dependencies, pytest & ruff configs
├── AGENTS.md                       # Standing agent instructions & honesty guardrails
├── STATUS.md                       # Project progress checklist & session logs
├── PLAN.md                         # Master architecture, schemas, and implementation plan (this file)
├── DATA_SOURCES.md                 # Provenance, licences, and consent tracking for all datasets
├── README.md                       # Top-level portfolio overview & human checklist
├── specs/
│   ├── 00_shared_contract.md       # Specification of shared CSV schemas
│   ├── 01_neuromuscular_pipeline.md# Spec for Project 1 (VALD-style jump engine)
│   ├── 02_reporting_engine.md      # Spec for Project 2 (SQL, baselines, readiness)
│   ├── 03_asymmetry_profiler.md    # Spec for Project 3 (RWE wearable analysis, asymmetry)
│   └── 05_input_data_profile.md    # Biomechanical characteristics of raw data
├── scripts/
│   ├── progress.py                 # Terminal progress tracker parsed from STATUS.md
│   ├── import_cmj_xlsx.py          # De-identifies & imports professor Excel files to CSV
│   ├── fetch_pmdata_summaries.py   # Ingests PMData summaries from OSF
│   ├── generate_single_trial_report.py # Generates VALD-style CMJ diagnostic report
│   ├── generate_p3_figures.py      # Generates Project 3 figures
│   └── run_all.py                  # End-to-end execution script reproducing all figures
├── shared/
│   ├── __init__.py
│   ├── contract.py                 # Schemas, validators, metric registry, directions
│   └── synth.py                    # Analytical force-time & longitudinal table generators
├── p1_nmt_pipeline/                # PROJECT 1: Automatic Jump Analysis Engine
│   ├── __init__.py
│   ├── config.yaml                 # Tunable thresholds, filter cutoffs, integration lead, offset flags
│   ├── config.py                   # Configuration dataclass loaders
│   ├── filter.py                   # 4th-order zero-phase Butterworth filter (scipy.signal)
│   ├── events.py                   # Phase detection (quiet baseline, onset, unweight, brake, prop, TO, land)
│   ├── metrics.py                  # Impulse-momentum, flight-time, RSImod, power, velocity
│   ├── open_data_adapter.py        # White et al. Zenodo npz data loader
│   ├── reliability.py              # Reliability metrics (ICC, CV%, SEM, MDC)
│   ├── validator.py                # Data validation & resampling
│   └── cli.py                      # Batch processor outputting shared sessions.csv
│   # Note: classifier.py, sj.py, dj.py, imtp.py, app.py deferred per STATUS.md
├── p2_nmt_reports/                 # PROJECT 2: Baselines, Readiness & SQL Layer
│   ├── __init__.py
│   ├── config.yaml                 # Baseline window (28d), alert thresholds, readiness weights
│   ├── config.py                   # Configuration loaders
│   ├── loader.py                   # Shared contract loader & validator
│   ├── sql/                        # SQL layer with DuckDB
│   │   ├── schema.sql              # Table definitions for DuckDB/SQLite
│   │   ├── rolling_baselines.sql   # Window functions for 28-day rolling mean & SD
│   │   └── session_aggregates.sql  # Best trial & session means
│   ├── sql_engine.py               # Python-DuckDB executor & Pandas equivalence tests
│   ├── baselines.py                # Rolling personal baselines, typical error & SWC
│   ├── readiness.py                # Direction-aware z-scores & composite heuristic readiness
│   ├── wearable.py                 # Wearable ingestion, date matching & recovery overlay
│   ├── stats.py                    # Mixed-effects session-to-session change model (statsmodels)
│   └── charts.py                   # Publication-ready longitudinal baseline and readiness plots
│   # Note: alerts.py, report.py, app.py deferred per STATUS.md
├── p3_nmt_profiler/                # PROJECT 3: Real-World Evidence & Asymmetry Profiler
│   ├── __init__.py
│   ├── config.yaml                 # Wearable column mappings, asymmetry thresholds
│   ├── config.py                   # Configuration loaders
│   ├── ANALYSIS_PLAN.md            # Documented research questions, hypotheses & DAG
│   ├── pmdata_adapter.py           # Wearable adapter mapping PMData to shared schema
│   ├── model.py                    # Linear mixed-effects regression & random slopes LRT
│   ├── sensitivity.py              # Rest-day imputation, trimming, and BLUP slope sensitivities
│   ├── asymmetry.py                # Bilateral Asymmetry Index (BAI), LSI, persistence & fatigue slopes
│   └── r_crosscheck.R              # Independent R lme4 cross-validation script
├── data/
│   ├── synthetic/                  # Contract-compliant generated trial and longitudinal data
│   ├── processed/                  # Derived sessions.csv and wearable_daily.csv
│   ├── raw/                        # Gitignored: de-identified real data (D1, D4)
│   └── private/                    # Gitignored: original professor files (D1, D1b)
└── tests/
    ├── test_contract.py            # Schema validation & metric registry tests
    ├── test_synth.py               # Analytical property verification of synthetic generator
    ├── test_p1_pipeline.py         # Filtering, phase detection, metrics accuracy
    ├── test_cmj.py                 # Comprehensive CMJ detection & Monte Carlo tolerance tests
    ├── test_cmj_real.py            # Professor cohort parsing & golden test checks
    ├── test_cmj_open.py            # White et al. open-data validation & reliability tests
    ├── test_p2_baselines.py        # Baseline calculation & SWC tests
    ├── test_p2_readiness.py        # Direction-aware readiness score tests
    ├── test_p2_sql.py              # Parity between SQL window queries and Pandas
    ├── test_p3_pmdata.py           # PMData ingestion & schema mapping tests
    ├── test_p3_model.py            # Linear mixed-effects model estimation & diagnostic tests
    └── test_p3_asymmetry.py        # Bilateral asymmetry index & persistence tests
```

---

## 3. Data Contracts & Schemas (`shared/contract.py`)

### 3.1 `sessions.csv` (Long Format)
Each row represents a single metric observation for a specific trial and limb side:

| Column | Type | Validation Rules / Notes |
| :--- | :--- | :--- |
| `session_id` | `str` | Unique session identifier (e.g. `A01_2026-03-01`) |
| `date` | `str` | ISO 8601 date (`YYYY-MM-DD`) |
| `athlete_id` | `str` | Anonymized athlete code (e.g. `A01`, `A02`) |
| `test` | `str` | Categorical: `CMJ`, `SJ`, `IMTP`, `DJ` |
| `trial` | `int` | 1-based trial integer within session |
| `metric` | `str` | Valid snake_case metric registered in `MetricRegistry` |
| `value` | `float` | Metric value; finite, non-null |
| `unit` | `str` | Standardized unit string (`m`, `s`, `N`, `W`, `W/kg`, `m/s`, `m/s`, `ratio`) |
| `side` | `str` | `both` (default), `left`, `right`. Per-limb rows only where dual plate data exists |
| `valid` | `bool` | `True` if passed QA; `False` if flagged (kept for audit trail) |
| `flags` | `str` | Semicolon-separated QA flags (e.g. `noisy_baseline;dip_detected`) |
| `block` | `str` | Optional training phase or week identifier (e.g. `pre_season_w1`) |

### 3.2 `wearable_daily.csv` (Daily Format)
Each row represents daily wearable telemetry for an athlete-day:

| Column | Type | Validation Rules / Notes |
| :--- | :--- | :--- |
| `date` | `str` | ISO 8601 date (`YYYY-MM-DD`) |
| `athlete_id` | `str` | Matches athlete_id in sessions table |
| `recovery_pct` | `float` | 0.0 to 100.0 (Zones: Red <34, Yellow 34–66, Green 67–100) |
| `hrv_ms` | `float` | rMSSD or equivalent vagal HRV index in milliseconds |
| `rhr_bpm` | `float` | Resting heart rate in beats per minute |
| `day_strain` | `float` | Daily exertion / load score (native strain or documented proxy) |
| `sleep_hours` | `float` | Total sleep duration in hours |
| `proxy_used` | `str` | Empty string if native wearable data; otherwise method description (e.g. `sRPE_x_duration`) |

### 3.3 Metric Registry & Optimization Directions
All metrics are registered in `shared/contract.py` with standard units and optimization directions:
- `cmj_jump_height_imp` (`m`, `higher_is_better`): Impulse-momentum jump height
- `cmj_jump_height_flight` (`m`, `higher_is_better`): Flight-time jump height
- `cmj_rsimod` (`ratio` or `m/s`, `higher_is_better`): Modified RSI ($h / \Delta t_{ct}$)
- `cmj_peak_power` (`W`, `higher_is_better`): Peak propulsion mechanical power
- `cmj_peak_power_rel` (`W/kg`, `higher_is_better`): Peak power normalized to body mass
- `cmj_peak_velocity` (`m/s`, `higher_is_better`): Peak vertical velocity before takeoff
- `cmj_peak_braking_force` (`N`, `higher_is_better`): Maximum force during braking phase
- `cmj_time_to_takeoff` (`s`, `lower_is_better`): Contraction time from onset to takeoff
- `cmj_flight_time` (`s`, `higher_is_better`): Duration between takeoff and landing
- `sj_jump_height` (`m`, `higher_is_better`): Concentric-only jump height
- `sj_peak_force` (`N`, `higher_is_better`): Maximum vertical force
- `sj_rfd_0_50ms`, `sj_rfd_0_100ms`, `sj_rfd_0_200ms` (`N/s`, `higher_is_better`): Rate of force development
- `dj_rsi` (`ratio`, `higher_is_better`): Reactive Strength Index ($h / GCT$)
- `dj_gct` (`s`, `lower_is_better`): Ground contact time
- `dj_jump_height` (`m`, `higher_is_better`): Flight-time height off drop landing
- `imtp_peak_force_gross` (`N`, `higher_is_better`): Absolute peak force
- `imtp_peak_force_net` (`N`, `higher_is_better`): Peak force minus body weight

---

## 4. Biomechanical Formulations & Mathematical Rigor

### 4.1 Signal Conditioning & Offset Handling
- **Low-pass Filter**: Zero-phase 4th-order low-pass Butterworth filter (`scipy.signal.filtfilt`), default cutoff $f_c = 30 \text{ Hz}$ (configurable in YAML).
- **Flight-Phase Force-Offset Correction**: Optional configuration flag (`correct_flight_phase_offset: false`, off by default). When enabled, calculates residual mean force during flight $[\overline{F}_{flight}]$ and subtracts it across the trace. Synthetic generator supports `flight_force_offset_n: float = 0.0` to inject offsets (e.g. $+4.0 \text{ N}$) for regression testing this correction.

### 4.2 Phase & Event Detection for CMJ
1. **Quiet Standing Baseline**:
   Computed over a stable window $[0, t_{quiet}]$ (default 0.5 s, avoiding initial 0.1 s settling transients):
   $$\overline{BW} = \frac{1}{N_{quiet}} \sum_{i=1}^{N_{quiet}} F_z(t_i), \quad s_{BW} = \sqrt{\frac{1}{N_{quiet}-1} \sum_{i=1}^{N_{quiet}} (F_z(t_i) - \overline{BW})^2}$$
   Body mass is derived as $m = \overline{BW} / g$ with standard gravitational acceleration $g = 9.80665 \text{ m/s}^2$.
2. **Movement Onset ($t_{onset}$)**:
   The earliest instant prior to peak velocity where vertical force deviates from $\overline{BW}$ by more than $k$ standard deviations (default $k = 5$):
   $$|F_z(t) - \overline{BW}| > k \cdot s_{BW}$$
3. **Integration Start ($t_{int\_start}$)**:
   Configurable lead time prior to detected onset: $t_{int\_start} = \max(0, t_{onset} - \Delta t_{lead})$, with default $\Delta t_{lead} = 0.030 \text{ s}$ ($30 \text{ ms}$) to capture subtle pre-threshold unweighting.
4. **Unweighting Minimum ($t_{unweight\_min}$)**:
   The global minimum force during the countermovement dip: $F_{min} = \min_{t \in [t_{onset}, t_{TO}]} F_z(t)$.
5. **Braking / Deceleration Phase**:
   Begins when $F_z(t) > \overline{BW}$ after the unweighting dip, continuing while velocity remains negative ($v(t) < 0$).
6. **Propulsion / Concentric Acceleration Phase**:
   Begins at lowest center-of-mass position where velocity transitions from negative to positive ($v(t) \ge 0$), and ends at takeoff.
7. **Takeoff ($t_{takeoff}$)**:
   The first instant where vertical force drops below the takeoff force threshold ($F_{thresh} \le 10 \text{ N}$ or $0.5\% \overline{BW}$), sustained for $\ge 5$ consecutive samples.
8. **Landing Impact ($t_{landing}$)**:
   The instant following flight where vertical force rises back above $F_{thresh}$.

### 4.3 Kinematic & Kinetic Integration
- **Net Acceleration**:
  $$a(t) = \frac{F_z(t) - \overline{BW}}{m}$$
- **Instantaneous Velocity**:
  $$v(t) = \int_{t_{int\_start}}^{t} a(\tau) d\tau \approx \sum_{i=int\_start}^{k} \frac{a(t_{i-1}) + a(t_i)}{2} \cdot \Delta t$$
- **Instantaneous Mechanical Power**:
  $$P(t) = F_z(t) \cdot v(t), \quad P_{rel}(t) = \frac{P(t)}{m}$$
- **Impulse-Momentum Jump Height (Primary)**:
  $$h_{imp} = \frac{v(t_{takeoff})^2}{2g}$$
- **Flight-Time Jump Height (Secondary)**:
  $$t_{flight} = t_{landing} - t_{takeoff}, \quad h_{flight} = \frac{g \cdot t_{flight}^2}{8}$$
- **Modified Reactive Strength Index**:
  $$RSImod = \frac{h_{imp}}{t_{takeoff} - t_{onset}}$$

### 4.4 Rule-Based Jump Classifier
Transparent, deterministic feature tree:
1. **Pre-movement Lead-in Check**: Checks if initial $0.4 \text{ s}$ has low variance ($s_F < 0.05 \overline{BW}$). If initial force $\approx 0 \text{ N}$ followed by rapid impact, classifies as **Drop Jump (DJ)**.
2. **Countermovement Dip**: Evaluates if minimum force before peak is significantly below baseline ($F_{min} < \overline{BW} - 5 s_{BW}$ and $\Delta F > 0.1 \overline{BW}$).
   - If dip present AND flight phase detected ($F_z < 10 \text{ N}$ for $>0.15 \text{ s}$), classifies as **Countermovement Jump (CMJ)**.
   - If dip absent, stable hold observed, and force rises monotonically to takeoff, classifies as **Squat Jump (SJ)**.
3. **Flight Phase Absent**:
   - If force rises substantially above body weight ($F_{max} > 1.5 \overline{BW}$) with sustained plateau and no flight phase, classifies as **Isometric Mid-Thigh Pull (IMTP)**.
4. If feature signatures conflict or confidence $<0.7$, outputs **`unknown`** with explicit explanatory reasoning.

---

## 5. Statistical Framework & Research Methodology

### 5.1 Project 1: Measurement Agreement & Sensitivity
- **Bland-Altman Agreement ($h_{imp}$ vs $h_{flight}$)**:
  $$\text{Difference } d_i = h_{imp, i} - h_{flight, i}, \quad \bar{d} = \frac{1}{n} \sum d_i, \quad s_d = \sqrt{\frac{1}{n-1}\sum (d_i - \bar{d})^2}$$
  $$95\% \text{ Limits of Agreement (LoA)} = \bar{d} \pm 1.96 \cdot s_d$$
  Confidence intervals for $\bar{d}$ and LoA calculated using standard error $SE(\bar{d}) = s_d / \sqrt{n}$ and $SE(LoA) \approx \sqrt{3 s_d^2 / n}$. Explicit acknowledgment: with $n=10$, confidence intervals are wide.
- **Analyst-Choice Sensitivity Grid**:
  Evaluates elasticity of $h_{imp}$, $RSImod$, and peak power across parameter permutations:
  - Integration start lead offset: $[0, 10, 30, 50, 100] \text{ ms}$
  - Baseline duration: $[0.3, 0.5, 0.8] \text{ s}$
  - Onset multiplier: $[3, 5, 8] \cdot s_{BW}$
  - Takeoff threshold: $[5, 10, 20] \text{ N}$
  - Filter cutoff: $[15, 30, 50] \text{ Hz}$ vs unfiltered
  - Flight-phase force offset correction: enabled vs disabled
- **Reliability on Open Data (D4: White et al.)**:
  Using repeated trials per participant:
  $$\text{Typical Error / SEM} = \frac{s_{diff}}{\sqrt{2}}, \quad CV\% = \frac{SEM}{\overline{X}} \times 100, \quad ICC(3,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E}$$
  $$MDC_{95} = SEM \times 1.96 \times \sqrt{2}$$

### 5.2 Project 2: Baselines, Heuristic Readiness & SQL Architecture
- **Rolling Baseline Engine**:
  Computed over a 28-day exponential/rolling window with minimum sample constraint ($N \ge 3$ sessions):
  $$z_{ij} = \text{direction} \times \frac{x_{ij} - \mu_{baseline, i}}{\sigma_{baseline, i}}$$
- **Smallest Worthwhile Change (SWC)**:
  $$SWC_i = 0.2 \times \sigma_{between\_athlete} \quad \text{or} \quad 0.5 \times \sigma_{within\_athlete}$$
- **Readiness Index**:
  Weighted composite of key indicators:
  $$ReadyIndex = w_1 z_{RSImod} + w_2 z_{JumpHeight} + w_3 z_{PeakPowerRel}$$
  Default weights: $w_1 = 0.4, w_2 = 0.3, w_3 = 0.3$. Labeled explicitly as a heuristic index.
- **Multi-Tier Alert Logic**:
  - *Tier 1 (Mild Alert)*: $x_i < \mu_{baseline} - \max(10\%, SWC_i)$
  - *Tier 2 (Compounded Alert)*: Tier 1 alert active AND $\ge 2$ consecutive days of wearable `recovery_pct < 34` (Red Zone).
  - *Tier 3 (Multi-Metric Deficit)*: Simultaneous drops across $\ge 2$ independent neuromuscular parameters across consecutive sessions.
- **SQL Implementation**:
  DuckDB queries utilizing advanced window frames:
  ```sql
  AVG(value) OVER (
      PARTITION BY athlete_id, metric 
      ORDER BY date 
      RANGE BETWEEN INTERVAL 28 DAYS PRECEDING AND CURRENT ROW
  )
  ```
  Verified for exact numerical equivalence ($<10^{-9}$ difference) against Pandas in `test_p2_sql.py`.

### 5.3 Project 3: Real-World Evidence & Mixed-Effects Causal Modeling
- **Analysis Plan Hypotheses**:
  $H_1$: Within an individual, prior-night sleep deficit ($\Delta Sleep < 0$) predicts diminished next-day readiness.
  $H_2$: Within an individual, high acute day strain predicts increased neuromuscular fatigue (lower RSImod).
- **Within-Person vs Between-Person Separation (Person-Mean Centering)**:
  $$\text{Outcome}_{ij} = \beta_0 + \beta_{within}(X_{ij} - \bar{X}_i) + \beta_{between}\bar{X}_i + u_i + \epsilon_{ij}$$
  where $u_i \sim \mathcal{N}(0, \sigma_u^2)$ is the random intercept per participant, capturing individual trait differences, while $\beta_{within}$ isolates the pure acute within-person effect.
- **Causal DAG & Confounder Analysis**:
  Directed Acyclic Graph constructed using DAGitty principles:
  - Variables: Prior Sleep, Acute Strain, Next-Day Recovery, Subjective Fatigue, Chronic Load, Match Day Indicator.
  - Identification of backdoor paths and non-causal observational bias.
- **Bilateral Asymmetry Index**:
  $$ASI = \frac{\text{Higher Limb} - \text{Lower Limb}}{\text{Higher Limb}} \times 100\%$$
  Tracked across sessions with persistent asymmetry flag fired if $ASI > 10\%$ for $\ge 3$ consecutive sessions. No imputation on single-plate data.

---

## 6. Implementation Roadmap & Milestones

### Phase 0: Project Setup & Shared Contracts
- [x] Establish single repository root and initialize git repository.
- [x] Establish `pyproject.toml` with Python 3.11+, pin dependencies, configure `pytest` and `ruff`.
- [x] Install `uv`, create local `.venv`, and install dependencies.
- [x] Implement `shared/contract.py` with schemas, validators, and complete `MetricRegistry`.
- [x] Implement `shared/synth.py` with physically consistent analytical force-time generator and session table simulators.
- [x] Write `tests/test_contract.py` and `tests/test_synth.py` verifying analytical ground truth.
- [x] Document `DATA_SOURCES.md` and ensure `scripts/progress.py` functions cleanly.

### Phase 1: Automatic Jump Analysis Engine (`p1_nmt_pipeline`)
- [x] **M1a**: Config YAML, input validator, 4th-order zero-phase Butterworth filter.
- [x] **M1b**: Synthetic trial generators with analytical ground truth.
- [x] **M2a**: CMJ phase detection (quiet baseline, onset, unweight, brake, prop, takeoff, landing).
- [x] **M2b**: Kinetic & kinematic metrics ($h_{imp}, h_{flight}, RSImod, P_{peak}, v_{peak}, F_{brake}$) + tests.
- [x] **M2c**: Professor data importer adapter (`scripts/import_cmj_xlsx.py`) & data-quality report.
- [x] **M2d**: Golden test replicating reference workbook within stated tolerance.
- [x] **M2e**: Method comparison ($h_{flight}$ vs $h_{imp}$) Bland-Altman analysis with 95% CIs.
- [x] **M2f**: Analyst-choice sensitivity grid on baseline windows, onset $k$, integration lead, cutoff frequencies.
- [x] **M2g**: Open data adapter for White et al. (Zenodo 19136480, CC BY 4.0) & reliability ($ICC, CV\%$).
- [ ] **M3 [Deferred / Future Work]**: Rule-based jump-type classifier (CMJ, SJ, DJ, IMTP, unknown).
- [ ] **M4a [Deferred / Future Work]**: Squat jump metrics & countermovement invalidation flag.
- [ ] **M4b [Deferred / Future Work]**: Drop jump metrics (GCT, RSI, height).
- [x] **M5a**: Single-trial report: VALD-style multi-panel figure.
- [ ] **M5b [Deferred / Future Work]**: Streamlit drop-a-file web application.
- [x] **M6a**: CLI generating contract-compliant `sessions.csv`.
- [x] **M6b**: Comprehensive README with research questions, methods, and AI disclosure.

### Phase 2: Baselines, Readiness & SQL Layer (`p2_nmt_reports`)
- [x] **M1a**: Loaders via shared contract & 28-day rolling personal baselines.
- [x] **M1b**: DuckDB SQL layer with window functions & Pandas equivalence test suite.
- [x] **M2a**: Direction-aware readiness index & documented heuristic weights.
- [x] **M2b**: Daily wearable telemetry join with standardized recovery metric names.
- [ ] **M2c [Deferred / Future Work]**: Multi-tier alert logic (SWC, red zone recovery, multi-metric decline).
- [x] **M2d**: Mixed-effects session change model in `statsmodels`.
- [x] **M3**: Visualization module (metric vs baseline bands, recovery overlays).
- [ ] **M4a [Deferred / Future Work]**: Streamlit longitudinal monitoring dashboard.
- [ ] **M4b [Deferred / Future Work]**: Self-contained HTML report export.
- [x] **M5**: Comprehensive test suite & Project 2 README.

### Phase 3: Real-World Evidence & Asymmetry Profiler (`p3_nmt_profiler`)
- [x] **M1a**: Public wearable data adapters (PMData) to `wearable_daily.csv` with strain proxies.
- [x] **M1b**: Pre-analysis plan document (`ANALYSIS_PLAN.md`).
- [x] **M2**: Mixed-effects regression (within-person vs between-person sleep/load vs readiness).
- [x] **M3**: Causal inference framework (DAG, confounder analysis, sensitivity bounds).
- [x] **M4**: Asymmetry module ($ASI$, direction, persistence flag on synthetic bilateral data).
- [x] **M5a**: Longitudinal strain-readiness publication figures & test suite.
- [x] **M5b**: Two-page executive research brief formatted for performance science research evaluation.

---

## 7. Assumptions & Technical Decisions

- **Assumption 1 (Repo Root)**: The workspace root is the single repository root. All project files sit directly at this root, and git is initialized there.
- **Sampling-Rate-Dependent Test Tolerances**:
  - At 2000 Hz ($\Delta t = 0.5 \text{ ms}$): tolerance $\le 1\text{--}2 \text{ mm}$ ($0.001\text{--}0.002 \text{ m}$) for jump height.
  - At 250 Hz ($\Delta t = 4.0 \text{ ms}$): looser tolerance $\le 5\text{--}10 \text{ mm}$ ($0.005\text{--}0.010 \text{ m}$), accounting for discrete sampling integration error.
- **Analytical Ground Truth Calculation**:
  The synthetic generator computes true jump height $h_{true} = \frac{v_{to}^2}{2g}$ and true event times ($t_{onset}, t_{takeoff}, t_{land}$) directly from the closed-form piecewise equations generating the acceleration profile, completely independent of the trapezoidal numerical integration used by the pipeline analysis code.
- **Integration Lead Offset**:
  Trapezoidal integration starts at $t_{int\_start} = \max(0, t_{onset} - \Delta t_{lead})$ (default $\Delta t_{lead} = 30 \text{ ms}$), with parameter sensitivity evaluated across $[0, 10, 30, 50, 100] \text{ ms}$.
- **Flight-Phase Force-Offset Correction**:
  Optional config flag (`correct_flight_phase_offset: false`, off by default). The synthetic generator can inject non-zero offsets (`flight_force_offset_n`) for testing.
- **Strict Synthetic-Only & Data Privacy Rule**:
  No `.xlsx` files, `data/private/`, or `data/raw/` files will be read, opened, copied, or processed. All development, validation, and testing will use synthetic data and authorized de-identified benchmarks only.
- **Standardized Wearable Schema**:
  Field names mirror publicly documented industry wearable terminology (recovery %, sleep duration, resting HR, HRV, day strain) for seamless integration with commercial wearable data.
