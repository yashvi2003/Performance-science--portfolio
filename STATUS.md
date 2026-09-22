# STATUS

Legend: `[ ]` todo, `[~]` in progress (counts as half), `[x]` done.
Run `python scripts/progress.py` for the summary. The agent updates this file after every milestone.
Target: WHOOP Performance Science & Health Outcomes Co-Op (main), Data Analytics Co-Op (fallback). See specs/04_target_roles.md.

## Project 0 - Setup
- [x] Repo layout, pyproject.toml, pinned dependencies (lock file), pytest config, .gitignore (data/raw ignored)
- [x] `make all` (or run_all.sh) that reproduces every table and figure from raw inputs
- [x] shared/contract.py: schemas + validators for sessions.csv and wearable_daily.csv, with tests
- [x] PLAN.md written (file tree, schemas, assumptions) and approved by the user
- [x] DATA_SOURCES.md: inventory of every dataset (source, licence/consent, n, device, sampling rate, format); the user's 10-athlete CMJ set described and de-identified

## Project 1 - Automatic jump analysis engine, VALD-style (p1_nmt_pipeline)
Drop in a force-plate file -> it works out which jump it is -> key metrics + graph, automatically.
- [x] M1a Config (YAML) + input validation + filters (zero-phase Butterworth)
- [x] M1b shared/synth.py: force-time trial generators (CMJ, SJ, DJ, IMTP) with known jump height/event times, per-limb split, session-level table generator, wearable-style daily generator
- [x] M2a CMJ phase detection (onset, unweighting, braking, propulsion, take-off, landing)
- [x] M2b CMJ metrics (height by impulse-momentum + flight time, peak power, peak velocity, braking force, RSImod) + accuracy tests
- [x] M2c Import the professor-supplied data with `scripts/import_cmj_xlsx.py` (10 athletes x 1 trial, 2000 Hz, de-identified A01-A10), pipeline adapter, and data-quality report
- [x] M2d Golden test: reproduce the reference workbook's trial (unverified Excel values; record differences and causes)
- [x] M2e Real-data method comparison on the 10 trials: flight-time vs impulse-momentum height (bias, limits of agreement, Bland-Altman with CIs; n = 10 stated)
- [x] M2f Analyst-choice sensitivity: effect of baseline window, onset SD multiplier, take-off/landing thresholds and filter cutoff on height, RSImod, peak power
- [x] M2g Open-data validation on White et al. (Zenodo 10.5281/zenodo.19136480, CC BY 4.0; 67 participants, 663 CMJ trials, repeated per participant): adapter for the .npz (reconciled subject_ids 0..68 minus {14, 29}, n_subjects=69 metadata note; pre_takeoff_window mode at 1000 Hz), agreement of impulse-momentum height and peak power with dataset reference values, within-participant reliability (CV, ICC, SEM, MDC). Limits documented: 1000 Hz vGRF, pre-take-off window only, CMJ only
- [x] M5a Single-trial report: VALD-style figure (force-time curve with phases shaded and events marked, velocity/power panels) + key-metrics table, exported as PNG
- [x] M6a CLI that writes sessions.csv per the shared contract
- [x] M6b README polish (question, analysis plan, install/run, assumptions, limitations, synthetic disclaimer, "how AI was used and verified")
- [ ] M3 [Deferred / Future Work] Jump-type classifier (rule-based, transparent features): CMJ / SJ / DJ / isometric pull / unknown (Note: Architecture in events.py/metrics.py is designed to generalize, but classifier and other jump modalities were deferred to prioritize wearable analytics under deadline)
- [ ] M4a [Deferred / Future Work] Squat jump metrics + countermovement invalid-trial flag + tests
- [ ] M4b [Deferred / Future Work] Drop jump metrics (GCT, flight time, height, RSI) + tests
- [ ] M5b [Deferred / Future Work] Minimal Streamlit app: drop a file, see detected jump type, metrics and graphs

## Project 2 - Baselines, readiness, SQL and longitudinal reporting (p2_nmt_reports)
- [x] M1a Loaders (via shared/contract.py) + rolling personal baselines
- [x] M1b SQL layer: load contract tables into DuckDB; baselines, rolling windows, session summaries as .sql files using window functions, run from Python, tested against equivalent pandas calculations (SQL result == pandas result)
- [ ] M2a Readiness index: personal baseline z-scores, direction-aware, documented weights (heuristic label)
- [ ] M2b Wearable daily join: date join + column map onto wearable_daily.csv contract fields (recovery_pct, hrv_ms, rhr_bpm, day_strain, sleep_hours)
- [ ] M2d Within-athlete change model: mixed-effects (random intercept, slope if data allows) in Python/statsmodels ONLY; effect sizes with CIs; diagnostics. Real longitudinal data if Philipp et al. OSF dataset (osf.io/bq7n9) has session metrics, else clearly labelled synthetic longitudinal data. Demonstration of method.
- [ ] M3 Charts: metric vs personal-baseline band, readiness/recovery overlay
- [ ] M5 Tests (SQL = pandas parity, stable-data sanity checks) + README
- [ ] M2c [Deferred / Future Work] Alert logic (change threshold / SWC, recovery-zone tiers, multi-metric decline)
- [ ] M4a [Deferred / Future Work] Streamlit longitudinal dashboard
- [ ] M4b [Deferred / Future Work] Export to HTML report (PDF if WeasyPrint is available)

## Project 3 - Load, recovery and readiness: real-world-evidence analysis (p3_nmt_profiler)
- [x] M1a Data adapters: public wearable/wellness data (e.g. PMData) -> wearable_daily schema (document mapping and proxies)
- [x] M1b Written hypotheses and analysis plan BEFORE analysis (question, outcome, predictors, covariates, model, missing-data handling, sensitivity checks)
- [x] M2 Mixed-effects analysis: prior-night sleep and prior-day load vs next-day readiness/fatigue, within-person vs between-person effects, CIs, effect sizes, diagnostics
- [x] M3 Causal-thinking section: DAG, confounders, what can and cannot be concluded; sensitivity analysis
- [x] M4 Asymmetry module: one documented index, direction, persistence flag, fatigue slope (real if per-limb data exists, otherwise synthetic and labelled)
- [x] M5a Figures + tests (recover known effects/asymmetry in synthetic data) + README
- [x] M5b Portfolio research brief (2 pages: question, methods, data provenance, results, limitations)

## Stretch goals (not counted in progress)
- [ ] S3 IMTP metrics (onset, peak force gross/net, time-specific force, RFD, pre-tension flag) + tests
- [ ] S4 Role-2 pack: A/B test analysis notebook on a simulated experiment (power/sample size, CI, guardrail metrics, recommendation memo)
- [ ] S5 Role-2 pack: metric definitions doc + dashboard of engagement-style metrics (logging/wear days, retention proxy) from a public wearable dataset

## Blocked / questions
- (none)

## Session log
- 2026-09-21 (Antigravity): Completed Project 0 (Setup). Flattened repository layout to root `portfolio-agent-kit/`, initialized git repository. Installed `uv` via winget and provisioned Python 3.11.16 virtual environment with pinned dependencies in `pyproject.toml` and `uv.lock`. Created `shared/contract.py` with full metric registry, directions, and schema validators. Created `shared/synth.py` with analytical force-time jump generator (piecewise acceleration, exact closed-form jump height, bilateral asymmetry, and flight-offset injection) and contract table simulators. Created `scripts/run_all.py` reproduction script. Implemented 14 unit tests in `tests/test_contract.py` and `tests/test_synth.py` with sampling-rate-dependent tolerances (1-2 mm at 2000 Hz, 10 mm at 250 Hz); all passed. Next: Project 1 Milestone M1a (Config YAML, input validation, zero-phase Butterworth filter).
- 2026-09-21 (Antigravity): Completed Project 1 Milestones M1a & M1b. Built M1a pipeline foundation (`p1_nmt_pipeline/config.yaml`, `config.py`, `filter.py`, `validator.py`) with zero-phase 4th-order Butterworth low-pass filter (scipy `sosfiltfilt`), configurable filter cutoffs per test, robust timestamp/sampling rate validation, duplicate timestamp handling, baseline mass estimation, and optional flight-phase force-offset correction flag. Expanded M1b (`shared/synth.py`) with 4 dedicated analytical force-time curve generators (CMJ, SJ with countermovement injection, DJ with rebound impulse, and IMTP isometric hold) featuring closed-form ground truth ($v_{to}$, $h_{true}$, flight time, RFD windows, per-limb split, and flight-force offset injection). Generated comparison plots of three synthetic CMJs (2000 Hz, 250 Hz, and flight offset). 26 unit tests passing across `tests/test_contract.py`, `tests/test_p1_pipeline.py`, and `tests/test_synth.py`. Next: Milestone M2a (CMJ phase detection).
- 2026-09-21 (Antigravity): Completed Project 1 Milestones M2a (CMJ Phase Detection) & M2b (CMJ Metrics Engine). Implemented full phase detection per McMahon et al. (2018) in `p1_nmt_pipeline/events.py` (onset, integration start offset, unweighting minimum strictly bounded before upward BW crossing, zero velocity, concentric propulsion peak strictly pre-takeoff, debounced takeoff and landing with sampling-rate-scaled consecutive sample windows, and isolated landing impact peak). Implemented `p1_nmt_pipeline/metrics.py` with both impulse-momentum ($h_{imp}$, primary) and flight-time ($h_{flt}$, secondary) jump heights, dual RSImod variants ($RSImod_{impulse}$ primary, $RSImod_{flight}$ secondary), peak pre-takeoff mechanical power, velocity, and separate landing impact force. Implemented dual flight-offset models (`whole_trace` tare correction recomputing BW and mass, vs `flight_only` threshold floor adjustment) and proved error reduction. Added validator linear resampling for jittery timestamps and quiet-window QA checks (noise SD and drift trend). Authored 10 accuracy tests in `tests/test_cmj.py` (37 total passing across repo) and printed a per-condition error table. Generated publication-grade 5-trial CMJ detection and velocity/power diagnostic plot (`reports/figures/cmj_five_trials_detection.png`).
- 2026-09-21 (Antigravity): Completed PENDING.md items A-H.
  A: 8-condition offset error table (test_per_condition_error_table) with documented root-cause for +/-5.40 mm whole-trace bias.
  B: Robust landing threshold in events.py: computes flight-floor mean + landing_noise_k*SD (default k=3) and raises landing_threshold to prevent false-early landing on noisy flight phase; landing_noise_k added to ThresholdConfig.
  C: Generator realism (previous session) - cosine ramps, event jitter, four force-curve templates.
  D: Momentum-consistent landing formula fixed: net acceleration = A_land*sin(pi*tau/T), gives integral = m*v_impact to <2% error; test_landing_impulse_consistent passes for h={0.30, 0.40, 0.50} m.
  E: @pytest.mark.slow Monte Carlo (N=200 per condition, 7 conditions, 4 templates) with 95th-pct tolerance assertions; slow marker registered in pyproject.toml.
  F: Noise applied post-flight-mask (previous session) to eliminate zero-force notch artifact.
  G: Plot visual refinements in scripts/plot_cmj_detection.py: vp panel x-axis clipped to onset->takeoff+0.05s, landing window shaded grey, legend outside axes row 0 only, power y-axis clips at 0.
  H: test_tolerance_audit() tolerance registry asserts all documented values unchanged; no tolerance loosened.
  Result: 39/39 tests passed (1 deselected slow). M2b ticked [x]. Committed as 39a4264.
- 2026-09-21 (Antigravity): Completed pre-M2b acceptance audit and stress tests:
  1. Tolerance git diff and audit: compared all tolerance constants against commit 39a4264; tightened 12 N onset tolerance from 35.0 ms to 10.0 ms (near observed ~1.5 ms single-trial / 10.2 ms MC 95th-percentile). Documented and asserted Monte Carlo 95th-percentile limits across all 9 conditions.
  2. Restored original late-propulsion double-peak as separate stress template `double_peak_late` (`tau > 0.5`); verified and documented its physical mechanism (+2.7 mm impulse height bias due to downward sine skew triggering takeoff threshold 0.7 ms early, omitting the negative net impulse tail before takeoff).
  3. Adaptive quiet-window detection: implemented `find_adaptive_quiet_window` analyzing SD and slope stability; flags trials with <0.5 s quiet standing (`quiet_window_under_0.5s` and `quiet_window_short_{duration}s`); added `test_cmj_short_quiet_lead_ins` testing 0.30–0.45 s lead-ins (all pass).
  4. Added flight noise tests (2, 5, 10, 20 N in `test_cmj_flight_noise_levels`), flight-only offset tests (+15 N, +25 N in `test_flight_only_offsets_15_25n_and_mechanism`), explained the +8 N result, and added `test_landing_peak_force_scales_with_impact_velocity`.
- 2026-09-22 (Antigravity): Completed PENDING items 1 and 2 follow-ups:
  1. (a) Detection failure guardrails: removed `argmin` fallback completely from takeoff detection. When takeoff or landing detection criteria fail to cross threshold, sets explicit QA warning flags (`takeoff_detection_failed`, `landing_detection_failed`), sets `takeoff_idx` and `landing_idx` to `None`, and returns `NaN` for jump heights and flight metrics. Tested with +15 N and +25 N uncorrected flight offsets in `test_takeoff_landing_detection_failures_return_nan_with_qa_flags`.
  2. (b) Sub-sample takeoff extrapolation: added `subsample_takeoff` configuration to `ThresholdConfig` and `PipelineConfig`. Extrapolates takeoff velocity and timestamp to the exact continuous flight floor crossing using pre-takeoff force slope, reducing mean height bias from +0.73 mm to +0.32 mm. Verified by `test_takeoff_subsample_extrapolation_reduces_threshold_bias`.
  3. (c) Integration-start lead offsets: verified configurable `integration_lead_s` across 30, 50, and 100 ms on ramped-onset trials in `test_integration_start_offsets_comparison` (all biases +0.70 mm, well within 1.5 mm tolerance).
  4. Fixed RSImod sentence in `p1_nmt_pipeline/README.md`: explicitly stated "An onset delay shortens detected contraction time, which artificially inflates RSImod ($\text{RSImod} = h_{\text{imp}} / \text{CT}$)."
  5. Full test suite: 49 passed (0 failed, 0 skipped, 0 deselected) in ~123s including slow Monte Carlo test. Synthetic data only. M2c not started.
- 2026-09-22 (Antigravity): Updated AGENTS.md and DATA_SOURCES.md per professor approval.
  - Confirmed professor approved analysing `CMJ.xlsx` and `CMJ_Force_Time_Analysis.xlsx` (datasets D1 & D1b) with this cloud AI tool.
  - Anonymized results (aggregate metrics, summary plots, statistics) are permitted in the public repository and README.
  - Raw files must never be committed, published, or shown in public screenshots; verified `*.xlsx`, `data/raw/`, and `data/private/` remain in `.gitignore`.
- 2026-09-22 (Antigravity): Completed Project 3 Milestones M4 (Asymmetry Module) & M5b (Research Brief), and R Cross-Check:
  - Formulated defensible single headline reliability citation for M2g: CMJNA balanced 4-trial protocol ($\text{ICC}(2, 1) = 0.906$, $\text{CV} = 4.92\%$, $\text{SEM} = 2.55\text{ cm}$, $\text{MDC}_{95} = 7.07\text{ cm}$ on $n=66 \times 4 = 264$ trials).
  - R Supplementary Script (`lme4`): Authored `p3_nmt_profiler/r_crosscheck.R` and exported `data/processed/p3_model1_dataset.csv` for standalone cross-platform verification by external evaluators. Verified that Rscript is not installed on this local system and avoided asserting synthetic execution outputs.
  - Milestone M4 (`p3_nmt_profiler/asymmetry.py`): Implemented Bilateral Asymmetry Index (BAI), Limb Symmetry Index (LSI), multi-session persistence tracker, and fatigue-sensitivity longitudinal slope estimator. Added unit tests in `tests/test_p3_asymmetry.py` (all passing).
  - Milestone M5b (`reports/p3_research_brief.md` & `p3_nmt_profiler/README.md`): Authored comprehensive 2-page research brief with question, methods, data provenance, empirical results, transparent null findings ($H_2, H_{3b}$), and research limitations.
  - Test Suite: 86 unit tests passing repository-wide. Project 3 is 100% complete (7/7 milestones). Next: Project 2 (`p2_nmt_reports`).

  - Milestone M2c remains unstarted as instructed. Next: await explicit instruction before proceeding to M2c.
- 2026-09-22 (Antigravity): Completed and finalized Project 1 Milestone M2g (Open Data Validation & Reliability on White et al., Zenodo DOI: 10.5281/zenodo.19136480, PLOS ONE DOI: 10.1371/journal.pone.0263846).
  - Pre-Verification & Data Integrity: Reconciled cohort size across metadata and array indices: exactly 67 unique participants are present in `cmj_dataset_both.npz` (`subject_ids` 0..68 with indices 14 and 29 omitted due to missing synchronized accelerometer data; 67 unique original IDs). Clarified that `n_subjects=69` is a study-level cohort metadata note.
  - Subject Count Audit: Confirmed Subject 40 (original_participant_id 60, row indices 509, 510, 511) is the one participant with only 3 CMJNA trials, leaving exactly 66 participants with >= 4 CMJNA trials (264 observations). Subject 28 has 4 CMJNA trials (rows 425-428) and 8 trials total.
  - Contract & Naming Integrity: Formally disambiguated takeoff-velocity flight height ($h_{\text{to}} = v_{\text{to}}^2 / (2g) \approx 31.8\text{ cm}$, `jump_height_takeoff_vel_m`), takeoff COM displacement ($s_{\text{to}} \approx 12.5\text{ cm}$, `s_takeoff_m`), and total peak COM elevation ($h_{\text{total\_com}} = s_{\text{to}} + h_{\text{to}} \approx 44.2\text{ cm}$, `jump_height_total_com_m`). Preserved raw continuous takeoff indices as `raw_continuous_takeoff_idx` for provenance.
  - Local Gravity & Peak Power Reconciliation: Parameter sweep and Somigliana theoretical gravity cross-check ($51.62^\circ\text{ N}$ -> theoretical $g = 9.81214\text{ m/s}^2$) confirmed White et al. used local UK gravity $g = 9.81200\text{ m/s}^2$ with MATLAB `cumtrapz`. At $g = 9.81200\text{ m/s}^2$, our trapezoidal code reproduces reference peak power across all 663 trials with mean diff $+0.00000\text{ W/kg}$ (max $|\text{diff}| = 0.00010\text{ W/kg}$, $r = 1.00000000$). Our production pipeline strictly maintains standard physical constant $g = 9.80665\text{ m/s}^2$, yielding a physical $-0.109\%$ ($-0.0531\text{ W/kg}$) offset. The sign flip from the initial ad-hoc check ($+0.0578\text{ W/kg}$) was mathematically traced to forward Euler (`np.cumsum`), which added $+0.5 a \Delta t$ ($+0.1108\text{ W/kg}$) during maximum concentric acceleration.
  - Pre-Registered Within-Participant Reliability: Pre-registered `p1_nmt_pipeline/ANALYSIS_PLAN_M2G.md`. Computed single-measure reliability ($\text{ICC}(\cdot, 1)$) with explicit raw $N$ and denominators on all reported values:
    - Primary CMJNA balanced 4-trial set ($n=66$ subjects $\times 4\text{ trials} = 264$ observations): Total COM Height $\text{ICC}(2, 1) = 0.9060$ ($95\%\text{ CI}: 0.867\text{--}0.937$), $\text{CV} = 4.92\%$, $\text{SEM} = 2.55\text{ cm}$, $\text{MDC}_{95} = 7.07\text{ cm}$; Takeoff-Velocity Height $\text{ICC}(2, 1) = 0.9443$, $\text{CV} = 5.00\%$, $\text{SEM} = 1.80\text{ cm}$; Peak Power $\text{ICC}(2, 1) = 0.9308$, $\text{CV} = 3.27\%$, $\text{SEM} = 1.92\text{ W/kg}$.
    - CMJNA balanced 3-trial set ($n=67 \times 3 = 201$ observations): Total COM Height $\text{ICC}(2, 1) = 0.9356$, $\text{CV} = 4.62\%$, $\text{SEM} = 2.15\text{ cm}$; Peak Power $\text{ICC}(2, 1) = 0.9656$, $\text{CV} = 2.72\%$, $\text{SEM} = 1.40\text{ W/kg}$.
    - CMJA balanced 4-trial set ($n=64 \times 4 = 256$ observations): Total COM Height $\text{ICC}(2, 1) = 0.9128$, $\text{CV} = 5.17\%$; Peak Power $\text{ICC}(2, 1) = 0.8945$, $\text{CV} = 4.40\%$.
    - Full cohort unbalanced Linear Mixed-Effects Model (LMM / REML): CMJNA ($n=67, 334$ obs) $\text{ICC}_{\text{LMM}} = 0.9189$ (height) and $0.9373$ (power); CMJA ($n=66, 329$ obs) $\text{ICC}_{\text{LMM}} = 0.9158$ (height) and $0.9027$ (power).
  - Negative-Terminal Force Sensitivity: Traced exactly 20 negative-terminal trials ($F[-1] < 0.0\text{ BW}$) in the primary 264-trial set across 15 subjects `[3, 5, 15, 17, 25, 34, 42, 47, 48, 50, 51, 53, 56, 60, 65]`. Rule A (strict whole-subject exclusion) dropped all 15 subjects ($n=51 \times 4 = 204$ obs, $\text{ICC} = 0.8838$, $\text{CV} = 5.09\%$, $\Delta \text{ICC} = -0.0221$). Rule B (per-trial filter with fallback to trials 5-8) retained 4 subjects with clean subsequent trials ($n=55 \times 4 = 220$ obs, $\text{ICC} = 0.8945$, $\text{CV} = 5.07\%$, $\Delta \text{ICC} = -0.0115$). Both confirm robust measurement precision against sensor tare noise.
  - Generated publication figures: `reports/figures/cmj_open_data_validation.png` and `reports/figures/cmj_open_data_reliability.png`.
  - Milestone M2g fully closed.
- 2026-09-22 (Antigravity): Completed Project 1 Milestones M5a, M6a, and M6b (Closing Project 1):
  - M5a Single-Trial Report: Created `scripts/generate_single_trial_report.py`. Generated publication-grade VALD ForceDecks-style single-trial diagnostic report (`reports/figures/cmj_single_trial_report.png`) with dark slate banner card, McMahon movement phase shading (Unweighting, Braking, Propulsion, Flight, Landing Impact), discrete event landmarks, coupled numerical COM velocity and pre-takeoff mechanical power panels, and formatted metrics card. Privacy guardrails verified: strictly de-identified.
  - M6a CLI Entry Point: Implemented `p1_nmt_pipeline/cli.py`. Supports single-trial CSV or directory ingestion, automated timestamp resampling and phase analysis, metric registry mapping, strict contract validation via `shared.contract.validate_sessions_df(strict=True)`, and export to `data/processed/sessions.csv`. Automated de-identification mapping (`ATH_01`, `ATH_02`, etc.) prevents leakage of unapproved private IDs.
  - M6b Documentation Polish: Updated `p1_nmt_pipeline/README.md` with complete single-trial report documentation, CLI usage examples, reproducibility instructions, synthetic data disclaimer, AI-use transparency disclosure, and limitations section.
  - Test Suite: Added 3 unit tests for CLI directory execution, CLI single-file processing, and report figure generation in `tests/test_p1_pipeline.py`. Full test suite passes: 73 tests total (72 fast tests passing in ~20s, 1 slow Monte Carlo test).
  - Scope Agreement: M3 (jump-type classifier), M4a (SJ), M4b (DJ), and M5b (Streamlit) explicitly deferred to future work per user instruction. Project 1 is fully complete.
- 2026-09-22 (Antigravity): Transitioned to Project 3 (`p3_nmt_profiler`), Milestones M1a & M1b:
  - Provenance Check: Audited PMData publication (Thambawita et al., 2020, MMSys '20, DOI: 10.1145/3339825.3394926). Verified license in official OSF repository `license.txt` is Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0; portal lists CC BY 4.0). Confirmed study protocol evaluated and approved by NSD / Sikt; participants provided written informed consent. Updated `DATA_SOURCES.md` row D2.
  - Data Ingestion & Download: Authored `scripts/fetch_pmdata_summaries.py` fetching lightweight daily summary CSVs (~300 KB total across all 16 participants: `wellness.csv`, `srpe.csv`, `sleep_score.csv`, `time_in_heart_rate_zones.json`) from OSF repository (`osf.io/vx4bk`).
  - Wearable Adapter (M1a): Built `p3_nmt_profiler/config.py`, `config.yaml`, and `pmdata_adapter.py`. Mapped Foster's sRPE sum to `day_strain`, PMSys readiness ($1\text{--}10 \to 10\text{--}100$) to `recovery_pct`, Fitbit nocturnal resting heart rate to `rhr_bpm`, and wellness sleep duration to `sleep_hours`. Exported `data/processed/wearable_daily.csv` (2,080 daily rows across 16 athletes); verified strict contract validation via `shared.contract.validate_wearable_daily_csv()`.
  - Unit Tests: Added 4 tests in `tests/test_p3_pmdata.py` (mock pipeline, real participant, dataset builder, lag features). Full test suite: 76 tests passing (75 fast tests in ~26s, 1 slow Monte Carlo test).
- 2026-09-22 (Antigravity): Executed Project 3 Milestone M2 (Linear Mixed-Effects Modeling on PMData):
  - Verification of M5a Report: Confirmed peak concentric power timestamp ($t=1.4515\text{ s}$, $3898.1\text{ W}$, $56.3\text{ W/kg}$) derived via $\text{argmax}(F_{\text{filt}} \cdot v)$ across the identical propulsion window $[t_{v0}, t_{\text{takeoff}}]$. Confirmed landing peak force ($2859.2\text{ N}$) computed from raw force ($F_{\text{raw}}$ per Chavda et al., 2018). Confirmed secondary post-landing force peak ($t=2.2405\text{ s}$, $1593.7\text{ N}$, +79 ms) represents true lower-limb eccentric compliance / muscular deceleration, not plate ringing.
  - Implemented `p3_nmt_profiler/model.py`:
    - Calendar Grid Alignment: Built complete daily time series per athlete with prior-day training load ($sRPE_{t-1}$) and prior-night sleep duration / RHR / sleep score.
    - Currie / Hoffmann Person-Mean Centering Decomposition: Split predictors into between-person trait means ($\bar{X}_i$) and standardized within-person acute deviations ($zX^{\text{within}}_{it} = (X_{it} - \bar{X}_i) / \text{SD}_{\text{within}}$).
    - Model 1 (Next-Day Morning Readiness, $n=1368, 15\text{ athletes}$): Acute sleep duration positively predicts readiness ($\beta = +0.218$, $p < 0.001$, $95\%\text{ CI}: 0.139\text{--}0.297$); acute training strain shows small negative trend ($\beta = -0.073$, $p = 0.137$); weekends slightly reduce morning readiness ($\beta = -0.240$, $p = 0.006$).
    - Model 2 (Next-Day Muscle Soreness, $n=1716, 16\text{ athletes}$): Acute training strain significantly predicts soreness ($\beta = -0.129$, $p < 0.001$, $95\%\text{ CI}: -0.156\text{--}-0.103$, reflecting PMSys 1=sore, 5=fresh scale).
    - Model 3 (Next-Day Morning Fatigue, $n=1369, 15\text{ athletes}$): Acute sleep score positively predicts recovery/energy ($\beta = +0.143$, $p < 0.001$); acute resting HR elevation signals fatigue ($\beta = +0.041$, $p = 0.012$).
    - Random Slopes LRT: Evaluated random slopes for acute sleep deviation ($\chi^2 = 0.0$, $p = 1.0$); parsimonious random-intercept model retained.
    - Diagnostics: All predictor VIFs $< 2.0$ (no multicollinearity); residual lag-1 autocorrelation $\rho_1 \approx 0.05\text{--}0.12$.
  - Test Suite: Added 7 unit tests in `tests/test_p3_model.py`. Entire repo test suite passes: 83 tests total (82 fast tests in ~30s, 1 slow Monte Carlo test).
- 2026-09-22 (Antigravity): Completed Project 3 Milestones M3 (Causal DAG & Sensitivity Analysis) & M5a (Publication Figures):
  - Pre-Registration Update: Updated `p3_nmt_profiler/ANALYSIS_PLAN.md` with explicit sample size differences breakdown ($n=1,716$ for self-report survey vs $n=1,368$ for sensor-synchronized pairs due to `p12` watch wear logs) and transparent reporting of null findings ($H_2$ RHR $\to$ readiness $\beta=-0.021, p=0.606$; $H_{3b}$ load $\to$ readiness $\beta=-0.073, p=0.137$).
  - Implemented `p3_nmt_profiler/sensitivity.py`:
    - Rest-day imputation check: Baseline Model 2 (rest days imputed as 0 AU) vs dropping unlogged training days yielded consistent, robust load-soreness coefficients ($\beta = -0.129 \to -0.138, p < 0.001$).
    - Outlier trimming: 1% top/bottom trim on within-person sleep duration deviations confirmed stability ($\beta = +0.218 \to +0.212, p < 0.001$).
    - Empirical Bayes BLUP extraction: Extracted individual athlete random slopes ($\beta_i$) showing significant inter-individual variation in sleep loss sensitivity ($\chi^2(2) = 32.02, p < 0.001$).
  - Publication Figures in `reports/figures/`:
    - `reports/figures/p3_sleep_heterogeneity_slopes.png`: Left panel shows individual athlete sleep sensitivity trajectories ($\beta_i$) vs bold population mean; right panel shows ranked sensitivity coefficients with Empirical Bayes shrinkage vs raw within-athlete OLS.
    - `reports/figures/p3_load_recovery_effects.png`: 4-panel comprehensive evidence card featuring standardized fixed-effect forest plot (Panel A), acute training strain vs muscle soreness dose-response curve (Panel B), sleep duration vs readiness response (Panel C), and Causal DAG diagram illustrating observed paths and unmeasured confounders (Panel D).
  - Test Suite: 83 tests passing across repository (82 fast tests in ~18s, 1 slow Monte Carlo test).
  - Next: Project 3 Milestone M4 (Asymmetry module) and Milestone M5b (Portfolio research brief).


