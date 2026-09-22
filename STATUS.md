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
- [ ] M2c Import the professor-supplied data with `scripts/import_cmj_xlsx.py` (10 athletes x 1 trial, 2000 Hz, de-identified A01-A10), pipeline adapter, and data-quality report
- [ ] M2d Golden test: reproduce the reference workbook's trial (unverified Excel values; record differences and causes)
- [ ] M2e Real-data method comparison on the 10 trials: flight-time vs impulse-momentum height (bias, limits of agreement, Bland-Altman with CIs; n = 10 stated)
- [ ] M2f Analyst-choice sensitivity: effect of baseline window, onset SD multiplier, take-off/landing thresholds and filter cutoff on height, RSImod, peak power
- [ ] M2g Open-data validation on White et al. (Zenodo 10.5281/zenodo.19136480, CC BY 4.0; 67 participants, 663 CMJ trials, repeated per participant): adapter for the .npz (inspect keys first; body-weight-normalised vGRF, 250 Hz, 2 s pre-take-off window, no flight phase), agreement of impulse-momentum height and peak power with the dataset's reference values, within-participant reliability (CV, ICC, SEM). State the limits: 250 Hz, pre-take-off window only, CMJ only
- [ ] M3 Jump-type classifier (rule-based, transparent features): CMJ / SJ / DJ / isometric pull / unknown, with a reason string and confidence; all 10 real trials must come out CMJ; other classes validated on synthetic data only and labelled so
- [ ] M4a Squat jump metrics + countermovement invalid-trial flag + tests
- [ ] M4b Drop jump metrics (GCT, flight time, height, RSI) + tests
- [ ] M5a Single-trial report: VALD-style figure (force-time curve with phases shaded and events marked, velocity/power panels) + key-metrics table, exported as PNG and HTML/PDF
- [ ] M5b Minimal Streamlit app: drop a file, see detected jump type, metrics and graphs
- [ ] M6a CLI that writes sessions.csv per the shared contract
- [ ] M6b README (question, analysis plan, install/run, assumptions, limitations, synthetic disclaimer, "how AI was used and verified")

## Project 2 - Baselines, readiness, SQL and longitudinal reporting (p2_nmt_reports)
- [ ] M1a Loaders (via shared contract) + rolling personal baselines
- [ ] M1b SQL layer: load contract tables into DuckDB/SQLite; baselines, rolling windows and session summaries as .sql files (window functions), run from Python and tested against the pandas results
- [ ] M2a Readiness index (personal baseline z-scores, direction-aware, documented weights, heuristic label)
- [ ] M2b Wearable daily join (date join, column map; WHOOP-compatible field names, no WHOOP data used)
- [ ] M2c Alert logic (change threshold / SWC, recovery-zone tiers, multi-metric decline)
- [ ] M2d Within-athlete change model: mixed-effects (random intercept, slope if data allows) in R (lme4) or statsmodels; effect sizes with CIs; diagnostics. The professor's CMJ data has ONE trial per athlete, so this runs on real longitudinal data only if the Philipp et al. OSF dataset (osf.io/bq7n9, 12 basketball players, 219 weekly screenings) turns out to contain usable session-level metrics; otherwise on labelled synthetic data. Either way it is a demonstration of the method, not a new finding
- [ ] M3 Charts (metric vs baseline band, recovery overlay)
- [ ] M4a Streamlit longitudinal dashboard
- [ ] M4b Export to HTML report (PDF if WeasyPrint is available)
- [ ] M5 Tests (alert tiers fire on the right days, no alerts on stable data, SQL = pandas) + README

## Project 3 - Load, recovery and readiness: real-world-evidence analysis (p3_nmt_profiler)
- [ ] M1a Data adapters: public wearable/wellness data (e.g. PMData) -> wearable_daily schema (document mapping and proxies)
- [ ] M1b Written hypotheses and analysis plan BEFORE analysis (question, outcome, predictors, covariates, model, missing-data handling, sensitivity checks)
- [ ] M2 Mixed-effects analysis: prior-night sleep and prior-day load vs next-day readiness/fatigue, within-person vs between-person effects, CIs, effect sizes, diagnostics
- [ ] M3 Causal-thinking section: DAG, confounders, what can and cannot be concluded; sensitivity analysis
- [ ] M4 Asymmetry module: one documented index, direction, persistence flag, fatigue slope (real if per-limb data exists, otherwise synthetic and labelled)
- [ ] M5a Figures + tests (recover known effects/asymmetry in synthetic data) + README
- [ ] M5b Portfolio research brief (2 pages: question, methods, data provenance, results, limitations)

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
  - Anonymization constraint: `A01-A10` is NOT a confirmed anonymization scheme; neither these IDs nor re-identifying details may appear in public outputs.
  - Milestone M2c remains unstarted as instructed. Next: await explicit instruction before proceeding to M2c.
