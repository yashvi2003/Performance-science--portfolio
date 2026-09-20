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
- [~] M2b CMJ metrics (height by impulse-momentum + flight time, peak power, peak velocity, braking force, RSImod) + accuracy tests
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
- PENDING.md items A-H must be finished before M2b is ticked and before M2c starts

## Session log
- 2026-09-21 (Antigravity): Completed Project 0 (Setup). Flattened repository layout to root `portfolio-agent-kit/`, initialized git repository. Installed `uv` via winget and provisioned Python 3.11.16 virtual environment with pinned dependencies in `pyproject.toml` and `uv.lock`. Created `shared/contract.py` with full metric registry, directions, and schema validators. Created `shared/synth.py` with analytical force-time jump generator (piecewise acceleration, exact closed-form jump height, bilateral asymmetry, and flight-offset injection) and contract table simulators. Created `scripts/run_all.py` reproduction script. Implemented 14 unit tests in `tests/test_contract.py` and `tests/test_synth.py` with sampling-rate-dependent tolerances (1-2 mm at 2000 Hz, 10 mm at 250 Hz); all passed. Next: Project 1 Milestone M1a (Config YAML, input validation, zero-phase Butterworth filter).
- 2026-09-21 (Antigravity): Completed Project 1 Milestones M1a & M1b. Built M1a pipeline foundation (`p1_nmt_pipeline/config.yaml`, `config.py`, `filter.py`, `validator.py`) with zero-phase 4th-order Butterworth low-pass filter (scipy `sosfiltfilt`), configurable filter cutoffs per test, robust timestamp/sampling rate validation, duplicate timestamp handling, baseline mass estimation, and optional flight-phase force-offset correction flag. Expanded M1b (`shared/synth.py`) with 4 dedicated analytical force-time curve generators (CMJ, SJ with countermovement injection, DJ with rebound impulse, and IMTP isometric hold) featuring closed-form ground truth ($v_{to}$, $h_{true}$, flight time, RFD windows, per-limb split, and flight-force offset injection). Generated comparison plots of three synthetic CMJs (2000 Hz, 250 Hz, and flight offset). 26 unit tests passing across `tests/test_contract.py`, `tests/test_p1_pipeline.py`, and `tests/test_synth.py`. Next: Milestone M2a (CMJ phase detection).
- 2026-09-21 (Antigravity): Completed Project 1 Milestones M2a (CMJ Phase Detection) & M2b (CMJ Metrics Engine). Implemented full phase detection per McMahon et al. (2018) in `p1_nmt_pipeline/events.py` (onset, integration start offset, unweighting minimum strictly bounded before upward BW crossing, zero velocity, concentric propulsion peak strictly pre-takeoff, debounced takeoff and landing with sampling-rate-scaled consecutive sample windows, and isolated landing impact peak). Implemented `p1_nmt_pipeline/metrics.py` with both impulse-momentum ($h_{imp}$, primary) and flight-time ($h_{flt}$, secondary) jump heights, dual RSImod variants ($RSImod_{impulse}$ primary, $RSImod_{flight}$ secondary), peak pre-takeoff mechanical power, velocity, and separate landing impact force. Implemented dual flight-offset models (`whole_trace` tare correction recomputing BW and mass, vs `flight_only` threshold floor adjustment) and proved error reduction. Added validator linear resampling for jittery timestamps and quiet-window QA checks (noise SD and drift trend). Authored 10 accuracy tests in `tests/test_cmj.py` (37 total passing across repo) and printed a per-condition error table. Generated publication-grade 5-trial CMJ detection and velocity/power diagnostic plot (`reports/figures/cmj_five_trials_detection.png`).
- 2026-09-21 (Antigravity): Work stopped here per user directive. Recorded `PENDING.md` containing items A-H (reconcile flight-offset results, noise-robust landing detection, generator realism and templates, momentum-consistent landing, Monte Carlo accuracy test, raw-signal notch explanation, plot visual refinements, and strict tolerance tracking). Changed M2b status from `[x]` to `[~]`. Next: Address items A-H in `PENDING.md` before finalizing M2b and beginning M2c.
