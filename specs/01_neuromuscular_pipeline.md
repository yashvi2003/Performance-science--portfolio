# Project 1: Automatic jump analysis engine, VALD-style (`p1_nmt_pipeline`)

Vision: when a force-plate file comes in, the system works out which jump it is, extracts the key metrics and draws
the standard graph, without manual steps. Build order: CMJ end to end (validated on the real data) -> jump-type
classifier -> SJ and DJ metrics -> single-trial report and drop-a-file app. IMTP metrics are a stretch goal (S3); the
classifier still recognises an isometric pull. Readiness, baselines and wearable joins live in Project 2.

Goal: turn raw vertical force-time data into a classified jump, validated metrics and a standard report, and write
`sessions.csv` per the shared contract.
Audience: performance-science and data-analytics researchers at a wearable-performance company
(thinking in measurement quality, personal baselines and trend visibility).

## Input spec
- CSV/JSON with `time` (s) and `force` (N, total vertical GRF). Optional `force_left` /
  `force_right`: sum for analysis but keep per-limb values in the output.
- Per-trial metadata: test type, body mass (kg; if missing, estimate from a quiet-standing
  window and flag it), sampling rate (infer from `time`, warn if irregular), drop height cm (DJ).
- Validate inputs with clear errors (units, NaNs, duplicate timestamps).

## Processing
- Zero-phase 4th-order Butterworth low-pass (scipy). Cutoff configurable per test in YAML;
  document default values and their rationale in the README.
- All thresholds (onset SD multiplier, takeoff/landing force threshold, windows) live in config.

## Test definitions (implement exactly; state the formula in each docstring)
- CMJ: body weight from the quiet-standing window; onset = force deviating from BW by more than a
  configurable multiple of the quiet-period SD; takeoff/landing via a force threshold. Velocity by
  integrating (F - BW)/m from onset. Jump height by impulse-momentum (primary) and flight time
  (secondary), report both. Extract peak power (absolute and relative), peak velocity, peak
  braking force, time to takeoff, and RSImod = jump height / time from onset to takeoff.
- SJ: require a stable hold before onset. Flag the trial invalid if a countermovement (force dipping
  below BW beyond a threshold) precedes propulsion. Extract peak force, peak velocity, jump height,
  RFD over 0-50, 0-100, 0-200 ms plus peak RFD.
- IMTP: baseline from a quiet pre-pull window. Onset = force exceeding baseline by a configurable
  multiple of baseline SD. Extract gross and net peak force, time to peak force, force at
  50/100/150/200/250 ms from onset, RFD windows. Flag pre-tension or unstable baseline.
- DJ: drop height as input. Contact/takeoff by force threshold. Extract GCT, flight time, jump
  height from flight time, RSI = jump height / GCT.

## Jump-type classifier (rule-based, transparent)
- Input: filtered force-time trace, body mass. Output: label in {CMJ, SJ, DJ, isometric_pull, unknown}, a confidence
  and a plain-language reason (which features fired).
- Features: quiet-standing lead-in present; force dips below body weight by more than a threshold before propulsion
  (countermovement); stable hold near body weight before a rise with no dip (squat jump); starts with a flight-free
  period at ~0 N then a sharp landing impact followed by take-off (drop jump); force rises well above body weight with
  no take-off (isometric pull); flight phase present or absent.
- Return `unknown` rather than guessing when features conflict. Thresholds live in the config.
- Validation honesty: the real data has only CMJ trials. All 10 real trials must classify as CMJ. SJ, DJ and isometric
  classes are validated on synthetic trials only, with a confusion matrix, and the README says so. Also allow the
  operator to set the test type explicitly (metadata); as far as I know commercial systems often have the test chosen
  up front, so auto-classification is an extra, not a replacement.

## Single-trial report (the VALD-style output)
- One figure per trial: force-time curve with phases shaded (unweighting, braking, propulsion, flight, landing) and
  events marked, plus velocity and power panels, and a key-metrics table (jump height by both methods, peak power,
  RSImod, contraction time, flight time, peak force before take-off, body weight/mass). Export PNG and HTML/PDF.
- Minimal Streamlit app: drop a file (CSV, or the professor's Excel layout via the importer), show detected jump type,
  metrics and graphs. Any screenshot used publicly must come from synthetic data unless permission is recorded.

## Output
- Trial dataclass: metrics, event indices (onset/takeoff/landing), quality flags, JSON-serialisable.
- `sessions.csv` per the shared contract (best trial and mean per session in addition to trials).
- QA plot per trial (matplotlib) with detected events annotated.

## Real-data validation (research core of this project)
What the real data is (see specs/05_input_data_profile.md): 10 athletes x ONE CMJ trial each, 2000 Hz, total vertical
force in N, supplied by the user's professor. There are NO repeated trials or sessions, so trial-to-trial reliability
(CV, ICC, SEM, MDC) and within-athlete change CANNOT be estimated from it. Do not fabricate or imply them.
What it does support:
- Data-quality report per trial: quiet-phase length and SD, timestamp irregularities, flight-phase force offset,
  negative force values, plausibility of body mass.
- Method comparison across the 10 trials: flight-time vs impulse-momentum jump height (bias, limits of agreement,
  Bland-Altman with confidence intervals). State that n = 10 gives wide intervals.
- Analyst-choice sensitivity: vary the baseline window, onset SD multiplier, take-off/landing thresholds and filter
  cutoff; report how much jump height, RSImod and peak power move. Trials with a noisy quiet phase are expected to be
  the most sensitive; say which and why.
- Golden test: the reference workbook's trial (data/raw/reference/, expected values in reference_expected.json)
  must be reproduced within stated tolerances. Its values are UNVERIFIED (provenance unknown, built with Excel
  formulas following Chavda et al. 2018), so any mismatch is investigated in both directions, not assumed to be a bug
  in the Python.
- Manual spot checks: overlay detected onset/takeoff/landing on the raw curve for every trial; record failures.
- Open data (specs/05, D4): once downloaded, add a `pre_takeoff_window` input mode (no flight phase, take-off given, body-weight units, 250 Hz) and
  validate on the 663 CMJ trials: agreement of impulse-momentum height and peak power with the dataset's reference values, and within-participant
  reliability (CV, ICC, SEM). This is the only real repeated-trial data available, so reliability claims come from it, not from the professor's files.
  Public demo figures and README screenshots should use this CC BY dataset (with attribution), not the professor's data.
- Peak force/power for take-off metrics are computed only BEFORE take-off; landing peaks are reported separately
  (several trials have a landing peak that is far higher than the propulsion peak).

## Tests (pytest)
Synthetic trials from `shared/synth.py` with analytically known height and event times. Assert jump
height within 1 cm, event times within a few samples, robustness across noise levels and sampling
rates (500-2000 Hz), correct flagging of invalid SJ trials, classifier confusion matrix on synthetic CMJ/SJ/DJ/isometric trials, all 10 real trials classified CMJ.

## Definition of done
All Project 1 tasks ticked; pytest green; dropping a file into the app or CLI gives jump type, metrics and a graph;
CLI writes a valid sessions.csv; README with install/run instructions, assumptions, limitations, synthetic disclaimer.
