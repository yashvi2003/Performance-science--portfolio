# Project 2: SQL layer, reporting and longitudinal analysis (`p2_nmt_reports`)

Goal: take sessions.csv + wearable_daily.csv and produce a scannable report/dashboard with personal
baselines and actionable alerts. No coupling to Project 1 code: only the shared contract.

## Logic
- Personal baseline per metric: rolling mean and SD over a configurable window (default 28 days,
  minimum N sessions before flagging). Percent change and z-score vs baseline. Typical error /
  smallest worthwhile change from the athlete's own variability, configurable.
- Exclude invalid trials. Direction-aware (lower GCT = better).
- Alert tiers, each stating which rule fired and why:
  (a) metric change beyond threshold (default -10%) or beyond SWC;
  (b) same, plus >= N consecutive Red recovery days (Red <34, Yellow 34-66, Green 67-100; configurable);
  (c) sustained multi-metric decline.
  Wording: "possible fatigue flag", never a diagnosis.

## Readiness + wearable layer
- `readiness.py`: per metric, z-score against a rolling personal baseline (configurable window and
  minimum sessions). Direction-aware. Composite index with documented, configurable weights.
  Flag a drop when deviation exceeds max(10%, smallest worthwhile change), both configurable.
  Label the composite a heuristic index, not a validated score.
- `wearable.py`: ingest wearable_daily.csv (column map via config), join to sessions by date, output a
  table and a plot of readiness index vs recovery/strain.

## SQL layer
- Load sessions.csv and wearable_daily.csv into DuckDB or SQLite. Express baselines, rolling windows and
  session summaries as `.sql` files using window functions; run them from Python; tests assert the SQL results
  equal the pandas results. Keep queries readable and commented (this is a skill to demonstrate).

## Within-athlete change model
- Mixed-effects model of session-to-session change (random intercept per athlete; random slope if the data
  support it), in R (lme4) or Python (statsmodels). Report fixed effects and variance components with CIs, effect
  sizes, residual diagnostics, and what n = 10 athletes limits. Use it to derive SEM/MDC-style thresholds.

## Output
- Streamlit dashboard: upload the two CSVs, choose athlete, metric-vs-baseline charts with shaded
  bands, recovery overlay, alert list.
- Export a self-contained HTML report (and PDF if WeasyPrint is installed; otherwise degrade
  gracefully). Executive summary at the top.
- Real data: the professor's CMJ data has one trial per athlete, so baselines, alerts and the within-athlete model run on
  labelled synthetic longitudinal data (or a public repeated-measures dataset if one is found; log it in DATA_SOURCES.md);
  the recovery overlay still comes from a public wearable dataset or synthetic data and is labelled as such.
- Demo mode using `shared/synth.py` with a built-in fatigue episode so an alert visibly fires.

## Tests
Known synthetic series: assert baseline maths, right alert tier on the right days, no alerts on
stable data, correct handling of gaps and missing recovery days.

## Definition of done
All Project 2 tasks ticked; pytest green; dashboard runs on synthetic demo data; HTML report
export works; README with disclaimer that synthetic data shows function, not findings.
