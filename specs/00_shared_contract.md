# Shared data contract (the ONLY interface between projects)

Implemented and validated in `shared/contract.py`. Projects import only from `shared/`.

## sessions.csv  (long format, one row per trial-metric-side)
| column      | type    | notes |
|-------------|---------|-------|
| session_id  | str     | unique per test session (athlete + date) |
| date        | ISO date| YYYY-MM-DD |
| athlete_id  | str     | |
| test        | str     | one of CMJ, SJ, IMTP, DJ |
| trial       | int     | 1-based trial number within the session |
| metric      | str     | snake_case, prefixed by test, e.g. `cmj_jump_height_imp`, `cmj_rsimod`, `imtp_peak_force_gross`, `imtp_force_100ms`, `dj_rsi`, `dj_gct`, `sj_rfd_0_100ms` |
| value       | float   | |
| unit        | str     | m, s, N, N/s, W, W/kg, m/s, ratio, ... |
| side        | str     | `both` (default), `left`, or `right`. Per-limb rows only where the source has per-limb data |
| valid       | bool    | false if the trial failed a quality check |
| flags       | str     | semicolon-separated quality flags, empty if none |
| block       | str     | optional training block / week label |

Rules: never impute missing per-limb data; invalid trials are kept but marked `valid=false`;
metric names and units are defined once in `shared/contract.py` (a registry with direction:
`higher_is_better` or `lower_is_better`).

## wearable_daily.csv  (one row per athlete-day)
| column        | type  | notes |
|---------------|-------|-------|
| date          | ISO   | |
| athlete_id    | str   | |
| recovery_pct  | float | 0-100 (zones configurable; default Red <34, Yellow 34-66, Green 67-100) |
| hrv_ms        | float | |
| rhr_bpm       | float | |
| day_strain    | float | a strain-like load measure: native wearable strain if ever available, otherwise a documented proxy (see `proxy_used` column) |
| sleep_hours   | float | |
| proxy_used    | str   | empty if native; otherwise which fields were proxied and how |

Field names mirror publicly documented WHOOP metrics so a WHOOP export could be adapted later, but NO WHOOP
device or WHOOP data is used in this portfolio. Any source (public wearable datasets, synthetic) must be converted to this
schema by an adapter. Missing fields stay NaN. Adapters document every mapping and every proxy.

## Synthetic data (shared/synth.py)
- `make_force_trial(test, ...)`: physically consistent force-time trial with KNOWN jump height
  and event times (built from a prescribed acceleration profile); optional left/right split with
  configurable asymmetry; configurable noise and sampling rate.
- `make_sessions_table(...)`: contract-compliant sessions.csv with configurable baseline,
  asymmetry, fatigue slope and a fatigue episode.
- `make_wearable_daily(...)`: contract-compliant daily table with a matching recovery dip.
- All synthetic outputs carry a `synthetic` marker in metadata and plot titles.
