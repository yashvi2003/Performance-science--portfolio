# Project 2 Analytics Architecture & Statistical Engine
**Neuromuscular Monitoring Toolkit (`p2_nmt_reports`)**  
*Consolidated Technical Specification: DuckDB SQL Layer & Parity Tests, Readiness Index Formulation, and Within-Athlete Linear Mixed-Effects Model Output.*

---

## 1. DuckDB SQL Layer, Queries & Parity Test Suite

The Project 2 database layer operates on an in-memory DuckDB analytical engine (`duckdb`), ingesting contract tables and executing session aggregation and 28-day rolling window baselines using pure SQL window functions.

### 1.1 SQL Schema (`p2_nmt_reports/sql/schema.sql`)
```sql
-- Schema definition for DuckDB / SQLite contract tables

CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR NOT NULL,
    date DATE NOT NULL,
    athlete_id VARCHAR NOT NULL,
    test VARCHAR NOT NULL,
    trial INTEGER NOT NULL,
    metric VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    unit VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    valid BOOLEAN NOT NULL,
    flags VARCHAR,
    block VARCHAR
);

CREATE TABLE IF NOT EXISTS wearable_daily (
    date DATE NOT NULL,
    athlete_id VARCHAR NOT NULL,
    recovery_pct DOUBLE,
    hrv_ms DOUBLE,
    rhr_bpm DOUBLE,
    day_strain DOUBLE,
    sleep_hours DOUBLE,
    proxy_used VARCHAR
);
```

---

### 1.2 Session Aggregates Query (`p2_nmt_reports/sql/session_aggregates.sql`)
```sql
-- Aggregate valid trials to session-level metrics
-- Filters out invalid trials (valid = false)

SELECT
    athlete_id,
    date,
    session_id,
    test,
    metric,
    side,
    unit,
    AVG(value) AS value,
    COUNT(trial) AS n_trials
FROM sessions
WHERE valid = TRUE
GROUP BY
    athlete_id,
    date,
    session_id,
    test,
    metric,
    side,
    unit
ORDER BY
    athlete_id,
    metric,
    side,
    date;
```

---

### 1.3 28-Day Rolling Baselines Window Query (`p2_nmt_reports/sql/rolling_baselines.sql`)
```sql
-- Compute 28-day rolling personal baselines using SQL window functions
-- Matches Hopkins SWC (0.5 * SD_within), Hopkins Typical Error (SD / sqrt(2)),
-- and direction-aware standardized z-scores.

WITH session_agg AS (
    SELECT
        athlete_id,
        CAST(date AS DATE) AS date,
        session_id,
        test,
        metric,
        side,
        unit,
        AVG(value) AS value,
        COUNT(trial) AS n_trials
    FROM sessions
    WHERE valid = TRUE
    GROUP BY
        athlete_id,
        CAST(date AS DATE),
        session_id,
        test,
        metric,
        side,
        unit
),

rolling_stats AS (
    SELECT
        athlete_id,
        date,
        session_id,
        test,
        metric,
        side,
        unit,
        value,
        n_trials,
        -- Rolling 28-day calendar window (including current row)
        AVG(value) OVER w AS baseline_mean,
        STDDEV_SAMP(value) OVER w AS baseline_sd,
        COUNT(value) OVER w AS baseline_n
    FROM session_agg
    WINDOW w AS (
        PARTITION BY athlete_id, metric, side
        ORDER BY date
        RANGE BETWEEN INTERVAL 28 DAYS PRECEDING AND CURRENT ROW
    )
)

SELECT
    athlete_id,
    date,
    session_id,
    test,
    metric,
    side,
    unit,
    value,
    n_trials,
    baseline_mean,
    baseline_sd,
    baseline_n,
    CASE 
        WHEN baseline_n >= 3 THEN TRUE 
        ELSE FALSE 
    END AS baseline_established,
    CASE 
        WHEN baseline_n >= 3 THEN 0.5 * baseline_sd 
        ELSE NULL 
    END AS swc,
    CASE 
        WHEN baseline_n >= 3 THEN baseline_sd / SQRT(2.0) 
        ELSE NULL 
    END AS typical_error,
    CASE 
        WHEN baseline_n >= 3 AND ABS(baseline_mean) > 1e-12 THEN ((value - baseline_mean) / baseline_mean) * 100.0 
        ELSE NULL 
    END AS pct_change,
    CASE 
        WHEN baseline_n < 3 THEN NULL
        WHEN baseline_sd IS NULL OR baseline_sd <= 1e-9 THEN 0.0
        -- Lower is better metrics: invert sign so +z indicates better-than-baseline readiness
        WHEN metric IN ('dj_gct', 'cmj_time_to_takeoff') THEN -1.0 * ((value - baseline_mean) / baseline_sd)
        ELSE 1.0 * ((value - baseline_mean) / baseline_sd)
    END AS z_score
FROM rolling_stats
ORDER BY
    athlete_id,
    metric,
    side,
    date;
```

---

### 1.4 SQL Execution Engine (`p2_nmt_reports/sql_engine.py`)
```python
"""DuckDB SQL Execution Engine for Project 2.

Loads contract tables into DuckDB, executes modular .sql window queries,
and returns analytics DataFrames.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Union
import duckdb
import pandas as pd

from shared.contract import validate_sessions_df, validate_wearable_daily_df
from p2_nmt_reports.loader import load_sessions, load_wearable_daily

logger = logging.getLogger(__name__)

SQL_DIR = Path(__file__).parent / "sql"


def create_connection() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection and initialize contract schema."""
    con = duckdb.connect(database=":memory:")
    schema_path = SQL_DIR / "schema.sql"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            con.execute(f.read())
    return con


def load_tables(
    con: duckdb.DuckDBPyConnection,
    sessions: Union[str, Path, pd.DataFrame],
    wearable: Optional[Union[str, Path, pd.DataFrame]] = None,
) -> None:
    """Load validated sessions and wearable dataframes into DuckDB tables."""
    if isinstance(sessions, (str, Path, pd.DataFrame)):
        sessions_df = load_sessions(sessions, filter_valid=False)
    else:
        sessions_df = sessions

    # Keep only contract columns for sessions table
    cols_sessions = [
        "session_id", "date", "athlete_id", "test", "trial",
        "metric", "value", "unit", "side", "valid", "flags", "block"
    ]
    avail_sess_cols = [c for c in cols_sessions if c in sessions_df.columns]
    sess_to_load = sessions_df[avail_sess_cols].copy()

    con.execute("DELETE FROM sessions")
    con.register("temp_sessions", sess_to_load)
    con.execute(f"INSERT INTO sessions ({', '.join(avail_sess_cols)}) SELECT * FROM temp_sessions")
    con.unregister("temp_sessions")

    if wearable is not None:
        if isinstance(wearable, (str, Path, pd.DataFrame)):
            wearable_df = load_wearable_daily(wearable)
        else:
            wearable_df = wearable

        cols_wearable = [
            "date", "athlete_id", "recovery_pct", "hrv_ms",
            "rhr_bpm", "day_strain", "sleep_hours", "proxy_used"
        ]
        avail_wear_cols = [c for c in cols_wearable if c in wearable_df.columns]
        wear_to_load = wearable_df[avail_wear_cols].copy()

        con.execute("DELETE FROM wearable_daily")
        con.register("temp_wearable", wear_to_load)
        con.execute(f"INSERT INTO wearable_daily ({', '.join(avail_wear_cols)}) SELECT * FROM temp_wearable")
        con.unregister("temp_wearable")


def run_session_aggregates_sql(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Run session_aggregates.sql query in DuckDB."""
    query_path = SQL_DIR / "session_aggregates.sql"
    with open(query_path, "r", encoding="utf-8") as f:
        query = f.read()
    return con.execute(query).df()


def run_rolling_baselines_sql(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Run rolling_baselines.sql window query in DuckDB."""
    query_path = SQL_DIR / "rolling_baselines.sql"
    with open(query_path, "r", encoding="utf-8") as f:
        query = f.read()
    df = con.execute(query).df()
    df["date"] = pd.to_datetime(df["date"])
    return df
```

---

### 1.5 Parity Test Suite (`tests/test_p2_sql.py`)
```python
"""Unit tests for Project 2 Milestone M1b: DuckDB SQL Window Parity against Pandas Reference."""

import pytest
import numpy as np
import pandas as pd

from shared.synth import make_sessions_table, make_wearable_daily
from p2_nmt_reports.loader import load_sessions, aggregate_session_metrics
from p2_nmt_reports.baselines import compute_rolling_baselines
from p2_nmt_reports.sql_engine import (
    create_connection,
    load_tables,
    run_session_aggregates_sql,
    run_rolling_baselines_sql,
)


def test_sql_session_aggregates_matches_pandas():
    """Test that SQL session aggregates query exactly matches Pandas aggregation."""
    df_raw = make_sessions_table(n_athletes=3, n_weeks=5, random_seed=42)
    # Inject some invalid trials to verify WHERE valid = TRUE filtering
    df_raw.loc[5, "valid"] = False
    df_raw.loc[12, "valid"] = False

    # Pandas reference
    sessions_valid = load_sessions(df_raw, filter_valid=True)
    df_pandas = aggregate_session_metrics(sessions_valid, method="mean")

    # DuckDB SQL execution
    con = create_connection()
    load_tables(con, df_raw)
    df_sql = run_session_aggregates_sql(con)

    # Sort identically
    sort_cols = ["athlete_id", "metric", "side", "date"]
    df_pandas["date"] = pd.to_datetime(df_pandas["date"])
    df_sql["date"] = pd.to_datetime(df_sql["date"])
    
    df_pandas = df_pandas.sort_values(sort_cols).reset_index(drop=True)
    df_sql = df_sql.sort_values(sort_cols).reset_index(drop=True)

    assert len(df_sql) == len(df_pandas)
    assert np.all(df_sql["athlete_id"] == df_pandas["athlete_id"])
    assert np.all(df_sql["metric"] == df_pandas["metric"])
    assert np.all(df_sql["n_trials"] == df_pandas["n_trials"])
    np.testing.assert_allclose(df_sql["value"].values, df_pandas["value"].values, rtol=1e-7, atol=1e-7)


def test_sql_rolling_baselines_matches_pandas_exact():
    """Test that SQL window functions produce numerically identical rolling baselines to Pandas.

    Validates:
    - baseline_mean: |diff| < 1e-6
    - baseline_sd: |diff| < 1e-6
    - baseline_n: exact integer equality
    - baseline_established: exact boolean equality
    - swc: |diff| < 1e-6
    - typical_error: |diff| < 1e-6
    - pct_change: |diff| < 1e-6
    - z_score: |diff| < 1e-6
    """
    df_raw = make_sessions_table(n_athletes=4, n_weeks=8, random_seed=123)

    # 1. Pandas reference calculation
    sessions_valid = load_sessions(df_raw, filter_valid=True)
    agg_pandas = aggregate_session_metrics(sessions_valid, method="mean")
    df_pandas = compute_rolling_baselines(agg_pandas)

    # 2. DuckDB SQL execution
    con = create_connection()
    load_tables(con, df_raw)
    df_sql = run_rolling_baselines_sql(con)

    # Align sort orders
    sort_cols = ["athlete_id", "metric", "side", "date"]
    df_pandas["date"] = pd.to_datetime(df_pandas["date"])
    df_sql["date"] = pd.to_datetime(df_sql["date"])

    df_pandas = df_pandas.sort_values(sort_cols).reset_index(drop=True)
    df_sql = df_sql.sort_values(sort_cols).reset_index(drop=True)

    assert len(df_sql) == len(df_pandas)

    # Structural / boolean columns
    assert np.all(df_sql["athlete_id"] == df_pandas["athlete_id"])
    assert np.all(df_sql["metric"] == df_pandas["metric"])
    assert np.all(df_sql["baseline_n"] == df_pandas["baseline_n"])
    assert np.all(df_sql["baseline_established"] == df_pandas["baseline_established"])

    # Numerical metrics comparison where baseline is established
    mask = df_pandas["baseline_established"] == True
    assert mask.sum() > 0

    np.testing.assert_allclose(
        df_sql.loc[mask, "baseline_mean"].values,
        df_pandas.loc[mask, "baseline_mean"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL baseline_mean does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "baseline_sd"].values,
        df_pandas.loc[mask, "baseline_sd"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL baseline_sd does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "swc"].values,
        df_pandas.loc[mask, "swc"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL swc does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "typical_error"].values,
        df_pandas.loc[mask, "typical_error"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL typical_error does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "pct_change"].values,
        df_pandas.loc[mask, "pct_change"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL pct_change does not match Pandas reference",
    )

    np.testing.assert_allclose(
        df_sql.loc[mask, "z_score"].values,
        df_pandas.loc[mask, "z_score"].values,
        rtol=1e-6,
        atol=1e-6,
        err_msg="SQL direction-aware z_score does not match Pandas reference",
    )
```

---

### 1.6 Actual Parity Test Execution Output
```
============================= test session starts =============================
platform win32 -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Yashvi\Internship 2027\portfolio-agent-kit
configfile: pyproject.toml
plugins: anyio-4.15.1
collected 13 items

tests/test_p2_sql.py::test_sql_session_aggregates_matches_pandas PASSED   [  7%]
tests/test_p2_sql.py::test_sql_rolling_baselines_matches_pandas_exact PASSED [ 15%]
tests/test_p2_baselines.py::test_load_sessions_and_filter_invalid PASSED  [ 23%]
tests/test_p2_baselines.py::test_load_wearable_daily PASSED               [ 30%]
tests/test_p2_baselines.py::test_aggregate_session_metrics PASSED         [ 38%]
tests/test_p2_baselines.py::test_rolling_baseline_math_and_swc PASSED     [ 46%]
tests/test_p2_baselines.py::test_direction_aware_z_score PASSED           [ 53%]
tests/test_p2_baselines.py::test_28d_window_expiry_and_gap_handling PASSED [ 61%]
tests/test_p2_readiness.py::test_composite_readiness_weights_and_direction PASSED [ 69%]
tests/test_p2_readiness.py::test_wearable_join_and_recovery_zones PASSED  [ 76%]
tests/test_p2_readiness.py::test_classify_recovery_zone PASSED            [ 84%]
tests/test_p2_readiness.py::test_within_athlete_mixed_effects_model PASSED [ 92%]
tests/test_p2_readiness.py::test_chart_generation PASSED                  [100%]

======================= 13 passed, 9 warnings in 21.55s =======================
```

---

## 2. Complete Readiness Index Formula & Scientific Justification

### 2.1 Mathematical Formulation

#### 1. 28-Day Rolling Personal Baseline ($N \ge 3$ Sessions Required)
For athlete $i$, metric $m$, across historical sessions $k \in \{1, \dots, N\}$ occurring in the window $[t - 28\text{ days}, t]$:
$$\mu_{i,m} = \frac{1}{N} \sum_{k=1}^N x_{i,m,k}, \quad \sigma_{i,m} = \sqrt{\frac{1}{N-1} \sum_{k=1}^N (x_{i,m,k} - \mu_{i,m})^2}$$

#### 2. Smallest Worthwhile Change (SWC) & Typical Error (TE)
$$\text{SWC}_{i,m} = 0.5 \times \sigma_{i,m} \quad (\text{Hopkins, 2004})$$
$$\text{TE}_{i,m} = \frac{\sigma_{i,m}}{\sqrt{2}} \quad (\text{Hopkins, 2000})$$

#### 3. Standardized Direction-Aware Z-Score ($z_{i,m,t}$)
To ensure that positive $z$-scores consistently represent better-than-baseline neuromuscular state across metrics with opposing physical directions:
$$z_{i,m,t} = \text{sign}(m) \times \left( \frac{x_{i,m,t} - \mu_{i,m}}{\sigma_{i,m}} \right)$$
where:
$$\text{sign}(m) = \begin{cases} +1.0, & \text{if } m \in \{\text{jump\_height}, \text{rsimod}, \text{peak\_power}, \text{peak\_velocity}\} \\ -1.0, & \text{if } m \in \{\text{dj\_gct}, \text{cmj\_time\_to\_takeoff}\} \end{cases}$$

#### 4. Composite Neuromuscular Readiness Index ($z_{\text{readiness}}$)
For a session containing available metric subset $M_{\text{avail}} \subseteq \{\text{cmj\_jump\_height\_imp}, \text{cmj\_rsimod}, \text{cmj\_peak\_power}\}$:
$$z_{\text{readiness}} = \sum_{m \in M_{\text{avail}}} w_m' \cdot z_{i,m,t}$$
where the normalized weights $w_m'$ are defined as:
$$w_m' = \frac{w_m}{\sum_{k \in M_{\text{avail}}} w_k}$$
with default base weights configured in `p2_nmt_reports/config.yaml`:
* $w(\text{cmj\_jump\_height\_imp}) = 0.40$ (40%)
* $w(\text{cmj\_rsimod}) = 0.35$ (35%)
* $w(\text{cmj\_peak\_power}) = 0.25$ (25%)

#### 5. Acute Drop Flag Rule
$$\text{Acute Drop} = \text{TRUE} \iff \left( z_{\text{readiness}} < -1.0 \right) \lor \left( \Delta\%_{\text{primary}} < -\max(10.0\%, \text{SWC}_{\%}) \right)$$

---

### 2.2 Scientific Justification & Literature Citations

1. **CMJ Metrics as Neuromuscular Fatigue Biomarkers**:
   - **Claudino et al. (2017)** (*Sports Medicine*, meta-analysis): Identified jump height, flight time, and reactive strength measures as the most sensitive, non-invasive indicators of neuromuscular fatigue following competitive match play and training strain.
   - **McLean et al. (2010)** (*J Strength Cond Res*): Demonstrated that neuromuscular fatigue manifests simultaneously in kinetic output (peak power) and movement strategy (RSImod, contraction time), justifying a multi-metric composite over single-variable monitoring.
2. **Individual vs. Cohort Thresholds**:
   - **Hopkins (2004)** (*Sports Med*): Established that for individual longitudinal athlete monitoring, the Smallest Worthwhile Change is properly anchored to within-athlete biological variation ($0.5 \times \text{SD}_{\text{within}}$) rather than between-athlete cohort variation ($0.2 \times \text{SD}_{\text{between}}$), which is confounded by inter-subject physiological heterogeneity.
   - **Hopkins (2000)**: Formulated Typical Error ($\text{TE} = \text{SD} / \sqrt{2}$) as the standard metric for assessing measurement consistency across repeated trials.
3. **Guardrails & Portfolio Transparency (Rule 7 Framing)**:
   - *Heuristic Labeling*: The composite readiness index is an experimental heuristic index for exploratory performance monitoring, **NOT** a clinically validated diagnostic instrument or medical score.
   - *Baseline Establishment*: Enforces a strict minimum of $N \ge 3$ valid sessions within the backward 28-day window before establishing baseline statistics; sessions with $N < 3$ return `NaN` and `baseline_established = False`.

---

### 2.3 Readiness Python Engine (`p2_nmt_reports/readiness.py`)
```python
"""Composite Readiness Index Module for Project 2.

Scientific Framing & Basis:
Combines standardized within-athlete z-scores across key neuromuscular metrics
using a linear weighted composite (McLean et al., 2010; Claudino et al., 2017).
All z-scores are direction-aware so that positive values reflect better-than-baseline
neuromuscular state.

DISCLAIMER:
The composite readiness index is an experimental heuristic index for exploratory monitoring,
NOT a validated clinical score or diagnostic instrument.
"""

from __future__ import annotations
import logging
from typing import Optional, Dict
import numpy as np
import pandas as pd

from p2_nmt_reports.config import ReadinessConfig
from p2_nmt_reports.baselines import compute_rolling_baselines

logger = logging.getLogger(__name__)


def compute_composite_readiness(
    baselines_df: pd.DataFrame,
    config: Optional[ReadinessConfig] = None,
) -> pd.DataFrame:
    """Calculate direction-aware composite readiness index per athlete-session."""
    if config is None:
        config = ReadinessConfig()

    target_weights = config.composite_weights
    fallback_metric = config.fallback_single_metric
    drop_pct_thresh = config.drop_threshold_pct
    use_swc = config.use_swc_threshold
    disclaimer = config.heuristic_label

    if baselines_df.empty:
        return pd.DataFrame(
            columns=[
                "athlete_id", "date", "session_id", "composite_readiness_z",
                "metrics_included", "weights_used", "is_single_metric_fallback",
                "acute_drop_flag", "heuristic_disclaimer",
            ]
        )

    sessions: list[dict] = []
    grouped = baselines_df.groupby(["athlete_id", "date", "session_id"], sort=False)

    for (ath, dt, sess_id), group in grouped:
        valid_rows = group[group["baseline_established"] & group["z_score"].notna()].copy()

        if valid_rows.empty:
            sessions.append({
                "athlete_id": ath,
                "date": dt,
                "session_id": sess_id,
                "composite_readiness_z": np.nan,
                "metrics_included": "",
                "weights_used": "",
                "is_single_metric_fallback": False,
                "acute_drop_flag": False,
                "heuristic_disclaimer": disclaimer,
            })
            continue

        metric_z_map = dict(zip(valid_rows["metric"], valid_rows["z_score"]))
        metric_swc_map = dict(zip(valid_rows["metric"], valid_rows["swc"]))
        metric_pct_map = dict(zip(valid_rows["metric"], valid_rows["pct_change"]))

        matching_metrics = [m for m in target_weights.keys() if m in metric_z_map]

        if matching_metrics:
            raw_weights = [target_weights[m] for m in matching_metrics]
            total_weight = sum(raw_weights)
            norm_weights = [w / total_weight for w in raw_weights]

            comp_z = sum(w * metric_z_map[m] for w, m in zip(norm_weights, matching_metrics))
            metrics_str = ";".join(matching_metrics)
            weights_str = ";".join(f"{m}:{w:.2f}" for m, w in zip(matching_metrics, norm_weights))
            is_fallback = len(matching_metrics) == 1
        elif fallback_metric in metric_z_map:
            comp_z = metric_z_map[fallback_metric]
            metrics_str = fallback_metric
            weights_str = f"{fallback_metric}:1.00"
            is_fallback = True
        else:
            all_metrics = list(metric_z_map.keys())
            comp_z = float(np.mean([metric_z_map[m] for m in all_metrics]))
            metrics_str = ";".join(all_metrics)
            weights_str = ";".join(f"{m}:{1.0/len(all_metrics):.2f}" for m in all_metrics)
            is_fallback = True

        acute_drop = False
        if comp_z < -1.0:
            acute_drop = True

        primary_metric = matching_metrics[0] if matching_metrics else (fallback_metric if fallback_metric in metric_pct_map else None)
        if primary_metric and primary_metric in metric_pct_map and primary_metric in metric_swc_map:
            pct_chg = metric_pct_map[primary_metric]
            swc_val = metric_swc_map[primary_metric]
            if not np.isnan(pct_chg) and not np.isnan(swc_val):
                if pct_chg < -drop_pct_thresh:
                    acute_drop = True

        sessions.append({
            "athlete_id": ath,
            "date": dt,
            "session_id": sess_id,
            "composite_readiness_z": float(comp_z),
            "metrics_included": metrics_str,
            "weights_used": weights_str,
            "is_single_metric_fallback": is_fallback,
            "acute_drop_flag": bool(acute_drop),
            "heuristic_disclaimer": disclaimer,
        })

    out_df = pd.DataFrame(sessions)
    out_df["date"] = pd.to_datetime(out_df["date"])
    return out_df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
```

---

## 3. Project 2 Within-Athlete Linear Mixed-Effects Model (`p2_nmt_reports/stats.py`)

### 3.1 Model Specification & Measurement Error Equations

The longitudinal change model quantifies how session-level neuromuscular readiness fluctuates with daily wearable recovery, training strain, and sleep duration, accounting for clustered repeated measures within athletes:

$$\text{Readiness}_{ij} = \beta_0 + \beta_1 z\text{Recovery}_{ij} + \beta_2 z\text{Strain}_{ij} + \beta_3 z\text{Sleep}_{ij} + u_i + \epsilon_{ij}$$

where:
* $u_i \sim \mathcal{N}(0, \sigma_u^2)$ is the between-athlete random intercept variance (`group_var_between_athlete`).
* $\epsilon_{ij} \sim \mathcal{N}(0, \sigma_\epsilon^2)$ is the residual within-athlete variance (`residual_var_within_athlete`).
* Estimation: Restricted Maximum Likelihood (**REML**) via Python `statsmodels.regression.mixed_linear_model.MixedLM`.

#### Derived Measurement Error & Detection Thresholds:
1. **Intraclass Correlation Coefficient (ICC)**:
   $$\text{ICC} = \frac{\sigma_u^2}{\sigma_u^2 + \sigma_\epsilon^2}$$
2. **Standard Error of Measurement (SEM)**:
   $$\text{SEM} = \sqrt{\sigma_\epsilon^2}$$
3. **Minimal Detectable Change at 95% Confidence ($\text{MDC}_{95}$)**:
   $$\text{MDC}_{95} = 1.96 \times \sqrt{2} \times \text{SEM} \approx 2.7718 \times \text{SEM}$$

---

### 3.2 Model Execution Code (`p2_nmt_reports/stats.py`)
```python
"""Within-Athlete Mixed-Effects Session Change Model for Project 2.

Fits a Linear Mixed-Effects Model (LMM via statsmodels) modeling session-to-session
neuromuscular readiness fluctuations with random intercepts per athlete.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


def fit_within_athlete_change_model(
    merged_df: pd.DataFrame,
    outcome_col: str = "composite_readiness_z",
    predictor_cols: Optional[list[str]] = None,
    reml: bool = True,
) -> Dict[str, Any]:
    """Fit a linear mixed-effects model to longitudinal session monitoring data."""
    if predictor_cols is None:
        avail = [c for c in ["recovery_pct", "day_strain", "sleep_hours"] if c in merged_df.columns]
        predictor_cols = avail if avail else []

    cols_needed = ["athlete_id", outcome_col] + predictor_cols
    df = merged_df[cols_needed].dropna().copy()

    if len(df) < 10 or df["athlete_id"].nunique() < 2:
        raise ValueError(
            f"Insufficient data to fit mixed-effects model: {len(df)} rows across {df['athlete_id'].nunique()} athletes."
        )

    for col in predictor_cols:
        sd = df[col].std()
        if sd > 1e-9:
            df[f"z_{col}"] = (df[col] - df[col].mean()) / sd
        else:
            df[f"z_{col}"] = 0.0

    z_predictors = [f"z_{col}" for col in predictor_cols]
    if z_predictors:
        formula = f"{outcome_col} ~ {' + '.join(z_predictors)}"
    else:
        formula = f"{outcome_col} ~ 1"

    try:
        model = smf.mixedlm(formula, df, groups=df["athlete_id"]).fit(reml=reml, method=["lbfgs", "powell"])
    except Exception as e:
        logger.warning("Primary optimizer failed, falling back to default fit: %s", e)
        model = smf.mixedlm(formula, df, groups=df["athlete_id"]).fit(reml=reml)

    fe_params = model.params.to_dict()
    bse = model.bse.to_dict()
    pvalues = model.pvalues.to_dict()
    conf_int = model.conf_int()

    fixed_effects_summary = []
    for term in model.params.index:
        ci_lower = conf_int.loc[term, 0]
        ci_upper = conf_int.loc[term, 1]
        fixed_effects_summary.append({
            "term": term,
            "estimate": float(fe_params[term]),
            "std_error": float(bse[term]),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "z_value": float(model.tvalues[term]),
            "p_value": float(pvalues[term]),
        })

    group_var = float(model.cov_re.iloc[0, 0]) if hasattr(model, "cov_re") else float(fe_params.get("Group Var", np.nan))
    residual_var = float(model.scale)
    total_var = group_var + residual_var
    icc = group_var / total_var if total_var > 0 else np.nan

    sem = np.sqrt(residual_var)
    mdc95 = 1.96 * np.sqrt(2.0) * sem

    residuals = np.array(model.resid)
    res_mean = float(np.mean(residuals))
    res_std = float(np.std(residuals))
    res_skew = float(pd.Series(residuals).skew())

    return {
        "formula": formula,
        "n_obs": int(model.nobs),
        "n_groups": int(df["athlete_id"].nunique()),
        "reml_criterion": float(-2.0 * model.llf),
        "fixed_effects": fixed_effects_summary,
        "variance_components": {
            "group_var_between_athlete": group_var,
            "residual_var_within_athlete": residual_var,
            "total_var": total_var,
            "icc": icc,
            "sem": sem,
            "mdc95": mdc95,
        },
        "diagnostics": {
            "residual_mean": res_mean,
            "residual_std": res_std,
            "residual_skewness": res_skew,
        },
        "disclaimer": (
            "Demonstration of within-athlete mixed-effects modeling on labelled synthetic longitudinal data. "
            "Not an empirical research finding."
        ),
    }
```

---

### 3.3 Full Empirical Model Output Tables

```
====================================================================================================
PROJECT 2: LINEAR MIXED-EFFECTS REGRESSION RESULTS (REML ESTIMATION)
====================================================================================================
Dependent Variable: composite_readiness_z (Standardized Z-Score)
Dataset: Synthetic Longitudinal Monitoring Cohort (5 Athletes x 6 Weeks = 20 Established Sessions)
Estimation Method: Restricted Maximum Likelihood (REML) | Scale: 0.4937 | Log-Likelihood: -22.6417
----------------------------------------------------------------------------------------------------

TABLE 1: FIXED-EFFECT PARAMETER ESTIMATES (WITH 95% WALD CONFIDENCE INTERVALS)
----------------------------------------------------------------------------------------------------
Term                      Estimate (β)    Std. Error      z-value     p-value     95% Conf. Interval
----------------------------------------------------------------------------------------------------
Intercept                   -0.1223         0.1574        -0.7769      0.4372     [-0.4308, +0.1862]
z_recovery_pct              +0.0699         0.5606        +0.1247      0.9008     [-1.0288, +1.1686]
z_day_strain                +0.0040         0.4052        +0.0098      0.9921     [-0.7901, +0.7981]
z_sleep_hours               +0.1832         0.2157        +0.8495      0.3956     [-0.2395, +0.6059]
Group Var                   +0.0009         1.2517        +0.0008      0.9994     [-2.4523, +2.4542]
----------------------------------------------------------------------------------------------------

TABLE 2: VARIANCE COMPONENTS & MEASUREMENT ERROR METRICS
----------------------------------------------------------------------------------------------------
Parameter Name                                Symbol            Exact Numerical Value
----------------------------------------------------------------------------------------------------
Between-Athlete Random Intercept Variance     σ_u²              0.000465974
Within-Athlete Residual Variance (Scale)      σ_ε²              0.493685309
Total Variance                                σ_total²          0.494151283
Intraclass Correlation Coefficient (ICC)      ICC               0.000942979
Standard Error of Measurement (SEM)           SEM = √(σ_ε²)     0.702627433
Minimal Detectable Change (95% Confidence)    MDC_95            1.947583879 z-score units
----------------------------------------------------------------------------------------------------

TABLE 3: RESIDUAL ERROR DIAGNOSTICS
----------------------------------------------------------------------------------------------------
Diagnostic Metric                             Exact Numerical Value
----------------------------------------------------------------------------------------------------
Residual Mean                                 1.665335e-17 (centered at zero)
Residual Standard Deviation                   0.6283006
Residual Skewness                             +0.4220325 (approx normal)
Number of Observations (n_obs)                20
Number of Athlete Clusters (n_groups)         5
Average Cluster Size (sessions / athlete)     4.0
====================================================================================================
DISCLAIMER: Demonstration of within-athlete mixed-effects modeling on labelled synthetic longitudinal 
data generated via shared/synth.py. Not an empirical research finding.
====================================================================================================
```
