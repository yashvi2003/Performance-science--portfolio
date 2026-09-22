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
