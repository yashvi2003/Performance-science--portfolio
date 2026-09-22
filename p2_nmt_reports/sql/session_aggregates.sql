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
