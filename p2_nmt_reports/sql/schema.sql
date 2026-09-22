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
