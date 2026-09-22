"""Data loading and aggregation utilities for Project 2 reporting engine.

Interacts strictly through shared/contract.py schemas and validators.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Union
import pandas as pd

from shared.contract import (
    validate_sessions_df,
    validate_wearable_daily_df,
)

logger = logging.getLogger(__name__)


def load_sessions(
    source: Union[str, Path, pd.DataFrame],
    filter_valid: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Load and validate sessions dataframe from file or DataFrame.

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        Path to sessions CSV or an existing DataFrame.
    filter_valid : bool, default True
        If True, returns only trials where valid == True.
    strict : bool, default True
        Whether to enforce strict metric registry validation.

    Returns
    -------
    pd.DataFrame
        Validated sessions DataFrame with parsed date column.
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = source.copy()

    # Validate against shared contract
    is_valid, errors = validate_sessions_df(df, strict=strict)
    if not is_valid:
        raise ValueError(f"Sessions contract validation failed: {errors}")

    df["date"] = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    if filter_valid and "valid" in df.columns:
        initial_len = len(df)
        df = df[df["valid"] == True].copy()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info("Filtered out %d invalid trials from sessions data.", dropped)

    return df


def load_wearable_daily(
    source: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Load and validate wearable daily telemetry dataframe.

    Parameters
    ----------
    source : str, Path, or pd.DataFrame
        Path to wearable daily CSV or an existing DataFrame.

    Returns
    -------
    pd.DataFrame
        Validated wearable telemetry DataFrame with parsed date column.
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = source.copy()

    is_valid, errors = validate_wearable_daily_df(df)
    if not is_valid:
        raise ValueError(f"Wearable daily contract validation failed: {errors}")

    df["date"] = pd.to_datetime(df["date"])
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    return df


def aggregate_session_metrics(
    sessions_df: pd.DataFrame,
    method: str = "mean",
) -> pd.DataFrame:
    """Aggregate multiple trials within a session to a single session value.

    Parameters
    ----------
    sessions_df : pd.DataFrame
        Validated sessions DataFrame.
    method : str, default 'mean'
        Aggregation method ('mean', 'max', or 'best').

    Returns
    -------
    pd.DataFrame
        DataFrame aggregated per (athlete_id, date, session_id, test, metric, side, unit).
    """
    if sessions_df.empty:
        return pd.DataFrame(
            columns=[
                "athlete_id",
                "date",
                "date_str",
                "session_id",
                "test",
                "metric",
                "side",
                "unit",
                "value",
                "n_trials",
            ]
        )

    possible_group_cols = [
        "athlete_id",
        "date",
        "date_str",
        "session_id",
        "test",
        "metric",
        "side",
        "unit",
    ]
    group_cols = [c for c in possible_group_cols if c in sessions_df.columns]
    
    if method == "mean":
        agg_df = (
            sessions_df.groupby(group_cols, as_index=False)
            .agg(value=("value", "mean"), n_trials=("trial", "count"))
        )
    elif method in ("max", "best"):
        agg_df = (
            sessions_df.groupby(group_cols, as_index=False)
            .agg(value=("value", "max"), n_trials=("trial", "count"))
        )
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    sort_cols = [c for c in ["athlete_id", "metric", "date"] if c in agg_df.columns]
    return agg_df.sort_values(sort_cols).reset_index(drop=True)
