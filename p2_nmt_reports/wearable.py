"""Wearable Daily Join and Recovery Zone Tagging Module for Project 2.

Merges daily wearable telemetry (recovery_pct, hrv_ms, rhr_bpm, day_strain, sleep_hours)
with neuromuscular readiness indices on athlete_id and date.
"""

from __future__ import annotations
import logging
from typing import Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

from p2_nmt_reports.config import WearableConfig
from p2_nmt_reports.loader import load_wearable_daily

logger = logging.getLogger(__name__)


def classify_recovery_zone(
    recovery_pct: Optional[float],
    config: Optional[WearableConfig] = None,
) -> str:
    """Classify recovery percentage into Red, Yellow, or Green zone.

    Zones:
    - Red: < 34.0%
    - Yellow: 34.0% to 66.0%
    - Green: >= 67.0%
    """
    if recovery_pct is None or np.isnan(recovery_pct):
        return "Unknown"

    if config is None:
        config = WearableConfig()

    zones = config.recovery_zones
    if recovery_pct <= zones.red_upper:
        return "Red"
    elif recovery_pct < zones.green_lower:
        return "Yellow"
    else:
        return "Green"


def join_wearable_to_readiness(
    readiness_df: pd.DataFrame,
    wearable_df: Union[str, Path, pd.DataFrame],
    config: Optional[WearableConfig] = None,
) -> pd.DataFrame:
    """Join daily wearable telemetry to session readiness table on athlete_id and date.

    Parameters
    ----------
    readiness_df : pd.DataFrame
        DataFrame from compute_composite_readiness.
    wearable_df : str, Path, or pd.DataFrame
        Wearable telemetry DataFrame or path to wearable_daily.csv.
    config : WearableConfig, optional
        Wearable configuration.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with recovery zone, strain, and autonomic indicators.
    """
    if config is None:
        config = WearableConfig()

    w_df = load_wearable_daily(wearable_df)

    r_df = readiness_df.copy()
    r_df["date"] = pd.to_datetime(r_df["date"])
    w_df["date"] = pd.to_datetime(w_df["date"])

    # Left join readiness with wearable telemetry
    merged = pd.merge(
        r_df,
        w_df,
        on=["athlete_id", "date"],
        how="left",
    )

    # Add recovery zone classification
    merged["recovery_zone"] = [
        classify_recovery_zone(val, config) for val in merged["recovery_pct"]
    ]

    # Flag combined autonomic + neuromuscular deficit (Tier 2 signature preview)
    # Fired if acute drop in readiness AND red recovery zone
    merged["neuromuscular_autonomic_deficit"] = (
        merged["acute_drop_flag"] & (merged["recovery_zone"] == "Red")
    )

    return merged.sort_values(["athlete_id", "date"]).reset_index(drop=True)
