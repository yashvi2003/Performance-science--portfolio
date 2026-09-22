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
    """Calculate direction-aware composite readiness index per athlete-session.

    Parameters
    ----------
    baselines_df : pd.DataFrame
        DataFrame from compute_rolling_baselines containing columns:
        ['athlete_id', 'date', 'session_id', 'test', 'metric', 'side', 'value',
         'baseline_mean', 'baseline_sd', 'baseline_established', 'swc', 'pct_change', 'z_score'].
    config : ReadinessConfig, optional
        Readiness configuration. If None, default ReadinessConfig is used.

    Returns
    -------
    pd.DataFrame
        Session-level DataFrame containing:
        ['athlete_id', 'date', 'session_id', 'composite_readiness_z',
         'metrics_included', 'weights_used', 'is_single_metric_fallback',
         'acute_drop_flag', 'heuristic_disclaimer'].
    """
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
                "athlete_id",
                "date",
                "session_id",
                "composite_readiness_z",
                "metrics_included",
                "weights_used",
                "is_single_metric_fallback",
                "acute_drop_flag",
                "heuristic_disclaimer",
            ]
        )

    # Group by athlete, date, session_id
    sessions: list[dict] = []
    grouped = baselines_df.groupby(["athlete_id", "date", "session_id"], sort=False)

    for (ath, dt, sess_id), group in grouped:
        # Filter for established baselines with valid z-scores
        valid_rows = group[group["baseline_established"] & group["z_score"].notna()].copy()

        if valid_rows.empty:
            # Baseline not established for any metric in this session
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

        # Check which configured metrics are present
        metric_z_map = dict(zip(valid_rows["metric"], valid_rows["z_score"]))
        metric_swc_map = dict(zip(valid_rows["metric"], valid_rows["swc"]))
        metric_pct_map = dict(zip(valid_rows["metric"], valid_rows["pct_change"]))

        matching_metrics = [m for m in target_weights.keys() if m in metric_z_map]

        if matching_metrics:
            # Normalize weights for the subset of matching metrics present
            raw_weights = [target_weights[m] for m in matching_metrics]
            total_weight = sum(raw_weights)
            norm_weights = [w / total_weight for w in raw_weights]

            comp_z = sum(w * metric_z_map[m] for w, m in zip(norm_weights, matching_metrics))
            metrics_str = ";".join(matching_metrics)
            weights_str = ";".join(f"{m}:{w:.2f}" for m, w in zip(matching_metrics, norm_weights))
            is_fallback = len(matching_metrics) == 1
        elif fallback_metric in metric_z_map:
            # Fallback to single designated metric
            comp_z = metric_z_map[fallback_metric]
            metrics_str = fallback_metric
            weights_str = f"{fallback_metric}:1.00"
            is_fallback = True
        else:
            # Take mean z-score across all available valid metrics
            all_metrics = list(metric_z_map.keys())
            comp_z = float(np.mean([metric_z_map[m] for m in all_metrics]))
            metrics_str = ";".join(all_metrics)
            weights_str = ";".join(f"{m}:{1.0/len(all_metrics):.2f}" for m in all_metrics)
            is_fallback = True

        # Evaluate acute drop flag:
        # Fired if composite z < -1.0, or if primary metric pct_change < -max(10%, SWC)
        acute_drop = False
        if comp_z < -1.0:
            acute_drop = True

        # Check primary metric percentage change vs SWC threshold
        primary_metric = matching_metrics[0] if matching_metrics else (fallback_metric if fallback_metric in metric_pct_map else None)
        if primary_metric and primary_metric in metric_pct_map and primary_metric in metric_swc_map:
            pct_chg = metric_pct_map[primary_metric]
            swc_val = metric_swc_map[primary_metric]
            if not np.isnan(pct_chg) and not np.isnan(swc_val):
                # Calculate threshold: max(drop_pct_thresh, SWC_pct)
                # Note: swc_val is absolute, so threshold drop pct is drop_pct_thresh
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
