"""Rolling personal baselines, Typical Error, SWC, and direction-aware z-scores.

Formulas:
- Rolling Mean: mu = (1 / N) * sum(x_i)
- Rolling Sample SD (ddof=1): sigma = sqrt((1 / (N - 1)) * sum((x_i - mu)^2))
- Smallest Worthwhile Change (Hopkins 2004): SWC = swc_multiplier * sigma (default 0.5 * SD_within)
- Typical Error (Hopkins 2000): TE = sigma / sqrt(2)
- Percent Change: pct_change = ((x_t - mu) / mu) * 100.0
- Direction-Aware Z-score:
    z = + (x_t - mu) / sigma  (if higher_is_better)
    z = - (x_t - mu) / sigma  (if lower_is_better)
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Optional

from shared.contract import METRIC_REGISTRY, MetricDirection
from p2_nmt_reports.config import BaselineConfig

logger = logging.getLogger(__name__)


def compute_rolling_baselines(
    session_agg_df: pd.DataFrame,
    config: Optional[BaselineConfig] = None,
    window_days: Optional[int] = None,
    min_sessions: Optional[int] = None,
) -> pd.DataFrame:
    """Compute rolling personal baseline statistics per athlete and metric.

    Parameters
    ----------
    session_agg_df : pd.DataFrame
        Session-aggregated DataFrame containing columns:
        ['athlete_id', 'date', 'session_id', 'metric', 'side', 'value'].
    config : BaselineConfig, optional
        Configuration object. If None, default BaselineConfig is used.
    window_days : int, optional
        Override for rolling window duration in calendar days (default 28).
    min_sessions : int, optional
        Override for minimum sessions required to establish baseline (default 3).

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with:
        ['baseline_mean', 'baseline_sd', 'baseline_n', 'baseline_established',
         'swc', 'typical_error', 'pct_change', 'z_score'].
    """
    if config is None:
        config = BaselineConfig()

    win_days = window_days if window_days is not None else config.rolling_window_days
    min_sess = min_sessions if min_sessions is not None else config.min_sessions_for_baseline
    swc_mult = config.swc_multiplier

    if session_agg_df.empty:
        out_df = session_agg_df.copy()
        for col in [
            "baseline_mean",
            "baseline_sd",
            "baseline_n",
            "baseline_established",
            "swc",
            "typical_error",
            "pct_change",
            "z_score",
        ]:
            out_df[col] = []
        return out_df

    df = session_agg_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["athlete_id", "metric", "side", "date"]).reset_index(drop=True)

    # Output columns to populate
    baseline_means = []
    baseline_sds = []
    baseline_ns = []
    baseline_established = []
    swcs = []
    typical_errors = []
    pct_changes = []
    z_scores = []

    # Process each athlete, metric, side group
    groups = df.groupby(["athlete_id", "metric", "side"], sort=False)

    for (ath, met, side), group in groups:
        dates = group["date"].values
        values = group["value"].values

        # Direction lookup from shared contract
        met_def = METRIC_REGISTRY.get(met)
        if met_def is not None and met_def.direction == MetricDirection.LOWER_IS_BETTER:
            dir_factor = -1.0
        else:
            dir_factor = 1.0

        for i in range(len(group)):
            cur_date = dates[i]
            cur_val = values[i]

            # Calendar window: [cur_date - win_days, cur_date]
            start_date = cur_date - np.timedelta64(win_days, "D")
            mask = (dates >= start_date) & (dates <= cur_date)
            window_vals = values[mask]
            n_in_win = len(window_vals)

            if n_in_win >= min_sess:
                mu = float(np.mean(window_vals))
                sd = float(np.std(window_vals, ddof=1)) if n_in_win > 1 else 0.0

                established = True
                swc = swc_mult * sd
                te = sd / np.sqrt(2.0)
                
                # Percent change from baseline mean
                pct_chg = ((cur_val - mu) / mu * 100.0) if abs(mu) > 1e-12 else 0.0
                
                # Direction-aware z-score
                if sd > 1e-9:
                    z = dir_factor * ((cur_val - mu) / sd)
                else:
                    z = 0.0
            else:
                mu = float(np.mean(window_vals)) if n_in_win > 0 else np.nan
                sd = float(np.std(window_vals, ddof=1)) if n_in_win > 1 else np.nan
                established = False
                swc = np.nan
                te = np.nan
                pct_chg = np.nan
                z = np.nan

            baseline_means.append(mu)
            baseline_sds.append(sd)
            baseline_ns.append(n_in_win)
            baseline_established.append(established)
            swcs.append(swc)
            typical_errors.append(te)
            pct_changes.append(pct_chg)
            z_scores.append(z)

    df["baseline_mean"] = baseline_means
    df["baseline_sd"] = baseline_sds
    df["baseline_n"] = baseline_ns
    df["baseline_established"] = baseline_established
    df["swc"] = swcs
    df["typical_error"] = typical_errors
    df["pct_change"] = pct_changes
    df["z_score"] = z_scores

    return df
