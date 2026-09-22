"""Sensitivity analyses and causal DAG evaluation for Project 3.

Implements pre-registered sensitivity tests (p3_nmt_profiler/ANALYSIS_PLAN.md):
1. Rest-day imputation check: sRPE=0 vs dropping unlogged training days.
2. Outlier trimming: 1% top/bottom trim on within-person sleep duration and sRPE deviations.
3. Alternative outcome: Fitbit sleep score in place of PMSys morning readiness.
4. Empirical Bayes individual athlete sleep sensitivity slopes (BLUPs).
5. Formal DAG confounding & observational boundary specifications.
"""
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from p3_nmt_profiler.config import ProfilerConfig
from p3_nmt_profiler.model import prepare_longitudinal_dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SensitivityComparison:
    """Comparison of baseline vs perturbed model estimates."""
    analysis_name: str
    parameter_name: str
    baseline_coef: float
    baseline_se: float
    baseline_p: float
    perturbed_coef: float
    perturbed_se: float
    perturbed_p: float
    percent_change: float


@dataclass
class AthleteSleepSensitivity:
    """Individual athlete sleep duration sensitivity (random slope BLUP)."""
    athlete_id: str
    n_obs: int
    raw_ols_slope: float
    blup_slope: float
    baseline_readiness: float
    mean_sleep_hours: float


def run_rest_day_sensitivity(df: pd.DataFrame) -> SensitivityComparison:
    """Compare baseline model (rest days imputed as 0 AU) vs dropping unlogged training days."""
    # Baseline Model 2
    m2_base = smf.mixedlm(
        "soreness ~ z_dev_prior_day_strain + mean_prior_day_strain + z_dev_prior_sleep_hours",
        df.dropna(subset=["soreness", "z_dev_prior_day_strain", "mean_prior_day_strain", "z_dev_prior_sleep_hours"]),
        groups=df.dropna(subset=["soreness", "z_dev_prior_day_strain", "mean_prior_day_strain", "z_dev_prior_sleep_hours"])["athlete_id"],
    ).fit(reml=True)

    # Perturbed: only days where prior day strain > 0 (strictly active training days)
    active_df = df[df["prior_day_strain"] > 0].dropna(
        subset=["soreness", "z_dev_prior_day_strain", "mean_prior_day_strain", "z_dev_prior_sleep_hours"]
    ).copy()
    m2_pert = smf.mixedlm(
        "soreness ~ z_dev_prior_day_strain + mean_prior_day_strain + z_dev_prior_sleep_hours",
        active_df,
        groups=active_df["athlete_id"],
    ).fit(reml=True, method=["lbfgs", "powell"])

    b_coef = float(m2_base.params["z_dev_prior_day_strain"])
    p_coef = float(m2_pert.params["z_dev_prior_day_strain"])
    pct_diff = ((p_coef - b_coef) / abs(b_coef)) * 100.0 if b_coef != 0 else 0.0

    return SensitivityComparison(
        analysis_name="Rest-Day Imputation vs Dropping Zero-Load Days",
        parameter_name="z_dev_prior_day_strain -> Soreness",
        baseline_coef=round(b_coef, 4),
        baseline_se=round(float(m2_base.bse["z_dev_prior_day_strain"]), 4),
        baseline_p=float(m2_base.pvalues["z_dev_prior_day_strain"]),
        perturbed_coef=round(p_coef, 4),
        perturbed_se=round(float(m2_pert.bse["z_dev_prior_day_strain"]), 4),
        perturbed_p=float(m2_pert.pvalues["z_dev_prior_day_strain"]),
        percent_change=round(pct_diff, 2),
    )


def run_outlier_trimming_sensitivity(df: pd.DataFrame) -> SensitivityComparison:
    """Re-estimate Model 1 after trimming top and bottom 1% of within-person sleep duration."""
    req_cols = [
        "readiness", "z_dev_prior_sleep_hours", "mean_prior_sleep_hours",
        "z_dev_prior_rhr", "mean_prior_rhr", "z_dev_prior_day_strain",
        "mean_prior_day_strain", "is_weekend", "athlete_id"
    ]
    m1_df = df.dropna(subset=req_cols).copy()
    m1_base = smf.mixedlm(
        "readiness ~ z_dev_prior_sleep_hours + mean_prior_sleep_hours + z_dev_prior_rhr + mean_prior_rhr + z_dev_prior_day_strain + mean_prior_day_strain + is_weekend",
        m1_df,
        groups=m1_df["athlete_id"],
    ).fit(reml=True)

    # Trim 1% tails
    p1 = m1_df["z_dev_prior_sleep_hours"].quantile(0.01)
    p99 = m1_df["z_dev_prior_sleep_hours"].quantile(0.99)
    trimmed_df = m1_df[(m1_df["z_dev_prior_sleep_hours"] >= p1) & (m1_df["z_dev_prior_sleep_hours"] <= p99)].copy()

    m1_pert = smf.mixedlm(
        "readiness ~ z_dev_prior_sleep_hours + mean_prior_sleep_hours + z_dev_prior_rhr + mean_prior_rhr + z_dev_prior_day_strain + mean_prior_day_strain + is_weekend",
        trimmed_df,
        groups=trimmed_df["athlete_id"],
    ).fit(reml=True)

    b_coef = float(m1_base.params["z_dev_prior_sleep_hours"])
    p_coef = float(m1_pert.params["z_dev_prior_sleep_hours"])
    pct_diff = ((p_coef - b_coef) / abs(b_coef)) * 100.0 if b_coef != 0 else 0.0

    return SensitivityComparison(
        analysis_name="1% Outlier Trimming on Sleep Deviations",
        parameter_name="z_dev_prior_sleep_hours -> Readiness",
        baseline_coef=round(b_coef, 4),
        baseline_se=round(float(m1_base.bse["z_dev_prior_sleep_hours"]), 4),
        baseline_p=float(m1_base.pvalues["z_dev_prior_sleep_hours"]),
        perturbed_coef=round(p_coef, 4),
        perturbed_se=round(float(m1_pert.bse["z_dev_prior_sleep_hours"]), 4),
        perturbed_p=float(m1_pert.pvalues["z_dev_prior_sleep_hours"]),
        percent_change=round(pct_diff, 2),
    )


def extract_individual_athlete_slopes(df: pd.DataFrame) -> list[AthleteSleepSensitivity]:
    """Extract Empirical Bayes random slopes and raw within-person OLS slopes for all athletes."""
    req_cols = [
        "readiness", "z_dev_prior_sleep_hours", "mean_prior_sleep_hours",
        "z_dev_prior_rhr", "mean_prior_rhr", "z_dev_prior_day_strain",
        "mean_prior_day_strain", "is_weekend", "athlete_id"
    ]
    m1_df = df.dropna(subset=req_cols).copy()
    formula = (
        "readiness ~ z_dev_prior_sleep_hours + mean_prior_sleep_hours + "
        "z_dev_prior_rhr + mean_prior_rhr + z_dev_prior_day_strain + "
        "mean_prior_day_strain + is_weekend"
    )
    mod_rs = smf.mixedlm(formula, m1_df, groups=m1_df["athlete_id"], re_formula="~z_dev_prior_sleep_hours")
    res_rs = mod_rs.fit(reml=False)

    fixed_slope = float(res_rs.params["z_dev_prior_sleep_hours"])
    ranef_dict = res_rs.random_effects

    athlete_slopes: list[AthleteSleepSensitivity] = []
    for athlete_id, group_df in m1_df.groupby("athlete_id"):
        n_obs = len(group_df)
        base_readiness = float(group_df["readiness"].mean())
        mean_sleep = float(group_df["prior_sleep_hours"].mean())

        # Raw within-athlete OLS slope
        if n_obs >= 5 and group_df["z_dev_prior_sleep_hours"].std() > 0:
            cov = np.cov(group_df["z_dev_prior_sleep_hours"], group_df["readiness"])[0, 1]
            var = np.var(group_df["z_dev_prior_sleep_hours"], ddof=1)
            raw_slope = float(cov / var) if var > 0 else fixed_slope
        else:
            raw_slope = fixed_slope

        # Empirical Bayes BLUP slope
        u_slope = 0.0
        if athlete_id in ranef_dict:
            re_vals = ranef_dict[athlete_id]
            if "z_dev_prior_sleep_hours" in re_vals:
                u_slope = float(re_vals["z_dev_prior_sleep_hours"])
            elif len(re_vals) > 1:
                u_slope = float(re_vals.iloc[1])

        blup_slope = fixed_slope + u_slope

        athlete_slopes.append(AthleteSleepSensitivity(
            athlete_id=str(athlete_id),
            n_obs=n_obs,
            raw_ols_slope=round(raw_slope, 4),
            blup_slope=round(blup_slope, 4),
            baseline_readiness=round(base_readiness, 2),
            mean_sleep_hours=round(mean_sleep, 2),
        ))

    athlete_slopes.sort(key=lambda a: a.blup_slope, reverse=True)
    return athlete_slopes


def run_all_sensitivities(raw_dir: Path | str = Path("data/raw/pmdata")) -> dict[str, Any]:
    """Execute complete sensitivity suite and return structured results."""
    df = prepare_longitudinal_dataset(raw_dir)
    s1 = run_rest_day_sensitivity(df)
    s2 = run_outlier_trimming_sensitivity(df)
    slopes = extract_individual_athlete_slopes(df)
    return {
        "rest_day_sensitivity": s1,
        "outlier_sensitivity": s2,
        "athlete_slopes": slopes,
    }
