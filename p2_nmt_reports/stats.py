"""Within-Athlete Mixed-Effects Session Change Model for Project 2.

Fits a Linear Mixed-Effects Model (LMM via statsmodels) modeling session-to-session
neuromuscular readiness fluctuations with random intercepts per athlete.

Scientific Framing & Purpose:
Quantifies within-athlete neuromuscular changes across repeated sessions,
decomposes variance components into between-athlete and residual within-athlete noise,
and derives individual Minimal Detectable Change (MDC95) thresholds.

DISCLAIMER:
This analysis runs on clearly labelled synthetic longitudinal data to demonstrate
the statistical modeling pipeline and error framework. It is NOT an empirical research finding.
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
    """Fit a linear mixed-effects model to longitudinal session monitoring data.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Longitudinal DataFrame containing athlete_id, date, outcome_col, and predictor_cols.
    outcome_col : str, default 'composite_readiness_z'
        Outcome variable to model.
    predictor_cols : list[str], optional
        Predictor columns. Defaults to ['recovery_pct', 'day_strain', 'sleep_hours'] if present.
    reml : bool, default True
        Whether to estimate parameters via Restricted Maximum Likelihood.

    Returns
    -------
    dict
        Model summary dictionary with fixed effects, variance components, MDC95, and diagnostics.
    """
    if predictor_cols is None:
        avail = [c for c in ["recovery_pct", "day_strain", "sleep_hours"] if c in merged_df.columns]
        predictor_cols = avail if avail else []

    cols_needed = ["athlete_id", outcome_col] + predictor_cols
    df = merged_df[cols_needed].dropna().copy()

    if len(df) < 10 or df["athlete_id"].nunique() < 2:
        raise ValueError(
            f"Insufficient data to fit mixed-effects model: {len(df)} rows across {df['athlete_id'].nunique()} athletes."
        )

    # Standardize predictors to mean 0, SD 1 for interpretable effect sizes
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

    logger.info("Fitting LMM formula: %s with groups=athlete_id (REML=%s)", formula, reml)
    try:
        model = smf.mixedlm(formula, df, groups=df["athlete_id"]).fit(reml=reml, method=["lbfgs", "powell"])
    except Exception as e:
        logger.warning("Primary optimizer failed, falling back to default fit: %s", e)
        model = smf.mixedlm(formula, df, groups=df["athlete_id"]).fit(reml=reml)

    # Extract fixed effects
    fe_params = model.params.to_dict()
    bse = model.bse.to_dict()
    pvalues = model.pvalues.to_dict()
    conf_int = model.conf_int()

    fixed_effects_summary = []
    for term in model.params.index:
        if term in ("Group Var", "Intercept"):
            pass
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

    # Extract variance components
    group_var = float(model.cov_re.iloc[0, 0]) if hasattr(model, "cov_re") else float(fe_params.get("Group Var", np.nan))
    residual_var = float(model.scale)
    total_var = group_var + residual_var
    icc = group_var / total_var if total_var > 0 else np.nan

    # Calculate Standard Error of Measurement (SEM) and MDC95
    # SEM = sqrt(residual_variance), MDC95 = 1.96 * sqrt(2) * SEM
    sem = np.sqrt(residual_var)
    mdc95 = 1.96 * np.sqrt(2.0) * sem

    # Residual diagnostics
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
