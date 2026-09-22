"""Linear Mixed-Effects Models for longitudinal load, recovery, and readiness profiling.

Implements the pre-registered analysis plan (p3_nmt_profiler/ANALYSIS_PLAN.md):
1. Prepares calendar-aligned complete daily time series from PMData.
2. Performs Currie / Hoffmann Person-Mean Centering Decomposition:
   - Between-person trait means (mean_X_i)
   - Within-person daily fluctuations (dev_X_it)
   - Standardized within-person deviations (z_dev_X_it)
3. Estimates Linear Mixed-Effects Models (LMM) via Restricted Maximum Likelihood (REML):
   - Model 1: Morning Readiness (Sleep duration, RHR, Training load, Weekend)
   - Model 2: Morning Muscle Soreness (Training load, Sleep duration)
   - Model 3: Morning Fatigue / Energy (Sleep score, Training load, RHR)
4. Evaluates random slopes via Likelihood Ratio Tests (LRT).
5. Computes diagnostic metrics (VIF multicollinearity, residual normality, autocorrelation).
"""
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from p3_nmt_profiler.config import ProfilerConfig
from p3_nmt_profiler.pmdata_adapter import (
    load_participant_wellness,
    load_participant_srpe,
    load_participant_sleep,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixedEffectResult:
    """Fixed-effect coefficient summary with inference statistics."""
    name: str
    coef: float
    std_err: float
    z_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class ModelDiagnostics:
    """Statistical diagnostics for linear mixed-effects models."""
    vif_dict: dict[str, float]
    residual_skewness: float
    residual_kurtosis: float
    normality_jb_pvalue: float
    residual_autocorr_lag1: float


@dataclass
class LMMFitResult:
    """Comprehensive container for a fitted Linear Mixed-Effects Model."""
    model_name: str
    dependent_var: str
    n_obs: int
    n_groups: int
    reml_loglik: float
    aic: float
    bic: float
    scale_residual_var: float
    group_var: float
    fixed_effects: list[FixedEffectResult]
    diagnostics: ModelDiagnostics
    fitted_model: Any = field(repr=False)
    data: pd.DataFrame = field(repr=False)


def prepare_longitudinal_dataset(
    raw_dir: Path | str = Path("data/raw/pmdata"),
    cfg: ProfilerConfig | None = None,
) -> pd.DataFrame:
    """Build a complete, calendar-aligned daily dataset with Currie-Hoffmann centering.

    Parameters
    ----------
    raw_dir : Path or str
        Path to PMData raw directory.
    cfg : ProfilerConfig, optional
        Configuration settings.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with raw, mean, dev, and z_dev columns.
    """
    if cfg is None:
        cfg = ProfilerConfig.load()

    raw_dir = Path(raw_dir)
    participant_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("p")])
    if not participant_dirs:
        raise ValueError(f"No participant folders found in {raw_dir}")

    records: list[pd.DataFrame] = []
    for p_dir in participant_dirs:
        pid = p_dir.name
        w = load_participant_wellness(p_dir / "pmsys")
        s = load_participant_sleep(p_dir / "fitbit")
        srpe = load_participant_srpe(p_dir / "pmsys")
        if w.empty:
            continue

        # Extract complete date coverage for this athlete
        dates_set = set(w["date"].dropna())
        if not s.empty:
            dates_set.update(s["date"].dropna())
        if not srpe.empty:
            dates_set.update(srpe["date"].dropna())

        if not dates_set:
            continue

        all_d = sorted(list(dates_set))
        min_date, max_date = pd.to_datetime(all_d[0]), pd.to_datetime(all_d[-1])
        full_dates = pd.date_range(min_date, max_date, freq="D").strftime("%Y-%m-%d")
        base_df = pd.DataFrame({"date": full_dates})

        # Merge daily tables
        m = pd.merge(base_df, w, on="date", how="left")
        if not s.empty:
            m = pd.merge(m, s, on="date", how="left")
        if not srpe.empty:
            m = pd.merge(m, srpe, on="date", how="left")
            m["day_strain"] = m["day_strain"].fillna(0.0)
        else:
            m["day_strain"] = 0.0

        m["athlete_id"] = pid
        m["date_dt"] = pd.to_datetime(m["date"])
        m["is_weekend"] = (m["date_dt"].dt.dayofweek >= 5).astype(int)

        # 1. Prior-day training load (sRPE from day D-1)
        m["prior_day_strain"] = m["day_strain"].shift(1)

        # 2. Prior-night sleep duration (from morning survey on day D)
        m["prior_sleep_hours"] = pd.to_numeric(m["sleep_duration_h"], errors="coerce")

        # 3. Prior-night resting heart rate & sleep score (from Fitbit on day D)
        m["prior_rhr"] = m["resting_heart_rate"] if "resting_heart_rate" in m.columns else np.nan
        m["prior_sleep_score"] = m["overall_score"] if "overall_score" in m.columns else np.nan

        records.append(m)

    if not records:
        raise RuntimeError("No participant records loaded from PMData.")

    df = pd.concat(records, ignore_index=True)

    # Biological plausibility filtering per pre-registration
    df.loc[(df["prior_sleep_hours"] < 2.0) | (df["prior_sleep_hours"] > 16.0), "prior_sleep_hours"] = np.nan
    df.loc[(df["prior_rhr"] < 35.0) | (df["prior_rhr"] > 110.0), "prior_rhr"] = np.nan
    df.loc[(df["prior_day_strain"] < 0.0) | (df["prior_day_strain"] > 3000.0), "prior_day_strain"] = np.nan

    # Currie / Hoffmann Person-Mean Centering Decomposition
    predictors = ["prior_sleep_hours", "prior_rhr", "prior_day_strain", "prior_sleep_score"]
    for col in predictors:
        if col in df.columns:
            df[f"mean_{col}"] = df.groupby("athlete_id")[col].transform("mean")
            df[f"dev_{col}"] = df[col] - df[f"mean_{col}"]
            sd_within = df[f"dev_{col}"].std()
            if pd.notna(sd_within) and sd_within > 0:
                df[f"z_dev_{col}"] = df[f"dev_{col}"] / sd_within
            else:
                df[f"z_dev_{col}"] = df[f"dev_{col}"]

    return df


def compute_model_diagnostics(
    res: Any,
    data: pd.DataFrame,
    formula: str,
) -> ModelDiagnostics:
    """Compute VIF, residual normality, and lag-1 autocorrelation diagnostics."""
    residuals = res.resid.dropna().to_numpy()
    
    # 1. Residual normality (Jarque-Bera test)
    jb_stat, jb_pvalue, skew, kurt = sm.stats.stattools.jarque_bera(residuals)

    # 2. Multicollinearity (VIF) on fixed-effects design matrix
    rhs = formula.split("~")[1].strip()
    terms = [t.strip() for t in rhs.split("+")]
    vif_data = data[terms].dropna()
    vif_dict: dict[str, float] = {}
    if len(vif_data) > len(terms):
        try:
            X_const = sm.add_constant(vif_data)
            for term in terms:
                col_idx = X_const.columns.get_loc(term)
                v = float(variance_inflation_factor(X_const.values, col_idx))
                vif_dict[term] = round(v, 2)
        except Exception as exc:
            logger.warning("VIF calculation encountered an issue: %s", exc)

    # 3. Lag-1 autocorrelation of residuals within athlete
    data_res = data.copy()
    data_res["resid"] = res.resid
    autocorrs = []
    for _, g in data_res.groupby("athlete_id"):
        r_series = g["resid"].dropna()
        if len(r_series) > 3:
            ac = r_series.autocorr(lag=1)
            if pd.notna(ac):
                autocorrs.append(ac)
    mean_ac = float(np.mean(autocorrs)) if autocorrs else float("nan")

    return ModelDiagnostics(
        vif_dict=vif_dict,
        residual_skewness=round(float(skew), 3),
        residual_kurtosis=round(float(kurt), 3),
        normality_jb_pvalue=float(jb_pvalue),
        residual_autocorr_lag1=round(mean_ac, 3),
    )


def extract_fit_result(
    model_name: str,
    dependent_var: str,
    res: Any,
    data: pd.DataFrame,
    formula: str,
) -> LMMFitResult:
    """Extract standard parameters and structured inference metrics from MixedLMResults."""
    conf = res.conf_int()
    fixed_effects: list[FixedEffectResult] = []
    
    for term in res.params.index:
        if term == "Group Var":
            continue
        fe = FixedEffectResult(
            name=term,
            coef=round(float(res.params[term]), 4),
            std_err=round(float(res.bse[term]), 4),
            z_stat=round(float(res.tvalues[term]), 3),
            p_value=float(res.pvalues[term]),
            ci_lower=round(float(conf.loc[term, 0]), 4),
            ci_upper=round(float(conf.loc[term, 1]), 4),
        )
        fixed_effects.append(fe)

    group_var = float(res.cov_re.iloc[0, 0]) if hasattr(res, "cov_re") and not res.cov_re.empty else 0.0
    diagnostics = compute_model_diagnostics(res, data, formula)

    # AIC / BIC estimation
    k = len(res.params) + 1  # Fixed effects + residual var + random intercept
    n = res.nobs
    loglik = float(res.llf)
    aic = round(-2.0 * loglik + 2.0 * k, 2)
    bic = round(-2.0 * loglik + k * np.log(n), 2)

    return LMMFitResult(
        model_name=model_name,
        dependent_var=dependent_var,
        n_obs=int(res.nobs),
        n_groups=int(data["athlete_id"].nunique()),
        reml_loglik=round(loglik, 2),
        aic=aic,
        bic=bic,
        scale_residual_var=round(float(res.scale), 4),
        group_var=round(group_var, 4),
        fixed_effects=fixed_effects,
        diagnostics=diagnostics,
        fitted_model=res,
        data=data,
    )


def fit_model_1_readiness(
    df: pd.DataFrame,
) -> LMMFitResult:
    """Fit Model 1: Determinants of Next-Day Morning Subjective Readiness.

    Formula:
    Readiness ~ z_dev_sleep + mean_sleep + z_dev_rhr + mean_rhr + z_dev_load + mean_load + is_weekend + (1|Athlete)
    """
    req_cols = [
        "readiness",
        "z_dev_prior_sleep_hours",
        "mean_prior_sleep_hours",
        "z_dev_prior_rhr",
        "mean_prior_rhr",
        "z_dev_prior_day_strain",
        "mean_prior_day_strain",
        "is_weekend",
        "athlete_id",
    ]
    model_df = df.dropna(subset=req_cols).copy()
    formula = (
        "readiness ~ z_dev_prior_sleep_hours + mean_prior_sleep_hours + "
        "z_dev_prior_rhr + mean_prior_rhr + z_dev_prior_day_strain + "
        "mean_prior_day_strain + is_weekend"
    )
    mod = smf.mixedlm(formula, model_df, groups=model_df["athlete_id"])
    res = mod.fit(reml=True)
    return extract_fit_result(
        model_name="Model 1: Next-Day Readiness",
        dependent_var="readiness",
        res=res,
        data=model_df,
        formula=formula,
    )


def fit_model_2_soreness(
    df: pd.DataFrame,
) -> LMMFitResult:
    """Fit Model 2: Determinants of Next-Day Muscle Soreness.

    Formula:
    Soreness ~ z_dev_load + mean_load + z_dev_sleep + (1|Athlete)
    """
    req_cols = [
        "soreness",
        "z_dev_prior_day_strain",
        "mean_prior_day_strain",
        "z_dev_prior_sleep_hours",
        "athlete_id",
    ]
    model_df = df.dropna(subset=req_cols).copy()
    formula = "soreness ~ z_dev_prior_day_strain + mean_prior_day_strain + z_dev_prior_sleep_hours"
    mod = smf.mixedlm(formula, model_df, groups=model_df["athlete_id"])
    res = mod.fit(reml=True)
    return extract_fit_result(
        model_name="Model 2: Next-Day Muscle Soreness",
        dependent_var="soreness",
        res=res,
        data=model_df,
        formula=formula,
    )


def fit_model_3_fatigue(
    df: pd.DataFrame,
) -> LMMFitResult:
    """Fit Model 3: Determinants of Next-Day Morning Fatigue.

    Formula:
    Fatigue ~ z_dev_sleep_score + z_dev_load + z_dev_rhr + (1|Athlete)
    """
    req_cols = [
        "fatigue",
        "z_dev_prior_sleep_score",
        "z_dev_prior_day_strain",
        "z_dev_prior_rhr",
        "athlete_id",
    ]
    model_df = df.dropna(subset=req_cols).copy()
    formula = "fatigue ~ z_dev_prior_sleep_score + z_dev_prior_day_strain + z_dev_prior_rhr"
    mod = smf.mixedlm(formula, model_df, groups=model_df["athlete_id"])
    res = mod.fit(reml=True)
    return extract_fit_result(
        model_name="Model 3: Next-Day Fatigue",
        dependent_var="fatigue",
        res=res,
        data=model_df,
        formula=formula,
    )


def evaluate_random_slopes(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Test whether random slopes for acute sleep deviation improve Model 1 fit."""
    req_cols = [
        "readiness",
        "z_dev_prior_sleep_hours",
        "mean_prior_sleep_hours",
        "z_dev_prior_rhr",
        "mean_prior_rhr",
        "z_dev_prior_day_strain",
        "mean_prior_day_strain",
        "is_weekend",
        "athlete_id",
    ]
    model_df = df.dropna(subset=req_cols).copy()
    formula = (
        "readiness ~ z_dev_prior_sleep_hours + mean_prior_sleep_hours + "
        "z_dev_prior_rhr + mean_prior_rhr + z_dev_prior_day_strain + "
        "mean_prior_day_strain + is_weekend"
    )
    
    # 1. Random Intercept model (via ML for likelihood ratio test)
    mod_ri = smf.mixedlm(formula, model_df, groups=model_df["athlete_id"])
    res_ri = mod_ri.fit(reml=False)

    # 2. Random Slope model for sleep deviation
    try:
        mod_rs = smf.mixedlm(
            formula,
            model_df,
            groups=model_df["athlete_id"],
            re_formula="~z_dev_prior_sleep_hours",
        )
        res_rs = mod_rs.fit(reml=False)
        # Likelihood Ratio Test
        lrt_stat = float(2.0 * (res_rs.llf - res_ri.llf))
        from scipy.stats import chi2
        p_val = float(chi2.sf(max(0.0, lrt_stat), df=2))
        return {
            "lrt_stat": round(lrt_stat, 3),
            "p_value": p_val,
            "ri_aic": round(float(res_ri.aic), 2),
            "rs_aic": round(float(res_rs.aic), 2),
            "supports_random_slopes": bool(p_val < 0.05 and lrt_stat > 0),
        }
    except Exception as exc:
        logger.info("Random slope model estimation note: %s", exc)
        return {
            "lrt_stat": 0.0,
            "p_value": 1.0,
            "supports_random_slopes": False,
            "note": str(exc),
        }


def run_all_models(
    raw_dir: Path | str = Path("data/raw/pmdata"),
    cfg: ProfilerConfig | None = None,
) -> tuple[LMMFitResult, LMMFitResult, LMMFitResult, dict[str, Any]]:
    """Execute the full pre-registered Linear Mixed-Effects modeling pipeline."""
    df = prepare_longitudinal_dataset(raw_dir, cfg=cfg)
    m1 = fit_model_1_readiness(df)
    m2 = fit_model_2_soreness(df)
    m3 = fit_model_3_fatigue(df)
    rs_test = evaluate_random_slopes(df)
    return m1, m2, m3, rs_test
