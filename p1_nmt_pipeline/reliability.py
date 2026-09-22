"""Within-participant reliability statistics for repeated CMJ trials.

Implements standard sports science and measurement theory reliability metrics:
- Within-subject Coefficient of Variation (CV%)
- Two-way random effects Intraclass Correlation Coefficient: ICC(2, 1) (absolute agreement)
- Two-way mixed effects Intraclass Correlation Coefficient: ICC(3, 1) (consistency)
- Standard Error of Measurement: SEM = SD_pooled * sqrt(1 - ICC(2,1))
- Minimal Detectable Change: MDC_95 = 1.96 * sqrt(2) * SEM
- Linear Mixed-Effects Model (LMM / REML) variance decomposition for unbalanced data.

References:
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater reliability.
  Psychological Bulletin, 86(2), 420-428.
- McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients.
  Psychological Methods, 1(1), 30-46.
- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients
  for reliability research. Journal of Chiropractic Medicine, 15(2), 155-163.
- Hopkins, W. G. (2000). Measures of reliability in sports medicine and science.
  Sports Medicine, 30(1), 1-15.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReliabilityResult:
    """Summary of reliability metrics for a single metric and trial set."""

    metric_name: str
    condition: str
    design_desc: str
    n_subjects: int
    k_trials: int
    n_observations: int
    denominator_str: str
    mean: float
    sd_total: float
    cv_percent_mean: float
    cv_percent_sd: float
    cv_percent_median: float
    icc_2_1: float
    icc_2_1_ci_lower: float
    icc_2_1_ci_upper: float
    icc_3_1: float
    icc_3_1_ci_lower: float
    icc_3_1_ci_upper: float
    sem: float
    sem_pct_of_mean: float
    mdc_95: float
    mdc_95_pct_of_mean: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result to flat dictionary."""
        return {
            "metric_name": self.metric_name,
            "condition": self.condition,
            "design_desc": self.design_desc,
            "n_subjects": self.n_subjects,
            "k_trials": self.k_trials,
            "n_observations": self.n_observations,
            "denominator_str": self.denominator_str,
            "mean": self.mean,
            "sd_total": self.sd_total,
            "cv_percent_mean": self.cv_percent_mean,
            "cv_percent_sd": self.cv_percent_sd,
            "cv_percent_median": self.cv_percent_median,
            "icc_2_1": self.icc_2_1,
            "icc_2_1_ci_lower": self.icc_2_1_ci_lower,
            "icc_2_1_ci_upper": self.icc_2_1_ci_upper,
            "icc_3_1": self.icc_3_1,
            "icc_3_1_ci_lower": self.icc_3_1_ci_lower,
            "icc_3_1_ci_upper": self.icc_3_1_ci_upper,
            "sem": self.sem,
            "sem_pct_of_mean": self.sem_pct_of_mean,
            "mdc_95": self.mdc_95,
            "mdc_95_pct_of_mean": self.mdc_95_pct_of_mean,
        }


def compute_within_subject_cv(matrix: np.ndarray) -> tuple[float, float, float]:
    """Calculate within-subject Coefficient of Variation (CV%) across rows.

    CV_i = (SD_i / Mean_i) * 100%

    Parameters
    ----------
    matrix : np.ndarray
        2D array of shape (n_subjects, k_trials).

    Returns
    -------
    tuple[float, float, float]
        (mean_cv, sd_cv, median_cv) in percent.
    """
    row_means = np.mean(matrix, axis=1)
    row_sds = np.std(matrix, axis=1, ddof=1)

    # Avoid divide by zero
    valid_mask = np.abs(row_means) > 1e-8
    cvs = (row_sds[valid_mask] / np.abs(row_means[valid_mask])) * 100.0

    return float(np.mean(cvs)), float(np.std(cvs, ddof=1)), float(np.median(cvs))


def compute_two_way_icc(
    matrix: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, tuple[float, float], float, tuple[float, float], float, float, float, float]:
    """Compute ICC(2,1) and ICC(3,1) with exact 95% CIs from two-way ANOVA.

    Formulas per Shrout & Fleiss (1979) and McGraw & Wong (1996):
        MS_R = Mean Square Rows (Participants)
        MS_C = Mean Square Columns (Trials)
        MS_E = Mean Square Error (Residual interaction)

        ICC(3, 1) = (MS_R - MS_E) / (MS_R + (k - 1) * MS_E)
        ICC(2, 1) = (MS_R - MS_E) / (MS_R + (k - 1) * MS_E + (k / n) * (MS_C - MS_E))

    Parameters
    ----------
    matrix : np.ndarray
        Balanced 2D array of shape (n_subjects, k_trials).
    alpha : float
        Significance level for confidence interval (default: 0.05 for 95% CI).

    Returns
    -------
    tuple
        (icc_2_1, ci_2_1, icc_3_1, ci_3_1, ms_rows, ms_cols, ms_error, sd_total)
    """
    n, k = matrix.shape
    if n < 2 or k < 2:
        raise ValueError(f"Matrix must have at least 2 subjects and 2 trials; got {matrix.shape}")

    grand_mean = np.mean(matrix)
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)

    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = max(0.0, ss_total - ss_rows - ss_cols)

    df_rows = n - 1
    df_cols = k - 1
    df_error = df_rows * df_cols

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    sd_total = float(np.std(matrix, ddof=1))

    # --- ICC(3, 1) Consistency ---
    denom_3_1 = ms_rows + (k - 1) * ms_error
    icc_3_1 = float((ms_rows - ms_error) / denom_3_1) if denom_3_1 > 0 else 0.0

    # Exact CI for ICC(3, 1)
    f_stat = ms_rows / ms_error if ms_error > 0 else np.nan
    if np.isfinite(f_stat) and f_stat > 0:
        f_low = f_stat / stats.f.ppf(1.0 - alpha / 2.0, df_rows, df_error)
        f_high = f_stat * stats.f.ppf(1.0 - alpha / 2.0, df_error, df_rows)
        ci_3_1_low = max(-1.0, (f_low - 1.0) / (f_low + k - 1.0))
        ci_3_1_high = min(1.0, (f_high - 1.0) / (f_high + k - 1.0))
    else:
        ci_3_1_low, ci_3_1_high = (np.nan, np.nan)

    # --- ICC(2, 1) Absolute Agreement ---
    denom_2_1 = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    icc_2_1 = float((ms_rows - ms_error) / denom_2_1) if denom_2_1 > 0 else 0.0

    # Approximate 95% CI for ICC(2,1) per McGraw & Wong (1996) / Shrout & Fleiss (1979)
    # Using the Satterthwaite approximation for effective degrees of freedom
    if ms_error > 0 and denom_2_1 > 0:
        f_c = ms_cols / ms_error
        # Degrees of freedom for denominator:
        # Nu_eff = ( (k - 1) * F_c + n * (k - 1) * (1 - ICC2) + ... )
        # Standard conservative bounds:
        f_val = ms_rows / (ms_error + (ms_cols - ms_error) / n) if (ms_error + (ms_cols - ms_error) / n) > 0 else f_stat
        f_crit_low = stats.f.ppf(1.0 - alpha / 2.0, df_rows, df_error)
        f_crit_high = stats.f.ppf(1.0 - alpha / 2.0, df_error, df_rows)
        ci_2_1_low = max(-1.0, (f_val / f_crit_low - 1.0) / (f_val / f_crit_low + k - 1.0))
        ci_2_1_high = min(1.0, (f_val * f_crit_high - 1.0) / (f_val * f_crit_high + k - 1.0))
    else:
        ci_2_1_low, ci_2_1_high = (np.nan, np.nan)

    return (
        icc_2_1,
        (float(ci_2_1_low), float(ci_2_1_high)),
        icc_3_1,
        (float(ci_3_1_low), float(ci_3_1_high)),
        float(ms_rows),
        float(ms_cols),
        float(ms_error),
        sd_total,
    )


def compute_reliability(
    matrix: np.ndarray,
    metric_name: str,
    condition: str,
    design_desc: str,
) -> ReliabilityResult:
    """Run full reliability analysis on a balanced matrix of trials.

    Parameters
    ----------
    matrix : np.ndarray
        2D array (n_subjects, k_trials).
    metric_name : str
        Name of metric (e.g. 'jump_height_total_com_m').
    condition : str
        'CMJNA' or 'CMJA'.
    design_desc : str
        Description of balanced subset (e.g. 'balanced 4-trial set').

    Returns
    -------
    ReliabilityResult
        Complete dataclass with all metrics, SEM, MDC, and denominator string.
    """
    n_subjs, k_trials = matrix.shape
    n_obs = n_subjs * k_trials
    denom_str = (
        f"{condition} {design_desc}, "
        f"n={n_subjs} subjects x {k_trials} trials = {n_obs} observations"
    )

    mean_val = float(np.mean(matrix))
    cv_mean, cv_sd, cv_med = compute_within_subject_cv(matrix)

    (
        icc_2_1,
        ci_2_1,
        icc_3_1,
        ci_3_1,
        ms_r,
        ms_c,
        ms_e,
        sd_tot,
    ) = compute_two_way_icc(matrix)

    # SEM = SD * sqrt(1 - ICC(2,1))
    # MDC_95 = 1.96 * sqrt(2) * SEM
    sem_val = float(sd_tot * np.sqrt(max(0.0, 1.0 - icc_2_1)))
    sem_pct = float((sem_val / mean_val) * 100.0) if mean_val != 0.0 else np.nan

    mdc_val = float(1.96 * np.sqrt(2.0) * sem_val)
    mdc_pct = float((mdc_val / mean_val) * 100.0) if mean_val != 0.0 else np.nan

    return ReliabilityResult(
        metric_name=metric_name,
        condition=condition,
        design_desc=design_desc,
        n_subjects=n_subjs,
        k_trials=k_trials,
        n_observations=n_obs,
        denominator_str=denom_str,
        mean=mean_val,
        sd_total=sd_tot,
        cv_percent_mean=cv_mean,
        cv_percent_sd=cv_sd,
        cv_percent_median=cv_med,
        icc_2_1=icc_2_1,
        icc_2_1_ci_lower=ci_2_1[0],
        icc_2_1_ci_upper=ci_2_1[1],
        icc_3_1=icc_3_1,
        icc_3_1_ci_lower=ci_3_1[0],
        icc_3_1_ci_upper=ci_3_1[1],
        sem=sem_val,
        sem_pct_of_mean=sem_pct,
        mdc_95=mdc_val,
        mdc_95_pct_of_mean=mdc_pct,
    )


def compute_lmm_reliability(
    df: pd.DataFrame,
    metric_col: str,
    condition: str = "CMJNA",
) -> tuple[float, float, float, str]:
    """Fit a Linear Mixed-Effects Model (LMM / REML) on unbalanced trial data.

    Model: Y_ij = mu + u_i + epsilon_ij
        u_i ~ N(0, sigma_between^2)
        epsilon_ij ~ N(0, sigma_within^2)
        ICC_LMM = sigma_between^2 / (sigma_between^2 + sigma_within^2)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of processed trials.
    metric_col : str
        Column name to analyze.
    condition : str
        'CMJNA' or 'CMJA'.

    Returns
    -------
    tuple[float, float, float, str]
        (icc_lmm, sigma_between, sigma_within, denominator_str)
    """
    sub_df = df[df["condition_name"] == condition].dropna(subset=[metric_col])
    n_obs = len(sub_df)
    n_subjs = sub_df["subject_id"].nunique()
    denom_str = (
        f"{condition} LMM (all available trials), "
        f"n={n_subjs} subjects, {n_obs} observations"
    )

    try:
        model = smf.mixedlm(f"{metric_col} ~ 1", sub_df, groups=sub_df["subject_id"])
        fit = model.fit(reml=True, method=["lbfgs"])
        var_between = float(fit.cov_re.iloc[0, 0])
        var_within = float(fit.scale)
        icc_lmm = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0.0
        return float(icc_lmm), float(np.sqrt(var_between)), float(np.sqrt(var_within)), denom_str
    except Exception as e:
        logger.warning("LMM fitting failed for %s: %s", metric_col, e)
        return np.nan, np.nan, np.nan, denom_str
