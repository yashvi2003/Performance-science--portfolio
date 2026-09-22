"""Unit tests for Project 3 Linear Mixed-Effects Models (p3_nmt_profiler.model)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from p3_nmt_profiler.model import (
    prepare_longitudinal_dataset,
    fit_model_1_readiness,
    fit_model_2_soreness,
    fit_model_3_fatigue,
    evaluate_random_slopes,
    run_all_models,
)


@pytest.fixture
def sample_pmdata_dir() -> Path:
    p = Path("data/raw/pmdata")
    if not p.exists() or not any(p.glob("p*")):
        pytest.skip("PMData raw directory data/raw/pmdata is not present.")
    return p


def test_prepare_longitudinal_dataset(sample_pmdata_dir: Path) -> None:
    df = prepare_longitudinal_dataset(sample_pmdata_dir)
    assert not df.empty
    assert "athlete_id" in df.columns
    assert "date" in df.columns
    assert "z_dev_prior_sleep_hours" in df.columns
    assert "mean_prior_sleep_hours" in df.columns
    assert "z_dev_prior_rhr" in df.columns
    assert "z_dev_prior_day_strain" in df.columns
    assert "is_weekend" in df.columns

    # Verify person-mean centering properties: within-person deviations should sum to ~0 per athlete
    for _, g in df.groupby("athlete_id"):
        dev_sleep = g["dev_prior_sleep_hours"].dropna()
        if len(dev_sleep) > 1:
            assert abs(dev_sleep.mean()) < 1e-4


def test_model_1_readiness(sample_pmdata_dir: Path) -> None:
    df = prepare_longitudinal_dataset(sample_pmdata_dir)
    res = fit_model_1_readiness(df)
    assert res.n_obs > 500
    assert res.n_groups >= 10
    assert len(res.fixed_effects) >= 7

    # Check key predictors exist
    param_names = [fe.name for fe in res.fixed_effects]
    assert "z_dev_prior_sleep_hours" in param_names
    assert "mean_prior_sleep_hours" in param_names
    assert "z_dev_prior_rhr" in param_names
    assert "z_dev_prior_day_strain" in param_names

    # Check diagnostics
    assert res.diagnostics.vif_dict
    assert not np.isnan(res.diagnostics.residual_skewness)
    assert not np.isnan(res.diagnostics.residual_autocorr_lag1)


def test_model_2_soreness(sample_pmdata_dir: Path) -> None:
    df = prepare_longitudinal_dataset(sample_pmdata_dir)
    res = fit_model_2_soreness(df)
    assert res.n_obs > 500
    assert res.n_groups >= 10
    param_names = [fe.name for fe in res.fixed_effects]
    assert "z_dev_prior_day_strain" in param_names
    assert "z_dev_prior_sleep_hours" in param_names


def test_model_3_fatigue(sample_pmdata_dir: Path) -> None:
    df = prepare_longitudinal_dataset(sample_pmdata_dir)
    res = fit_model_3_fatigue(df)
    assert res.n_obs > 500
    assert res.n_groups >= 10
    param_names = [fe.name for fe in res.fixed_effects]
    assert "z_dev_prior_sleep_score" in param_names
    assert "z_dev_prior_day_strain" in param_names


def test_random_slopes_evaluation(sample_pmdata_dir: Path) -> None:
    df = prepare_longitudinal_dataset(sample_pmdata_dir)
    rs_res = evaluate_random_slopes(df)
    assert "lrt_stat" in rs_res
    assert "p_value" in rs_res
    assert "supports_random_slopes" in rs_res


def test_run_all_models_pipeline(sample_pmdata_dir: Path) -> None:
    m1, m2, m3, rs = run_all_models(sample_pmdata_dir)
    assert m1.model_name == "Model 1: Next-Day Readiness"
    assert m2.model_name == "Model 2: Next-Day Muscle Soreness"
    assert m3.model_name == "Model 3: Next-Day Fatigue"
    assert "supports_random_slopes" in rs
