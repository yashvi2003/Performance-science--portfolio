"""Tests for Milestone M2g: Open Data Adapter & Within-Participant Reliability.

Validates the White et al. (2022/2025) open dataset processing:
1. Dataset loading and subject count audit (67 participants, subject_ids 0..68 excluding {14, 29}).
2. Naming contract integrity (jump_height_takeoff_vel vs jump_height_total_com).
3. Provenance preservation of raw_continuous_takeoff_idx.
4. Numerical reproduction of peak power and total COM jump height.
5. Balanced matrix dimensions and denominator strings.
6. Reliability statistical functions against analytical ground truth.
7. QA flags assignment for terminal force offsets.
"""
from pathlib import Path

import numpy as np
import pytest

from p1_nmt_pipeline.open_data_adapter import (
    load_open_dataset,
    analyze_open_trial,
    process_all_open_trials,
    build_balanced_trial_matrix,
    OpenDataset,
    STANDARD_G_M_PER_S2,
)
from p1_nmt_pipeline.reliability import (
    compute_within_subject_cv,
    compute_two_way_icc,
    compute_reliability,
    compute_lmm_reliability,
)

OPEN_DATASET_PATH = Path("data/raw/open/cmj_dataset_both.npz")


@pytest.fixture(scope="module")
def open_dataset() -> OpenDataset:
    """Fixture to load the open dataset once for all module tests."""
    if not OPEN_DATASET_PATH.exists():
        pytest.skip(f"Open dataset file not found at {OPEN_DATASET_PATH}")
    return load_open_dataset(OPEN_DATASET_PATH)


class TestOpenDatasetLoadingAndAudit:
    """Verifies dataset structure and subject count reconciliation."""

    def test_reconciled_subject_counts(self, open_dataset: OpenDataset):
        """Verify the 67 unique participants and the study-level metadata scalar."""
        assert open_dataset.n_trials == 663
        assert open_dataset.n_unique_subjects == 67
        assert open_dataset.n_subjects_metadata == 69

        # Verify subject_ids range 0..68 with 14 and 29 missing
        unique_subjs = set(np.unique(open_dataset.subject_ids))
        assert len(unique_subjs) == 67
        assert min(unique_subjs) == 0
        assert max(unique_subjs) == 68
        assert 14 not in unique_subjs
        assert 29 not in unique_subjs

        # Verify original_participant_ids has 67 unique values
        unique_orig = np.unique(open_dataset.original_participant_ids)
        assert len(unique_orig) == 67


class TestNamingContractAndProvenance:
    """Verifies unambiguous naming and provenance preservation."""

    def test_jump_height_naming_and_math(self, open_dataset: OpenDataset):
        """Verify jump_height_total_com_m == s_takeoff_m + jump_height_takeoff_vel_m."""
        res = analyze_open_trial(open_dataset, 0)

        assert res.s_takeoff_m > 0.05  # upward displacement at takeoff (~12 cm)
        assert res.jump_height_takeoff_vel_m > 0.20  # flight height (~30-35 cm)
        assert res.jump_height_total_com_m > res.jump_height_takeoff_vel_m

        # Exact identity
        expected_total = res.s_takeoff_m + res.jump_height_takeoff_vel_m
        assert np.isclose(res.jump_height_total_com_m, expected_total, atol=1e-6)

    def test_provenance_preservation_of_raw_takeoff_idx(self, open_dataset: OpenDataset):
        """Verify raw continuous takeoff index is preserved, while internal index points to last sample."""
        for idx in [0, 50, 100, 200, 500]:
            res = analyze_open_trial(open_dataset, idx)
            # Internal sample index is strictly the last sample
            assert res.takeoff_sample_idx == res.n_samples - 1
            # Raw continuous index from continuous recording is preserved
            assert res.raw_continuous_takeoff_idx == int(open_dataset.grf_takeoff[idx])
            assert res.raw_continuous_takeoff_idx > res.n_samples - 1


class TestGoldenReproduction:
    """Verifies numerical reproduction of peak power and total COM jump height."""

    def test_peak_power_reproduction_accuracy(self, open_dataset: OpenDataset):
        """Verify calculated peak power matches reference with r > 0.9999 and mean error < 0.2%."""
        ref_powers = open_dataset.peak_power
        calc_powers = []
        for i in range(len(ref_powers)):
            res = analyze_open_trial(open_dataset, i)
            calc_powers.append(res.peak_power_w_per_kg)

        calc_powers = np.array(calc_powers)
        diff = calc_powers - ref_powers
        pct_err = (diff / ref_powers) * 100.0

        # Pearson r should be virtually 1.0
        corr = np.corrcoef(calc_powers, ref_powers)[0, 1]
        assert corr > 0.99999

        # Mean percent error is approximately -0.109% (due to g=9.80 vs 9.80665)
        assert np.abs(np.mean(pct_err) - (-0.109)) < 0.05
        assert np.max(np.abs(pct_err)) < 0.35

    def test_total_com_jump_height_reproduction(self, open_dataset: OpenDataset):
        """Verify total COM jump height matches reference with r > 0.9999 and mean diff < 0.2 cm."""
        ref_jh = open_dataset.jump_height
        calc_jh = []
        for i in range(len(ref_jh)):
            res = analyze_open_trial(open_dataset, i)
            calc_jh.append(res.jump_height_total_com_m)

        calc_jh = np.array(calc_jh)
        diff_cm = (calc_jh - ref_jh) * 100.0

        corr = np.corrcoef(calc_jh, ref_jh)[0, 1]
        assert corr > 0.99999
        assert np.abs(np.mean(diff_cm)) < 0.20  # Mean diff around +0.10 cm (+1.0 mm)


class TestBalancedMatrixAndDenominators:
    """Verifies balanced matrix dimensions and explicit denominator strings."""

    def test_cmjna_balanced_matrices(self, open_dataset: OpenDataset):
        """Verify CMJNA 3-trial and 4-trial balanced set dimensions and denominators."""
        _, df = process_all_open_trials(open_dataset)

        # 4-trial set
        m4, subjs4, denom4 = build_balanced_trial_matrix(df, condition="CMJNA", k_trials=4)
        assert m4.shape == (66, 4)
        assert len(subjs4) == 66
        assert 40 not in subjs4  # Subject 40 has only 3 trials
        assert "n=66 subjects x 4 trials = 264 observations" in denom4

        # 3-trial set
        m3, subjs3, denom3 = build_balanced_trial_matrix(df, condition="CMJNA", k_trials=3)
        assert m3.shape == (67, 3)
        assert len(subjs3) == 67
        assert "n=67 subjects x 3 trials = 201 observations" in denom3

    def test_cmja_balanced_matrices(self, open_dataset: OpenDataset):
        """Verify CMJA 3-trial and 4-trial balanced set dimensions and denominators."""
        _, df = process_all_open_trials(open_dataset)

        # 4-trial set
        m4, subjs4, denom4 = build_balanced_trial_matrix(df, condition="CMJA", k_trials=4)
        assert m4.shape == (64, 4)
        assert "n=64 subjects x 4 trials = 256 observations" in denom4

        # 3-trial set
        m3, subjs3, denom3 = build_balanced_trial_matrix(df, condition="CMJA", k_trials=3)
        assert m3.shape == (65, 3)
        assert "n=65 subjects x 3 trials = 195 observations" in denom3


class TestReliabilityStatistics:
    """Verifies analytical properties of reliability functions."""

    def test_icc_analytical_ground_truth(self):
        """Test ICC(2,1) and ICC(3,1) on an exact synthetic matrix with known variance."""
        # Create perfectly reliable matrix (no within-subject variation)
        base = np.array([[10.0], [20.0], [30.0], [40.0]])
        perfect_mat = np.tile(base, (1, 4))
        icc2, ci2, icc3, ci3, msr, msc, mse, sd = compute_two_way_icc(perfect_mat)

        assert np.isclose(icc2, 1.0, atol=1e-5)
        assert np.isclose(icc3, 1.0, atol=1e-5)
        assert np.isclose(mse, 0.0, atol=1e-5)

        # Create perfectly noisy matrix (pure noise, no subject effect)
        rng = np.random.default_rng(123)
        noise_mat = rng.normal(0, 1, size=(200, 4))
        icc2_n, _, icc3_n, _, _, _, _, _ = compute_two_way_icc(noise_mat)
        assert np.abs(icc2_n) < 0.10
        assert np.abs(icc3_n) < 0.10

    def test_cv_and_sem_formulas(self):
        """Test within-subject CV and SEM calculations."""
        # Row 1: [10, 10], Row 2: [20, 20] -> CV = 0%
        mat = np.array([[10.0, 10.0], [20.0, 20.0]])
        cv_mean, cv_sd, cv_med = compute_within_subject_cv(mat)
        assert cv_mean == 0.0

        rel = compute_reliability(mat, "test_metric", "CMJNA", "test set")
        assert np.isclose(rel.sem, 0.0, atol=1e-5)
        assert np.isclose(rel.mdc_95, 0.0, atol=1e-5)


class TestQAFlagsAssignment:
    """Verifies QA flag detection."""

    def test_negative_terminal_force_flagged(self, open_dataset: OpenDataset):
        """Verify Trial 187 is flagged for negative terminal force without causing exceptions."""
        res187 = analyze_open_trial(open_dataset, 187)
        assert res187.flag_negative_terminal_force is True
        assert res187.terminal_force_bw < 0.0
        assert any("NEGATIVE_TERMINAL_FORCE" in f for f in res187.qa_flags)

        # Verify a normal trial (Trial 0) has flag False
        res0 = analyze_open_trial(open_dataset, 0)
        assert res0.flag_negative_terminal_force is False

    def test_no_terminal_force_outliers(self, open_dataset: OpenDataset):
        """Verify no trials in the dataset exceed 0.02 BW at takeoff."""
        for i in range(open_dataset.n_trials):
            res = analyze_open_trial(open_dataset, i)
            assert res.flag_terminal_force_outlier is False


class TestReconciliationVerifications:
    """Verifications requested for Milestone M2g final sign-off."""

    def test_peak_power_integration_mechanism_and_swansea_gravity(self, open_dataset: OpenDataset):
        """Verify the exact mathematical mechanism reconciling Euler vs Trapezoid and local Swansea g."""
        # 1. With trapezoidal integration and Swansea local gravity g=9.81200, reproduction is exact
        sample_trials = [0, 1, 10, 55, 100, 280, 500]
        for t in sample_trials:
            f_bw = np.asarray(open_dataset.grf_signals[t], dtype=np.float64)
            dt = 1.0 / open_dataset.grf_sampling_rate
            ref_pp = float(open_dataset.peak_power[t])

            # Swansea local g trapezoid
            acc_sw = (f_bw - 1.0) * 9.81200
            from scipy.integrate import cumulative_trapezoid
            v_sw = cumulative_trapezoid(acc_sw, dx=dt, initial=0.0)
            pp_sw = np.max((f_bw * 9.81200) * v_sw)
            assert np.isclose(pp_sw, ref_pp, atol=2e-4)

            # Standard g trapezoid (underestimates ref due to 9.80665 < 9.81200)
            res_std = analyze_open_trial(open_dataset, t, g=STANDARD_G_M_PER_S2)
            assert res_std.peak_power_w_per_kg < ref_pp

            # Standard g Euler (np.cumsum overestimates ref due to +0.5*a*dt offset)
            acc_std = (f_bw - 1.0) * STANDARD_G_M_PER_S2
            v_euler = np.cumsum(acc_std) * dt
            pp_euler = np.max((f_bw * STANDARD_G_M_PER_S2) * v_euler)
            assert pp_euler > ref_pp

    def test_negative_terminal_force_sensitivity_traceability(self, open_dataset: OpenDataset):
        """Verify the exact count of 20 negative-terminal trials across 15 subjects in CMJNA primary set."""
        _, df = process_all_open_trials(open_dataset)
        df_cmjna = df[df["condition_name"] == "CMJNA"].sort_values("trial_idx")

        mat_primary, subjs_primary, _ = build_balanced_trial_matrix(
            df, condition="CMJNA", k_trials=4, metric_col="jump_height_total_com_m"
        )
        assert len(subjs_primary) == 66
        assert mat_primary.size == 264

        # Extract the exact 264 observations
        primary_indices = []
        for s in subjs_primary:
            primary_indices.extend(df_cmjna[df_cmjna["subject_id"] == s].iloc[:4]["trial_idx"].tolist())
        df_primary = df_cmjna[df_cmjna["trial_idx"].isin(primary_indices)]

        # Exactly 20 trials with negative terminal force
        neg_trials = df_primary[df_primary["flag_negative_terminal_force"]]
        assert len(neg_trials) == 20

        # Spanning exactly 15 unique subjects
        neg_subjs = sorted(neg_trials["subject_id"].unique())
        assert len(neg_subjs) == 15
        assert neg_subjs == [3, 5, 15, 17, 25, 34, 42, 47, 48, 50, 51, 53, 56, 60, 65]

        # Rule A: Strict whole-subject exclusion drops all 15 subjects -> 66 - 15 = 51 subjects
        strict_subjs = [s for s in subjs_primary if s not in neg_subjs]
        assert len(strict_subjs) == 51

        # Rule B: Per-trial exclusion retains 4 subjects with clean trials 5-8 -> 66 - 11 = 55 subjects
        df_clean = df[~df["flag_negative_terminal_force"]]
        _, subjs_clean, _ = build_balanced_trial_matrix(
            df_clean, condition="CMJNA", k_trials=4, metric_col="jump_height_total_com_m"
        )
        assert len(subjs_clean) == 55
        saved_subjs = set(neg_subjs).intersection(set(subjs_clean))
        assert saved_subjs == {17, 42, 60, 65}
