"""
Unit tests for statistical significance testing functions.

Covers effect sizes, p-values and multiple comparison corrections.

Test Coverage:
    - TestBonferroniCorrection: Family-wise error rate control (7 tests)
    - TestCohensD: Effect size calculation with pooled standard deviation (6 tests)
    - TestWilcoxonSignedRankTest: Non-parametric paired comparison (5 tests)
    - TestComprehensiveComparisonReport: Full analysis pipeline (5 tests)

Total: 23 test methods across 4 test classes

Implementation: src/evaluation/statistical_tests.py
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical_tests import (
    bonferroni_correction,
    calculate_cohens_d,
    comprehensive_comparison_report,
    wilcoxon_signed_rank_test,
)


class TestBonferroniCorrection:
    """Test Bonferroni correction for multiple comparisons."""

    def test_bonferroni_5_comparisons(self):
        """
        Test standard usage with 5 comparisons.

        Setup:
            - p_values: [0.001, 0.005, 0.02, 0.03, 0.001]
            - alpha: 0.05
            - corrected_alpha: 0.05 / 5 = 0.01

        Expected:
            - Tests at indices 0, 1, 4 are significant (p < 0.01)
            - Tests at indices 2, 3 are not significant (p >= 0.01)
        """
        p_values = [0.001, 0.005, 0.02, 0.03, 0.001]
        alpha = 0.05

        result = bonferroni_correction(p_values, alpha=alpha)

        assert result["success"] is True
        assert result["data"]["corrected_alpha"] == 0.01
        assert result["data"]["n_comparisons"] == 5
        assert result["data"]["n_significant"] == 3
        assert result["data"]["significant_tests"] == [0, 1, 4]
        assert result["error"] is None

    def test_bonferroni_all_significant(self):
        """
        Test where all tests pass correction.

        Setup:
            - p_values: [0.001, 0.002, 0.003, 0.004]
            - alpha: 0.05
            - corrected_alpha: 0.05 / 4 = 0.0125

        Expected:
            - All 4 tests significant (all p < 0.0125)
        """
        p_values = [0.001, 0.002, 0.003, 0.004]
        alpha = 0.05

        result = bonferroni_correction(p_values, alpha=alpha)

        assert result["success"] is True
        assert result["data"]["corrected_alpha"] == 0.0125
        assert result["data"]["n_significant"] == 4
        assert len(result["data"]["significant_tests"]) == 4
        assert result["data"]["significant_tests"] == [0, 1, 2, 3]
        assert result["error"] is None

    def test_bonferroni_none_significant(self):
        """
        Test where no tests pass correction.

        Setup:
            - p_values: [0.02, 0.03, 0.04, 0.05]
            - alpha: 0.05
            - corrected_alpha: 0.05 / 4 = 0.0125

        Expected:
            - No tests significant (all p >= 0.0125)
        """
        p_values = [0.02, 0.03, 0.04, 0.05]
        alpha = 0.05

        result = bonferroni_correction(p_values, alpha=alpha)

        assert result["success"] is True
        assert result["data"]["n_significant"] == 0
        assert len(result["data"]["significant_tests"]) == 0
        assert result["data"]["significant_tests"] == []
        assert result["error"] is None

    def test_bonferroni_empty_pvalues(self):
        """
        Test validation failure with empty p_values list.

        Expected:
            - success = False
            - error message contains 'empty'
        """
        p_values = []

        result = bonferroni_correction(p_values)

        assert result["success"] is False
        assert result["data"] is None
        assert "empty" in result["error"].lower()

    def test_bonferroni_invalid_pvalue_too_high(self):
        """
        Test validation failure with p-value > 1.

        Setup:
            - p_values: [0.01, 1.5, 0.03]

        Expected:
            - success = False
            - error message contains 'range' or 'invalid'
        """
        p_values = [0.01, 1.5, 0.03]

        result = bonferroni_correction(p_values)

        assert result["success"] is False
        assert result["data"] is None
        assert ("range" in result["error"].lower() or "invalid" in result["error"].lower())

    def test_bonferroni_invalid_pvalue_negative(self):
        """
        Test validation failure with negative p-value.

        Setup:
            - p_values: [0.01, -0.5, 0.03]

        Expected:
            - success = False
            - error message contains 'range' or 'invalid'
        """
        p_values = [0.01, -0.5, 0.03]

        result = bonferroni_correction(p_values)

        assert result["success"] is False
        assert result["data"] is None
        assert ("range" in result["error"].lower() or "invalid" in result["error"].lower())

    def test_bonferroni_invalid_alpha(self):
        """
        Test validation failure with alpha outside (0, 1).

        Setup:
            - p_values: [0.01, 0.02]
            - alpha: 1.5 (invalid)

        Expected:
            - success = False
            - error message contains 'alpha'
        """
        p_values = [0.01, 0.02]
        alpha = 1.5

        result = bonferroni_correction(p_values, alpha=alpha)

        assert result["success"] is False
        assert result["data"] is None
        assert "alpha" in result["error"].lower()


class TestCohensD:
    """Test Cohen's d effect size calculation for paired sample comparison."""

    def test_cohens_d_large_effect(self):
        """
        Test large effect size (d > 0.8).

        Setup:
            - Baseline: mean=0.55, std=0.05, n=100
            - Treatment: mean=0.85, std=0.05, n=100
            - Expected difference: 0.30 (30 percentage points)

        Expected:
            - success = True
            - cohens_d > 0.8 (large effect threshold)
            - magnitude in ["large", "very_large"]
        """
        np.random.seed(42)
        baseline = np.random.normal(0.55, 0.05, 100)
        treatment = np.random.normal(0.85, 0.05, 100)

        result = calculate_cohens_d(baseline.tolist(), treatment.tolist())

        assert result["success"] is True, f"Expected success=True, got {result}"
        assert (
            result["data"]["cohens_d"] > 0.8
        ), f"Expected Cohen's d > 0.8, got {result['data']['cohens_d']}"
        assert result["data"]["magnitude"] in [
            "large",
            "very_large",
        ], f"Expected magnitude in ['large', 'very_large'], got {result['data']['magnitude']}"

    def test_cohens_d_small_effect(self):
        """
        Test small effect size (0.2 <= d < 0.5).

        Setup:
            - Baseline: mean=0.70, std=0.1, n=100
            - Treatment: mean=0.73, std=0.1, n=100
            - Expected difference: 0.03 (3 percentage points - small improvement)

        Expected:
            - success = True
            - 0.2 <= cohens_d < 0.5
            - magnitude == "small"
        """
        np.random.seed(42)
        baseline = np.random.normal(0.70, 0.1, 100)
        treatment = np.random.normal(0.73, 0.1, 100)

        result = calculate_cohens_d(baseline.tolist(), treatment.tolist())

        assert result["success"] is True, f"Expected success=True, got {result}"
        assert (
            0.2 <= result["data"]["cohens_d"] < 0.5
        ), f"Expected 0.2 <= Cohen's d < 0.5, got {result['data']['cohens_d']}"
        assert (
            result["data"]["magnitude"] == "small"
        ), f"Expected magnitude='small', got {result['data']['magnitude']}"

    def test_cohens_d_negligible_effect(self):
        """
        Test negligible effect size (d < 0.2).

        Setup:
            - Baseline: [0.850, 0.840, 0.860]
            - Treatment: [0.851, 0.841, 0.861]
            - Difference: 0.001 (0.1 percentage points - minimal)

        Expected:
            - success = True
            - cohens_d < 0.2 (negligible threshold)
            - magnitude == "negligible"
        """
        baseline = [0.850, 0.840, 0.860]
        treatment = [0.851, 0.841, 0.861]

        result = calculate_cohens_d(baseline, treatment)

        assert result["success"] is True, f"Expected success=True, got {result}"
        assert (
            result["data"]["cohens_d"] < 0.2
        ), f"Expected Cohen's d < 0.2, got {result['data']['cohens_d']}"
        assert (
            result["data"]["magnitude"] == "negligible"
        ), f"Expected magnitude='negligible', got {result['data']['magnitude']}"

    def test_cohens_d_length_mismatch(self):
        """
        Test validation failure for mismatched array lengths.

        Setup:
            - Baseline: 3 values
            - Treatment: 2 values (length mismatch)

        Expected:
            - success = False
            - error message contains "length"
        """
        baseline = [0.5, 0.6, 0.55]
        treatment = [0.85, 0.88]

        result = calculate_cohens_d(baseline, treatment)

        assert result["success"] is False, f"Expected success=False, got {result}"
        assert (
            "length" in result["error"].lower()
        ), f"Expected 'length' in error message, got: {result['error']}"

    def test_cohens_d_nan_values(self):
        """
        Test NaN handling in input arrays.

        Setup:
            - Baseline: Contains np.nan
            - Treatment: Valid values

        Expected:
            - success = False
            - error message contains "nan" or "finite"
        """
        baseline = [0.5, 0.6, np.nan]
        treatment = [0.85, 0.88, 0.82]

        result = calculate_cohens_d(baseline, treatment)

        assert result["success"] is False, f"Expected success=False, got {result}"
        assert (
            "nan" in result["error"].lower() or "finite" in result["error"].lower()
        ), f"Expected 'nan' or 'finite' in error message, got: {result['error']}"

    def test_cohens_d_constant_arrays(self):
        """
        Test error handling for constant arrays (zero variance).

        Setup:
            - Baseline: [0.5, 0.5, 0.5] (std = 0)
            - Treatment: [0.5, 0.5, 0.5] (std = 0)
            - Pooled std = 0 (division by zero case)

        Expected:
            - success = False
            - error message contains "deviation" or "constant"
        """
        baseline = [0.5, 0.5, 0.5]
        treatment = [0.5, 0.5, 0.5]

        result = calculate_cohens_d(baseline, treatment)

        assert result["success"] is False, f"Expected success=False, got {result}"
        assert (
            "deviation" in result["error"].lower()
            or "constant" in result["error"].lower()
        ), f"Expected 'deviation' or 'constant' in error message, got: {result['error']}"


class TestWilcoxonSignedRankTest:
    """Test Wilcoxon signed-rank test implementation for paired sample comparisons."""

    def test_wilcoxon_treatment_significantly_better(self):
        """
        Test case where treatment clearly outperforms baseline.

        Setup:
            - Baseline: 10 samples with mean ~0.55 (moderate performance)
            - Treatment: 10 samples with mean ~0.85 (high performance)
            - Improvement: ~0.30 (30 percentage points) - clinically significant

        Expected:
            - success = True (no validation errors)
            - p_value < 0.05 (statistically significant improvement)
            - median_difference ≈ 0.30 (matches expected improvement)
            - interpretation contains "significant" (human-readable confirmation)

        Statistical Context:
            This represents a realistic scenario where fine-tuned model shows
            substantial improvement over baseline in wound segmentation metrics.
        """
        baseline = [0.5, 0.6, 0.55, 0.52, 0.58, 0.50, 0.57, 0.54, 0.56, 0.53]
        treatment = [0.85, 0.88, 0.82, 0.87, 0.86, 0.84, 0.87, 0.83, 0.86, 0.85]

        result = wilcoxon_signed_rank_test(baseline, treatment, alternative="greater")

        assert result["success"] is True, "Function should succeed with valid input"
        assert result["data"]["p_value"] < 0.05, "p-value should be < 0.05 for significant improvement"
        assert (
            0.25 < result["data"]["median_difference"] < 0.35
        ), "Median difference should be approximately 0.30"
        assert (
            "significant" in result["data"]["interpretation"].lower()
        ), "Interpretation should mention significance"

    def test_wilcoxon_no_difference(self):
        """
        Test case where treatment equals baseline (no meaningful improvement).

        Setup:
            - Baseline: 10 samples with mean ~0.55
            - Treatment: 10 samples with mean ~0.55 (tiny noise only)
            - Difference: < 0.05 (noise level, not meaningful)

        Expected:
            - success = True (valid comparison)
            - p_value > 0.05 (not statistically significant)
            - |median_difference| < 0.1 (negligible difference)

        Clinical Context:
            This represents a null result where experimental modification
            provides no measurable benefit over baseline.
        """
        baseline = [0.5, 0.6, 0.55, 0.52, 0.58, 0.50, 0.57, 0.54, 0.56, 0.53]
        treatment = [0.51, 0.59, 0.56, 0.53, 0.57, 0.51, 0.58, 0.55, 0.55, 0.54]

        result = wilcoxon_signed_rank_test(baseline, treatment, alternative="greater")

        assert result["success"] is True, "Function should succeed with valid input"
        assert result["data"]["p_value"] > 0.05, "p-value should be > 0.05 for no significant difference"
        assert (
            abs(result["data"]["median_difference"]) < 0.1
        ), "Median difference should be negligible (< 0.1)"

    def test_wilcoxon_length_mismatch(self):
        """
        Test validation failure when baseline and treatment have different lengths.

        Validation Rule:
            Wilcoxon signed-rank test requires PAIRED samples - each baseline value
            must have a corresponding treatment value from the same image/subject.

        Expected:
            - success = False (validation failed)
            - error message mentions "length" (clear failure reason)
            - data = None (no results on validation failure)
        """
        baseline = [0.5, 0.6, 0.55]
        treatment = [0.85, 0.88]  # Different length - invalid pairing

        result = wilcoxon_signed_rank_test(baseline, treatment)

        assert result["success"] is False, "Function should fail validation with mismatched lengths"
        assert "length" in result["error"].lower(), "Error message should mention 'length'"
        assert result["data"] is None, "Data should be None on validation failure"

    def test_wilcoxon_insufficient_samples(self):
        """
        Test validation failure when sample size is too small for statistical power.

        Validation Rule:
            Minimum 10 samples required for adequate statistical power in clinical
            wound segmentation analysis. Fewer samples cannot reliably detect
            meaningful differences.

        Contract Reference:
            VALIDATION RULES - Point 2: "Minimum Samples: n >= 10 (statistical power requirement)"

        Expected:
            - success = False (validation failed)
            - error message mentions "samples" or "sample size" (clear failure reason)
            - data = None (no results on validation failure)
        """
        baseline = [0.5, 0.6]
        treatment = [0.85, 0.88]

        result = wilcoxon_signed_rank_test(baseline, treatment)

        assert result["success"] is False, "Function should fail validation with too few samples"
        assert "sample" in result["error"].lower(), "Error message should mention 'samples' or 'sample size'"
        assert result["data"] is None, "Data should be None on validation failure"

    def test_wilcoxon_nan_handling(self):
        """
        Test validation failure when input contains NaN values.

        Validation Rule:
            All values must be finite numeric values. NaN values indicate missing
            or corrupted data that cannot be used in statistical analysis.

        Contract Reference:
            VALIDATION RULES - Point 3: "No NaN/Inf: All values must be finite numeric values"

        Expected:
            - success = False (validation failed)
            - error message mentions "nan" or "finite" (clear failure reason)
            - data = None (no results on validation failure)

        Clinical Context:
            NaN values may occur from failed predictions, corrupted images, or
            processing errors - these must be flagged and handled before analysis.
        """
        baseline = [0.5, 0.6, np.nan, 0.52, 0.58, 0.50, 0.57, 0.54, 0.56, 0.53]
        treatment = [0.85, 0.88, 0.82, 0.87, 0.86, 0.84, 0.87, 0.83, 0.86, 0.85]

        result = wilcoxon_signed_rank_test(baseline, treatment)

        assert result["success"] is False, "Function should fail validation with NaN values"
        assert (
            "nan" in result["error"].lower() or "finite" in result["error"].lower()
        ), "Error message should mention NaN or finite values"
        assert result["data"] is None, "Data should be None on validation failure"


class TestComprehensiveComparisonReport:
    """Test comprehensive statistical comparison workflow."""

    def test_comprehensive_report_significant_improvement(self):
        """
        Test full comparison where treatment significantly beats baseline.

        Setup:
            - 100 paired images
            - Baseline metrics: ~0.50-0.65 (moderate performance)
            - Treatment metrics: ~0.75-0.88 (high performance)
            - Large effect sizes with highly significant p-values

        Expected:
            - success = True
            - n_images = 100
            - bonferroni_alpha = 0.01 (0.05 / 5 metrics)
            - all_significant = True
            - recommendation includes "adopt" or "superior"
        """
        from src.evaluation.statistical_tests import comprehensive_comparison_report

        np.random.seed(42)
        n_images = 100

        # Create baseline DataFrame (moderate performance)
        baseline_df = pd.DataFrame({
            "image_id": [f"img_{i:03d}" for i in range(n_images)],
            "dice": np.random.normal(0.55, 0.05, n_images),
            "iou": np.random.normal(0.45, 0.05, n_images),
            "precision": np.random.normal(0.50, 0.08, n_images),
            "recall": np.random.normal(0.65, 0.08, n_images),
            "boundary_f1": np.random.normal(0.48, 0.06, n_images),
        })

        # Create treatment DataFrame (high performance - significant improvement)
        treatment_df = pd.DataFrame({
            "image_id": [f"img_{i:03d}" for i in range(n_images)],
            "dice": np.random.normal(0.85, 0.04, n_images),
            "iou": np.random.normal(0.75, 0.04, n_images),
            "precision": np.random.normal(0.82, 0.05, n_images),
            "recall": np.random.normal(0.88, 0.05, n_images),
            "boundary_f1": np.random.normal(0.80, 0.05, n_images),
        })

        result = comprehensive_comparison_report(
            baseline_df,
            treatment_df,
            metric_names=["dice", "iou", "precision", "recall", "boundary_f1"],
            alpha=0.05
        )

        assert result["success"] is True
        assert result["data"]["n_images"] == n_images
        assert result["data"]["bonferroni_alpha"] == 0.01
        assert result["data"]["summary"]["all_significant"] is True

        recommendation = result["data"]["summary"]["recommendation"].lower()
        assert "adopt" in recommendation or "superior" in recommendation

    def test_comprehensive_report_missing_image_id(self):
        """
        Test validation error when baseline DataFrame lacks 'image_id' column.

        Expected:
            - success = False
            - error message mentions "image_id"
        """
        from src.evaluation.statistical_tests import comprehensive_comparison_report

        # Create baseline DataFrame WITHOUT 'image_id' column
        baseline_df = pd.DataFrame({
            "dice": [0.85, 0.87, 0.83, 0.86, 0.84],
            "iou": [0.75, 0.77, 0.73, 0.76, 0.74],
        })

        treatment_df = pd.DataFrame({
            "image_id": ["img1", "img2", "img3", "img4", "img5"],
            "dice": [0.91, 0.93, 0.89, 0.92, 0.90],
            "iou": [0.85, 0.87, 0.83, 0.86, 0.84],
        })

        result = comprehensive_comparison_report(
            baseline_df,
            treatment_df,
            metric_names=["dice", "iou"]
        )

        assert result["success"] is False
        assert "image_id" in result["error"].lower()

    def test_comprehensive_report_missing_metric_column(self):
        """
        Test validation error when requested metric column does not exist in DataFrame.

        Expected:
            - success = False
            - error message mentions "column" or "metric"
        """
        from src.evaluation.statistical_tests import comprehensive_comparison_report

        baseline_df = pd.DataFrame({
            "image_id": ["img1", "img2", "img3", "img4", "img5"],
            "dice": [0.85, 0.87, 0.83, 0.86, 0.84],
        })

        treatment_df = pd.DataFrame({
            "image_id": ["img1", "img2", "img3", "img4", "img5"],
            "dice": [0.91, 0.93, 0.89, 0.92, 0.90],
        })

        result = comprehensive_comparison_report(
            baseline_df,
            treatment_df,
            metric_names=["dice", "iou"]  # 'iou' column does not exist
        )

        assert result["success"] is False
        error_lower = result["error"].lower()
        assert "column" in error_lower or "metric" in error_lower

    def test_comprehensive_report_no_overlapping_images(self):
        """
        Test error handling when no image IDs overlap between DataFrames.

        Expected:
            - success = False
            - error mentions "overlap" or "empty" or "paired"
        """
        from src.evaluation.statistical_tests import comprehensive_comparison_report

        baseline_df = pd.DataFrame({
            "image_id": ["img1", "img2", "img3"],
            "dice": [0.85, 0.87, 0.83],
            "iou": [0.75, 0.77, 0.73],
        })

        # Treatment DataFrame with COMPLETELY DIFFERENT image IDs (no overlap)
        treatment_df = pd.DataFrame({
            "image_id": ["img4", "img5", "img6"],
            "dice": [0.91, 0.93, 0.89],
            "iou": [0.85, 0.87, 0.83],
        })

        result = comprehensive_comparison_report(
            baseline_df,
            treatment_df,
            metric_names=["dice", "iou"]
        )

        assert result["success"] is False
        error_lower = result["error"].lower()
        assert "overlap" in error_lower or "empty" in error_lower or "paired" in error_lower

    def test_comprehensive_report_partial_significance(self):
        """
        Test case where some metrics show significant improvement but not all.

        Setup:
            - 50 paired images
            - Dice: Small improvement (0.80 -> 0.85) - significant
            - IoU: Tiny improvement (0.70 -> 0.705) - not significant

        Expected:
            - success = True
            - all_significant = False
            - n_significant < total number of metrics
            - recommendation includes "conditional"
        """
        from src.evaluation.statistical_tests import comprehensive_comparison_report

        np.random.seed(42)
        n_images = 50

        baseline_df = pd.DataFrame({
            "image_id": [f"img_{i:03d}" for i in range(n_images)],
            "dice": np.random.normal(0.80, 0.03, n_images),
            "iou": np.random.normal(0.70, 0.05, n_images),
        })

        treatment_df = pd.DataFrame({
            "image_id": [f"img_{i:03d}" for i in range(n_images)],
            "dice": np.random.normal(0.85, 0.03, n_images),    # Small improvement - significant
            "iou": np.random.normal(0.705, 0.05, n_images),    # Tiny improvement - not significant
        })

        result = comprehensive_comparison_report(
            baseline_df,
            treatment_df,
            metric_names=["dice", "iou"]
        )

        assert result["success"] is True
        assert result["data"]["summary"]["all_significant"] is False
        assert result["data"]["summary"]["n_significant"] < len(["dice", "iou"])
        assert "conditional" in result["data"]["summary"]["recommendation"].lower()
