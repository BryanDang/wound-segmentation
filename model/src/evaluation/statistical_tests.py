"""
Statistical significance testing for model comparisons.

Statistical validation for ablation studies and model comparisons.

Key Functions:
    - wilcoxon_signed_rank_test: Non-parametric paired test for statistical significance
    - calculate_cohens_d: Effect size calculation (sample-size independent)
    - bonferroni_correction: Multiple comparison correction for family-wise error control
    - comprehensive_comparison_report: Full statistical analysis pipeline

Tests: tests/unit/evaluation/test_statistical_tests.py
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.io import failure, success


def wilcoxon_signed_rank_test(
    baseline_values: List[float],
    treatment_values: List[float],
    alternative: str = "greater",
) -> Dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test for paired samples comparison.

    CONTRACT DEFINITION
    -------------------
    Inputs:
        baseline_values: List[float]
            Paired observations from baseline condition (e.g., baseline model metrics).
            Must contain numeric values only, no NaN/infinite values.

        treatment_values: List[float]
            Paired observations from treatment condition (e.g., experimental model metrics).
            Must have same length as baseline_values, no NaN/infinite values.

        alternative: str (default: "greater")
            Alternative hypothesis direction:
            - "greater": treatment > baseline (one-sided, typical for improvement)
            - "less": treatment < baseline (one-sided)
            - "two-sided": treatment != baseline

    Outputs:
        On Success: {
            "success": True,
            "data": {
                "statistic": float,           # Wilcoxon signed-rank test statistic
                "p_value": float,             # One-sided or two-sided p-value
                "n_samples": int,             # Number of paired samples
                "n_ties": int,                # Number of zero differences (ties)
                "median_difference": float,   # Median of (treatment - baseline)
                "interpretation": str         # Human-readable result interpretation
            },
            "error": None
        }

        On Failure: {
            "success": False,
            "data": None,
            "error": str  # Descriptive error message
        }

    VALIDATION RULES
    ----------------
    1. Length Match: len(baseline_values) == len(treatment_values)
    2. Minimum Samples: n >= 10 (statistical power requirement)
    3. No NaN/Inf: All values must be finite numeric values
    4. Numeric Types: Both arrays must contain only int or float types

    STATISTICAL DETAILS
    -------------------
    Null Hypothesis (H₀):
        The median difference between paired samples is zero.
        Distribution of (treatment - baseline) is symmetric around zero.

    Alternative Hypothesis (H₁):
        - "greater": median(treatment - baseline) > 0
        - "less": median(treatment - baseline) < 0
        - "two-sided": median(treatment - baseline) ≠ 0

    Test Assumptions:
        1. Paired observations (same subjects/items in both conditions)
        2. Differences are continuous or ordinal
        3. Differences are symmetric around median (robust to skewness)
        4. Independence of paired differences

    Implementation Notes:
        - Uses scipy.stats.wilcoxon with method='auto'
        - method='auto' selects 'exact' for n < 50, 'approx' otherwise
        - Zero differences (ties) are excluded from test statistic calculation
        - **CRITICAL LIMITATION**: For n=3 samples, minimum p-value = 0.125
          (cannot achieve statistical significance at α=0.05)
        - For clinical wound segmentation: Recommended n >= 10 for adequate power

    Interpretation Guidance:
        - p < 0.001: Highly significant improvement (strong evidence)
        - p < 0.05: Statistically significant improvement
        - p >= 0.05: No statistically significant difference
        - Bonferroni correction: Use α = 0.0125 for 4-way comparisons

    References:
        - scipy.stats.wilcoxon: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        - PRD Statistical Framework: Section on multiple comparisons and effect sizes

    Example:
        >>> baseline = [0.85, 0.87, 0.83, 0.86, 0.84, 0.88, 0.85, 0.87, 0.86, 0.85]
        >>> treatment = [0.91, 0.93, 0.89, 0.92, 0.90, 0.94, 0.91, 0.93, 0.92, 0.91]
        >>> result = wilcoxon_signed_rank_test(baseline, treatment, alternative="greater")
        >>> print(result["data"]["p_value"])  # Expected: p < 0.001 (improvement)
        >>> print(result["data"]["interpretation"])
        'Highly significant improvement: treatment significantly outperforms baseline (p < 0.001)'
    """
    # Convert to numpy arrays for validation and computation
    baseline_arr = np.array(baseline_values, dtype=float)
    treatment_arr = np.array(treatment_values, dtype=float)

    # Validation 1: Check for length mismatch
    if len(baseline_arr) != len(treatment_arr):
        return failure(
            f"Array length mismatch: baseline has {len(baseline_arr)} values, "
            f"treatment has {len(treatment_arr)} values. "
            "Paired samples must have equal length."
        )

    # Validation 2: Check for minimum sample size (n >= 10)
    if len(baseline_arr) < 10:
        return failure(
            f"Insufficient samples: {len(baseline_arr)} values provided. "
            "Minimum of 10 samples required for adequate statistical power."
        )

    # Validation 3: Check for NaN or infinite values in baseline
    if not np.all(np.isfinite(baseline_arr)):
        return failure(
            "Baseline array contains NaN or infinite values. "
            "All values must be finite numbers."
        )

    # Validation 4: Check for NaN or infinite values in treatment
    if not np.all(np.isfinite(treatment_arr)):
        return failure(
            "Treatment array contains NaN or infinite values. "
            "All values must be finite numbers."
        )

    # Calculate differences
    differences = treatment_arr - baseline_arr

    # Count ties (zero differences)
    n_ties = int(np.sum(differences == 0))

    # Calculate median difference
    median_difference = float(np.median(differences))

    # Perform Wilcoxon signed-rank test
    # scipy.stats.wilcoxon automatically handles zero differences by excluding them
    # NOTE: For alternative="greater", the test is treatment > baseline
    # scipy requires (x, y, alternative="greater") to test if x > y
    test_result = stats.wilcoxon(
        treatment_arr,
        baseline_arr,
        alternative=alternative,
        method='auto'
    )

    # Extract statistic and p-value
    statistic = float(test_result.statistic)
    p_value = float(test_result.pvalue)

    # Generate human-readable interpretation
    if p_value < 0.001:
        significance_level = "highly significant"
        interpretation = (
            f"Highly significant improvement: treatment significantly outperforms baseline "
            f"(p < 0.001, median difference = {median_difference:.4f})"
        )
    elif p_value < 0.05:
        significance_level = "statistically significant"
        interpretation = (
            f"Statistically significant improvement: treatment outperforms baseline "
            f"(p = {p_value:.4f}, median difference = {median_difference:.4f})"
        )
    else:
        significance_level = "not significant"
        interpretation = (
            f"No statistically significant difference detected "
            f"(p = {p_value:.4f}, median difference = {median_difference:.4f})"
        )

    # Return success with all required fields
    return success(
        data={
            "statistic": statistic,
            "p_value": p_value,
            "n_samples": len(baseline_arr),
            "n_ties": n_ties,
            "median_difference": median_difference,
            "interpretation": interpretation,
        }
    )


def calculate_cohens_d(
    baseline_values: List[float], treatment_values: List[float]
) -> Dict[str, Any]:
    """
    Calculate Cohen's d effect size for paired sample comparison.

    CONTRACT DEFINITION
    -------------------
    Computes standardized effect size using pooled standard deviation approach,
    enabling interpretation independent of sample size (valid even for n=3).

    MATHEMATICAL FORMULA
    --------------------
    Cohen's d = (mean_treatment - mean_baseline) / pooled_std

    where:
        pooled_std = sqrt((std_baseline^2 + std_treatment^2) / 2)

    INTERPRETATION THRESHOLDS (Cohen, 1988)
    ----------------------------------------
    | Cohen's d | Magnitude    | Interpretation                    |
    |-----------|--------------|-----------------------------------|
    | d < 0.2   | Negligible   | Minimal practical impact          |
    | 0.2 ≤ d < 0.5 | Small    | Noticeable but modest effect      |
    | 0.5 ≤ d < 0.8 | Medium   | Moderate practical importance     |
    | 0.8 ≤ d < 1.2 | Large    | Strong practical significance     |
    | d ≥ 1.2   | Very Large   | Exceptional impact                |

    VALIDATION RULES
    ----------------
    1. Both arrays must have the same length (paired samples)
    2. Both arrays must contain only numeric values
    3. No NaN or infinite values allowed
    4. Minimum of 2 samples required per array
    5. Standard deviations must be > 0 (no constant arrays)

    SAMPLE SIZE INDEPENDENCE
    ------------------------
    Unlike p-values, Cohen's d is independent of sample size. It measures
    standardized difference in standard deviation units, making it valid for
    interpretation whether n=3 or n=1000. This makes it ideal for ablation
    studies with limited random seeds (n=3: seeds 42, 123, 456).

    Args:
        baseline_values: List of baseline model scores (e.g., Dice scores from
                        3 random seeds)
        treatment_values: List of treatment model scores (same length as baseline)

    Returns:
        Dict[str, Any]: Standardized response with structure:

        SUCCESS CASE:
        {
            "success": True,
            "data": {
                "cohens_d": float,              # Effect size value
                "magnitude": str,               # "negligible"|"small"|"medium"|"large"|"very_large"
                "mean_difference": float,       # mean_treatment - mean_baseline
                "pooled_std": float,            # Pooled standard deviation
                "interpretation": str,          # Human-readable interpretation
                "baseline_mean": float,         # Mean of baseline values
                "baseline_std": float,          # Std of baseline values
                "treatment_mean": float,        # Mean of treatment values
                "treatment_std": float          # Std of treatment values
            },
            "error": None
        }

        FAILURE CASE:
        {
            "success": False,
            "data": None,
            "error": str  # Descriptive error message
        }

    Examples:
        >>> # Negligible effect
        >>> result = calculate_cohens_d(
        ...     baseline_values=[0.850, 0.840, 0.860],
        ...     treatment_values=[0.855, 0.845, 0.865]
        ... )
        >>> result["data"]["magnitude"]
        'negligible'

        >>> # Large effect
        >>> result = calculate_cohens_d(
        ...     baseline_values=[0.850, 0.840, 0.860],
        ...     treatment_values=[0.880, 0.870, 0.890]
        ... )
        >>> result["data"]["magnitude"]
        'large'

    Notes:
        - Uses sample standard deviation (ddof=1) for unbiased estimation
        - Pooled SD approach ensures consistency with traditional Cohen's d
          interpretation guidelines
        - For ablation studies, compare against clinical threshold (3% absolute
          improvement) alongside effect size
        - Works with n=3 samples (unlike Wilcoxon which requires n≥5 for
          meaningful p-values)

    References:
        - Cohen, J. (1988). Statistical Power Analysis for the Behavioral
          Sciences (2nd ed.)
    """
    # Convert to numpy arrays
    baseline_arr = np.array(baseline_values, dtype=float)
    treatment_arr = np.array(treatment_values, dtype=float)

    # Validation 1: Check for length mismatch
    if len(baseline_arr) != len(treatment_arr):
        return failure(
            f"Array length mismatch: baseline has {len(baseline_arr)} values, "
            f"treatment has {len(treatment_arr)} values. Paired samples must have equal length."
        )

    # Validation 2: Check for minimum sample size
    if len(baseline_arr) < 2:
        return failure(
            f"Insufficient samples: {len(baseline_arr)} values provided. "
            "Minimum of 2 samples required per array."
        )

    # Validation 3: Check for NaN or infinite values
    if not np.all(np.isfinite(baseline_arr)):
        return failure(
            "Baseline array contains NaN or infinite values. "
            "All values must be finite numbers."
        )

    if not np.all(np.isfinite(treatment_arr)):
        return failure(
            "Treatment array contains NaN or infinite values. "
            "All values must be finite numbers."
        )

    # Calculate descriptive statistics
    baseline_mean = float(np.mean(baseline_arr))
    baseline_std = float(np.std(baseline_arr, ddof=1))
    treatment_mean = float(np.mean(treatment_arr))
    treatment_std = float(np.std(treatment_arr, ddof=1))

    # Validation 4: Check for constant arrays (zero variance)
    if baseline_std == 0 or treatment_std == 0:
        return failure(
            "Standard deviation is zero (constant array detected). "
            "Cohen's d requires non-zero variance in both arrays."
        )

    # Calculate pooled standard deviation
    pooled_std = np.sqrt((baseline_std**2 + treatment_std**2) / 2)

    # Validation 5: Additional check for pooled std (should be caught above)
    if pooled_std == 0:
        return failure(
            "Pooled standard deviation is zero. "
            "Cohen's d requires non-constant arrays."
        )

    # Calculate Cohen's d
    mean_difference = treatment_mean - baseline_mean
    cohens_d = mean_difference / pooled_std

    # Determine magnitude using absolute value for classification
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    elif abs_d < 1.2:
        magnitude = "large"
    else:
        magnitude = "very_large"

    # Generate human-readable interpretation
    direction = "improvement" if cohens_d > 0 else "decline" if cohens_d < 0 else "no change"
    interpretation = (
        f"{magnitude.capitalize()} effect size (d={cohens_d:.3f}): "
        f"Treatment shows {direction} of {abs(mean_difference):.4f} "
        f"({abs(cohens_d):.2f} pooled standard deviations)."
    )

    return success(
        data={
            "cohens_d": float(cohens_d),
            "magnitude": magnitude,
            "mean_difference": float(mean_difference),
            "pooled_std": float(pooled_std),
            "interpretation": interpretation,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "treatment_mean": treatment_mean,
            "treatment_std": treatment_std,
        }
    )


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply Bonferroni correction for multiple hypothesis testing to control family-wise error rate.

    Contract Definition
    -------------------
    The Bonferroni correction is a conservative method to adjust significance thresholds when
    performing multiple statistical comparisons simultaneously. It controls the family-wise
    error rate (FWER) - the probability of making at least one Type I error (false positive)
    across all comparisons.

    Mathematical Formula
    --------------------
    corrected_alpha = alpha / n_comparisons

    Where:
    - alpha: Original significance level (typically 0.05)
    - n_comparisons: Number of statistical tests being performed
    - corrected_alpha: Adjusted significance threshold for each individual test

    Decision Rule:
    Test i is statistically significant if: p_values[i] < corrected_alpha

    Family-Wise Error Rate (FWER) Control
    --------------------------------------
    Without correction, performing k independent tests at alpha=0.05 gives:
        P(at least one false positive) = 1 - (1 - alpha)^k

    For k=4 tests: P(FWER) = 1 - 0.95^4 ≈ 0.185 (18.5% chance of false positive)

    Bonferroni correction ensures: P(FWER) ≤ alpha regardless of number of tests
    For k=4 tests with alpha=0.05: corrected_alpha = 0.05/4 = 0.0125

    Trade-off: Increased protection against false positives at cost of reduced
    statistical power (increased false negatives for true effects).

    Parameters
    ----------
    p_values : List[float]
        List of p-values from multiple statistical tests. Each value must be in [0, 1].
        Must contain at least one p-value.

    alpha : float, default=0.05
        Family-wise significance level. Must be in the open interval (0, 1).
        Common values: 0.05 (standard), 0.01 (stringent), 0.10 (exploratory).

    Validation Rules
    ----------------
    1. p_values must not be empty (len(p_values) >= 1)
    2. All p-values must be in valid range: 0 <= p <= 1
    3. alpha must be in open interval: 0 < alpha < 1

    Returns
    -------
    Dict[str, Any]
        Success case (all validations pass):
        {
            "success": True,
            "data": {
                "corrected_alpha": float,           # Bonferroni-corrected significance threshold
                "n_comparisons": int,               # Number of statistical tests
                "significant_tests": List[int],     # Indices of tests that remain significant
                "n_significant": int,               # Count of significant tests after correction
                "interpretation": str               # Human-readable summary of results
            },
            "error": None
        }

        Failure case (validation fails):
        {
            "success": False,
            "data": None,
            "error": str  # Descriptive error message explaining validation failure
        }

    Interpretation Output Format
    ----------------------------
    The interpretation string provides:
    - Corrected alpha threshold (e.g., "0.0125")
    - Which tests remain significant after correction (e.g., "Tests [0, 2] remain significant")
    - Statistical power note if no tests significant (conservative nature of correction)

    Examples
    --------
    >>> # Standard usage: 4 comparisons at alpha=0.05
    >>> result = bonferroni_correction([0.001, 0.03, 0.008, 0.15], alpha=0.05)
    >>> result["data"]["corrected_alpha"]
    0.0125
    >>> result["data"]["significant_tests"]
    [0, 2]  # Only tests with p < 0.0125 remain significant

    >>> # Validation failure: invalid p-value
    >>> result = bonferroni_correction([0.01, 1.5, 0.03])
    >>> result["success"]
    False
    >>> result["error"]
    'All p-values must be in range [0, 1]. Found invalid value: 1.5'

    >>> # Edge case: empty p-values list
    >>> result = bonferroni_correction([])
    >>> result["success"]
    False
    >>> result["error"]
    'p_values list cannot be empty'

    Clinical Wound Segmentation Context
    ------------------------------------
    Bonferroni correction is applied when comparing 4 experimental factors
    (focal_tversky loss, augmentation, multi-scale training, optimizer) against baseline:
    - Original alpha: 0.05
    - Corrected alpha: 0.05/4 = 0.0125
    - Only factors with p < 0.0125 are considered statistically significant improvements

    Family-wise confidence stays at 95% across all 4 comparisons.

    References
    ----------
    - Bonferroni, C.E. (1936). "Teoria statistica delle classi e calcolo delle probabilità"
    - Armstrong, R.A. (2014). "When to use the Bonferroni correction."
      Ophthalmic Physiol Opt. 34(5):502-8.
    """
    # Validation: Empty p_values list
    if not p_values:
        return failure("p_values list cannot be empty")

    # Validation: Invalid alpha (must be in open interval (0, 1))
    if alpha <= 0 or alpha >= 1:
        return failure(f"alpha must be in range (0, 1), got: {alpha}")

    # Validation: All p-values must be in range [0, 1]
    for p in p_values:
        if p < 0 or p > 1:
            return failure(f"All p-values must be in range [0, 1]. Found invalid value: {p}")

    # Calculate corrected alpha
    n_comparisons = len(p_values)
    corrected_alpha = alpha / n_comparisons

    # Identify significant tests after correction
    significant_tests = [i for i, p in enumerate(p_values) if p < corrected_alpha]
    n_significant = len(significant_tests)

    # Generate human-readable interpretation
    if n_significant == 0:
        interpretation = (
            f"After Bonferroni correction (alpha={corrected_alpha:.4f}), "
            f"no tests remain significant. This may indicate insufficient statistical power "
            f"or no true effects present."
        )
    elif n_significant == 1:
        interpretation = (
            f"After Bonferroni correction (alpha={corrected_alpha:.4f}), "
            f"test {significant_tests[0]} remains significant."
        )
    else:
        interpretation = (
            f"After Bonferroni correction (alpha={corrected_alpha:.4f}), "
            f"tests {significant_tests} remain significant."
        )

    # Return success with all required fields
    return success({
        "corrected_alpha": corrected_alpha,
        "n_comparisons": n_comparisons,
        "significant_tests": significant_tests,
        "n_significant": n_significant,
        "interpretation": interpretation
    })


def comprehensive_comparison_report(
    baseline_metrics: pd.DataFrame,
    treatment_metrics: pd.DataFrame,
    metric_names: List[str] = ["dice", "iou", "precision", "recall", "boundary_f1"],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Generate a statistical comparison report between baseline and treatment models.

    CONTRACT DEFINITION
    -------------------
    Purpose:
        Orchestrates complete statistical analysis pipeline to determine if treatment model
        shows statistically significant and clinically meaningful improvements over baseline.

    Process:
        1. Merge baseline and treatment DataFrames on 'image_id' to create paired samples
        2. For each metric in metric_names:
           a. Extract paired arrays (baseline_values, treatment_values)
           b. Run Wilcoxon signed-rank test for statistical significance
           c. Calculate Cohen's d for effect size (clinical relevance)
           d. Apply Bonferroni correction: adjusted_alpha = alpha / n_metrics
        3. Aggregate results across all metrics to generate summary recommendation

    Parameters:
        baseline_metrics (pd.DataFrame):
            Per-image metrics from baseline model. Must contain:
            - 'image_id' column (str): Unique identifier for each image
            - One column per metric in metric_names (float): Metric values [0.0, 1.0]

        treatment_metrics (pd.DataFrame):
            Per-image metrics from treatment model. Must contain same structure as baseline_metrics.

        metric_names (List[str], optional):
            List of metric column names to compare. Default: ["dice", "iou", "precision", "recall", "boundary_f1"]

        alpha (float, optional):
            Family-wise error rate before Bonferroni correction. Default: 0.05

    VALIDATION RULES
    ----------------
    Input Validation:
        - baseline_metrics must have 'image_id' column
        - treatment_metrics must have 'image_id' column
        - All metric_names must exist as columns in both DataFrames
        - Merged DataFrame (on 'image_id') must be non-empty (paired samples required)
        - All metric values must be numeric

    Statistical Validation:
        - Wilcoxon test requires n >= 3 paired samples per metric
        - Cohen's d calculation requires non-zero pooled standard deviation

    DECISION CRITERIA
    -----------------
    Per-Metric Classification:
        - SIGNIFICANT: p-value < bonferroni_alpha (alpha / n_metrics)
        - CLINICAL_RELEVANCE:
            * "Large": Cohen's d > 0.8 (substantial improvement)
            * "Medium": 0.5 < Cohen's d <= 0.8 (moderate improvement)
            * "Small": 0.2 < Cohen's d <= 0.5 (minor improvement)
            * "Negligible": Cohen's d <= 0.2 (no meaningful improvement)

    Overall Recommendation:
        - ADOPT: ALL metrics show (Cohen's d > 0.8 AND p < bonferroni_alpha)
        - CONDITIONAL: Some metrics meet adoption criteria
        - REJECT: No metrics meet adoption criteria OR negative effects detected

    RETURNS
    -------
    Success:
        {
            "success": True,
            "data": {
                "n_images": int,  # Number of paired images analyzed
                "metrics_compared": List[str],  # List of metrics analyzed
                "bonferroni_alpha": float,  # Adjusted alpha after correction (alpha / n_metrics)
                "results": {
                    "dice": {
                        "wilcoxon": {
                            "statistic": float,
                            "p_value": float,
                            "significant": bool  # p < bonferroni_alpha
                        },
                        "cohens_d": {
                            "d": float,
                            "interpretation": str  # "Large", "Medium", "Small", "Negligible"
                        },
                        "significant": bool,  # p < bonferroni_alpha
                        "clinical_relevance": str  # "Large", "Medium", "Small", "Negligible"
                    },
                    # ... same structure for each metric in metric_names
                },
                "summary": {
                    "all_significant": bool,  # All metrics pass statistical test
                    "n_significant": int,  # Count of significant metrics
                    "mean_effect_size": float,  # Mean Cohen's d across all metrics
                    "recommendation": str  # "ADOPT", "CONDITIONAL", or "REJECT"
                }
            },
            "error": None
        }

    Failure:
        {
            "success": False,
            "data": None,
            "error": str  # Human-readable error message explaining validation failure
        }

    FAILURE CASES
    -------------
    - Missing 'image_id' column in either DataFrame
    - Missing metric columns in either DataFrame
    - No overlapping image_ids between DataFrames (cannot create paired samples)
    - Insufficient sample size (n < 3) after merging
    - Non-numeric values in metric columns
    - Wilcoxon test failure (e.g., all differences are zero)
    - Cohen's d calculation failure (e.g., zero pooled standard deviation)

    USAGE EXAMPLE
    -------------
    >>> baseline = pd.DataFrame({
    ...     'image_id': ['img1', 'img2', 'img3'],
    ...     'dice': [0.85, 0.80, 0.88],
    ...     'iou': [0.75, 0.70, 0.78]
    ... })
    >>> treatment = pd.DataFrame({
    ...     'image_id': ['img1', 'img2', 'img3'],
    ...     'dice': [0.92, 0.89, 0.94],
    ...     'iou': [0.85, 0.82, 0.88]
    ... })
    >>> result = comprehensive_comparison_report(baseline, treatment, ['dice', 'iou'])
    >>> if result['success']:
    ...     print(result['data']['summary']['recommendation'])

    DEPENDENCIES
    ------------
    Orchestrates:
        - wilcoxon_signed_rank_test(baseline, treatment, alpha)
        - calculate_cohens_d(baseline, treatment)
        - bonferroni_correction(alpha, n_comparisons)

    From:
        - src/utils.io: success(), failure()
    """
    # Step 1: Validate 'image_id' column exists in both DataFrames
    if "image_id" not in baseline_metrics.columns:
        return failure("Baseline DataFrame must contain 'image_id' column")

    if "image_id" not in treatment_metrics.columns:
        return failure("Treatment DataFrame must contain 'image_id' column")

    # Step 2: Validate all metric columns exist in both DataFrames
    for metric_name in metric_names:
        if metric_name not in baseline_metrics.columns:
            return failure(
                f"Baseline DataFrame missing metric column: '{metric_name}'"
            )
        if metric_name not in treatment_metrics.columns:
            return failure(
                f"Treatment DataFrame missing metric column: '{metric_name}'"
            )

    # Step 3: Merge DataFrames on 'image_id' (inner join for paired samples)
    merged_df = baseline_metrics.merge(
        treatment_metrics,
        on="image_id",
        suffixes=("_baseline", "_treatment"),
        how="inner"
    )

    # Step 4: Validate non-empty merge (overlapping images required)
    if len(merged_df) == 0:
        return failure(
            "No overlapping image IDs found between baseline and treatment DataFrames. "
            "Paired samples required for comparison."
        )

    # Step 5: Calculate Bonferroni-corrected alpha
    n_metrics = len(metric_names)
    bonferroni_alpha = alpha / n_metrics

    # Step 6: Initialize results dictionary
    results = {}
    all_p_values = []
    all_effect_sizes = []

    # Step 7: For each metric, perform statistical tests
    for metric_name in metric_names:
        # Extract paired values
        baseline_values = merged_df[f"{metric_name}_baseline"].tolist()
        treatment_values = merged_df[f"{metric_name}_treatment"].tolist()

        # Run Wilcoxon signed-rank test
        wilcoxon_result = wilcoxon_signed_rank_test(
            baseline_values,
            treatment_values,
            alternative="greater"
        )

        # Check if Wilcoxon test failed
        if not wilcoxon_result["success"]:
            return failure(
                f"Wilcoxon test failed for metric '{metric_name}': {wilcoxon_result['error']}"
            )

        # Calculate Cohen's d effect size
        cohens_d_result = calculate_cohens_d(baseline_values, treatment_values)

        # Check if Cohen's d calculation failed
        if not cohens_d_result["success"]:
            return failure(
                f"Cohen's d calculation failed for metric '{metric_name}': {cohens_d_result['error']}"
            )

        # Extract results
        p_value = wilcoxon_result["data"]["p_value"]
        statistic = wilcoxon_result["data"]["statistic"]
        cohens_d = cohens_d_result["data"]["cohens_d"]
        magnitude = cohens_d_result["data"]["magnitude"]

        # Determine if metric is statistically significant after Bonferroni correction
        is_significant = p_value < bonferroni_alpha

        # Store p-value and effect size for summary calculations
        all_p_values.append(p_value)
        all_effect_sizes.append(cohens_d)

        # Store per-metric results
        results[metric_name] = {
            "wilcoxon": {
                "statistic": statistic,
                "p_value": p_value,
                "significant": is_significant
            },
            "cohens_d": {
                "d": cohens_d,
                "interpretation": magnitude.capitalize()
            },
            "significant": is_significant,
            "clinical_relevance": magnitude.capitalize()
        }

    # Step 8: Generate summary statistics
    n_significant = sum(1 for metric_name in metric_names if results[metric_name]["significant"])
    all_significant = (n_significant == n_metrics)
    mean_effect_size = float(np.mean(all_effect_sizes))

    # Step 9: Generate recommendation based on decision criteria
    # ADOPT: ALL metrics show (Cohen's d > 0.8 AND p < bonferroni_alpha)
    # CONDITIONAL: Some metrics meet criteria
    # REJECT: No metrics meet criteria

    if all_significant and mean_effect_size > 0.8:
        recommendation = (
            "ADOPT: Treatment model shows statistically significant improvements "
            f"across all {n_metrics} metrics with large effect sizes (mean d={mean_effect_size:.3f}). "
            "Recommended for clinical deployment."
        )
    elif n_significant > 0:
        recommendation = (
            f"CONDITIONAL: Treatment model shows significant improvements in {n_significant}/{n_metrics} metrics. "
            f"Mean effect size: {mean_effect_size:.3f}. "
            "Further validation recommended before full deployment."
        )
    else:
        recommendation = (
            f"REJECT: Treatment model shows no statistically significant improvements "
            f"after Bonferroni correction (alpha={bonferroni_alpha:.4f}). "
            "Consider alternative approaches or larger sample size."
        )

    # Step 10: Return success with complete results
    return success({
        "n_images": len(merged_df),
        "metrics_compared": metric_names,
        "bonferroni_alpha": bonferroni_alpha,
        "results": results,
        "summary": {
            "all_significant": all_significant,
            "n_significant": n_significant,
            "mean_effect_size": mean_effect_size,
            "recommendation": recommendation
        }
    })
