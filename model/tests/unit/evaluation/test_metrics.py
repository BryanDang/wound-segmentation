"""
Unit Tests for Baseline Model Metrics
======================================
Tests validate contract compliance for segmentation metrics calculations.

Test Philosophy:
- Test contracts BEFORE implementation
- Each test validates one edge case or requirement from contract
- Tests must pass with correct implementation
- Tests must fail with incorrect implementation

Run with: pytest tests/unit/test_baseline_metrics.py -v
"""

import numpy as np
import pytest

# Import functions to test
from src.evaluation.metrics import (
    calculate_bootstrap_ci,
    calculate_boundary_f1,
    calculate_segmentation_metrics,
)

FUNCTIONS_IMPLEMENTED = True


class TestSegmentationMetrics:
    """
    Test Suite: calculate_segmentation_metrics
    Contract: MetricsCalculationContract
    """

    def test_perfect_match(self):
        """Contract: Empty pred + empty GT = Dice 1.0, IoU 1.0"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        gt = np.zeros((512, 512), dtype=np.uint8)

        result = calculate_segmentation_metrics(pred, gt, "test_001.jpg")

        assert result["dice"] == 1.0, "Empty masks should have Dice = 1.0"
        assert result["iou"] == 1.0, "Empty masks should have IoU = 1.0"
        assert result["precision"] == 0.0, "No predictions gives Precision = 0.0"
        assert result["recall"] == 0.0, "No ground truth gives Recall = 0.0"

    def test_complete_mismatch(self):
        """Contract: Non-overlapping masks = Dice 0.0, IoU 0.0"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        pred[100:200, 100:200] = 1  # Wound in upper-left

        gt = np.zeros((512, 512), dtype=np.uint8)
        gt[300:400, 300:400] = 1  # Wound in lower-right (no overlap)

        result = calculate_segmentation_metrics(pred, gt, "test_002.jpg")

        assert result["dice"] == 0.0, "No overlap gives Dice = 0.0"
        assert result["iou"] == 0.0, "No overlap gives IoU = 0.0"
        assert result["tp"] == 0, "No true positives"
        assert result["fp"] == 10000, "All predictions are false positives (100×100)"
        assert result["fn"] == 10000, "All ground truth missed (100×100)"

    def test_perfect_overlap(self):
        """Contract: Identical masks = Dice 1.0, IoU 1.0"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        pred[100:200, 100:200] = 1

        gt = pred.copy()  # Identical

        result = calculate_segmentation_metrics(pred, gt, "test_003.jpg")

        assert result["dice"] == 1.0, "Perfect overlap gives Dice = 1.0"
        assert result["iou"] == 1.0, "Perfect overlap gives IoU = 1.0"
        assert result["precision"] == 1.0, "All predictions correct"
        assert result["recall"] == 1.0, "All ground truth found"

    def test_dice_iou_relationship(self):
        """Contract: Dice = 2×IoU / (1+IoU) mathematical relationship"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        pred[100:150, 100:150] = 1  # 50×50 = 2500 pixels

        gt = np.zeros((512, 512), dtype=np.uint8)
        gt[100:200, 100:200] = 1  # 100×100 = 10000 pixels (partial overlap)

        result = calculate_segmentation_metrics(pred, gt, "test_004.jpg")

        # Verify mathematical relationship
        iou = result["iou"]
        dice = result["dice"]
        expected_dice = (2 * iou) / (1 + iou)

        assert (
            abs(dice - expected_dice) < 1e-6
        ), f"Dice-IoU relationship violated: {dice} vs {expected_dice}"

    def test_division_by_zero_precision(self):
        """Contract: No predictions gives Precision = 0.0 (not undefined)"""
        pred = np.zeros((512, 512), dtype=np.uint8)  # No predictions

        gt = np.zeros((512, 512), dtype=np.uint8)
        gt[100:200, 100:200] = 1  # Has ground truth

        result = calculate_segmentation_metrics(pred, gt, "test_005.jpg")

        assert result["precision"] == 0.0, "Division by zero should return 0.0, not raise error"

    def test_division_by_zero_recall(self):
        """Contract: No ground truth gives Recall = 0.0 (not undefined)"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        pred[100:200, 100:200] = 1  # Has predictions

        gt = np.zeros((512, 512), dtype=np.uint8)  # No ground truth

        result = calculate_segmentation_metrics(pred, gt, "test_006.jpg")

        assert result["recall"] == 0.0, "Division by zero should return 0.0, not raise error"

    def test_output_structure(self):
        """Contract: Output must contain all required keys"""
        pred = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        gt = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

        result = calculate_segmentation_metrics(pred, gt, "test_007.jpg")

        required_keys = ["image_id", "dice", "iou", "precision", "recall", "tp", "fp", "tn", "fn"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

    def test_mismatched_shapes(self):
        """Contract: Mismatched shapes raise ValueError"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        gt = np.zeros((256, 256), dtype=np.uint8)  # Different shape

        with pytest.raises(ValueError):
            calculate_segmentation_metrics(pred, gt, "test_008.jpg")

    def test_metric_ranges(self):
        """Contract: All metrics in valid ranges [0.0, 1.0]"""
        pred = np.random.randint(0, 2, (512, 512), dtype=np.uint8)
        gt = np.random.randint(0, 2, (512, 512), dtype=np.uint8)

        result = calculate_segmentation_metrics(pred, gt, "test_009.jpg")

        assert 0.0 <= result["dice"] <= 1.0, "Dice out of range"
        assert 0.0 <= result["iou"] <= 1.0, "IoU out of range"
        assert 0.0 <= result["precision"] <= 1.0, "Precision out of range"
        assert 0.0 <= result["recall"] <= 1.0, "Recall out of range"

    def test_empty_prediction_nonempty_gt_metrics(self):
        """Gap Test: Empty prediction + non-empty GT gives Dice=0.0, IoU=0.0"""
        pred = np.zeros((512, 512), dtype=np.uint8)  # No predictions

        gt = np.zeros((512, 512), dtype=np.uint8)
        gt[100:200, 100:200] = 1  # Has ground truth

        result = calculate_segmentation_metrics(pred, gt, "test_010.jpg")

        assert result["dice"] == 0.0, "Empty prediction gives Dice = 0.0"
        assert result["iou"] == 0.0, "Empty prediction gives IoU = 0.0"
        assert result["precision"] == 0.0, "No predictions gives Precision = 0.0"
        assert result["recall"] == 0.0, "No true positives gives Recall = 0.0"
        assert result["tp"] == 0, "No true positives"
        assert result["fn"] == 10000, "All ground truth missed (100×100)"

    def test_nonempty_prediction_empty_gt_metrics(self):
        """Gap Test: Non-empty prediction + empty GT gives Dice=0.0, IoU=0.0"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        pred[100:200, 100:200] = 1  # Has predictions

        gt = np.zeros((512, 512), dtype=np.uint8)  # No ground truth

        result = calculate_segmentation_metrics(pred, gt, "test_011.jpg")

        assert result["dice"] == 0.0, "No ground truth gives Dice = 0.0"
        assert result["iou"] == 0.0, "No ground truth gives IoU = 0.0"
        assert result["precision"] == 0.0, "All predictions are false positives"
        assert result["recall"] == 0.0, "No ground truth gives Recall = 0.0"
        assert result["tp"] == 0, "No true positives"
        assert result["fp"] == 10000, "All predictions are false positives (100×100)"

    def test_tp_fp_tn_fn_counts(self):
        """Gap Test: Validate TP/FP/TN/FN counts for known configuration"""
        # Create specific test case with known pixel counts
        pred = np.zeros((100, 100), dtype=np.uint8)  # Smaller for easy calculation
        pred[20:40, 20:40] = 1  # 20×20 = 400 predicted pixels

        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[30:50, 30:50] = 1  # 20×20 = 400 ground truth pixels

        # Overlap region: [30:40, 30:40] = 10×10 = 100 pixels
        # TP = 100 (overlap - both predicted AND in GT)
        # FP = 300 (predicted but NOT in GT: 400 pred - 100 overlap)
        # FN = 300 (in GT but NOT predicted: 400 GT - 100 overlap)
        # TN = 9300 (neither predicted NOR in GT: 10000 - 400 - 300 = 9300)
        #      Calculation: Total - pred_positive - fn = 10000 - 400 - 300 = 9300

        result = calculate_segmentation_metrics(pred, gt, "test_012.jpg")

        assert result["tp"] == 100, f"Expected TP=100, got {result['tp']}"
        assert result["fp"] == 300, f"Expected FP=300, got {result['fp']}"
        assert result["fn"] == 300, f"Expected FN=300, got {result['fn']}"
        assert result["tn"] == 9300, f"Expected TN=9300, got {result['tn']}"

        # Verify confusion matrix sums to total pixels
        total = result["tp"] + result["fp"] + result["tn"] + result["fn"]
        assert total == 10000, f"Confusion matrix should sum to 10000, got {total}"


class TestBoundaryF1:
    """
    Test Suite: calculate_boundary_f1
    Contract: BoundaryF1Contract
    """

    def test_perfect_boundary_match(self):
        """Contract: Identical boundaries = Boundary F1 = 1.0"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 1  # Square wound

        result = calculate_boundary_f1(mask, mask, tolerance=2)

        assert result == 1.0, "Identical boundaries should have F1 = 1.0"

    def test_empty_ground_truth(self):
        """Contract: No GT boundary returns 0.0"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        pred[100:200, 100:200] = 1

        gt = np.zeros((512, 512), dtype=np.uint8)  # Empty

        result = calculate_boundary_f1(pred, gt, tolerance=2)

        assert result == 0.0, "No ground truth boundary gives F1 = 0.0"

    def test_empty_prediction(self):
        """Contract: No predicted boundary returns 0.0"""
        pred = np.zeros((512, 512), dtype=np.uint8)  # Empty

        gt = np.zeros((512, 512), dtype=np.uint8)
        gt[100:200, 100:200] = 1

        result = calculate_boundary_f1(pred, gt, tolerance=2)

        assert result == 0.0, "No predicted boundary gives F1 = 0.0"

    def test_both_empty(self):
        """Contract: Both empty returns 1.0 (perfect match)"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        gt = np.zeros((512, 512), dtype=np.uint8)

        result = calculate_boundary_f1(pred, gt, tolerance=2)

        assert result == 1.0, "Both empty gives perfect match, F1 = 1.0"

    def test_tolerance_effect(self):
        """Contract: Higher tolerance gives higher F1 for slightly misaligned boundaries"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        pred[100:200, 100:200] = 1

        gt = np.zeros((512, 512), dtype=np.uint8)
        gt[102:202, 102:202] = 1  # Shifted by 2 pixels

        f1_tight = calculate_boundary_f1(pred, gt, tolerance=1)
        f1_loose = calculate_boundary_f1(pred, gt, tolerance=5)

        assert f1_loose > f1_tight, "Higher tolerance should give higher F1 for shifted boundaries"

    def test_invalid_tolerance(self):
        """Contract: tolerance <= 0 gives ValueError"""
        pred = np.zeros((512, 512), dtype=np.uint8)
        gt = np.zeros((512, 512), dtype=np.uint8)

        with pytest.raises(ValueError):
            calculate_boundary_f1(pred, gt, tolerance=0)

        with pytest.raises(ValueError):
            calculate_boundary_f1(pred, gt, tolerance=-1)


class TestBootstrapCI:
    """
    Test Suite: calculate_bootstrap_ci
    Contract: BootstrapCIContract
    """

    def test_single_value(self):
        """Contract: Single value gives CI = [value, value]"""
        metric_values = [0.85]

        result = calculate_bootstrap_ci(metric_values, confidence_level=0.95, n_bootstrap=1000)

        assert result["mean"] == 0.85
        assert result["ci_lower"] == 0.85
        assert result["ci_upper"] == 0.85

    def test_identical_values(self):
        """Contract: All identical gives CI = [value, value]"""
        metric_values = [0.90] * 100

        result = calculate_bootstrap_ci(metric_values, confidence_level=0.95, n_bootstrap=1000)

        assert abs(result["mean"] - 0.90) < 1e-6, "Mean should be 0.90"
        assert abs(result["ci_lower"] - 0.90) < 0.01, "Lower bound should be ≈0.90"
        assert abs(result["ci_upper"] - 0.90) < 0.01, "Upper bound should be ≈0.90"

    def test_ci_contains_mean(self):
        """Contract: Mean must fall within CI"""
        np.random.seed(42)
        metric_values = np.random.uniform(0.7, 0.9, 100).tolist()

        result = calculate_bootstrap_ci(metric_values, confidence_level=0.95, n_bootstrap=10000)

        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"], "Mean must be within CI"

    def test_ci_width_increases_with_variance(self):
        """Contract: Higher variance gives wider CI"""
        # Low variance data
        low_var = np.random.normal(0.85, 0.02, 100).tolist()
        # High variance data
        high_var = np.random.normal(0.85, 0.10, 100).tolist()

        result_low = calculate_bootstrap_ci(low_var, confidence_level=0.95, n_bootstrap=5000)
        result_high = calculate_bootstrap_ci(high_var, confidence_level=0.95, n_bootstrap=5000)

        width_low = result_low["ci_upper"] - result_low["ci_lower"]
        width_high = result_high["ci_upper"] - result_high["ci_lower"]

        assert width_high > width_low, "Higher variance should produce wider CI"

    def test_reproducibility(self):
        """Contract: Same seed gives same CI"""
        metric_values = np.random.uniform(0.7, 0.9, 50).tolist()

        result1 = calculate_bootstrap_ci(metric_values, random_seed=42, n_bootstrap=1000)
        result2 = calculate_bootstrap_ci(metric_values, random_seed=42, n_bootstrap=1000)

        assert result1["ci_lower"] == result2["ci_lower"], "Same seed should give identical results"
        assert result1["ci_upper"] == result2["ci_upper"], "Same seed should give identical results"

    def test_empty_input(self):
        """Contract: Empty list gives ValueError"""
        with pytest.raises(ValueError):
            calculate_bootstrap_ci([], confidence_level=0.95)

    def test_invalid_confidence_level(self):
        """Contract: Confidence not in (0, 1) gives ValueError"""
        metric_values = [0.8, 0.85, 0.9]

        with pytest.raises(ValueError):
            calculate_bootstrap_ci(metric_values, confidence_level=0.0)

        with pytest.raises(ValueError):
            calculate_bootstrap_ci(metric_values, confidence_level=1.0)

        with pytest.raises(ValueError):
            calculate_bootstrap_ci(metric_values, confidence_level=1.5)

    def test_output_structure(self):
        """Contract: Output must contain all required keys"""
        metric_values = np.random.uniform(0.7, 0.9, 50).tolist()

        result = calculate_bootstrap_ci(metric_values)

        required_keys = ["mean", "std", "ci_lower", "ci_upper", "confidence_level", "n_samples"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"


# xfail until the functions exist.
if not FUNCTIONS_IMPLEMENTED:
    pytestmark = pytest.mark.xfail(
        reason="Functions not yet implemented"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
