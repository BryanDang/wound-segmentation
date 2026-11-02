"""
Evaluation metrics for wound segmentation.
"""

from typing import Any, Dict, List

import cv2
import numpy as np


def calculate_segmentation_metrics(
    pred_mask: np.ndarray, gt_mask: np.ndarray, image_id: str
) -> Dict[str, Any]:
    """
    Calculate segmentation metrics (Dice, IoU, Precision, Recall).

    Contract: MetricsCalculationContract
    Tests: tests/unit/evaluation/test_metrics.py

    Args:
        pred_mask: Predicted binary mask (H×W, values 0 or 1)
        gt_mask: Ground truth binary mask (H×W, values 0 or 1)
        image_id: Image identifier for record keeping

    Returns:
        {
            "image_id": str,
            "dice": float [0.0, 1.0],
            "iou": float [0.0, 1.0],
            "precision": float [0.0, 1.0],
            "recall": float [0.0, 1.0],
            "tp": int,
            "fp": int,
            "tn": int,
            "fn": int
        }

    Raises:
        ValueError: If masks have mismatched shapes
    """
    # Validate shapes
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")

    # Ensure binary (0 or 1)
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # Calculate confusion matrix elements
    tp = int(np.sum((pred_mask == 1) & (gt_mask == 1)))
    fp = int(np.sum((pred_mask == 1) & (gt_mask == 0)))
    tn = int(np.sum((pred_mask == 0) & (gt_mask == 0)))
    fn = int(np.sum((pred_mask == 0) & (gt_mask == 1)))

    # Calculate metrics with division-by-zero handling
    # Dice: 2×TP / (2×TP + FP + FN)
    dice_denominator = 2 * tp + fp + fn
    if dice_denominator == 0:
        dice = 1.0  # Both masks empty = perfect match
    else:
        dice = (2.0 * tp) / dice_denominator

    # IoU: TP / (TP + FP + FN)
    iou_denominator = tp + fp + fn
    if iou_denominator == 0:
        iou = 1.0  # Both masks empty = perfect match
    else:
        iou = float(tp) / iou_denominator

    # Precision: TP / (TP + FP)
    precision_denominator = tp + fp
    if precision_denominator == 0:
        precision = 0.0  # No predictions
    else:
        precision = float(tp) / precision_denominator

    # Recall: TP / (TP + FN)
    recall_denominator = tp + fn
    if recall_denominator == 0:
        recall = 0.0  # No ground truth
    else:
        recall = float(tp) / recall_denominator

    return {
        "image_id": image_id,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def calculate_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance: int = 2) -> float:
    """
    Calculate boundary F1 score for edge accuracy.

    Contract: BoundaryF1Contract
    Tests: tests/unit/evaluation/test_metrics.py

    Args:
        pred_mask: Predicted binary mask (H×W, values 0 or 1)
        gt_mask: Ground truth binary mask (H×W, values 0 or 1)
        tolerance: Boundary tolerance in pixels (default=2)

    Returns:
        Float value [0.0, 1.0] representing boundary F1 score

    Raises:
        ValueError: If tolerance <= 0
    """
    # Validate tolerance
    if tolerance <= 0:
        raise ValueError(f"Tolerance must be > 0, got {tolerance}")

    # Ensure binary
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # Handle empty masks
    pred_empty = np.sum(pred_mask) == 0
    gt_empty = np.sum(gt_mask) == 0

    if pred_empty and gt_empty:
        return 1.0  # Both empty = perfect match

    if pred_empty or gt_empty:
        return 0.0  # One empty, one not = no match

    # Extract boundaries using Canny edge detection
    pred_boundary = cv2.Canny(pred_mask * 255, 100, 200)
    gt_boundary = cv2.Canny(gt_mask * 255, 100, 200)

    # Dilate ground truth boundary by tolerance to create boundary zone
    kernel = np.ones((tolerance * 2 + 1, tolerance * 2 + 1), np.uint8)
    gt_zone = cv2.dilate(gt_boundary, kernel, iterations=1)

    # Calculate TP, FP, FN within boundary zone
    pred_boundary_pixels = pred_boundary > 0
    gt_zone_pixels = gt_zone > 0

    # TP: predicted boundary pixels within GT zone
    tp = np.sum(pred_boundary_pixels & gt_zone_pixels)

    # FP: predicted boundary pixels outside GT zone
    fp = np.sum(pred_boundary_pixels & ~gt_zone_pixels)

    # FN: GT boundary pixels not covered by predicted boundary
    # Dilate predicted boundary and check coverage
    pred_zone = cv2.dilate(pred_boundary, kernel, iterations=1)
    gt_boundary_pixels = gt_boundary > 0
    fn = np.sum(gt_boundary_pixels & ~(pred_zone > 0))

    # Calculate F1 = 2×precision×recall / (precision + recall)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_bootstrap_ci(
    metric_values: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence intervals for metrics.

    Contract: BootstrapCIContract
    Tests: tests/unit/evaluation/test_metrics.py

    Args:
        metric_values: List of metric values from test set
        confidence_level: Confidence level (default=0.95 for 95% CI)
        n_bootstrap: Number of bootstrap iterations (default=10000)
        random_seed: Random seed for reproducibility (default=42)

    Returns:
        {
            "mean": float,
            "std": float,
            "ci_lower": float,
            "ci_upper": float,
            "confidence_level": float,
            "n_samples": int
        }

    Raises:
        ValueError: If metric_values is empty or confidence_level not in (0, 1)
    """
    # Validate inputs
    if len(metric_values) == 0:
        raise ValueError("metric_values cannot be empty")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    # Convert to numpy array
    values = np.array(metric_values)

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(resampled))

    bootstrap_means = np.array(bootstrap_means)

    # Calculate statistics
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))  # Sample standard deviation

    # Calculate confidence interval percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(np.percentile(bootstrap_means, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_means, upper_percentile))

    return {
        "mean": mean,
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "n_samples": len(values),
    }
