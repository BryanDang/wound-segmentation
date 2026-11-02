"""
YOLO prediction format utilities for wound segmentation evaluation.

Contract: YOLOPredictionExtractionContract

This module provides functions to extract segmentation masks from YOLO
prediction results and convert them to formats compatible with our
metrics calculation functions.
"""

from typing import Any, Dict, Tuple

import numpy as np


def extract_mask_from_yolo_result(
    yolo_result: Any,
    target_size: Tuple[int, int] = (512, 512),
    confidence_threshold: float = 0.5,
    merge_masks: bool = True,
) -> Dict[str, Any]:
    """
    Extract binary segmentation mask from YOLO prediction result.

    Contract Definition:
    --------------------
    Inputs:
        - yolo_result: YOLO Results object from model.predict()
        - target_size: (height, width) for output mask (default: 512×512)
        - confidence_threshold: Minimum confidence for mask inclusion (default: 0.5)
        - merge_masks: Whether to merge multiple masks into one (default: True)

    Expected Outputs (Success):
        {
            "success": True,
            "data": {
                "mask": np.ndarray,      # Binary mask (H×W, dtype=uint8, values 0 or 1)
                "n_masks_found": int,    # Number of mask predictions in result
                "n_masks_used": int,     # Number above confidence threshold
                "max_confidence": float, # Highest confidence score
                "has_wound": bool        # Whether any wound detected
            },
            "error": None
        }

    Expected Outputs (Failure):
        {
            "success": False,
            "data": None,
            "error": str  # Error message describing extraction failure
        }

    Behavior:
    ---------
    1. Check if yolo_result contains masks (result.masks is not None)
    2. Filter masks by confidence_threshold
    3. Resize each mask to target_size
    4. If merge_masks=True: combine with logical OR
    5. If merge_masks=False: return highest-confidence mask only
    6. Convert to binary (0 or 1) uint8 array
    7. Return structured result with metadata

    Test Coverage:
    --------------
    - tests/unit/evaluation/test_yolo_utils.py::test_extract_mask_with_predictions
    - tests/unit/evaluation/test_yolo_utils.py::test_extract_mask_no_predictions
    - tests/unit/evaluation/test_yolo_utils.py::test_extract_mask_below_threshold
    - tests/unit/evaluation/test_yolo_utils.py::test_extract_mask_multiple_merged
    - tests/unit/evaluation/test_yolo_utils.py::test_extract_mask_invalid_result

    Args:
        yolo_result: YOLO Results object from model.predict()
        target_size: Output mask dimensions (height, width)
        confidence_threshold: Minimum confidence for inclusion
        merge_masks: Whether to merge multiple detections

    Returns:
        Result dictionary with success status, extracted mask, and metadata

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO('yolov8n-seg.pt')
        >>> results = model.predict('wound_image.png')
        >>> mask_result = extract_mask_from_yolo_result(results[0])
        >>> if mask_result["success"]:
        ...     pred_mask = mask_result["data"]["mask"]
        ...     print(f"Shape: {pred_mask.shape}, dtype: {pred_mask.dtype}")
    """
    import cv2

    # Validate yolo_result is a proper object (not string, int, etc.)
    if not hasattr(yolo_result, "__dict__"):
        return {
            "success": False,
            "data": None,
            "error": f"Invalid YOLO result object: expected Results object, got {type(yolo_result).__name__}",
        }

    # Validate yolo_result has expected structure
    try:
        # Check if result has masks attribute
        has_masks = hasattr(yolo_result, "masks") and yolo_result.masks is not None
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Invalid YOLO result object: {e}",
        }

    # Initialize empty mask
    final_mask = np.zeros(target_size, dtype=np.uint8)

    # If no masks detected, return empty mask
    if not has_masks:
        return {
            "success": True,
            "data": {
                "mask": final_mask,
                "n_masks_found": 0,
                "n_masks_used": 0,
                "max_confidence": 0.0,
                "has_wound": False,
            },
            "error": None,
        }

    # Extract masks and confidences
    masks = yolo_result.masks.data
    confidences = yolo_result.masks.conf if hasattr(yolo_result.masks, "conf") else [1.0] * len(masks)

    n_masks_found = len(masks)

    # Filter by confidence threshold
    filtered_indices = [i for i, conf in enumerate(confidences) if conf >= confidence_threshold]
    n_masks_used = len(filtered_indices)

    # If all filtered out, return empty mask
    if n_masks_used == 0:
        return {
            "success": True,
            "data": {
                "mask": final_mask,
                "n_masks_found": n_masks_found,
                "n_masks_used": 0,
                "max_confidence": max(confidences) if len(confidences) > 0 else 0.0,
                "has_wound": False,
            },
            "error": None,
        }

    # Process filtered masks
    max_confidence = max([confidences[i] for i in filtered_indices])

    for idx in filtered_indices:
        mask = masks[idx]

        # Convert to numpy if needed (YOLO returns torch tensors)
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()

        # Resize to target size
        mask_resized = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

        # Binarize
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Merge with existing mask (logical OR)
        if merge_masks:
            final_mask = np.maximum(final_mask, mask_binary)
        else:
            # Keep only highest confidence mask
            if confidences[idx] == max_confidence:
                final_mask = mask_binary

    return {
        "success": True,
        "data": {
            "mask": final_mask,
            "n_masks_found": n_masks_found,
            "n_masks_used": n_masks_used,
            "max_confidence": float(max_confidence),
            "has_wound": np.sum(final_mask) > 0,
        },
        "error": None,
    }


def yolo_result_to_metrics_input(
    yolo_result: Any,
    ground_truth_mask_path: str,
    image_id: str,
    target_size: Tuple[int, int] = (512, 512),
) -> Dict[str, Any]:
    """
    Convert YOLO result to format compatible with calculate_segmentation_metrics().

    Contract Definition:
    --------------------
    Inputs:
        - yolo_result: YOLO Results object from model.predict()
        - ground_truth_mask_path: Path to ground truth PNG mask
        - image_id: Image identifier for metrics record
        - target_size: Mask dimensions (default: 512×512)

    Expected Outputs (Success):
        {
            "success": True,
            "data": {
                "pred_mask": np.ndarray,  # Binary prediction (H×W, uint8, 0 or 1)
                "gt_mask": np.ndarray,    # Binary ground truth (H×W, uint8, 0 or 1)
                "image_id": str,          # Image identifier
                "ready_for_metrics": bool # Always True on success
            },
            "error": None
        }

    Expected Outputs (Failure):
        {
            "success": False,
            "data": None,
            "error": str  # Error message describing conversion failure
        }

    Behavior:
    ---------
    1. Extract prediction mask from yolo_result using extract_mask_from_yolo_result()
    2. Load ground truth mask from file
    3. Binarize ground truth at threshold 127
    4. Validate both masks have same shape
    5. Validate both masks are binary (only 0 and 1 values)
    6. Return structured data ready for calculate_segmentation_metrics()

    Integration:
    ------------
    Output can be directly unpacked into calculate_segmentation_metrics():

    >>> result = yolo_result_to_metrics_input(yolo_result, gt_path, img_id)
    >>> if result["success"]:
    >>>     metrics = calculate_segmentation_metrics(
    >>>         pred_mask=result["data"]["pred_mask"],
    >>>         gt_mask=result["data"]["gt_mask"],
    >>>         image_id=result["data"]["image_id"]
    >>>     )

    Test Coverage:
    --------------
    - tests/unit/evaluation/test_yolo_utils.py::test_yolo_to_metrics_success
    - tests/unit/evaluation/test_yolo_utils.py::test_yolo_to_metrics_invalid_gt_path
    - tests/unit/evaluation/test_yolo_utils.py::test_yolo_to_metrics_shape_mismatch
    - tests/unit/evaluation/test_yolo_utils.py::test_yolo_to_metrics_integration

    Args:
        yolo_result: YOLO Results object
        ground_truth_mask_path: Path to ground truth mask image
        image_id: Image identifier string
        target_size: Mask dimensions

    Returns:
        Result dictionary with prediction and ground truth masks ready for metrics

    Example:
        >>> result = yolo_result_to_metrics_input(
        ...     yolo_result=results[0],
        ...     ground_truth_mask_path="/data/masks/wound_001.png",
        ...     image_id="wound_001"
        ... )
        >>> if result["success"]:
        ...     # Ready to pass to calculate_segmentation_metrics()
        ...     metrics = calculate_segmentation_metrics(**result["data"])
    """
    import cv2
    from pathlib import Path

    # Extract prediction mask from YOLO result
    pred_result = extract_mask_from_yolo_result(yolo_result, target_size=target_size)

    if not pred_result["success"]:
        return {
            "success": False,
            "data": None,
            "error": f"Failed to extract prediction mask: {pred_result['error']}",
        }

    pred_mask = pred_result["data"]["mask"]

    # Load ground truth mask
    gt_path = Path(ground_truth_mask_path)
    if not gt_path.exists():
        return {
            "success": False,
            "data": None,
            "error": f"Ground truth mask not found: {gt_path}",
        }

    try:
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to read ground truth mask: {gt_path}",
            }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Error loading ground truth: {e}",
        }

    # Resize GT to target size if needed (handle shape mismatches)
    if gt_mask.shape != target_size:
        gt_mask = cv2.resize(gt_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    # Binarize ground truth at threshold 127
    _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

    # Validate masks have same shape
    if pred_mask.shape != gt_mask.shape:
        return {
            "success": False,
            "data": None,
            "error": f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}",
        }

    # Return data ready for metrics calculation
    return {
        "success": True,
        "data": {
            "pred_mask": pred_mask,
            "gt_mask": gt_mask,
            "image_id": image_id,
            "ready_for_metrics": True,
        },
        "error": None,
    }
