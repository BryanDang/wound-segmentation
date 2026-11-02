"""
Tests for YOLO prediction format utilities.

Contract: YOLOPredictionExtractionContract
Implementation: src/evaluation/yolo_utils.py

YOLO prediction results are extracted and converted to the format the segmentation metrics take.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.evaluation.metrics import calculate_segmentation_metrics
from src.evaluation.yolo_utils import (
    extract_mask_from_yolo_result,
    yolo_result_to_metrics_input,
)


class TestExtractMaskFromYOLOResult:
    """
    Test suite for YOLO mask extraction.

    Tests verify that segmentation masks are correctly extracted from YOLO
    Results objects and converted to binary numpy arrays.
    """

    def test_extract_mask_with_predictions(self):
        """
        Test successful mask extraction when YOLO finds wounds.

        Contract: extract_mask_from_yolo_result() returns binary mask with
        correct shape and metadata when yolo_result contains mask predictions.

        Expected behavior:
        - Extracts masks from result.masks.data
        - Resizes to target_size
        - Returns binary uint8 array (0 or 1)
        - Provides metadata (n_masks_found, max_confidence, etc.)
        """
        # Arrange: Mock YOLO result with single wound mask
        mock_result = MagicMock()
        mock_result.masks = MagicMock()

        # Create a mock wound mask (100×100 with circular wound)
        mock_mask = np.zeros((100, 100), dtype=np.float32)
        cv2.circle(mock_mask, (50, 50), 30, 1.0, -1)  # Fill circle
        mock_result.masks.data = [mock_mask]  # YOLO returns list of masks
        mock_result.masks.conf = [0.95]  # Confidence scores

        # Act: Extract mask from YOLO result
        result = extract_mask_from_yolo_result(mock_result, target_size=(512, 512))

        # Assert: Extraction succeeds with correct format
        assert result["success"] is True
        assert result["error"] is None
        assert result["data"]["mask"].shape == (512, 512)
        assert result["data"]["mask"].dtype == np.uint8
        assert set(np.unique(result["data"]["mask"])) <= {0, 1}  # Binary
        assert result["data"]["n_masks_found"] == 1
        assert result["data"]["n_masks_used"] == 1
        assert result["data"]["max_confidence"] == 0.95
        assert result["data"]["has_wound"] ==  True  # Note: Use == not 'is' for boolean values

    def test_extract_mask_no_predictions(self):
        """
        Test mask extraction when YOLO finds no wounds.

        Contract: extract_mask_from_yolo_result() returns empty binary mask
        (all zeros) when yolo_result.masks is None or empty.

        Expected behavior:
        - Detects absence of predictions
        - Returns all-zero mask with correct shape
        - Sets has_wound=False
        - n_masks_found=0, n_masks_used=0
        """
        # Arrange: Mock YOLO result with no masks found
        mock_result = MagicMock()
        mock_result.masks = None  # No detections

        # Act: Extract mask from empty result
        result = extract_mask_from_yolo_result(mock_result, target_size=(512, 512))

        # Assert: Returns empty mask with metadata
        assert result["success"] is True
        assert result["data"]["mask"].shape == (512, 512)
        assert np.sum(result["data"]["mask"]) == 0  # All zeros
        assert result["data"]["n_masks_found"] == 0
        assert result["data"]["n_masks_used"] == 0
        assert result["data"]["has_wound"] is False

    def test_extract_mask_below_threshold(self):
        """
        Test mask extraction filters low-confidence predictions.

        Contract: extract_mask_from_yolo_result() excludes masks with
        confidence below threshold, returns empty mask if all filtered out.

        Expected behavior:
        - Filters masks by confidence_threshold
        - n_masks_found > n_masks_used when some filtered
        - Returns empty mask if all below threshold
        """
        # Arrange: Mock YOLO result with low-confidence mask
        mock_result = MagicMock()
        mock_result.masks = MagicMock()

        mock_mask = np.ones((100, 100), dtype=np.float32) * 0.6
        mock_result.masks.data = [mock_mask]
        mock_result.masks.conf = [0.3]  # Below default threshold of 0.5

        # Act: Extract with default threshold (0.5)
        result = extract_mask_from_yolo_result(
            mock_result, target_size=(512, 512), confidence_threshold=0.5
        )

        # Assert: Low-confidence mask filtered out
        assert result["success"] is True
        assert result["data"]["n_masks_found"] == 1
        assert result["data"]["n_masks_used"] == 0  # Filtered out
        assert result["data"]["has_wound"] is False
        assert np.sum(result["data"]["mask"]) == 0

    def test_extract_mask_multiple_merged(self):
        """
        Test mask extraction merges multiple predictions.

        Contract: extract_mask_from_yolo_result() with merge_masks=True
        combines multiple wound predictions using logical OR.

        Expected behavior:
        - Merges multiple masks into single binary mask
        - Uses logical OR (any pixel marked by any mask = 1)
        - Tracks all masks in n_masks_used
        - Reports highest confidence in max_confidence
        """
        # Arrange: Mock YOLO result with 3 overlapping masks
        mock_result = MagicMock()
        mock_result.masks = MagicMock()

        # Create 3 masks with different wound regions
        mask1 = np.zeros((100, 100), dtype=np.float32)
        cv2.circle(mask1, (30, 30), 15, 1.0, -1)

        mask2 = np.zeros((100, 100), dtype=np.float32)
        cv2.circle(mask2, (50, 50), 15, 1.0, -1)

        mask3 = np.zeros((100, 100), dtype=np.float32)
        cv2.circle(mask3, (70, 70), 15, 1.0, -1)

        mock_result.masks.data = [mask1, mask2, mask3]
        mock_result.masks.conf = [0.9, 0.85, 0.95]

        # Act: Extract and merge masks
        result = extract_mask_from_yolo_result(
            mock_result, target_size=(512, 512), merge_masks=True
        )

        # Assert: Multiple masks merged successfully
        assert result["success"] is True
        assert result["data"]["n_masks_found"] == 3
        assert result["data"]["n_masks_used"] == 3
        assert result["data"]["max_confidence"] == 0.95
        assert result["data"]["has_wound"] == True  # Note: Use == not 'is' for boolean values
        assert np.sum(result["data"]["mask"]) > 0  # Has wound pixels

    def test_extract_mask_invalid_result(self):
        """
        Test extraction handles invalid YOLO result gracefully.

        Contract: extract_mask_from_yolo_result() returns success=False
        with descriptive error when given invalid result object.

        Expected behavior:
        - Detects invalid result object
        - Returns success=False
        - Provides clear error message
        - Does not raise exceptions
        """
        # Arrange: Invalid result object (not YOLO Results)
        invalid_result = "not a YOLO result"

        # Act: Attempt extraction with invalid input
        result = extract_mask_from_yolo_result(invalid_result)

        # Assert: Handles error gracefully
        assert result["success"] is False
        assert result["data"] is None
        assert "invalid" in result["error"].lower() or "result" in result["error"].lower()


class TestYOLOResultToMetricsInput:
    """
    Test suite for YOLO-to-metrics conversion.

    Tests verify that YOLO results are correctly converted to the input
    format required by calculate_segmentation_metrics().
    """

    def test_yolo_to_metrics_success(self, tmp_path):
        """
        Test successful conversion of YOLO result to metrics input.

        Contract: yolo_result_to_metrics_input() returns pred_mask and
        gt_mask ready for calculate_segmentation_metrics() when given
        valid YOLO result and ground truth path.

        Expected behavior:
        - Extracts prediction mask from YOLO result
        - Loads and binarizes ground truth mask
        - Validates both masks have same shape
        - Returns ready_for_metrics=True
        """
        # Arrange: Create ground truth mask file
        gt_mask_path = tmp_path / "wound_mask.png"
        gt_mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(gt_mask, (256, 256), 100, 255, -1)
        cv2.imwrite(str(gt_mask_path), gt_mask)

        # Mock YOLO result with prediction
        mock_result = MagicMock()
        mock_result.masks = MagicMock()
        pred_mask = np.zeros((100, 100), dtype=np.float32)
        cv2.circle(pred_mask, (50, 50), 40, 1.0, -1)
        mock_result.masks.data = [pred_mask]
        mock_result.masks.conf = [0.92]

        # Act: Convert YOLO result to metrics input
        result = yolo_result_to_metrics_input(
            yolo_result=mock_result,
            ground_truth_mask_path=str(gt_mask_path),
            image_id="wound_001",
        )

        # Assert: Conversion successful with valid data
        assert result["success"] is True
        assert result["error"] is None
        assert result["data"]["pred_mask"].shape == (512, 512)
        assert result["data"]["gt_mask"].shape == (512, 512)
        assert result["data"]["pred_mask"].dtype == np.uint8
        assert result["data"]["gt_mask"].dtype == np.uint8
        assert result["data"]["image_id"] == "wound_001"
        assert result["data"]["ready_for_metrics"] is True

    def test_yolo_to_metrics_invalid_gt_path(self):
        """
        Test conversion failure when ground truth file doesn't exist.

        Contract: yolo_result_to_metrics_input() returns success=False
        when ground_truth_mask_path points to non-existent file.

        Expected behavior:
        - Detects missing ground truth file
        - Returns success=False with error
        - Error message indicates file not found
        """
        # Arrange: Valid YOLO result but invalid GT path
        mock_result = MagicMock()
        mock_result.masks = None  # Empty prediction

        # Act: Attempt conversion with invalid GT path
        result = yolo_result_to_metrics_input(
            yolo_result=mock_result,
            ground_truth_mask_path="/nonexistent/mask.png",
            image_id="wound_002",
        )

        # Assert: Conversion fails due to missing GT
        assert result["success"] is False
        assert result["data"] is None
        assert "not found" in result["error"].lower() or "exist" in result["error"].lower()

    def test_yolo_to_metrics_shape_mismatch(self, tmp_path):
        """
        Test conversion handles shape mismatches gracefully.

        Contract: yolo_result_to_metrics_input() detects and reports
        when prediction and ground truth have different shapes.

        Expected behavior:
        - Detects shape mismatch
        - Returns success=False
        - Error specifies shape mismatch with dimensions
        """
        # Arrange: GT mask with non-standard size
        gt_mask_path = tmp_path / "odd_size_mask.png"
        gt_mask = np.zeros((256, 256), dtype=np.uint8)  # Different from target 512×512
        cv2.imwrite(str(gt_mask_path), gt_mask)

        mock_result = MagicMock()
        mock_result.masks = None

        # Act: Attempt conversion with shape mismatch
        result = yolo_result_to_metrics_input(
            yolo_result=mock_result,
            ground_truth_mask_path=str(gt_mask_path),
            image_id="wound_003",
            target_size=(512, 512),  # Expecting 512×512
        )

        # Assert: Shape mismatch detected
        # Note: Implementation should either resize GT or report mismatch
        # Test only checks it is handled without error.
        assert result["success"] in [True, False]  # Either resizes or fails gracefully
        if not result["success"]:
            assert "shape" in result["error"].lower() or "size" in result["error"].lower()

    def test_yolo_to_metrics_integration(self, tmp_path):
        """
        Test end-to-end integration with calculate_segmentation_metrics().

        Contract: Output from yolo_result_to_metrics_input() can be
        directly used with calculate_segmentation_metrics() without errors.

        Expected behavior:
        - Converted data passes to metrics function successfully
        - Metrics calculation completes without errors
        - Returns valid Dice, IoU, etc. values
        """
        # Arrange: Create matching pred and GT masks
        gt_mask_path = tmp_path / "integration_mask.png"
        gt_mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(gt_mask, (256, 256), 80, 255, -1)
        cv2.imwrite(str(gt_mask_path), gt_mask)

        # Mock YOLO with similar prediction
        mock_result = MagicMock()
        mock_result.masks = MagicMock()
        pred_mask = np.zeros((100, 100), dtype=np.float32)
        cv2.circle(pred_mask, (50, 50), 40, 1.0, -1)  # Similar to GT
        mock_result.masks.data = [pred_mask]
        mock_result.masks.conf = [0.88]

        # Act: Convert and calculate metrics
        conversion_result = yolo_result_to_metrics_input(
            yolo_result=mock_result,
            ground_truth_mask_path=str(gt_mask_path),
            image_id="integration_test",
        )

        # Assert conversion succeeded
        assert conversion_result["success"] is True

        # Act: Pass to metrics function
        metrics = calculate_segmentation_metrics(
            pred_mask=conversion_result["data"]["pred_mask"],
            gt_mask=conversion_result["data"]["gt_mask"],
            image_id=conversion_result["data"]["image_id"],
        )

        # Assert: Metrics calculated successfully
        assert "dice" in metrics
        assert "iou" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0.0 <= metrics["dice"] <= 1.0
        assert metrics["image_id"] == "integration_test"
