"""
Unit Tests for Dataset Format Conversion
=========================================
Tests validate contract compliance for mask to YOLO format conversion.

Test Philosophy:
- Test contracts BEFORE implementation
- Each test validates one edge case or requirement from contract
- Tests must pass with correct implementation

Run with: pytest tests/unit/test_baseline_dataset_format.py -v
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import os
import pytest

# Import function to test
from src.data.preprocessing import convert_mask_to_yolo_format

FUNCTIONS_IMPLEMENTED = True


class TestDatasetFormat:
    """
    Test Suite: convert_mask_to_yolo_format
    Contract: DatasetFormatContract
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_empty_mask(self, temp_dir):
        """Contract: Empty mask gives empty .txt file"""
        # Create empty mask (all zeros)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask_path = temp_dir / "empty_mask.png"
        cv2.imwrite(str(mask_path), mask)

        output_path = temp_dir / "empty_label.txt"

        result = convert_mask_to_yolo_format(mask_path, 512, 512, output_path)

        assert result["success"] is True
        assert result["num_polygons"] == 0
        assert output_path.exists()

        # Check file is empty
        with open(output_path, "r") as f:
            content = f.read()
            assert content == ""

    def test_single_wound_region(self, temp_dir):
        """Contract: Single wound gives single polygon in .txt"""
        # Create mask with single square wound
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255  # 100×100 square

        mask_path = temp_dir / "single_wound.png"
        cv2.imwrite(str(mask_path), mask)

        output_path = temp_dir / "single_label.txt"

        result = convert_mask_to_yolo_format(mask_path, 512, 512, output_path)

        assert result["success"] is True
        assert result["num_polygons"] == 1
        assert output_path.exists()

        # Verify file content
        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert lines[0].startswith("0 ")  # Class 0

    def test_multiple_disconnected_regions(self, temp_dir):
        """Contract: Multiple regions gives multiple polygons"""
        # Create mask with two separate wounds
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:150, 100:150] = 255  # First wound
        mask[300:350, 300:350] = 255  # Second wound

        mask_path = temp_dir / "multiple_wounds.png"
        cv2.imwrite(str(mask_path), mask)

        output_path = temp_dir / "multiple_labels.txt"

        result = convert_mask_to_yolo_format(mask_path, 512, 512, output_path)

        assert result["success"] is True
        assert result["num_polygons"] == 2

        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert all(line.startswith("0 ") for line in lines)

    def test_normalized_coordinates(self, temp_dir):
        """Contract: Coordinates normalized to [0, 1] range"""
        # Create simple mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[0:100, 0:100] = 255  # Top-left corner

        mask_path = temp_dir / "normalized.png"
        cv2.imwrite(str(mask_path), mask)

        output_path = temp_dir / "normalized_label.txt"

        result = convert_mask_to_yolo_format(mask_path, 512, 512, output_path)
        assert result["success"] is True

        # Read and parse coordinates
        with open(output_path, "r") as f:
            line = f.readline().strip()
            parts = line.split()
            coords = [float(x) for x in parts[1:]]  # Skip class label

            # All coordinates should be in [0, 1]
            assert all(0.0 <= c <= 1.0 for c in coords)

    def test_file_not_found(self, temp_dir):
        """Contract: Non-existent mask gives FileNotFoundError"""
        mask_path = temp_dir / "nonexistent.png"
        output_path = temp_dir / "output.txt"

        with pytest.raises(FileNotFoundError):
            convert_mask_to_yolo_format(mask_path, 512, 512, output_path)

    def test_invalid_dimensions(self, temp_dir):
        """Contract: Invalid dimensions (<=0) gives ValueError"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask_path = temp_dir / "test.png"
        cv2.imwrite(str(mask_path), mask)

        output_path = temp_dir / "output.txt"

        with pytest.raises(ValueError):
            convert_mask_to_yolo_format(mask_path, 0, 512, output_path)

        with pytest.raises(ValueError):
            convert_mask_to_yolo_format(mask_path, 512, -1, output_path)

    def test_creates_output_directory(self, temp_dir):
        """Contract: Creates output directory if doesn't exist"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        mask_path = temp_dir / "test.png"
        cv2.imwrite(str(mask_path), mask)

        # Output path in non-existent directory
        output_path = temp_dir / "new_dir" / "subdir" / "output.txt"

        result = convert_mask_to_yolo_format(mask_path, 512, 512, output_path)

        assert result["success"] is True
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_grayscale_antialiased_mask(self, temp_dir):
        """Gap Test: Grayscale masks with anti-aliased edges gives binarize at threshold 127"""
        # REALITY: Kaggle masks have grayscale anti-aliased edges (values 0-255)
        # Create mask with anti-aliased edges (grayscale gradient)
        mask = np.zeros((512, 512), dtype=np.uint8)
        # Core wound region (full intensity)
        mask[100:200, 100:200] = 255
        # Anti-aliased edge (gradient)
        mask[99, 100:200] = 128  # Edge pixels
        mask[200, 100:200] = 128
        mask[100:200, 99] = 128
        mask[100:200, 200] = 128

        mask_path = temp_dir / "antialiased.png"
        cv2.imwrite(str(mask_path), mask)

        output_path = temp_dir / "antialiased_label.txt"

        # Should NOT raise ValueError - should binarize at threshold 127
        result = convert_mask_to_yolo_format(mask_path, 512, 512, output_path)

        assert result["success"] is True
        assert result["num_polygons"] >= 1  # At least one polygon detected
        assert output_path.exists()

    def test_string_vs_path_input(self, temp_dir):
        """Gap Test: Handle both string and Path inputs for Kaggle compatibility"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        mask_path = temp_dir / "test.png"
        cv2.imwrite(str(mask_path), mask)

        output_path = temp_dir / "output.txt"

        # Test with string inputs
        result = convert_mask_to_yolo_format(str(mask_path), 512, 512, str(output_path))

        assert result["success"] is True
        assert result["num_polygons"] == 1
        assert output_path.exists()

    def test_return_structure(self, temp_dir):
        """Contract: Return dict with required keys"""
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        mask_path = temp_dir / "test.png"
        cv2.imwrite(str(mask_path), mask)

        output_path = temp_dir / "output.txt"

        result = convert_mask_to_yolo_format(mask_path, 512, 512, output_path)

        assert "success" in result
        assert "num_polygons" in result
        assert "label_file_path" in result
        assert isinstance(result["success"], bool)
        assert isinstance(result["num_polygons"], int)
        assert isinstance(result["label_file_path"], Path)

    @pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="Root ignores directory permissions.",
    )
    def test_readonly_output_directory(self, temp_dir):
        """Gap Test: Read-only output directory gives PermissionError"""
        import os
        import stat

        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 255

        mask_path = temp_dir / "test.png"
        cv2.imwrite(str(mask_path), mask)

        # Create read-only directory
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()
        os.chmod(readonly_dir, stat.S_IRUSR | stat.S_IXUSR)  # Read + execute only

        output_path = readonly_dir / "output.txt"

        try:
            with pytest.raises(PermissionError):
                convert_mask_to_yolo_format(mask_path, 512, 512, output_path)
        finally:
            # Restore write permissions for cleanup
            os.chmod(readonly_dir, stat.S_IRWXU)


# Mark tests as expected to fail until implementation complete
if not FUNCTIONS_IMPLEMENTED:
    pytestmark = pytest.mark.xfail(
        reason="Functions not yet implemented"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
