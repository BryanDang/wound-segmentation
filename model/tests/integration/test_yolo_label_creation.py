"""
Integration tests for YOLO label creation from masks.

Tests the complete workflow:
1. Copy dataset
2. Convert masks to YOLO .txt format
3. Validate YOLO structure is correct
4. Validate label file format
"""

import pytest
from pathlib import Path
import shutil
import tempfile

from src.data.preprocessing import convert_mask_to_yolo_format


class TestYOLOLabelCreationWorkflow:
    """Test complete mask-to-label conversion workflow."""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset structure."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create expected structure
        (temp_dir / "train" / "images").mkdir(parents=True)
        (temp_dir / "train" / "labels").mkdir(parents=True)
        (temp_dir / "test" / "images").mkdir(parents=True)
        (temp_dir / "test" / "labels").mkdir(parents=True)
        (temp_dir / "train_masks").mkdir(parents=True)
        (temp_dir / "test_masks").mkdir(parents=True)

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_yolo_label_directory_structure(self, temp_dataset):
        """Test that correct YOLO directory structure is created."""
        # Verify expected directories exist
        assert (temp_dataset / "train" / "labels").exists()
        assert (temp_dataset / "test" / "labels").exists()

        # Verify they're writable
        assert (temp_dataset / "train" / "labels").is_dir()
        assert (temp_dataset / "test" / "labels").is_dir()

    def test_convert_single_mask_creates_label_file(self, temp_dataset):
        """Test converting a single mask creates corresponding .txt label."""
        import numpy as np
        import cv2

        # Create a simple test mask (white circle on black background)
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(mask, (256, 256), 100, 255, -1)

        mask_path = temp_dataset / "train_masks" / "test_001.png"
        cv2.imwrite(str(mask_path), mask)

        label_path = temp_dataset / "train" / "labels" / "test_001.txt"

        # Convert mask to YOLO format
        result = convert_mask_to_yolo_format(
            mask_path=str(mask_path),
            image_width=512,
            image_height=512,
            output_path=str(label_path)
        )

        # Validate result
        assert result["success"] is True
        assert result["num_polygons"] == 1
        assert label_path.exists()

        # Validate label file format
        with open(label_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1

            parts = lines[0].strip().split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Validate class ID
            assert class_id == 0

            # Validate coordinates are normalized [0, 1]
            assert all(0.0 <= c <= 1.0 for c in coords)

            # Validate even number of coords (x, y pairs)
            assert len(coords) % 2 == 0

    def test_convert_all_masks_in_directory(self, temp_dataset):
        """Test batch conversion of all masks in a directory."""
        import numpy as np
        import cv2

        # Create 3 test masks
        for i in range(3):
            mask = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(mask, (256, 256), 50 + i*20, 255, -1)

            mask_path = temp_dataset / "train_masks" / f"test_{i:03d}.png"
            cv2.imwrite(str(mask_path), mask)

        # Convert all masks
        mask_dir = temp_dataset / "train_masks"
        label_dir = temp_dataset / "train" / "labels"

        converted_count = 0
        for mask_path in mask_dir.glob("*.png"):
            label_path = label_dir / f"{mask_path.stem}.txt"
            result = convert_mask_to_yolo_format(
                mask_path=str(mask_path),
                image_width=512,
                image_height=512,
                output_path=str(label_path)
            )
            if result["success"]:
                converted_count += 1

        # Validate all converted
        assert converted_count == 3
        assert len(list(label_dir.glob("*.txt"))) == 3

    def test_empty_mask_creates_empty_label(self, temp_dataset):
        """Test that mask with no wound creates empty label file."""
        import numpy as np
        import cv2

        # Create empty mask (all black)
        mask = np.zeros((512, 512), dtype=np.uint8)

        mask_path = temp_dataset / "test_masks" / "no_wound.png"
        cv2.imwrite(str(mask_path), mask)

        label_path = temp_dataset / "test" / "labels" / "no_wound.txt"

        # Convert empty mask
        result = convert_mask_to_yolo_format(
            mask_path=str(mask_path),
            image_width=512,
            image_height=512,
            output_path=str(label_path)
        )

        # Validate result
        assert result["success"] is True
        assert result["num_polygons"] == 0
        assert label_path.exists()

        # Validate label file is empty
        with open(label_path, 'r') as f:
            content = f.read().strip()
            assert content == ""

    def test_yolo_config_points_to_correct_paths(self, temp_dataset):
        """Test that YOLO config references label directories."""
        from src.data.yolo_config import create_yolo_config

        # Create YOLO config
        config_result = create_yolo_config(
            dataset_root=str(temp_dataset),
            train_dir="train",
            val_dir="test",
            test_dir="test"
        )

        assert config_result["success"] is True

        # Read generated YAML
        import yaml
        config_path = Path(config_result["data"]["config_path"])
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate paths reference correct structure
        assert "train" in config["train"]
        assert "test" in config["val"]

        # YOLO expects images in train/ and labels in train/labels
        # Config should point to train/ directory (YOLO auto-finds labels/)
