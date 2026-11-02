"""
Tests for YOLO dataset configuration validation and creation.

Contract: YOLOConfigContract
Implementation: src/data/yolo_config.py

These tests verify that YOLO dataset configuration files are correctly
validated and created for wound segmentation training.
"""

import tempfile
from pathlib import Path

import pytest

from src.data.yolo_config import create_yolo_config, validate_yolo_config


class TestValidateYOLOConfig:
    """
    Test suite for YOLO configuration validation.

    Tests verify that configuration files are properly validated according to
    YOLOv8 requirements and the wound segmentation specification.
    """

    def test_validate_yolo_config_success(self, tmp_path):
        """
        Test successful validation of correctly formatted YOLO config.

        Contract: validate_yolo_config() returns success=True with parsed data
        when given a valid YAML file containing all required keys.

        Expected behavior:
        - Parses YAML successfully
        - Validates all required keys present
        - Returns structured data dictionary
        - Sets success=True, error=None
        """
        # Arrange: Create valid YAML config
        config_path = tmp_path / "valid_config.yaml"
        yaml_content = """path: /kaggle/working/data_wound_seg
train: train_images
val: test_images
test: test_images
nc: 1
names:
  0: wound
"""
        config_path.write_text(yaml_content)

        # Act: Validate configuration
        result = validate_yolo_config(config_path)

        # Assert: Validation succeeds with correct data
        assert result["success"] is True
        assert result["error"] is None
        assert result["data"]["path"] == "/kaggle/working/data_wound_seg"
        assert result["data"]["train"] == "train_images"
        assert result["data"]["val"] == "test_images"
        assert result["data"]["test"] == "test_images"
        assert result["data"]["nc"] == 1
        assert result["data"]["names"] == {0: "wound"}

    def test_validate_yolo_config_missing_file(self):
        """
        Test validation failure when config file doesn't exist.

        Contract: validate_yolo_config() returns success=False with descriptive
        error message when file is not found.

        Expected behavior:
        - Detects missing file
        - Returns success=False
        - Provides clear error message
        - Sets data=None
        """
        # Arrange: Non-existent file path
        config_path = Path("/nonexistent/config.yaml")

        # Act: Attempt validation
        result = validate_yolo_config(config_path)

        # Assert: Validation fails with file not found error
        assert result["success"] is False
        assert result["data"] is None
        assert "not found" in result["error"].lower()

    def test_validate_yolo_config_missing_keys(self, tmp_path):
        """
        Test validation failure when required keys are missing.

        Contract: validate_yolo_config() returns success=False when YAML
        is missing required keys (path, train, val, test, nc, names).

        Expected behavior:
        - Detects missing required keys
        - Returns success=False
        - Error message specifies which keys are missing
        """
        # Arrange: YAML missing 'nc' and 'names' keys
        config_path = tmp_path / "incomplete_config.yaml"
        yaml_content = """path: /data
train: train
val: val
test: test
"""
        config_path.write_text(yaml_content)

        # Act: Validate incomplete configuration
        result = validate_yolo_config(config_path)

        # Assert: Validation fails due to missing keys
        assert result["success"] is False
        assert result["data"] is None
        assert "missing" in result["error"].lower()
        assert "nc" in result["error"] or "names" in result["error"]

    def test_validate_yolo_config_invalid_nc(self, tmp_path):
        """
        Test validation failure when 'nc' is not a positive integer.

        Contract: validate_yolo_config() returns success=False when 'nc'
        (number of classes) is zero, negative, or non-integer.

        Expected behavior:
        - Validates nc > 0
        - Validates nc is integer type
        - Returns descriptive error message
        """
        # Arrange: YAML with invalid nc=0
        config_path = tmp_path / "invalid_nc.yaml"
        yaml_content = """path: /data
train: train
val: val
test: test
nc: 0
names:
  0: wound
"""
        config_path.write_text(yaml_content)

        # Act: Validate configuration with invalid nc
        result = validate_yolo_config(config_path)

        # Assert: Validation fails due to invalid nc
        assert result["success"] is False
        assert result["data"] is None
        assert "nc" in result["error"].lower()

    def test_validate_yolo_config_names_mismatch(self, tmp_path):
        """
        Test validation failure when names dict length doesn't match nc.

        Contract: validate_yolo_config() returns success=False when the number
        of entries in 'names' dictionary doesn't equal 'nc' value.

        Expected behavior:
        - Validates len(names) == nc
        - Returns error specifying mismatch
        - Includes expected vs actual counts in error
        """
        # Arrange: YAML with nc=2 but only 1 name defined
        config_path = tmp_path / "names_mismatch.yaml"
        yaml_content = """path: /data
train: train
val: val
test: test
nc: 2
names:
  0: wound
"""
        config_path.write_text(yaml_content)

        # Act: Validate configuration with name count mismatch
        result = validate_yolo_config(config_path)

        # Assert: Validation fails due to names/nc mismatch
        assert result["success"] is False
        assert result["data"] is None
        assert "names" in result["error"].lower() or "mismatch" in result["error"].lower()


class TestCreateYOLOConfig:
    """
    Test suite for YOLO configuration file creation.

    Tests verify that configuration files are correctly generated with proper
    structure and validation for wound segmentation training.
    """

    def test_create_yolo_config_success(self, tmp_path):
        """
        Test successful creation of YOLO config with default parameters.

        Contract: create_yolo_config() creates valid YAML file at default
        location (dataset_root/wound_dataset.yaml) when given valid inputs.

        Expected behavior:
        - Creates YAML file at expected location
        - File contains all required keys
        - File passes validation
        - Returns config_path and yaml_content in result
        """
        # Arrange: Create mock dataset root with subdirectories
        dataset_root = tmp_path / "data_wound_seg"
        dataset_root.mkdir()
        (dataset_root / "train_images").mkdir()
        (dataset_root / "test_images").mkdir()

        # Act: Create YOLO configuration
        result = create_yolo_config(dataset_root)

        # Assert: Config created successfully and is valid
        assert result["success"] is True
        assert result["error"] is None
        assert result["data"]["config_path"].exists()
        assert result["data"]["config_path"].name == "wound_dataset.yaml"

        # Verify created config passes validation
        validation_result = validate_yolo_config(result["data"]["config_path"])
        assert validation_result["success"] is True
        assert validation_result["data"]["nc"] == 1
        assert validation_result["data"]["names"] == {0: "wound"}

    def test_create_yolo_config_invalid_root(self):
        """
        Test creation failure when dataset root doesn't exist.

        Contract: create_yolo_config() returns success=False when given
        non-existent dataset_root path.

        Expected behavior:
        - Detects missing dataset root
        - Returns success=False
        - Provides clear error message
        - Does not create any files
        """
        # Arrange: Non-existent dataset root
        dataset_root = Path("/nonexistent/dataset")

        # Act: Attempt to create config with invalid root
        result = create_yolo_config(dataset_root)

        # Assert: Creation fails with descriptive error
        assert result["success"] is False
        assert result["data"] is None
        assert "not found" in result["error"].lower() or "exist" in result["error"].lower()

    def test_create_yolo_config_custom_output(self, tmp_path):
        """
        Test config creation with custom output path.

        Contract: create_yolo_config() respects custom output_path parameter
        and creates YAML file at specified location.

        Expected behavior:
        - Creates file at custom output_path location
        - File contains correct dataset_root reference
        - Custom path is returned in result
        """
        # Arrange: Dataset root and custom output location
        dataset_root = tmp_path / "data_wound_seg"
        dataset_root.mkdir()
        (dataset_root / "train_images").mkdir()
        (dataset_root / "test_images").mkdir()

        custom_output = tmp_path / "configs" / "custom_wound.yaml"
        custom_output.parent.mkdir()

        # Act: Create config with custom output path
        result = create_yolo_config(dataset_root, output_path=custom_output)

        # Assert: Config created at custom location
        assert result["success"] is True
        assert result["data"]["config_path"] == custom_output
        assert custom_output.exists()

    def test_create_yolo_config_validates_output(self, tmp_path):
        """
        Test that created config passes validation automatically.

        Contract: create_yolo_config() always produces YAML that passes
        validate_yolo_config() checks.

        Expected behavior:
        - Created config has all required keys
        - nc and names are correctly set
        - Paths are properly formatted
        - Result includes validation confirmation
        """
        # Arrange: Valid dataset root
        dataset_root = tmp_path / "data_wound_seg"
        dataset_root.mkdir()
        (dataset_root / "train_images").mkdir()
        (dataset_root / "test_images").mkdir()

        # Act: Create and immediately validate config
        create_result = create_yolo_config(dataset_root)
        validation_result = validate_yolo_config(create_result["data"]["config_path"])

        # Assert: Created config passes all validation checks
        assert create_result["success"] is True
        assert validation_result["success"] is True
        assert validation_result["data"]["path"] == str(dataset_root)
        assert validation_result["data"]["nc"] == 1
        assert len(validation_result["data"]["names"]) == 1
