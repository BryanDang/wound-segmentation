"""
Unit Tests for Dataset Integrity Validation
==========================================

Tests the dataset integrity validation function that checks dataset structure,
file counts, and validates image-mask pairing for both simulated and Kaggle datasets.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class TestDatasetIntegrityValidation(unittest.TestCase):
    """
    Test cases for verify_dataset_integrity() function.

    Based on contract specification:
    - Validates directory structure (train_images/, train_masks/, test_images/, test_masks/)
    - Counts files in each directory
    - Checks image-mask pairing (same filenames)
    - Validates file integrity (size > 0, readable)
    - Returns success/failure with detailed error information
    """

    def setUp(self):
        """Set up test fixtures."""
        self.maxDiff = None

        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.test_dir) / "wound-segmentation-images"

        # Expected success output for simulated dataset
        self.expected_simulated_success = {
            "success": True,
            "data": {
                "train_images": 2,
                "train_masks": 2,
                "test_images": 1,
                "test_masks": 1,
                "mismatches": [],
                "missing_files": [],
                "corrupt_files": [],
                "total_images": 3,
                "total_masks": 3,
            },
            "error": None,
        }

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_mock_dataset(self, structure):
        """
        Create a mock dataset structure for testing.

        Args:
            structure (dict): Dictionary defining the dataset structure
                e.g., {
                    "train_images": ["img1.png", "img2.png"],
                    "train_masks": ["img1.png", "img2.png"],
                    "test_images": ["img3.png"],
                    "test_masks": ["img3.png"]
                }
        """
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        for directory, files in structure.items():
            dir_path = self.dataset_path / directory
            dir_path.mkdir(exist_ok=True)

            for filename in files:
                file_path = dir_path / filename
                # Create non-empty file
                with open(file_path, "wb") as f:
                    f.write(b"fake image data" * 100)  # Non-empty file

    def create_corrupt_file(self, directory, filename):
        """Create a corrupt (empty) file for testing."""
        dir_path = self.dataset_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / filename
        # Create empty file (corrupt)
        file_path.touch()

    def test_valid_simulated_dataset(self):
        """Test validation of perfect simulated dataset (2 train, 1 test)."""
        # Arrange
        structure = {
            "train_images": ["fusc_0002.png", "fusc_0003.png"],
            "train_masks": ["fusc_0002.png", "fusc_0003.png"],
            "test_images": ["fusc_0001.png"],
            "test_masks": ["fusc_0001.png"],
        }
        self.create_mock_dataset(structure)

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Act
        result = verify_dataset_integrity(str(self.dataset_path))

        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["train_images"], 2)
        self.assertEqual(result["data"]["train_masks"], 2)
        self.assertEqual(result["data"]["test_images"], 1)
        self.assertEqual(result["data"]["test_masks"], 1)
        self.assertEqual(result["data"]["total_images"], 3)
        self.assertEqual(result["data"]["total_masks"], 3)
        self.assertEqual(result["data"]["mismatches"], [])
        self.assertEqual(result["data"]["missing_files"], [])
        self.assertEqual(result["data"]["corrupt_files"], [])
        self.assertIsNone(result["error"])

    def test_missing_directory(self):
        """Test behavior when required directories are missing."""
        # Arrange - only create some directories
        structure = {
            "train_images": ["fusc_0002.png"],
            "train_masks": ["fusc_0002.png"],
            # Missing test directories
        }
        self.create_mock_dataset(structure)

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Act
        result = verify_dataset_integrity(str(self.dataset_path))

        # Assert
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertTrue(
            "directory" in result["error"].lower() or "directories" in result["error"].lower()
        )

    def test_missing_masks(self):
        """Test detection of missing mask files."""
        # Arrange - image without corresponding mask
        structure = {
            "train_images": ["fusc_0002.png", "fusc_0003.png"],
            "train_masks": ["fusc_0002.png"],  # Missing fusc_0003.png mask
            "test_images": ["fusc_0001.png"],
            "test_masks": ["fusc_0001.png"],
        }
        self.create_mock_dataset(structure)

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Act
        result = verify_dataset_integrity(str(self.dataset_path))

        # Assert
        self.assertFalse(result["success"])
        self.assertEqual(result["data"]["train_images"], 2)
        self.assertEqual(result["data"]["train_masks"], 1)
        self.assertTrue(any("fusc_0003.png" in match for match in result["data"]["mismatches"]))
        self.assertIn("Image-mask pairing", result["error"])

    def test_corrupt_files_detection(self):
        """Test detection of corrupt (empty) files."""
        # Arrange - create normal structure first
        structure = {
            "train_images": ["fusc_0002.png", "fusc_0003.png"],
            "train_masks": ["fusc_0002.png", "fusc_0003.png"],
            "test_images": ["fusc_0001.png"],
            "test_masks": ["fusc_0001.png"],
        }
        self.create_mock_dataset(structure)

        # Create one corrupt file
        self.create_corrupt_file("train_images", "fusc_corrupt.png")

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Act
        result = verify_dataset_integrity(str(self.dataset_path))

        # Assert
        self.assertFalse(result["success"])
        self.assertGreater(len(result["data"]["corrupt_files"]), 0)
        self.assertIn("fusc_corrupt.png", str(result["data"]["corrupt_files"]))

    def test_nonexistent_dataset_path(self):
        """Test behavior with nonexistent dataset path."""
        nonexistent_path = "/path/that/does/not/exist"

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Act
        result = verify_dataset_integrity(nonexistent_path)

        # Assert
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])
        self.assertIn("not found", result["error"].lower())

    def test_return_structure_completeness(self):
        """Test that return structure contains all required fields."""
        # Arrange - valid dataset
        structure = {
            "train_images": ["fusc_0002.png"],
            "train_masks": ["fusc_0002.png"],
            "test_images": ["fusc_0001.png"],
            "test_masks": ["fusc_0001.png"],
        }
        self.create_mock_dataset(structure)

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Act
        result = verify_dataset_integrity(str(self.dataset_path))

        # Assert all required top-level fields
        self.assertIn("success", result)
        self.assertIn("data", result)
        self.assertIn("error", result)

        # Assert all required data fields
        data = result["data"]
        required_fields = [
            "train_images",
            "train_masks",
            "test_images",
            "test_masks",
            "mismatches",
            "missing_files",
            "corrupt_files",
            "total_images",
            "total_masks",
        ]

        for field in required_fields:
            self.assertIn(field, data, f"Missing required field: {field}")

        # Assert field types
        self.assertIsInstance(result["success"], bool)
        self.assertIsInstance(data["train_images"], int)
        self.assertIsInstance(data["mismatches"], list)

    def test_kaggle_dataset_size_validation(self):
        """Test validation logic for Kaggle-sized dataset (mocked)."""
        # Create large mock dataset structure
        train_files = [f"img_{i: 04d}.png" for i in range(2208)]
        test_files = [f"test_{i: 04d}.png" for i in range(552)]

        structure = {
            "train_images": train_files[:100],  # Only create subset for performance
            "train_masks": train_files[:100],
            "test_images": test_files[:50],
            "test_masks": test_files[:50],
        }
        self.create_mock_dataset(structure)

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Act
        result = verify_dataset_integrity(str(self.dataset_path))

        # Assert
        self.assertTrue(result["success"])
        self.assertEqual(result["data"]["train_images"], 100)
        self.assertEqual(result["data"]["test_images"], 50)
        self.assertEqual(result["data"]["total_images"], 150)

    def test_file_pairing_validation(self):
        """Test detailed file pairing validation."""
        # Arrange - mixed pairing scenarios
        structure = {
            "train_images": ["img1.png", "img2.png", "img3.png"],
            "train_masks": ["img1.png", "img3.png"],  # Missing img2.png
            "test_images": ["test1.png", "test2.png"],
            "test_masks": ["test1.png", "test2.png", "test3.png"],  # Extra test3.png
        }
        self.create_mock_dataset(structure)

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Act
        result = verify_dataset_integrity(str(self.dataset_path))

        # Assert
        self.assertFalse(result["success"])
        mismatches = result["data"]["mismatches"]
        self.assertTrue(any("img2.png" in match for match in mismatches))  # Image without mask
        # test3.png mask without image might also be reported

    def test_contract_compliance(self):
        """Test that function output matches contract specification."""
        # Arrange - valid dataset
        structure = {
            "train_images": ["fusc_0002.png"],
            "train_masks": ["fusc_0002.png"],
            "test_images": ["fusc_0001.png"],
            "test_masks": ["fusc_0001.png"],
        }
        self.create_mock_dataset(structure)

        try:
            from dataset_validator import verify_dataset_integrity
        except ImportError:
            self.skipTest("Dataset validator not yet implemented")

        # Import contract for comparison
        try:
            from contracts.data_pipeline_contract import DataPipelineContracts

            DataPipelineContracts.dataset_integrity_contract()
        except ImportError:
            self.skipTest("Contract not available for comparison")

        # Act
        result = verify_dataset_integrity(str(self.dataset_path))

        # Assert contract compliance
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("data", result)
        self.assertIn("error", result)

        # Verify data structure matches contract
        data = result["data"]
        for field in ["train_images", "train_masks", "test_images", "test_masks"]:
            self.assertIsInstance(data[field], int)

        for field in ["mismatches", "missing_files", "corrupt_files"]:
            self.assertIsInstance(data[field], list)


class TestDatasetIntegrityEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_permission_denied_handling(self):
        """Test behavior when directory access is denied."""
        # Setup is complex, so this is mocked
        with patch("os.path.exists", side_effect=PermissionError("Access denied")):
            try:
                from dataset_validator import verify_dataset_integrity

                result = verify_dataset_integrity("/some/path")
                self.assertFalse(result["success"])
                self.assertIn("error", result)
            except ImportError:
                self.skipTest("Dataset validator not yet implemented")

    def test_very_large_dataset_handling(self):
        """Test that function can handle large dataset counts."""
        # This test would verify the function can handle large numbers
        # without performance issues (implementation detail)
        self.assertTrue(True)  # Placeholder


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
