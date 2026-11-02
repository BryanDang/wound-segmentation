"""
Unit Tests for Environment Detection
===================================

Tests the environment detection function that identifies whether code is running
in Kaggle or simulated environment and returns appropriate paths and counts.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.environment import detect_environment


class TestEnvironmentDetection(unittest.TestCase):
    """
    Test cases for detect_environment() function.

    Based on contract specification:
    - Returns environment type ("kaggle" or "simulated_kaggle")
    - Returns appropriate paths for each environment
    - Returns correct expected counts (2208/552 for Kaggle, 2/1 for simulated)
    - Never raises exceptions
    """

    def setUp(self):
        """Set up test fixtures."""
        self.maxDiff = None  # Show full diff for assertion failures

        # Expected outputs based on contract
        self.expected_kaggle = {
            "environment": "kaggle",
            "dataset_path": "/kaggle/input/wound-segmentation-images",
            "working_path": "/kaggle/working",
            "expected_counts": {
                "train_images": 2208,
                "train_masks": 2208,
                "test_images": 552,
                "test_masks": 552,
            },
        }

        # Note: simulated path will be resolved dynamically in tests
        self.expected_simulated_structure = {
            "environment": "simulated_kaggle",
            "expected_counts": {
                "train_images": 2,
                "train_masks": 2,
                "test_images": 1,
                "test_masks": 1,
            },
        }

    @patch("os.path.exists")
    def test_detect_kaggle_environment(self, mock_exists):
        """detect_environment() returns "kaggle" when /kaggle/input exists."""
        mock_exists.return_value = True

        result = detect_environment()

        mock_exists.assert_called_with("/kaggle/input")
        self.assertEqual(result, "kaggle")

    @patch("os.path.exists")
    def test_detect_local_environment(self, mock_exists):
        """detect_environment() returns "local" when /kaggle/input is missing."""
        mock_exists.return_value = False

        result = detect_environment()

        mock_exists.assert_called_with("/kaggle/input")
        self.assertEqual(result, "local")

    @patch("os.path.exists")
    def test_detect_simulated_environment(self, mock_exists):
        """Test detection of simulated Kaggle environment."""
        # Arrange
        mock_exists.return_value = False  # /kaggle/input does not exist

        # Import function after mocking
        try:
            from environment_detector import detect_environment
        except ImportError:
            self.skipTest("Environment detector not yet implemented")

        # Act
        result = detect_environment()

        # Assert
        mock_exists.assert_called_with("/kaggle/input")
        self.assertEqual(result["environment"], "simulated_kaggle")
        self.assertIn("simulated_kaggle_data", result["dataset_path"])
        self.assertIn("wound-segmentation-images", result["dataset_path"])
        self.assertIn("simulated_kaggle_data", result["working_path"])
        self.assertEqual(result["expected_counts"]["train_images"], 2)
        self.assertEqual(result["expected_counts"]["test_images"], 1)

    @patch("os.path.exists")
    def test_simulated_environment_paths_are_absolute(self, mock_exists):
        """Test that simulated environment returns absolute paths."""
        # Arrange
        mock_exists.return_value = False

        try:
            from environment_detector import detect_environment
        except ImportError:
            self.skipTest("Environment detector not yet implemented")

        # Act
        result = detect_environment()

        # Assert
        self.assertTrue(os.path.isabs(result["dataset_path"]), "Dataset path should be absolute")
        self.assertTrue(os.path.isabs(result["working_path"]), "Working path should be absolute")

    @patch("os.path.exists")
    def test_return_structure_completeness(self, mock_exists):
        """Test that return structure contains all required fields."""
        # Test both environments
        for kaggle_exists in [True, False]:
            with self.subTest(kaggle_environment=kaggle_exists):
                mock_exists.return_value = kaggle_exists

                try:
                    from environment_detector import detect_environment
                except ImportError:
                    self.skipTest("Environment detector not yet implemented")

                result = detect_environment()

                # Assert all required fields are present
                self.assertIn("environment", result)
                self.assertIn("dataset_path", result)
                self.assertIn("working_path", result)
                self.assertIn("expected_counts", result)

                # Assert expected_counts structure
                expected_counts = result["expected_counts"]
                self.assertIn("train_images", expected_counts)
                self.assertIn("train_masks", expected_counts)
                self.assertIn("test_images", expected_counts)
                self.assertIn("test_masks", expected_counts)

                # Assert all counts are positive integers
                for count_key, count_value in expected_counts.items():
                    self.assertIsInstance(count_value, int, f"{count_key} should be integer")
                    self.assertGreater(count_value, 0, f"{count_key} should be positive")

    @patch("os.path.exists")
    def test_no_exceptions_raised(self, mock_exists):
        """Test that function never raises exceptions."""
        # Test various scenarios that might cause errors
        test_scenarios = [
            True,  # Kaggle environment
            False,  # Simulated environment
        ]

        for scenario in test_scenarios:
            with self.subTest(kaggle_exists=scenario):
                mock_exists.return_value = scenario

                try:
                    from environment_detector import detect_environment

                    # Should not raise any exceptions
                    result = detect_environment()
                    self.assertIsInstance(result, dict)
                except ImportError:
                    self.skipTest("Environment detector not yet implemented")
                except Exception as e:
                    self.fail(f"detect_environment() raised {type(e).__name__}: {e}")

    @patch("os.path.exists")
    def test_consistent_train_test_mask_counts(self, mock_exists):
        """Test that image and mask counts are consistent."""
        for kaggle_exists in [True, False]:
            with self.subTest(kaggle_environment=kaggle_exists):
                mock_exists.return_value = kaggle_exists

                try:
                    from environment_detector import detect_environment
                except ImportError:
                    self.skipTest("Environment detector not yet implemented")

                result = detect_environment()
                counts = result["expected_counts"]

                # Train images should equal train masks
                self.assertEqual(
                    counts["train_images"],
                    counts["train_masks"],
                    "Train image and mask counts should match",
                )

                # Test images should equal test masks
                self.assertEqual(
                    counts["test_images"],
                    counts["test_masks"],
                    "Test image and mask counts should match",
                )

    @patch("os.path.exists")
    def test_dataset_split_ratio(self, mock_exists):
        """Test that dataset split maintains reasonable ratios."""
        mock_exists.return_value = True  # Test Kaggle environment

        try:
            from environment_detector import detect_environment
        except ImportError:
            self.skipTest("Environment detector not yet implemented")

        result = detect_environment()
        counts = result["expected_counts"]

        total_images = counts["train_images"] + counts["test_images"]
        train_ratio = counts["train_images"] / total_images

        # Should be approximately 80/20 split
        self.assertAlmostEqual(
            train_ratio,
            0.8,
            delta=0.05,
            msg="Train/test split should be approximately 80/20",
        )

    def test_contract_compliance(self):
        """Test that function output matches contract specification."""
        # This test verifies the implementation matches the contract
        try:
            from environment_detector import detect_environment
        except ImportError:
            self.skipTest("Environment detector not yet implemented")

        # Import contract for comparison
        try:
            from contracts.data_pipeline_contract import DataPipelineContracts

            DataPipelineContracts.environment_detection_contract()
        except ImportError:
            self.skipTest("Contract not available for comparison")

        with patch("os.path.exists", return_value=True):
            result = detect_environment()

            # Verify output structure matches contract
            self.assertIsInstance(result["environment"], str)
            self.assertIn(result["environment"], ["kaggle", "simulated_kaggle"])
            self.assertIsInstance(result["dataset_path"], str)
            self.assertIsInstance(result["working_path"], str)
            self.assertIsInstance(result["expected_counts"], dict)


class TestEnvironmentDetectionEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    @patch("os.path.exists")
    def test_os_path_exists_failure(self, mock_exists):
        """Test behavior when os.path.exists() fails."""
        # Simulate os.path.exists raising an exception
        mock_exists.side_effect = OSError("Permission denied")

        try:
            from environment_detector import detect_environment

            # Should handle the exception gracefully
            result = detect_environment()
            # Should default to simulated environment
            self.assertEqual(result["environment"], "simulated_kaggle")
        except ImportError:
            self.skipTest("Environment detector not yet implemented")

    def test_import_requirements(self):
        """Test that all required modules can be imported."""
        required_modules = ["os", "pathlib"]

        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                self.fail(f"Required module '{module_name}' cannot be imported")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
