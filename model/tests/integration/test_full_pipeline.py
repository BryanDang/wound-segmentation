"""
Integration Tests for Full EDA Pipeline
========================================

Tests the complete EDA pipeline from data loading through visualization generation
using the actual simulated dataset files.

These tests validate end-to-end functionality of the run_eda_pipeline orchestrator.
"""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent

# Import from the actual EDA pipeline module
from src.pipeline.eda import EDA_CONFIG, run_eda_pipeline


SAMPLE_DATA_MISSING = not (
    Path(__file__).resolve().parents[2] / "simulated_kaggle_data" / "input"
    / "wound-segmentation-images" / "test_images"
).exists()
SAMPLE_REASON = "Sample images not in the repo (only 2 of the 5 train images and no test_images)."


class TestFullEDAPipeline(unittest.TestCase):
    """
    Integration tests for the complete EDA pipeline.

    These tests use the actual simulated dataset in simulated_kaggle_data/
    to verify the entire pipeline works end-to-end.
    """

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.project_root = project_root
        cls.dataset_path = (
            cls.project_root / "simulated_kaggle_data" / "input" / "wound-segmentation-images"
        )
        cls.source_map_path = (
            cls.project_root
            / "archive"
            / "Original Data sets"
            / "Wound images segmentation [2760 samples]"
            / "correspondence_table.xlsx"
        )

        # Verify dataset exists
        if not cls.dataset_path.exists():
            raise unittest.SkipTest("Simulated Kaggle dataset not found")

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        self.output_csv_path = os.path.join(self.temp_dir, "metadata.csv")
        self.output_viz_dir = os.path.join(self.temp_dir, "visualizations")

    def tearDown(self):
        """Clean up after each test."""
        # Clean up temporary files
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @unittest.skipIf(SAMPLE_DATA_MISSING, SAMPLE_REASON)
    def test_full_pipeline_execution(self):
        """
        Test the complete EDA pipeline orchestrator with real dataset.

        This is the primary integration test that validates:
        1. Pipeline processes all images in train and test sets
        2. Metadata CSV is generated with all expected columns
        3. All visualizations are created successfully
        4. No critical errors occur during execution
        """
        # Act: Run the full pipeline
        result = run_eda_pipeline(
            data_path=str(self.dataset_path),
            source_map_path=str(self.source_map_path),
            output_csv_path=self.output_csv_path,
            output_viz_dir=self.output_viz_dir,
            config=EDA_CONFIG,
        )

        # Assert: Pipeline completed successfully
        self.assertTrue(
            result["success"], f"Pipeline failed: {result.get('error', 'Unknown error')}"
        )

        # Assert: Metadata CSV was created
        self.assertTrue(os.path.exists(self.output_csv_path), "Metadata CSV should be created")

        # Assert: Validate CSV content
        df = pd.read_csv(self.output_csv_path)

        # Dataset has: 4 train images, 2 test images = 6 total images
        # But fusc_0004_no_mask.png will be processed even though it has no mask
        expected_min_rows = 5  # At minimum, images with readable data
        self.assertGreaterEqual(
            len(df),
            expected_min_rows,
            f"CSV should have at least {expected_min_rows} rows for processed images",
        )

        # Assert: Required columns exist
        required_columns = [
            "image_id",
            "set",
            "source",
            "width",
            "height",
            "wound_pixel_count",
            "wound_percentage",
            "bounding_box_aspect_ratio",
            "mean_intensity",
            "mean_r",
            "mean_g",
            "mean_b",
            "centroid_x",
            "centroid_y",
        ]
        for col in required_columns:
            self.assertIn(col, df.columns, f"Column '{col}' should exist in metadata CSV")

        # Assert: Both train and test sets are represented
        self.assertIn("train", df["set"].values, "Train set should be present")
        self.assertIn("test", df["set"].values, "Test set should be present")

    @unittest.skipIf(SAMPLE_DATA_MISSING, SAMPLE_REASON)
    def test_pipeline_visualization_generation(self):
        """
        Test that the pipeline generates all expected visualization files.
        """
        # Act: Run the full pipeline
        _ = run_eda_pipeline(
            data_path=str(self.dataset_path),
            source_map_path=str(self.source_map_path),
            output_csv_path=self.output_csv_path,
            output_viz_dir=self.output_viz_dir,
            config=EDA_CONFIG,
        )

        # Assert: Visualization directory was created
        self.assertTrue(
            os.path.exists(self.output_viz_dir), "Visualization directory should be created"
        )

        # Assert: Core visualizations exist
        expected_visualizations = [
            "wound_size_distribution.png",
            "train_test_split.png",
            "class_imbalance.png",
            "brightness_distribution.png",
            "color_channel_distribution.png",
            "size_stratification.png",
            "correlation_heatmap.png",
            "positional_heatmap.png",
        ]

        for viz_file in expected_visualizations:
            viz_path = os.path.join(self.output_viz_dir, viz_file)
            self.assertTrue(
                os.path.exists(viz_path), f"Visualization '{viz_file}' should be created"
            )
            # Verify file has content
            self.assertGreater(
                os.path.getsize(viz_path), 0, f"Visualization '{viz_file}' should not be empty"
            )

        # Assert: At least one overlay image exists
        overlays = [f for f in os.listdir(self.output_viz_dir) if f.startswith("overlay_")]
        self.assertGreater(len(overlays), 0, "At least one sample overlay image should be created")

        # Assert: Extrema examples exist (3 files expected)
        extrema_files = [f for f in os.listdir(self.output_viz_dir) if f.startswith("extrema_")]
        self.assertEqual(
            len(extrema_files),
            3,
            "Three extrema example files should be created "
            "(wound_pixel_count, aspect_ratio, brightness)",
        )

    def test_pipeline_handles_missing_masks_gracefully(self):
        """
        Test that the pipeline handles images with missing or empty masks correctly.

        The simulated dataset includes:
        - fusc_0004_no_mask.png: Image with no corresponding mask file
        - fusc_0005_empty_mask.png: Image with empty (all-black) mask
        """
        # Act: Run the full pipeline
        result = run_eda_pipeline(
            data_path=str(self.dataset_path),
            source_map_path=str(self.source_map_path),
            output_csv_path=self.output_csv_path,
            output_viz_dir=self.output_viz_dir,
            config=EDA_CONFIG,
        )

        # Assert: Pipeline completes despite missing/empty masks
        self.assertTrue(result["success"], "Pipeline should handle missing/empty masks gracefully")

        # Assert: Images with missing/empty masks have wound_pixel_count = 0
        df = pd.read_csv(self.output_csv_path)

        # Check for the image with no mask
        no_mask_rows = df[df["image_id"] == "fusc_0004_no_mask.png"]
        if not no_mask_rows.empty:
            self.assertEqual(
                no_mask_rows.iloc[0]["wound_pixel_count"],
                0,
                "Image with no mask should have wound_pixel_count = 0",
            )

        # Check for the image with empty mask
        empty_mask_rows = df[df["image_id"] == "fusc_0005_empty_mask.png"]
        if not empty_mask_rows.empty:
            self.assertEqual(
                empty_mask_rows.iloc[0]["wound_pixel_count"],
                0,
                "Image with empty mask should have wound_pixel_count = 0",
            )

    def test_pipeline_error_reporting(self):
        """
        Test that the pipeline reports errors appropriately when given invalid inputs.
        """
        # Test with nonexistent data path
        invalid_result = run_eda_pipeline(
            data_path="/nonexistent/path",
            source_map_path=str(self.source_map_path),
            output_csv_path=self.output_csv_path,
            output_viz_dir=self.output_viz_dir,
            config=EDA_CONFIG,
        )

        # Pipeline should still complete but with warnings/errors
        # The pipeline is designed to be resilient and skip missing directories
        self.assertIn("warnings", invalid_result["data"], "Pipeline should report warnings")

    def test_pipeline_summary_statistics(self):
        """
        Test that the pipeline generates summary statistics correctly.
        """
        # Act: Run the full pipeline
        result = run_eda_pipeline(
            data_path=str(self.dataset_path),
            source_map_path=str(self.source_map_path),
            output_csv_path=self.output_csv_path,
            output_viz_dir=self.output_viz_dir,
            config=EDA_CONFIG,
        )

        # Assert: Summary table result is present
        self.assertIn(
            "summary_table_result", result["data"], "Pipeline should generate summary table"
        )

        # If summary table was generated successfully, validate it
        summary_result = result["data"]["summary_table_result"]
        if summary_result and summary_result.get("success"):
            summary_df = summary_result["data"]
            self.assertIsInstance(summary_df, pd.DataFrame, "Summary should be a DataFrame")
            self.assertGreater(len(summary_df), 0, "Summary should have statistics")

    def test_pipeline_with_minimal_config(self):
        """
        Test that the pipeline works with minimal configuration.
        """
        # Use a minimal config with just essential parameters
        minimal_config = {
            "heatmap_size": (256, 256),
            "plot_bins": 50,
            "overlay_sample_size": 3,  # Reduced from default
            "overlay_color": [0, 0, 255],
        }

        # Act: Run pipeline with minimal config
        result = run_eda_pipeline(
            data_path=str(self.dataset_path),
            source_map_path=str(self.source_map_path),
            output_csv_path=self.output_csv_path,
            output_viz_dir=self.output_viz_dir,
            config=minimal_config,
        )

        # Assert: Pipeline completes successfully
        self.assertTrue(result["success"], "Pipeline should work with minimal configuration")


class TestDatasetIntegrity(unittest.TestCase):
    """
    Validate the actual simulated dataset structure and integrity.

    These tests ensure the test data itself is correctly structured.
    """

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        cls.project_root = project_root
        cls.dataset_path = (
            cls.project_root / "simulated_kaggle_data" / "input" / "wound-segmentation-images"
        )

    @unittest.skipIf(SAMPLE_DATA_MISSING, SAMPLE_REASON)
    def test_simulated_dataset_structure(self):
        """Verify the simulated dataset has the expected directory structure."""
        # Check main directory exists
        self.assertTrue(self.dataset_path.exists(), "Simulated dataset directory should exist")

        # Check subdirectories
        required_dirs = ["train_images", "train_masks", "test_images", "test_masks"]
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            self.assertTrue(dir_path.exists(), f"Directory {dir_name} should exist")

    @unittest.skipIf(SAMPLE_DATA_MISSING, SAMPLE_REASON)
    def test_dataset_file_counts(self):
        """Verify the dataset has the expected number of files."""
        train_images = list((self.dataset_path / "train_images").glob("*.png"))
        train_masks = list((self.dataset_path / "train_masks").glob("*.png"))
        test_images = list((self.dataset_path / "test_images").glob("*.png"))
        test_masks = list((self.dataset_path / "test_masks").glob("*.png"))

        # Based on actual simulated dataset
        self.assertEqual(len(train_images), 4, "Should have 4 training images")
        self.assertEqual(
            len(train_masks), 3, "Should have 3 training masks (one missing intentionally)"
        )
        self.assertEqual(len(test_images), 1, "Should have 1 test image")
        self.assertEqual(len(test_masks), 1, "Should have 1 test mask")

    @unittest.skipIf(SAMPLE_DATA_MISSING, SAMPLE_REASON)
    def test_specific_test_files_exist(self):
        """Verify specific expected test files exist."""
        expected_files = {
            "train_images": [
                "fusc_0002.png",
                "fusc_0003.png",
                "fusc_0004_no_mask.png",
                "fusc_0005_empty_mask.png",
            ],
            "train_masks": ["fusc_0002.png", "fusc_0003.png", "fusc_0005_empty_mask.png"],
            "test_images": ["fusc_0001.png"],  # At least this one
            "test_masks": ["fusc_0001.png"],
        }

        for subdir, filenames in expected_files.items():
            dir_path = self.dataset_path / subdir
            for filename in filenames:
                file_path = dir_path / filename
                self.assertTrue(
                    file_path.exists(), f"Expected file {filename} should exist in {subdir}"
                )

    def test_file_integrity(self):
        """Verify files are not empty and readable."""
        for subdir in ["train_images", "train_masks", "test_images", "test_masks"]:
            dir_path = self.dataset_path / subdir
            for file_path in dir_path.glob("*.png"):
                # Check file is not empty
                self.assertGreater(
                    file_path.stat().st_size, 0, f"File {file_path.name} should not be empty"
                )

                # Check file is readable
                try:
                    with open(file_path, "rb") as f:
                        data = f.read(100)  # Read first 100 bytes
                        self.assertGreater(
                            len(data), 0, f"File {file_path.name} should be readable"
                        )
                except Exception as e:
                    self.fail(f"Could not read file {file_path.name}: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
