from pathlib import Path
import os
import sys

import pandas as pd
import pytest

# Add the src directory to the Python path to allow importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from src.pipeline.eda import EDA_CONFIG, Visualizer


SAMPLE_DATA_MISSING = not (
    Path(__file__).resolve().parents[3] / "simulated_kaggle_data" / "input"
    / "wound-segmentation-images" / "test_images"
).exists()
SAMPLE_REASON = "Sample images not in the repo (only 2 of the 5 train images and no test_images)."


@pytest.fixture
def visualizer_instance():
    """Provides a Visualizer instance for testing."""
    return Visualizer(config=EDA_CONFIG)


def test_generate_class_imbalance_chart_creates_file(visualizer_instance, tmp_path):
    """
    Tests that the `_generate_class_imbalance_chart` method successfully
    creates a pie chart PNG file in the specified output directory.

    **Suggestions for Test Coverage Enhancement:**
    -   **Empty DataFrame:** Create a test case where an empty DataFrame is passed.
        Assert that the method returns a `failure` result and that no file is created.
    -   **Missing Columns:** Create a test case with a DataFrame that lacks the required
        `wound_pixel_count` column. Assert that the method handles this gracefully
        and returns a `failure` result.
    -   **Zero Pixels:** Create a test case where `width` and `height` are zero, leading
        to `total_pixels` being zero. Assert that a `failure` is returned to prevent
        a division-by-zero error.
    """
    # Arrange: Create a valid DataFrame with required columns
    df = pd.DataFrame(
        {
            "image_id": ["img1.png", "img2.png", "img3.png"],
            "width": [512, 512, 512],
            "height": [512, 512, 512],
            "wound_pixel_count": [10000, 5000, 0],
            "wound_percentage": [3.81, 1.91, 0.0],
        }
    )

    output_dir = str(tmp_path)

    # Act: Call the method
    result = visualizer_instance._generate_class_imbalance_chart(df, output_dir)

    # Assert: Verify success and file creation
    assert result["success"] is True, f"Expected success, got: {result}"
    assert "files" in result["data"], "Expected 'files' key in result data"
    assert len(result["data"]["files"]) == 1, "Expected exactly one file to be created"

    file_path = result["data"]["files"][0]
    expected_path = os.path.join(output_dir, "class_imbalance.png")
    assert file_path == expected_path, f"Expected file path to be {expected_path}, got {file_path}"
    assert os.path.exists(file_path), f"Expected file to exist at {file_path}"
    assert os.path.isfile(file_path), f"Expected {file_path} to be a file"

    # Verify file is a valid PNG (basic check - file size > 0)
    assert os.path.getsize(file_path) > 0, "Expected file to have non-zero size"


def test_generate_color_analysis_creates_files(visualizer_instance, tmp_path):
    """
    Tests that `generate_color_analysis` successfully creates the two expected
    PNG files: `brightness_distribution.png` and `color_channel_distribution.png`.

    **Suggestions for Test Coverage Enhancement:**
    -   **No Wound Data:** Pass a DataFrame where all images have a `wound_pixel_count` of 0.
        Assert that the method returns success but that the `files` list is empty,
        as no plots should be generated.
    -   **Missing Columns:** Test that the function returns a `failure` result if the
        input DataFrame is missing the `mean_intensity` or other required color columns.
    """
    # Arrange: Create a valid DataFrame with required columns
    df = pd.DataFrame(
        {
            "image_id": ["img1.png", "img2.png", "img3.png"],
            "wound_pixel_count": [10000, 5000, 3000],
            "mean_intensity": [128.5, 145.2, 110.8],
            "mean_r": [150.0, 160.5, 120.3],
            "mean_g": [130.2, 140.8, 115.6],
            "mean_b": [105.7, 125.3, 95.2],
        }
    )

    output_dir = str(tmp_path)

    # Act: Call the method
    result = visualizer_instance.generate_color_analysis(df, output_dir)

    # Assert: Verify success and both files created
    assert result["success"] is True, f"Expected success, got: {result}"
    assert "files" in result["data"], "Expected 'files' key in result data"
    assert len(result["data"]["files"]) == 2, "Expected exactly two files to be created"

    # Check for brightness distribution file
    brightness_file = os.path.join(output_dir, "brightness_distribution.png")
    assert (
        brightness_file in result["data"]["files"]
    ), "Expected brightness_distribution.png in files list"
    assert os.path.exists(brightness_file), f"Expected file to exist at {brightness_file}"
    assert os.path.isfile(brightness_file), f"Expected {brightness_file} to be a file"
    assert os.path.getsize(brightness_file) > 0, "Expected brightness file to have non-zero size"

    # Check for color channel distribution file
    color_file = os.path.join(output_dir, "color_channel_distribution.png")
    assert (
        color_file in result["data"]["files"]
    ), "Expected color_channel_distribution.png in files list"
    assert os.path.exists(color_file), f"Expected file to exist at {color_file}"
    assert os.path.isfile(color_file), f"Expected {color_file} to be a file"
    assert os.path.getsize(color_file) > 0, "Expected color file to have non-zero size"


def test_generate_size_stratification_chart(visualizer_instance, tmp_path):
    """
    Tests that `generate_size_stratification_chart` creates a bar chart of size categories.
    """
    # Arrange: Create a DataFrame with wound_percentage values covering all categories
    df = pd.DataFrame(
        {
            "image_id": ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png", "img6.png"],
            "wound_percentage": [
                0.0,
                0.5,
                1.5,
                5.0,
                8.0,
                15.0,
            ],  # None, Small, Small, Medium, Medium, Large
        }
    )

    # Add size_category column based on wound_percentage
    def categorize_size(wound_pct):
        if wound_pct == 0.0:
            return "None"
        elif 0.1 <= wound_pct < 2.0:
            return "Small"
        elif 2.0 <= wound_pct < 10.0:
            return "Medium"
        else:  # >= 10.0
            return "Large"

    df["size_category"] = df["wound_percentage"].apply(categorize_size)

    output_dir = str(tmp_path)

    # Act: Call the method
    result = visualizer_instance.generate_size_stratification_chart(df, output_dir)

    # Assert: Verify success and file creation
    assert result["success"] is True, f"Expected success, got: {result}"
    assert "files" in result["data"], "Expected 'files' key in result data"
    assert len(result["data"]["files"]) == 1, "Expected exactly one file to be created"

    file_path = result["data"]["files"][0]
    expected_path = os.path.join(output_dir, "size_stratification.png")
    assert file_path == expected_path, f"Expected file path to be {expected_path}, got {file_path}"
    assert os.path.exists(file_path), f"Expected file to exist at {file_path}"
    assert os.path.isfile(file_path), f"Expected {file_path} to be a file"
    assert (
        os.path.getsize(file_path) > 0
    ), f"Expected file to have content, got size: {os.path.getsize(file_path)}"


def test_generate_correlation_heatmap(visualizer_instance, tmp_path):
    """
    Tests that `generate_correlation_heatmap` creates a correlation matrix heatmap
    for all numerical metrics, filtering to only wound images.
    """
    # Arrange: Create a valid DataFrame with all numerical columns
    df = pd.DataFrame(
        {
            "image_id": ["img1.png", "img2.png", "img3.png", "img4.png"],
            "wound_pixel_count": [10000, 5000, 8000, 0],  # One no-wound image
            "wound_percentage": [3.81, 1.91, 3.05, 0.0],
            "bounding_box_aspect_ratio": [1.2, 0.8, 1.5, 0.0],
            "mean_intensity": [120.5, 115.3, 125.7, 100.0],
            "mean_r": [130.2, 125.8, 135.1, 110.5],
            "mean_g": [115.4, 110.2, 120.3, 95.2],
            "mean_b": [105.8, 100.5, 110.2, 90.8],
            "centroid_x": [256.3, 250.1, 260.5, 0.0],
            "centroid_y": [245.8, 240.2, 250.3, 0.0],
        }
    )

    output_dir = str(tmp_path)

    # Act: Call the method
    result = visualizer_instance.generate_correlation_heatmap(df, output_dir)

    # Assert: Verify success and file creation
    assert result["success"] is True, f"Expected success, got: {result}"
    assert "files" in result["data"], "Expected 'files' key in result data"
    assert len(result["data"]["files"]) == 1, "Expected exactly one file to be created"

    file_path = result["data"]["files"][0]
    expected_path = os.path.join(output_dir, "correlation_heatmap.png")
    assert file_path == expected_path, f"Expected file path to be {expected_path}, got {file_path}"
    assert os.path.exists(file_path), f"Expected file to exist at {file_path}"
    assert os.path.isfile(file_path), f"Expected {file_path} to be a file"

    # Verify file is a valid PNG (basic check - file size > 0)
    assert os.path.getsize(file_path) > 0, "Expected file to have non-zero size"


def test_generate_no_wound_examples(visualizer_instance, tmp_path):
    """
    Tests that `generate_no_wound_examples` creates a composite image of no-wound examples.
    """
    # Arrange: Create a DataFrame with images having wound_pixel_count == 0
    df = pd.DataFrame(
        {
            "image_id": [
                "fusc_0004_no_mask.png",
                "fusc_0005_empty_mask.png",
                "fusc_0002.png",  # Has wound
                "fusc_0003.png",  # Has wound
            ],
            "wound_pixel_count": [0, 0, 5000, 8000],
            "width": [512, 512, 512, 512],
            "height": [512, 512, 512, 512],
        }
    )

    # Use actual test data path (simulated Kaggle environment)
    test_data_path = "simulated_kaggle_data/input/wound-segmentation-images/"
    output_dir = str(tmp_path)

    # Act: Call the method
    result = visualizer_instance.generate_no_wound_examples(df, test_data_path, output_dir)

    # Assert: Verify success and file creation
    assert result["success"] is True, f"Expected success, got: {result}"
    assert "files" in result["data"], "Expected 'files' key in result data"
    assert len(result["data"]["files"]) == 1, "Expected exactly one file to be created"

    file_path = result["data"]["files"][0]
    expected_path = os.path.join(output_dir, "no_wound_examples.png")
    assert file_path == expected_path, f"Expected file path to be {expected_path}, got {file_path}"
    assert os.path.exists(file_path), f"Expected file to exist at {file_path}"
    assert os.path.isfile(file_path), f"Expected {file_path} to be a file"
    assert os.path.getsize(file_path) > 0, "Expected file to have content"


def test_generate_positional_heatmap(visualizer_instance, tmp_path):
    """
    Tests that `generate_positional_heatmap` creates a positional heatmap of wound centroids.
    """
    # Arrange: Create a DataFrame with wound images having non-zero centroids
    df = pd.DataFrame(
        {
            "image_id": ["wound1.png", "wound2.png", "wound3.png", "wound4.png", "no_wound.png"],
            "width": [512, 512, 512, 512, 512],
            "height": [512, 512, 512, 512, 512],
            "wound_pixel_count": [5000, 8000, 12000, 3000, 0],
            "centroid_x": [256.5, 128.3, 384.7, 200.1, 0.0],
            "centroid_y": [200.1, 350.8, 150.3, 256.5, 0.0],
        }
    )

    output_dir = str(tmp_path)

    # Act: Call the method
    result = visualizer_instance.generate_positional_heatmap(df, output_dir)

    # Assert: Verify success and file creation
    assert result["success"] is True, f"Expected success, got: {result}"
    assert "files" in result["data"], "Expected 'files' key in result data"
    assert len(result["data"]["files"]) == 1, "Expected exactly one file to be created"

    file_path = result["data"]["files"][0]
    expected_path = os.path.join(output_dir, "positional_heatmap.png")
    assert file_path == expected_path, f"Expected file path to be {expected_path}, got {file_path}"
    assert os.path.exists(file_path), f"Expected file to exist at {file_path}"
    assert os.path.isfile(file_path), f"Expected {file_path} to be a file"
    assert os.path.getsize(file_path) > 0, f"Expected {file_path} to have content"

    def test_generate_extrema_examples(visualizer_instance, tmp_path):
        """
        Tests that `generate_extrema_examples` creates composite images for metric extrema.
        """
        pass


def test_generate_confusion_matrix(visualizer_instance, tmp_path):
    """
    Tests that `generate_confusion_matrix` successfully creates a
    confusion matrix visualization PNG file.

    **Suggestions for Test Coverage Enhancement:**
    -   Test with a DataFrame where the sum of all confusion matrix
        elements is zero and ensure it handles this edge case gracefully.
    -   Test with a DataFrame missing one of the required columns (TP, FP, TN, FN)
        and assert that the method returns a `failure` result.
    """
    # Arrange: Create a DataFrame with confusion matrix values
    df = pd.DataFrame(
        {
            "image_id": ["img1.png", "img2.png", "img3.png", "img4.png"],
            "TP": [1500, 2000, 1800, 2200],
            "FP": [200, 150, 180, 120],
            "TN": [8000, 8200, 8100, 8300],
            "FN": [300, 250, 220, 180],
        }
    )

    output_dir = str(tmp_path)

    # Act: Call the method
    result = visualizer_instance.generate_confusion_matrix(df, output_dir)

    # Assert: Verify success and file creation
    assert result["success"] is True, f"Expected success, got: {result}"
    assert "files" in result["data"], "Expected 'files' key in result data"
    assert len(result["data"]["files"]) == 1, "Expected exactly one file to be created"

    file_path = result["data"]["files"][0]
    expected_path = os.path.join(output_dir, "confusion_matrix.png")
    assert file_path == expected_path, f"Expected file path to be {expected_path}, got {file_path}"
    assert os.path.exists(file_path), f"Expected file to exist at {file_path}"
    assert os.path.isfile(file_path), f"Expected {file_path} to be a file"

    # Verify file is a valid PNG (basic check - file size > 0)
    assert os.path.getsize(file_path) > 0, "Expected file to have non-zero size"

@pytest.mark.skipif(SAMPLE_DATA_MISSING, reason=SAMPLE_REASON)
def test_generate_example_predictions(visualizer_instance, tmp_path):
    """
    Tests that `generate_example_predictions` creates a composite image
    showing best, worst, and average Dice score examples.

    **Suggestions for Test Coverage Enhancement:**
    -   Test with a DataFrame with only one image (edge case for finding best/worst/avg)
    -   Test with missing 'dice' or 'image_id' column
    -   Test with an empty DataFrame
    """
    # Arrange: Create a DataFrame with dice scores covering a range
    df = pd.DataFrame(
        {
            "image_id": [
                "fusc_0001.png",
                "fusc_0002.png",
                "fusc_0003.png",
                "fusc_0004.png",
                "fusc_0005.png",
            ],
            "dice": [0.95, 0.45, 0.72, 0.88, 0.60],
        }
    )

    # Use actual test data path (simulated Kaggle environment)
    test_data_path = "simulated_kaggle_data/input/wound-segmentation-images/"
    output_dir = str(tmp_path)

    # Act: Call the method
    result = visualizer_instance.generate_example_predictions(df, test_data_path, output_dir)

    # Assert: Verify success and file creation
    assert result["success"] is True, f"Expected success, got: {result}"
    assert "files" in result["data"], "Expected 'files' key in result data"
    assert len(result["data"]["files"]) == 1, "Expected exactly one file to be created"

    file_path = result["data"]["files"][0]
    expected_path = os.path.join(output_dir, "example_predictions.png")
    assert file_path == expected_path, f"Expected file path to be {expected_path}, got {file_path}"
    assert os.path.exists(file_path), f"Expected file to exist at {file_path}"
    assert os.path.isfile(file_path), f"Expected {file_path} to be a file"
    assert os.path.getsize(file_path) > 0, "Expected file to have content"

