from pathlib import Path
import os
import sys

import pandas as pd
import pytest

# Add the src directory to the Python path to allow importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# Import from the new implementation file in src/
from src.pipeline.eda import EDA_CONFIG, Analyzer, Reporter, run_eda_pipeline

# --- Fixtures ---


SAMPLE_DATA_MISSING = not (
    Path(__file__).resolve().parents[3] / "simulated_kaggle_data" / "input"
    / "wound-segmentation-images" / "test_images"
).exists()
SAMPLE_REASON = "Sample images not in the repo (only 2 of the 5 train images and no test_images)."


@pytest.fixture(scope="module")
def test_data_path():
    """Returns the path to the simulated data directory."""
    return "simulated_kaggle_data/input/wound-segmentation-images/"


@pytest.fixture(scope="module")
def source_map_fixture(tmpdir_factory):
    """Creates a dummy correspondence_table.xlsx for testing."""
    path = tmpdir_factory.mktemp("data").join("correspondence_table.xlsx")
    df = pd.DataFrame(
        [
            {"image_id": "fusc_0001.png", "source": "Test_Source_1"},
            {"image_id": "fusc_0002.png", "source": "Test_Source_1"},
            {"image_id": "fusc_0003.png", "source": "Test_Source_2"},
            {"image_id": "fusc_0005_empty_mask.png", "source": "Test_Source_2"},
        ]
    )
    df.to_excel(path, index=False, engine="openpyxl")
    return str(path)


# --- Unit Tests ---


def test_analyzer_init(source_map_fixture):
    """Tests that the Analyzer initializes and loads the source map."""
    analyzer = Analyzer(source_map_path=source_map_fixture)
    assert analyzer.source_map["fusc_0001.png"] == "Test_Source_1"
    assert len(analyzer.source_map) == 4


def test_reporter(tmp_path):
    """Tests the Reporter class's file operations."""
    output_csv = os.path.join(tmp_path, "test.csv")
    fieldnames = ["col1", "col2"]

    reporter = Reporter(output_path=output_csv, fieldnames=fieldnames)
    reporter.write_row({"col1": "a", "col2": "b"})
    reporter.close_report()

    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv)
    assert len(df) == 1
    assert df.iloc[0]["col1"] == "a"


# --- Integration Test ---


@pytest.mark.skipif(SAMPLE_DATA_MISSING, reason=SAMPLE_REASON)
def test_full_pipeline_run(tmp_path, test_data_path, source_map_fixture):
    """
    An integration test for the full run_eda_pipeline orchestrator.
    """
    output_csv = os.path.join(tmp_path, "metadata.csv")
    output_viz = os.path.join(tmp_path, "visualizations")

    run_eda_pipeline(
        data_path=test_data_path,
        source_map_path=source_map_fixture,
        output_csv_path=output_csv,
        output_viz_dir=output_viz,
        config=EDA_CONFIG,
    )

    # Test if metadata CSV was created and has the correct number of rows
    assert os.path.exists(output_csv)
    df = pd.read_csv(output_csv)
    assert len(df) == 5  # 1 in test, 4 in train

    # Test if visualization directory and some files were created
    assert os.path.exists(output_viz)
    assert os.path.exists(os.path.join(output_viz, "wound_size_distribution.png"))
    assert os.path.exists(os.path.join(output_viz, "train_test_split.png"))
    # Check for at least one overlay image
    overlays = [f for f in os.listdir(output_viz) if f.startswith("overlay_")]
    assert len(overlays) > 0


@pytest.mark.skipif(SAMPLE_DATA_MISSING, reason=SAMPLE_REASON)
def test_analyze_image_returns_new_metrics(source_map_fixture, test_data_path):
    """
    Tests that the `analyze_image` method returns a dictionary containing the new
    per-image metrics (color and centroid) as defined in its contract.

    **Suggestions for Test Coverage Enhancement:**
    -   Test an image with an empty mask and assert that the new metric values are 0.0.
    -   Test an image with no corresponding mask file and assert the new metrics are 0.0.
    """
    # Initialize analyzer with source map
    analyzer = Analyzer(source_map_path=source_map_fixture)

    # Use fusc_0002.png which has a valid wound mask
    image_path = os.path.join(test_data_path, "train_images", "fusc_0002.png")
    mask_path = os.path.join(test_data_path, "train_masks", "fusc_0002.png")

    # Call analyze_image
    result = analyzer.analyze_image(image_path, mask_path)

    # Assert success
    assert result["success"] is True
    assert "data" in result

    data = result["data"]

    # Assert basic metrics exist
    assert "image_id" in data
    assert "source" in data
    assert "width" in data
    assert "height" in data

    # Assert wound metrics exist
    assert "wound_pixel_count" in data
    assert "wound_percentage" in data
    assert "bounding_box_aspect_ratio" in data

    # Assert NEW color metrics exist and are in valid range [0-255]
    assert "mean_intensity" in data
    assert "mean_r" in data
    assert "mean_g" in data
    assert "mean_b" in data

    # Color values should be in valid range for images with wounds
    assert 0 <= data["mean_r"] <= 255
    assert 0 <= data["mean_g"] <= 255
    assert 0 <= data["mean_b"] <= 255
    assert 0 <= data["mean_intensity"] <= 255

    # Assert NEW centroid metrics exist
    assert "centroid_x" in data
    assert "centroid_y" in data

    # Centroid should be within image bounds (for images with wounds)
    # Since this image has a wound (wound_pixel_count > 0), centroid should be > 0
    if data["wound_pixel_count"] > 0:
        assert 0 <= data["centroid_x"] <= data["width"]
        assert 0 <= data["centroid_y"] <= data["height"]
        # For images with wounds, color values should be non-zero
        assert data["mean_intensity"] > 0
        assert data["mean_r"] > 0 or data["mean_g"] > 0 or data["mean_b"] > 0
