import os
import sys

import pandas as pd
import pytest

# src/ on the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from src.pipeline.eda import SummaryReporter


@pytest.fixture
def dummy_metadata_csv(tmp_path):
    """Dummy image_metadata.csv."""
    data = {
        "image_id": ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"],
        "set": ["train", "train", "test", "train", "test"],
        "source": ["Medetec", "Wang", "Medetec", "Wang", "Oota"],
        "width": [100, 200, 150, 100, 200],
        "height": [100, 200, 150, 100, 200],
        "wound_pixel_count": [0, 500, 100, 0, 200],
        "wound_percentage": [0.0, 1.25, 0.44, 0.0, 0.5],
        "bounding_box_aspect_ratio": [0.0, 1.0, 0.5, 0.0, 2.0],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "image_metadata.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_summary_reporter_init(dummy_metadata_csv):
    """Tests that SummaryReporter initializes correctly."""
    reporter = SummaryReporter(metadata_csv_path=dummy_metadata_csv)
    assert reporter.metadata_csv_path == dummy_metadata_csv
    # The dataframe loads on init.
    assert not reporter.metadata_df.empty


def test_generate_summary_table_content(dummy_metadata_csv):
    """Tests the content of the generated summary table."""
    reporter = SummaryReporter(metadata_csv_path=dummy_metadata_csv)
    result = reporter.generate_summary_table()
    assert result["success"] is True
    summary_df = result["data"]
    assert isinstance(summary_df, pd.DataFrame)

    # Counts.
    assert summary_df.loc["Total Images", "Value"] == 5
    assert summary_df.loc["Images (Source: Medetec)", "Value"] == 2
    assert summary_df.loc["Images (Source: Wang)", "Value"] == 2
    assert summary_df.loc["Images (Source: Oota)", "Value"] == 1
    assert summary_df.loc["Images (Set: train)", "Value"] == 3
    assert summary_df.loc["Images (Set: test)", "Value"] == 2
    assert summary_df.loc["Images with No Wound", "Value"] == 2

    # Averages, approximate.
    assert abs(summary_df.loc["Avg Wound Pixel Count", "Value"] - (500 + 100 + 200) / 5) < 0.001
    assert abs(summary_df.loc["Avg Wound Percentage", "Value"] - (1.25 + 0.44 + 0.5) / 5) < 0.001
    assert abs(summary_df.loc["Avg Aspect Ratio", "Value"] - (1.0 + 0.5 + 2.0) / 5) < 0.001
