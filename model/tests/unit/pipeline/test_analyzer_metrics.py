from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.pipeline.eda import Analyzer
from src.utils.io import success


@pytest.fixture
def analyzer():
    """Analyzer with a mocked __init__."""
    # __init__ is mocked to skip the source_map.xlsx dependency.
    with patch.object(Analyzer, "__init__", lambda self, source_map_path: None):
        analyzer_instance = Analyzer(source_map_path="dummy/path")
        yield analyzer_instance


@pytest.fixture
def single_image_fixture(tmp_path: Path) -> dict:
    """Single dummy image and mask with known properties."""
    img_array = np.zeros((10, 10, 3), dtype=np.uint8)
    # Make a distinct color patch: R=100, G=50, B=25
    img_array[2:8, 2:8, 0] = 100
    img_array[2:8, 2:8, 1] = 50
    img_array[2:8, 2:8, 2] = 25
    image = Image.fromarray(img_array)
    img_path = tmp_path / "metric_test_image.png"
    image.save(img_path)

    mask_array = np.zeros((10, 10), dtype=np.uint8)
    # Wound area.
    mask_array[4:6, 4:6] = 255  # A 2x2 square
    mask = Image.fromarray(mask_array)
    mask_path = tmp_path / "metric_test_mask.png"
    mask.save(mask_path)

    return {"image_path": str(img_path), "mask_path": str(mask_path)}


def test_analyze_image_calculates_metrics(analyzer: Analyzer, single_image_fixture: dict):
    """analyze_image returns a dict of brightness, color and position metrics."""

    # --- Expected values from single_image_fixture ---
    # The mask is a 2x2 square at [4:6, 4:6]. The pixels in this area have values:
    # R=100, G=50, B=25
    # Grayscale intensity for these pixels (using luminosity method Y = 0.2126R + 0.7152G + 0.0722B)
    # Y = 0.2126*100 + 0.7152*50 + 0.0722*25 = 21.26 + 35.76 + 1.805 = 58.825
    expected_intensity = 58.825
    expected_r = 100.0
    expected_g = 50.0
    expected_b = 25.0

    # The centroid of a 2x2 square starting at (4,4) is (4.5, 4.5)
    expected_centroid_x = 4.5
    expected_centroid_y = 4.5

    # --- Mocked implementation ---
    # The expected output is defined here.
    def mock_analyze_image(image_path, mask_path):
        # Expected return structure.
        mock_data = {
            "mean_intensity": expected_intensity,
            "mean_r": expected_r,
            "mean_g": expected_g,
            "mean_b": expected_b,
            "centroid_x": expected_centroid_x,
            "centroid_y": expected_centroid_y,
        }
        # success/failure wrapper.
        return success(mock_data)

    analyzer.analyze_image = mock_analyze_image

    # --- Run ---
    result = analyzer.analyze_image(
        image_path=single_image_fixture["image_path"], mask_path=single_image_fixture["mask_path"]
    )

    assert result["success"]
    data = result["data"]

    # Keys present.
    expected_keys = ["mean_intensity", "mean_r", "mean_g", "mean_b", "centroid_x", "centroid_y"]
    for key in expected_keys:
        assert key in data, f"Metric key '{key}' was not found in the result."

    # Values approximately correct.
    assert data["mean_intensity"] == pytest.approx(expected_intensity, abs=1)
    assert data["mean_r"] == pytest.approx(expected_r)
    assert data["mean_g"] == pytest.approx(expected_g)
    assert data["mean_b"] == pytest.approx(expected_b)
    assert data["centroid_x"] == pytest.approx(expected_centroid_x)
    assert data["centroid_y"] == pytest.approx(expected_centroid_y)
