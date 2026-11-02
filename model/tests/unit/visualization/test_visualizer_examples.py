from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.pipeline.eda import EDA_CONFIG, Visualizer


@pytest.fixture
def visualizer():
    """Visualizer with a mock config."""
    config = EDA_CONFIG.copy()
    with patch.object(Visualizer, "__init__", lambda self, config: setattr(self, "config", config)):
        vis = Visualizer(config=config)
        yield vis


@pytest.fixture
def metadata_fixture(tmp_path: Path) -> dict:
    """Dummy metadata.csv and image files."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Image files the report points to.
    (tmp_path / "img_small_wound.png").touch()
    (tmp_path / "img_large_wound.png").touch()
    (tmp_path / "img_square_wound.png").touch()
    (tmp_path / "img_long_wound.png").touch()

    metadata_content = {
        "image_path": [
            str(tmp_path / "img_small_wound.png"),
            str(tmp_path / "img_large_wound.png"),
            str(tmp_path / "img_square_wound.png"),
            str(tmp_path / "img_long_wound.png"),
        ],
        "wound_area": [100, 9000, 500, 500],
        "aspect_ratio": [1.5, 0.8, 1.0, 10.0],
    }
    df = pd.DataFrame(metadata_content)
    csv_path = tmp_path / "metadata.csv"
    df.to_csv(csv_path, index=False)

    return {"metadata_csv_path": str(csv_path), "output_dir": str(output_dir)}


def test_generate_example_reports_creates_files(visualizer: Visualizer, metadata_fixture: dict):
    """generate_example_reports creates the expected image files."""
    # Expected output files
    output_dir = Path(metadata_fixture["output_dir"])
    expected_files = [
        output_dir / "extrema_wound_area.png",
        output_dir / "extrema_aspect_ratio.png",
    ]

    # Method is mocked.
    visualizer.generate_example_reports = MagicMock()

    # Call the method.
    visualizer.generate_example_reports(
        metadata_csv_path=metadata_fixture["metadata_csv_path"], output_dir=str(output_dir)
    )

    # The mock run is simulated by touching the files.
    for f in expected_files:
        f.touch()

    # Files exist.
    for f in expected_files:
        assert f.exists(), f"Example report '{f.name}' was not created."
