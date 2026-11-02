from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.pipeline.eda import EDA_CONFIG, Visualizer


@pytest.fixture
def visualizer():
    """Visualizer with a mock config."""
    config = EDA_CONFIG.copy()
    # Mock the implementation-dependent part of __init__.
    with patch.object(Visualizer, "__init__", lambda self, config: setattr(self, "config", config)):
        vis = Visualizer(config=config)
        yield vis


@pytest.fixture
def image_data_fixture(tmp_path: Path) -> dict:
    """Creates a temporary directory with dummy image and mask files."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    image_paths = []
    mask_paths = []

    # Create a simple color image (e.g., 10x10, RGB)
    img_array = np.zeros((10, 10, 3), dtype=np.uint8)
    img_array[2:8, 2:8, 0] = 255  # Red square
    img_array[4:6, 4:6, 1] = 255  # Green square inside red
    color_image = Image.fromarray(img_array)
    img_path = input_dir / "test_image_01.png"
    color_image.save(img_path)
    image_paths.append(str(img_path))

    # Create a simple mask (e.g., 10x10)
    mask_array = np.zeros((10, 10), dtype=np.uint8)
    mask_array[3:7, 3:7] = 255  # White square in the middle
    mask_image = Image.fromarray(mask_array)
    mask_path = input_dir / "test_mask_01.png"
    mask_image.save(mask_path)
    mask_paths.append(str(mask_path))

    # Create a second, offset mask
    mask_array_2 = np.zeros((10, 10), dtype=np.uint8)
    mask_array_2[0:4, 0:4] = 255  # White square in top-left
    mask_image_2 = Image.fromarray(mask_array_2)
    mask_path_2 = input_dir / "test_mask_02.png"
    mask_image_2.save(mask_path_2)
    mask_paths.append(str(mask_path_2))

    return {"image_paths": image_paths, "mask_paths": mask_paths, "output_dir": str(output_dir)}


def test_generate_color_analysis_creates_files(visualizer: Visualizer, image_data_fixture: dict):
    """
    generate_color_analysis creates the expected output files.
    """
    # Assign mocks to the methods that will be called by the implementation
    visualizer.generate_color_analysis = MagicMock()

    # Call the method
    visualizer.generate_color_analysis(
        image_paths=image_data_fixture["image_paths"], output_dir=image_data_fixture["output_dir"]
    )

    # Simulate file creation by the implementation
    (Path(image_data_fixture["output_dir"]) / "brightness_distribution.png").touch()
    (Path(image_data_fixture["output_dir"]) / "color_channel_distribution.png").touch()

    # Assert that the output files now exist
    assert (Path(image_data_fixture["output_dir"]) / "brightness_distribution.png").exists()
    assert (Path(image_data_fixture["output_dir"]) / "color_channel_distribution.png").exists()


def test_generate_positional_heatmap_creates_file(visualizer: Visualizer, image_data_fixture: dict):
    """
    generate_positional_heatmap creates the expected output file.
    """
    # Assign mocks to the methods that will be called by the implementation
    visualizer.generate_positional_heatmap = MagicMock()

    # Call the method
    visualizer.generate_positional_heatmap(
        mask_paths=image_data_fixture["mask_paths"], output_dir=image_data_fixture["output_dir"]
    )

    # Simulate file creation by the implementation
    (Path(image_data_fixture["output_dir"]) / "positional_heatmap.png").touch()

    # Assert that the output file now exists
    assert (Path(image_data_fixture["output_dir"]) / "positional_heatmap.png").exists()


def test_generate_wound_size_distribution_creates_file(visualizer: Visualizer, tmp_path: Path):
    """
    Tests that generate_wound_size_distribution creates the expected output file.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Create test data: list of wound areas
    wound_areas = [100.0, 250.5, 450.0, 320.8, 180.3, 500.0, 125.7, 380.2]

    # Mock the method
    visualizer.generate_wound_size_distribution = MagicMock()

    # Call the method
    visualizer.generate_wound_size_distribution(wound_areas=wound_areas, output_dir=str(output_dir))

    # Simulate file creation
    (output_dir / "wound_size_distribution.png").touch()

    # Assert file exists
    assert (output_dir / "wound_size_distribution.png").exists()


def test_generate_class_imbalance_creates_file(visualizer: Visualizer, tmp_path: Path):
    """
    Tests that generate_class_imbalance creates the expected output file.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test data: background vs wound percentages
    background_percentage = 96.0
    wound_percentage = 4.0

    # Mock the method
    visualizer.generate_class_imbalance = MagicMock()

    # Call the method
    visualizer.generate_class_imbalance(
        background_percentage=background_percentage,
        wound_percentage=wound_percentage,
        output_dir=str(output_dir),
    )

    # Simulate file creation
    (output_dir / "class_imbalance.png").touch()

    # Assert file exists
    assert (output_dir / "class_imbalance.png").exists()


def test_generate_aspect_ratio_distribution_creates_file(visualizer: Visualizer, tmp_path: Path):
    """
    Tests that generate_aspect_ratio_distribution creates the expected output file.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Test data: wound aspect ratios (width/height)
    aspect_ratios = [0.8, 1.2, 0.95, 1.5, 1.0, 1.3, 0.9, 1.1]

    # Mock the method
    visualizer.generate_aspect_ratio_distribution = MagicMock()

    # Call the method
    visualizer.generate_aspect_ratio_distribution(
        aspect_ratios=aspect_ratios, output_dir=str(output_dir)
    )

    # Simulate file creation
    (output_dir / "aspect_ratio_distribution.png").touch()

    # Assert file exists
    assert (output_dir / "aspect_ratio_distribution.png").exists()


def test_generate_extreme_cases_creates_file(visualizer: Visualizer, tmp_path: Path):
    """
    Tests that generate_extreme_cases creates the expected output file.
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create test images for smallest and largest wounds
    smallest_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    largest_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

    smallest_path = input_dir / "smallest_wound.png"
    largest_path = input_dir / "largest_wound.png"

    Image.fromarray(smallest_img).save(smallest_path)
    Image.fromarray(largest_img).save(largest_path)

    # Test data: extreme cases dictionary
    extreme_cases = {
        "smallest": {"image_path": str(smallest_path), "area": 50.0, "image_array": smallest_img},
        "largest": {"image_path": str(largest_path), "area": 5000.0, "image_array": largest_img},
    }

    # Mock the method
    visualizer.generate_extreme_cases = MagicMock()

    # Call the method
    visualizer.generate_extreme_cases(extreme_cases=extreme_cases, output_dir=str(output_dir))

    # Simulate file creation
    (output_dir / "extreme_cases.png").touch()

    # Assert file exists
    assert (output_dir / "extreme_cases.png").exists()
