"""
Data preprocessing functions.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import cv2
import pandas as pd


def convert_mask_to_yolo_format(
    mask_path: Union[str, Path],
    image_width: int,
    image_height: int,
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Convert binary wound segmentation mask to YOLO format.

    Contract: DatasetFormatContract
    Tests: tests/unit/data/test_baseline_dataset_format.py

    Args:
        mask_path: Path to binary mask image (PNG, 0=background, 255=wound)
        image_width: Image width in pixels (512 for this dataset)
        image_height: Image height in pixels (512 for this dataset)
        output_path: Path where YOLO label file will be saved

    Returns:
        {
            "success": bool,
            "num_polygons": int,
            "label_file_path": Path
        }

    Raises:
        FileNotFoundError: If mask_path doesn't exist
        ValueError: If image dimensions <= 0
        PermissionError: If cannot write to output_path
    """
    # Convert to Path objects
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    # Validate inputs
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid dimensions: {image_width}x{image_height}")

    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    # Binarize mask at threshold 127 (handles anti-aliased edges)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours (wound regions)
    # Note: NumPy 2.x compatibility - cv2.findContours returns correct format
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write YOLO format labels
    num_polygons = 0
    with open(output_path, "w") as f:
        for contour in contours:
            # Skip tiny contours (noise)
            if len(contour) < 3:
                continue

            # Normalize coordinates to [0, 1]
            normalized_coords = []
            for point in contour:
                x, y = point[0]  # contour point format: [[x, y]]
                norm_x = x / image_width
                norm_y = y / image_height
                normalized_coords.extend([norm_x, norm_y])

            # Write: class_id (0 for wound) + normalized coordinates
            coords_str = " ".join(f"{c:.6f}" for c in normalized_coords)
            f.write(f"0 {coords_str}\n")
            num_polygons += 1

    return {
        "success": True,
        "num_polygons": num_polygons,
        "label_file_path": output_path,
    }


def create_ddi_train_test_split(
    metadata_path: Union[str, Path],
    train_output_path: Union[str, Path],
    test_output_path: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Create stratified train/test split for DDI dataset.

    Args:
        metadata_path: Path to DDI metadata CSV with 'disease' column
        train_output_path: Output path for training split CSV
        test_output_path: Output path for test split CSV
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        {
            "success": bool,
            "train_count": int,
            "test_count": int,
            "train_path": Path,
            "test_path": Path,
            "disease_distribution": dict
        }

    Raises:
        FileNotFoundError: If metadata_path doesn't exist
        ValueError: If 'disease' column missing or test_size invalid
    """
    metadata_path = Path(metadata_path)
    train_output_path = Path(train_output_path)
    test_output_path = Path(test_output_path)

    # Validate inputs
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got: {test_size}")

    # Load metadata
    df = pd.read_csv(metadata_path)

    if "disease" not in df.columns:
        raise ValueError(f"'disease' column not found in {metadata_path}")

    # Perform stratified split manually (avoid sklearn dependency)
    import numpy as np
    np.random.seed(random_state)

    train_indices = []
    test_indices = []

    for disease in df["disease"].unique():
        disease_indices = df[df["disease"] == disease].index.tolist()
        np.random.shuffle(disease_indices)

        n_test = max(1, int(len(disease_indices) * test_size))
        test_indices.extend(disease_indices[:n_test])
        train_indices.extend(disease_indices[n_test:])

    train_df = df.loc[train_indices].sort_index()
    test_df = df.loc[test_indices].sort_index()

    # Create output directories if needed
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save splits
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    # Calculate disease distribution
    disease_distribution = {
        "train": train_df["disease"].value_counts().to_dict(),
        "test": test_df["disease"].value_counts().to_dict(),
    }

    return {
        "success": True,
        "train_count": len(train_df),
        "test_count": len(test_df),
        "train_path": train_output_path,
        "test_path": test_output_path,
        "disease_distribution": disease_distribution,
    }


def create_empty_yolo_label(
    output_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Create an empty YOLO label file (for hard negative samples).

    Empty label file indicates image contains no objects of interest.
    Used for training model to reject false positives.

    Args:
        output_path: Path where empty label file will be saved

    Returns:
        {
            "success": bool,
            "label_file_path": Path
        }

    Raises:
        PermissionError: If cannot write to output_path
    """
    output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create empty file (YOLO interprets empty file as no objects)
    output_path.touch()

    return {
        "success": True,
        "label_file_path": output_path,
    }
