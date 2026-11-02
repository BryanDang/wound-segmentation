"""
Kaggle dataset creation and versioning.

Contracts are in the function docstrings.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def validate_kaggle_credentials() -> Dict[str, Any]:
    """
    Check that Kaggle API credentials are set in the environment.

    Returns:
        Dictionary with:
        - valid: bool
        - username: str (if valid=True)
        - error: str (if valid=False)
    """
    username = os.environ.get("KAGGLE_USERNAME")
    api_key = os.environ.get("KAGGLE_KEY")

    if not username and not api_key:
        return {
            "valid": False,
            "error": "KAGGLE_USERNAME and KAGGLE_KEY environment variables not set",
        }

    if not username:
        return {"valid": False, "error": "KAGGLE_USERNAME environment variable not set"}

    if not api_key:
        return {"valid": False, "error": "KAGGLE_KEY environment variable not set"}

    return {"valid": True, "username": username}


def create_kaggle_dataset_metadata(
    title: str,
    username: str,
    slug: str,
    output_folder: Path,
    subtitle: str = "",
    description: str = "",
    keywords: List[str] = None,
    license_name: str = "CC0-1.0",
    is_private: bool = False,
) -> Dict[str, Any]:
    """
    Create Kaggle dataset metadata JSON structure.

    Args:
        title: Dataset title (6-50 characters)
        username: Kaggle username
        slug: Dataset URL slug (3-50 characters, lowercase, hyphens only)
        output_folder: Path to folder containing dataset files
        subtitle: Brief description (20-80 characters)
        description: Detailed markdown description
        keywords: List of searchable tags
        license_name: License identifier (default: CC0-1.0 public domain)
        is_private: Whether dataset should be private

    Returns:
        Dictionary with:
        - success: bool
        - metadata: DatasetMetadata dict
        - metadata_path: Path to saved metadata file
        - error: str (if success=False)

    Raises:
        ValueError: If title/slug/subtitle length invalid
        FileNotFoundError: If output_folder doesn't exist
    """
    # Title length
    if len(title) < 6 or len(title) > 50:
        raise ValueError(f"title must be 6-50 characters, got {len(title)}")

    # Slug length
    if len(slug) < 3 or len(slug) > 50:
        raise ValueError(f"slug must be 3-50 characters, got {len(slug)}")

    # Slug format: lowercase, digits, hyphens
    if not re.match(r"^[a-z0-9-]+$", slug):
        raise ValueError("slug must contain only lowercase letters, numbers, and hyphens")

    # Subtitle length, if given
    if subtitle and (len(subtitle) < 20 or len(subtitle) > 80):
        raise ValueError(f"subtitle must be 20-80 characters, got {len(subtitle)}")

    # Output folder
    output_folder = Path(output_folder)
    if not output_folder.exists():
        raise FileNotFoundError(f"output_folder does not exist: {output_folder}")

    # Metadata dict
    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": license_name}],
    }

    # Optional fields
    if subtitle:
        metadata["subtitle"] = subtitle

    if description:
        metadata["description"] = description

    if keywords:
        metadata["keywords"] = keywords

    if is_private:
        metadata["isPrivate"] = is_private

    # Write file
    metadata_path = output_folder / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"success": True, "metadata": metadata, "metadata_path": metadata_path}


def create_kaggle_dataset(
    metadata_path: Path,
    new_dataset: bool = True,
    version_notes: str = "",
) -> Dict[str, Any]:
    """
    Create or update Kaggle dataset using API.

    Args:
        metadata_path: Path to dataset-metadata.json file
        new_dataset: True for first creation, False for versioning
        version_notes: Commit message for new version (if updating)

    Returns:
        Dictionary with:
        - success: bool
        - dataset_url: str
        - dataset_id: str
        - version: int
        - message: str
        - metadata_path: Path

    Raises:
        FileNotFoundError: If metadata_path doesn't exist
        EnvironmentError: If KAGGLE_USERNAME or KAGGLE_KEY not set
        RuntimeError: If kaggle CLI command fails
    """
    # Metadata file
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Credentials
    creds = validate_kaggle_credentials()
    if not creds["valid"]:
        raise EnvironmentError(creds["error"])

    # Dataset ID
    with open(metadata_path) as f:
        metadata = json.load(f)
    dataset_id = metadata["id"]

    # Dataset folder
    folder_path = metadata_path.parent

    # CLI command
    if new_dataset:
        cmd = ["kaggle", "datasets", "create", "-p", str(folder_path)]
        version = 1
    else:
        cmd = ["kaggle", "datasets", "version", "-p", str(folder_path)]
        if version_notes:
            cmd.extend(["-m", version_notes])
        version = -1  # Version number comes from the API response

    # Run
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=False  # Non-zero exit handled below
    )

    # Errors
    if result.returncode != 0:
        error_msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Kaggle dataset command failed: {error_msg}")

    # Result
    dataset_url = f"https://www.kaggle.com/datasets/{dataset_id}"
    message = result.stdout.strip() if result.stdout else "Dataset operation successful"

    return {
        "success": True,
        "dataset_url": dataset_url,
        "dataset_id": dataset_id,
        "version": version,
        "message": message,
        "metadata_path": metadata_path,
    }


def generate_baseline_dataset_metadata(
    username: str, output_folder: Path, notebook_version: str = "v1"
) -> Dict[str, Any]:
    """
    Generate metadata for baseline model results dataset.

    Args:
        username: Kaggle username
        output_folder: Path to /kaggle/working/baseline_final/ or equivalent
        notebook_version: Version identifier (e.g., "v1", "v2")

    Returns:
        Same as create_kaggle_dataset_metadata()
    """
    title = "Wound Segmentation Zero-Shot Baseline Results"
    slug = f"wound-seg-baseline-results-{notebook_version}"

    subtitle = "YOLOv8n-seg zero-shot evaluation on clinical wound dataset"

    description = """# Zero-Shot Baseline Evaluation Results

## Overview
This dataset contains comprehensive evaluation results from a pretrained YOLOv8n-seg model
(COCO weights, no fine-tuning) applied to a clinical wound segmentation dataset.

## Contents
- `baseline_per_image_results.json` - Per-image metrics (Dice, IoU, Precision, Recall, Boundary F1)
- `baseline_per_image_results.csv` - Same data in CSV format
- `baseline_summary_statistics.json` - Aggregate statistics with 95% bootstrap confidence intervals
- `confusion_matrix.png` - Pixel-level classification visualization
- `example_predictions.png` - Best/middle/worst prediction examples
- `completion_status.json` - Notebook execution metadata

## Key Findings
- **Zero-shot Dice Score**: ~5% (95% CI: 4-6%)
- **High Recall**: ~77% (detects most wound pixels)
- **Low Precision**: ~3% (severe over-segmentation)
- **Clinical Implication**: Pretrained COCO weights are insufficient for medical segmentation

## Methodology
- **Model**: YOLOv8n-seg with COCO pretrained weights
- **Dataset**: 552 specialist-annotated wound images (test set)
- **Metrics**: Dice, IoU, Precision, Recall, Boundary F1
- **Statistical Validation**: 10,000 bootstrap iterations for confidence intervals

## Next Steps
This baseline establishes the performance gap that domain-specific fine-tuning must address.
Target performance: Dice > 85% with Focal Tversky Loss and clinical augmentation.

## Reproducibility
All results generated from Kaggle notebook: `02_baseline_model.ipynb`
NumPy version locked at 1.26.4 to prevent binary incompatibility issues.
"""

    keywords = [
        "wound",
        "segmentation",
        "yolo",
        "baseline",
        "medical-imaging",
        "computer-vision",
        "deep-learning",
    ]

    return create_kaggle_dataset_metadata(
        title=title,
        username=username,
        slug=slug,
        output_folder=Path(output_folder),
        subtitle=subtitle,
        description=description,
        keywords=keywords,
        license_name="CC0-1.0",
        is_private=False,
    )
