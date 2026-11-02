"""
Kaggle Dataset Management

Handles creation and versioning of Kaggle datasets for baseline results.
"""

import os
from pathlib import Path
from typing import Dict, Any


def validate_kaggle_credentials() -> Dict[str, Any]:
    """
    Validate Kaggle API credentials are available.

    Contract:
        Inputs: None (reads from environment variables)
        Outputs:
            Success: {"valid": True, "username": str}
            Failure: {"valid": False, "error": str}

        Environment Variables Required:
            - KAGGLE_USERNAME: Kaggle username
            - KAGGLE_KEY: Kaggle API key

        Behavior:
            - Checks for presence of both environment variables
            - Returns success with username if both present
            - Returns failure with error message if either missing

        Edge Cases:
            - Empty string values are treated as missing
            - Case-sensitive environment variable names
    """
    username = os.environ.get("KAGGLE_USERNAME")
    api_key = os.environ.get("KAGGLE_KEY")

    if not username or not api_key:
        return {"valid": False, "error": "Missing credentials"}
    return {"valid": True, "username": username}


def create_baseline_dataset_metadata(
    username: str, output_folder: Path
) -> Dict[str, Any]:
    """
    Generate metadata for baseline results dataset.

    Contract:
        Inputs:
            - username (str): Kaggle username for dataset ID
            - output_folder (Path): Directory to write metadata file

        Outputs:
            Success: {
                "success": True,
                "metadata": {
                    "title": str,
                    "id": str,  # format: "username/slug"
                    "subtitle": str,
                    "description": str,
                    "keywords": List[str],
                    "licenses": List[Dict],
                    "isPrivate": bool
                },
                "metadata_path": Path
            }
            Failure: {
                "success": False,
                "error": str
            }

        Behavior:
            - Creates dataset-metadata.json in output_folder
            - Uses fixed slug: "wound-seg-baseline-results-v1"
            - Validates slug format (lowercase, alphanumeric, hyphens only)
            - Overwrites existing metadata file

        Validations:
            - username must be non-empty string
            - output_folder must exist
            - slug must match pattern: ^[a-z0-9-]+$

        Edge Cases:
            - Creates parent directories if needed
            - Raises ValueError for invalid slug
            - Returns error dict if write fails
    """
    import json
    import re

    title = "Wound Segmentation Zero-Shot Baseline Results"
    slug = "wound-seg-baseline-results-v1"
    subtitle = "YOLOv8n-seg zero-shot evaluation on clinical wound dataset"

    description = """# Zero-Shot Baseline Evaluation Results

## Overview
Comprehensive evaluation results from pretrained YOLOv8n-seg (COCO weights, no fine-tuning)
applied to clinical wound segmentation dataset.

## Contents
- Per-image metrics (Dice, IoU, Precision, Recall, Boundary F1)
- Aggregate statistics with 95% bootstrap confidence intervals
- Confusion matrix visualization
- Example predictions (best/average/worst)

## Key Findings
- Zero-shot Dice: ~5% (95% CI: 4-6%)
- High recall (~77%) but low precision (~3%)
- Clinical implication: COCO weights insufficient for medical segmentation

## Methodology
- Model: YOLOv8n-seg (COCO pretrained)
- Dataset: 552 specialist-annotated wound images
- Statistical validation: 10,000 bootstrap iterations
"""

    keywords = [
        "wound",
        "segmentation",
        "yolo",
        "baseline",
        "medical-imaging",
        "computer-vision",
    ]

    # Validate slug format
    if not re.match(r"^[a-z0-9-]+$", slug):
        return {"success": False, "error": "Invalid slug format"}

    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "subtitle": subtitle,
        "description": description,
        "keywords": keywords,
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": False,
    }

    try:
        # Create parent directories if needed
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Write metadata file
        metadata_path = output_folder / "dataset-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "success": True,
            "metadata": metadata,
            "metadata_path": metadata_path,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
