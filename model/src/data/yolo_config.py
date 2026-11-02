"""
YOLO dataset configuration validation and creation.

Contract: YOLOConfigContract

This module provides functions to validate and create YOLO dataset configuration
files for wound segmentation training.
"""

from pathlib import Path
from typing import Any, Dict, Union


def validate_yolo_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate YOLO dataset configuration file structure and paths.

    Contract Definition:
    --------------------
    Inputs:
        - config_path: Path to YAML configuration file

    Expected Outputs (Success):
        {
            "success": True,
            "data": {
                "path": str,              # Dataset root path
                "train": str,             # Train directory name
                "val": str,               # Validation directory name
                "test": str,              # Test directory name
                "nc": int,                # Number of classes
                "names": Dict[int, str],  # Class ID to name mapping
                "all_dirs_exist": bool    # Whether all directories exist
            },
            "error": None
        }

    Expected Outputs (Failure):
        {
            "success": False,
            "data": None,
            "error": str  # Error message describing validation failure
        }

    Validation Rules:
    -----------------
    1. File must exist and be readable
    2. Must contain required keys: path, train, val, test, nc, names
    3. 'path' must be a valid string
    4. 'train', 'val', 'test' must be valid directory names
    5. 'nc' must be positive integer
    6. 'names' must be dict mapping int to str
    7. len(names) must equal nc

    Test Coverage:
    --------------
    - tests/unit/data/test_yolo_config.py::test_validate_yolo_config_success
    - tests/unit/data/test_yolo_config.py::test_validate_yolo_config_missing_file
    - tests/unit/data/test_yolo_config.py::test_validate_yolo_config_missing_keys
    - tests/unit/data/test_yolo_config.py::test_validate_yolo_config_invalid_nc
    - tests/unit/data/test_yolo_config.py::test_validate_yolo_config_names_mismatch

    Args:
        config_path: Path to YOLO YAML configuration file

    Returns:
        Result dictionary with success status, validated data, and error message

    Example:
        >>> result = validate_yolo_config("wound_dataset.yaml")
        >>> if result["success"]:
        ...     print(f"Dataset root: {result['data']['path']}")
        ...     print(f"Classes: {result['data']['names']}")
    """
    import yaml

    # Convert to Path object for validation
    config_path = Path(config_path)

    # Validate file exists
    if not config_path.exists():
        return {
            "success": False,
            "data": None,
            "error": f"Configuration file not found: {config_path}",
        }

    # Load YAML content
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return {"success": False, "data": None, "error": f"Failed to parse YAML: {e}"}

    # Validate required keys present
    required_keys = ["path", "train", "val", "test", "nc", "names"]
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        return {
            "success": False,
            "data": None,
            "error": f"Missing required keys: {', '.join(missing_keys)}",
        }

    # Validate nc is positive integer
    nc = config["nc"]
    if not isinstance(nc, int) or nc <= 0:
        return {
            "success": False,
            "data": None,
            "error": f"'nc' must be positive integer, got: {nc}",
        }

    # Validate names is dict and matches nc
    names = config["names"]
    if not isinstance(names, dict):
        return {
            "success": False,
            "data": None,
            "error": f"'names' must be dict, got: {type(names).__name__}",
        }

    if len(names) != nc:
        return {
            "success": False,
            "data": None,
            "error": f"'names' count ({len(names)}) doesn't match 'nc' ({nc})",
        }

    # Check if all directories exist (optional - directory may not exist yet)
    dataset_root = Path(config["path"])
    all_dirs_exist = False
    if dataset_root.exists():
        train_dir = dataset_root / config["train"]
        val_dir = dataset_root / config["val"]
        test_dir = dataset_root / config["test"]
        all_dirs_exist = train_dir.exists() and val_dir.exists() and test_dir.exists()

    # Validation successful - return parsed data
    return {
        "success": True,
        "data": {
            "path": config["path"],
            "train": config["train"],
            "val": config["val"],
            "test": config["test"],
            "nc": nc,
            "names": names,
            "all_dirs_exist": all_dirs_exist,
        },
        "error": None,
    }


def create_yolo_config(
    dataset_root: Union[str, Path],
    train_dir: str = "train_images",
    val_dir: str = "test_images",
    test_dir: str = "test_images",
    output_path: Union[str, Path] = None,
) -> Dict[str, Any]:
    """
    Create YOLO dataset configuration file for wound segmentation.

    Contract Definition:
    --------------------
    Inputs:
        - dataset_root: Root directory containing image/mask subdirectories
        - train_dir: Name of training images directory (default: "train_images")
        - val_dir: Name of validation images directory (default: "test_images")
        - test_dir: Name of test images directory (default: "test_images")
        - output_path: Where to save YAML file (default: dataset_root/wound_dataset.yaml)

    Expected Outputs (Success):
        {
            "success": True,
            "data": {
                "config_path": Path,  # Where YAML was written
                "yaml_content": str   # YAML content that was written
            },
            "error": None
        }

    Expected Outputs (Failure):
        {
            "success": False,
            "data": None,
            "error": str  # Error message describing failure
        }

    Behavior:
    ---------
    1. Validates dataset_root exists
    2. Creates YAML with structure:
       - path: <dataset_root>
       - train: <train_dir>
       - val: <val_dir>
       - test: <test_dir>
       - nc: 1 (wound class only)
       - names: {0: "wound"}
    3. Writes to output_path (or default location)
    4. Returns validation result

    Test Coverage:
    --------------
    - tests/unit/data/test_yolo_config.py::test_create_yolo_config_success
    - tests/unit/data/test_yolo_config.py::test_create_yolo_config_invalid_root
    - tests/unit/data/test_yolo_config.py::test_create_yolo_config_custom_output
    - tests/unit/data/test_yolo_config.py::test_create_yolo_config_validates_output

    Args:
        dataset_root: Root directory containing image/mask subdirectories
        train_dir: Training images directory name
        val_dir: Validation images directory name
        test_dir: Test images directory name
        output_path: Output YAML file path (optional)

    Returns:
        Result dictionary with success status, config path, and error message

    Example:
        >>> result = create_yolo_config("/kaggle/working/data_wound_seg")
        >>> if result["success"]:
        ...     print(f"Config created: {result['data']['config_path']}")
    """
    import yaml

    # Convert to Path object
    dataset_root = Path(dataset_root)

    # Validate dataset root exists
    if not dataset_root.exists():
        return {
            "success": False,
            "data": None,
            "error": f"Dataset root not found: {dataset_root}",
        }

    # Determine output path (default or custom)
    if output_path is None:
        output_path = dataset_root / "wound_dataset.yaml"
    else:
        output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create YAML configuration structure
    config = {
        "path": str(dataset_root),
        "train": train_dir,
        "val": val_dir,
        "test": test_dir,
        "nc": 1,  # Single class: wound
        "names": {0: "wound"},
    }

    # Convert to YAML string
    yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)

    # Write to file
    try:
        with open(output_path, "w") as f:
            f.write(yaml_content)
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": f"Failed to write config file: {e}",
        }

    # Return success with config path
    return {
        "success": True,
        "data": {"config_path": output_path, "yaml_content": yaml_content},
        "error": None,
    }
