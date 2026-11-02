"""
Generic Ablation Smoke Test Builder

Creates CPU smoke test notebooks for any ablation experiment.
Validates complete pipeline with TINY dataset before GPU deployment.
"""

import json
import textwrap
import nbformat as nbf
from pathlib import Path
from typing import Dict, Optional


SMOKE_TEST_CONFIG = {
    "1b": {
        "name": "Ablation 1b (Dice++ Loss)",
        "slug": "ablation-1b-smoke-test",
        "intervention": "Dice++ Loss",
        "training_params": """
        # Dice++ Loss Training Parameters
        model = YOLO("yolov8n-seg.pt")
        model.train(
            data=str(CONFIG_PATH),
            epochs=1,  # Smoke test: 1 epoch only
            imgsz=512,
            batch=4,  # TINY dataset: small batch
            lr0=0.015,
            workers=2,
            amp=False,
            project=str(TRAINING_OUTPUT),
            name=f"seed_{seed}",
            exist_ok=True,
            device=0,
            verbose=True,
            # INTERVENTION: Dice++ Loss
            loss="dicepp",  # Custom Dice++ implementation
        )
        """,
    },
    "5": {
        "name": "Ablation 5 (Test-Time Augmentation)",
        "slug": "ablation-5-smoke-test",
        "intervention": "Test-Time Augmentation (TTA)",
        "training_params": """
        # BASELINE training (no intervention during training)
        model = YOLO("yolov8n-seg.pt")
        model.train(
            data=str(CONFIG_PATH),
            epochs=1,  # Smoke test: 1 epoch only
            imgsz=512,
            batch=4,  # TINY dataset: small batch
            lr0=0.015,
            workers=2,
            amp=False,
            project=str(TRAINING_OUTPUT),
            name=f"seed_{seed}",
            exist_ok=True,
            device=0,
            verbose=True,
        )
        """,
        "eval_modification": """
        # INTERVENTION: Test-Time Augmentation (D4 group)
        # Apply 8 transformations and average predictions
        transformations = [
            lambda x: x,  # Identity
            lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
            lambda x: cv2.rotate(x, cv2.ROTATE_180),
            lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
            lambda x: cv2.flip(x, 1),  # Horizontal flip
            lambda x: cv2.flip(x, 0),  # Vertical flip
            lambda x: cv2.flip(cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE), 1),  # Diagonal
            lambda x: cv2.flip(cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE), 0),  # Anti-diagonal
        ]

        # Average predictions from all transformations
        tta_predictions = []
        for transform in transformations:
            transformed_img = transform(img_rgb)
            pred = model.predict(transformed_img, verbose=False)[0]
            tta_predictions.append(pred)

        # Average and threshold
        yolo_results = [np.mean(tta_predictions, axis=0)]
        """,
    },
    "6": {
        "name": "Ablation 6 (Label Smoothing)",
        "slug": "ablation-6-smoke-test",
        "intervention": "Label Smoothing",
        "training_params": """
        # Label Smoothing Training Parameters
        model = YOLO("yolov8n-seg.pt")
        model.train(
            data=str(CONFIG_PATH),
            epochs=1,  # Smoke test: 1 epoch only
            imgsz=512,
            batch=4,  # TINY dataset: small batch
            lr0=0.015,
            workers=2,
            amp=False,
            project=str(TRAINING_OUTPUT),
            name=f"seed_{seed}",
            exist_ok=True,
            device=0,
            verbose=True,
            # INTERVENTION: Label Smoothing
            label_smoothing=0.1,  # Soft targets (0.1, 0.9) instead of (0, 1)
            mixup=0.0,  # Disabled (unsafe for medical imaging)
        )
        """,
    },
}


def create_smoke_test_notebook(ablation_id: str, config: Dict) -> nbf.NotebookNode:
    """Create smoke test notebook for specified ablation."""

    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    }

    cells = []

    # Cell 0: NumPy Locking + Ultralytics Install
    cells.append(nbf.v4.new_code_cell("""# Fix: Kaggle's default NumPy 1.26.4 is CORRECT - prevent upgrades to 2.x
import sys
print("Step 1: Record Kaggle's default NumPy version...")
import numpy as np
default_numpy = np.__version__
print(f"   Default NumPy: {default_numpy}")

print("\\nStep 2: Install ultralytics...")
import subprocess
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)

print("\\nStep 3: LOCK NumPy back to Kaggle's default...")
subprocess.run(["pip", "install", "-q", "--force-reinstall", f"numpy=={default_numpy}"], check=True)

print("\\nInstallation complete!")
import numpy as np
from ultralytics import YOLO
print(f"   NumPy: {np.__version__}")
print("   PASS: Ultralytics imported successfully")"""))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell("""import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Environment detection
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent

# KAGGLE DATASET SETUP
if os.path.exists("/kaggle/input/wound-seg-code-v1"):
    sys.path.insert(0, "/kaggle/input/wound-seg-code-v1")
    print("[INFO] Code dataset added to Python path")
    from evaluation.metrics import calculate_segmentation_metrics, calculate_boundary_f1, calculate_bootstrap_ci
    from pipeline.eda import EDA_CONFIG, Visualizer
    from data.yolo_config import create_yolo_config, validate_yolo_config
    from evaluation.yolo_utils import yolo_result_to_metrics_input
else:
    sys.path.insert(0, str(PROJECT_ROOT))
    print("[INFO] Using local src directory")
    from src.evaluation.metrics import calculate_segmentation_metrics, calculate_boundary_f1, calculate_bootstrap_ci
    from src.pipeline.eda import EDA_CONFIG, Visualizer
    from src.data.yolo_config import create_yolo_config, validate_yolo_config
    from src.evaluation.yolo_utils import yolo_result_to_metrics_input

print("PASS: All modules imported successfully")"""))

    # Cell 2: Environment Setup
    cells.append(nbf.v4.new_code_cell("""# Configure paths for Kaggle
if os.path.exists("/kaggle/input"):
    IN_KAGGLE = True
    BASE_INPUT_PATH = Path("/kaggle/input")
    OUTPUT_PATH = Path("/kaggle/working")
    DATASET_PATH = BASE_INPUT_PATH / "tiny-wound-seg-smoke-test"
    CODE_PATH = BASE_INPUT_PATH / "wound-seg-code-v1"
else:
    IN_KAGGLE = False
    BASE_INPUT_PATH = PROJECT_ROOT / "data"
    OUTPUT_PATH = PROJECT_ROOT / "outputs"
    DATASET_PATH = BASE_INPUT_PATH / "TINY_TEST"
    CODE_PATH = PROJECT_ROOT / "src"

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local'}")
print(f"Dataset path: {DATASET_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# Verify dataset exists
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

print("PASS: Environment configured")"""))

    # Cell 3: YOLO Config
    cells.append(nbf.v4.new_code_cell("""# Create YOLO config
CONFIG_PATH = OUTPUT_PATH / "wound_dataset_smoke.yaml"
config_result = create_yolo_config(
    dataset_root=DATASET_PATH,
    output_path=CONFIG_PATH
)

if not config_result["success"]:
    raise RuntimeError(f"YOLO config creation failed: {config_result['error']}")

print(f"PASS: YOLO config created: {CONFIG_PATH}")

# Validate config
validation_result = validate_yolo_config(CONFIG_PATH)
if not validation_result["success"]:
    raise RuntimeError(f"YOLO config validation failed: {validation_result['error']}")

print("PASS: YOLO config validated")"""))

    # Cell 4: Training Setup
    cells.append(nbf.v4.new_markdown_cell(f"""## Training Setup

**Intervention**: {config['intervention']}

This smoke test validates the training pipeline with 1 epoch on TINY dataset (2-5 images).
"""))

    # Cell 5: Training (parameterized)
    training_params_dedented = textwrap.dedent(config['training_params']).strip()
    training_cell = f"""# Single-seed smoke test (seed 42 only)
TRAINING_OUTPUT = OUTPUT_PATH / "yolo_training"
seed = 42

{training_params_dedented}

print(f"PASS: Training completed for seed {{seed}}")

# Verify weights exist
weights_path = TRAINING_OUTPUT / f"seed_{{seed}}" / "weights" / "best.pt"
if not weights_path.exists():
    raise FileNotFoundError(f"Weights not found at {{weights_path}}")

print(f"PASS: Weights saved: {{weights_path}}")
"""
    cells.append(nbf.v4.new_code_cell(training_cell))

    # Cell 6: Evaluation (with optional TTA modification)
    eval_modification = config.get("eval_modification", "")
    if eval_modification:
        eval_modification = textwrap.dedent(eval_modification).strip()

    # Build eval cell without f-string to avoid escaping issues
    eval_base = """# Evaluation on TINY test set
TEST_IMAGES_DIR = DATASET_PATH / "test_images"
TEST_MASKS_DIR = DATASET_PATH / "test_masks"

# Load trained model
model = YOLO(str(weights_path))

# Evaluate on first test image
test_images = sorted(TEST_IMAGES_DIR.glob("*.png"))[:1]  # Just 1 image for smoke test

for img_path in test_images:
    image_id = img_path.stem
    mask_path = TEST_MASKS_DIR / f"{image_id}.png"

    if not mask_path.exists():
        print(f"WARNING: Skipping {image_id} - no ground truth mask")
        continue

    # Load image
    img_rgb = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    """

    if eval_modification:
        eval_base += "\n" + eval_modification
    else:
        eval_base += """# Standard evaluation (no TTA)
    yolo_results = model.predict(img_rgb, verbose=False)
"""

    eval_base += """
    # Convert to metrics input
    mask_path_str = str(mask_path)
    conversion_result = yolo_result_to_metrics_input(
        yolo_results[0],
        mask_path_str,  # ground_truth_mask_path parameter
        image_id
    )

    if not conversion_result["success"]:
        raise RuntimeError(f"Conversion failed: {conversion_result['error']}")

    print(f"PASS: Evaluation completed for {image_id}")

    # Test metrics calculation
    data = conversion_result["data"]
    metrics_result = calculate_segmentation_metrics(
        data["pred_mask"],
        data["gt_mask"],
        image_id
    )

    print(f"  Dice: {metrics_result['dice_score']:.4f}")

print("PASS: Evaluation pipeline validated")
"""
    cells.append(nbf.v4.new_code_cell(eval_base))

    # Cell 7: Schema Validation
    cells.append(nbf.v4.new_code_cell("""# Validate metrics schema (lowercase column names)
expected_keys = {'tp', 'fp', 'tn', 'fn', 'dice_score', 'iou', 'precision', 'recall'}
actual_keys = set(metrics_result.keys())

missing_keys = expected_keys - actual_keys
extra_keys = actual_keys - expected_keys

if missing_keys:
    raise AssertionError(f"Missing keys in metrics: {missing_keys}")

if extra_keys:
    print(f"WARNING: Extra keys in metrics: {extra_keys} (not an error, just FYI)")

print("PASS: Schema validation passed")
print(f"  Metrics keys: {sorted(actual_keys)}")"""))

    # Cell 8: Output Files
    cells.append(nbf.v4.new_code_cell("""# Verify output files created
smoke_test_report = {
    "ablation": "%s",
    "intervention": "%s",
    "validation_checks": {
        "numpy_locking": {"passed": True},
        "imports": {"passed": True},
        "yolo_config": {"passed": True},
        "training": {"passed": True, "epochs": 1},
        "evaluation": {"passed": True, "images_tested": 1},
        "schema_validation": {"passed": True},
    },
    "metrics_sample": {
        "dice": float(metrics_result['dice_score']),
        "iou": float(metrics_result['iou']),
        "recall": float(metrics_result['recall']),
        "precision": float(metrics_result['precision']),
    }
}

report_path = OUTPUT_PATH / "smoke_test_report.json"
with open(report_path, 'w') as f:
    json.dump(smoke_test_report, f, indent=2)

print(f"PASS: Smoke test report saved: {report_path}")
print("\\nALL SMOKE TEST VALIDATIONS PASSED!")
""" % (ablation_id, config['intervention'])))

    # Add all cells
    nb["cells"] = cells

    # Normalize and validate
    nb = nbf.validator.normalize(nb)[1]
    nbf.validate(nb)

    return nb


def create_kernel_metadata(ablation_id: str, config: Dict, output_dir: Path) -> None:
    """Create kernel-metadata.json for Kaggle deployment."""

    metadata = {
        "id": f"bkdkaggle/{config['slug']}",
        "title": f"{config['name']} Smoke Test (CPU)",
        "code_file": f"ablation_{ablation_id}_smoke_test.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",  # Need to install ultralytics
        "dataset_sources": [
            "bkdkaggle/wound-seg-code-v1",
            "bkdkaggle/baseline-outputs",  # Contains ALL training data
            "bkdkaggle/tiny-wound-seg-smoke-test"  # TINY test dataset
        ],
        "kernel_sources": []
    }

    metadata_path = output_dir / "kernel-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"PASS: Kernel metadata created: {metadata_path}")


def main():
    """Build smoke tests for Ablations 1b, 5, 6."""

    project_root = Path(__file__).parent.parent
    ablations_dir = project_root / "notebooks" / "03_ablations"

    for ablation_id in ["1b", "5", "6"]:
        config = SMOKE_TEST_CONFIG[ablation_id]

        print(f"\n{'='*60}")
        print(f"Building Smoke Test: {config['name']}")
        print(f"{'='*60}")

        # Create output directory
        output_dir = ablations_dir / f"ablation_{ablation_id}_smoke"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Build notebook
        nb = create_smoke_test_notebook(ablation_id, config)

        # Write notebook
        notebook_path = output_dir / f"ablation_{ablation_id}_smoke_test.ipynb"
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)

        print(f"PASS: Notebook created: {notebook_path}")

        # Create kernel metadata
        create_kernel_metadata(ablation_id, config, output_dir)

        print(f"PASS: Smoke test for Ablation {ablation_id} complete!")

    print(f"\n{'='*60}")
    print("ALL SMOKE TESTS CREATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Deploy smoke tests to Kaggle CPU:")
    for ablation_id in ["1b", "5", "6"]:
        slug = SMOKE_TEST_CONFIG[ablation_id]['slug']
        print(f"   cd notebooks/03_ablations/ablation_{ablation_id}_smoke && kaggle kernels push -p .")
    print("\n2. Validate all pass (3 min each)")
    print("3. Deploy production notebooks to GPU")


if __name__ == "__main__":
    main()
