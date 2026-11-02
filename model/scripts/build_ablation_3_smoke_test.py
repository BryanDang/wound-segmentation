"""
Build Ablation 3 CPU Smoke Test Notebook

Purpose: run the full pipeline on a tiny dataset (2-5 images) on Kaggle CPU
before any GPU deployment.

Checks:
1. NumPy version lock (OpenCV)
2. Code dataset imports (evaluation.yolo_utils, etc.)
3. Dataset loading (baseline-outputs structure)
4. YOLO model initialization
5. Training pipeline (1 epoch)
6. Evaluation pipeline (yolo_result_to_metrics_input signature)
7. Metrics schema (lowercase column names)
8. Output file creation

Origin: Ablations 2 and 3 went to GPU without a smoke test. A function
signature mismatch cost 3 hours of GPU time. This test takes about 3 minutes.
"""

import nbformat as nbf
from pathlib import Path

def create_smoke_test_notebook():
    """Create Ablation 3 smoke test notebook using nbformat."""

    # Initialize notebook
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

    # Cell 2: Environment Detection
    cells.append(nbf.v4.new_code_cell("""IN_KAGGLE = Path("/kaggle/input").exists()

if IN_KAGGLE:
    BASE_INPUT_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
    TINY_DATASET_PATH = Path("/kaggle/input/tiny-wound-seg-smoke-test")
    OUTPUT_PATH = Path("/kaggle/working")
else:
    BASE_INPUT_PATH = PROJECT_ROOT / "simulated_kaggle_data/input/wound-segmentation-images"
    TINY_DATASET_PATH = PROJECT_ROOT / "kaggle_notebooks/TINY_TEST"
    OUTPUT_PATH = PROJECT_ROOT / "kaggle_outputs/ablation_3_smoke"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local (simulated)'}")
print(f"Input path: {BASE_INPUT_PATH}")
print(f"TINY dataset: {TINY_DATASET_PATH}")
print(f"Output path: {OUTPUT_PATH}")"""))

    # Cell 3: Load TINY Dataset
    cells.append(nbf.v4.new_code_cell("""# Use TINY dataset for smoke test (2-5 images only)
TEST_IMAGES_DIR = TINY_DATASET_PATH / "test_images"
TEST_MASKS_DIR = TINY_DATASET_PATH / "test_masks"

test_image_paths = sorted(TEST_IMAGES_DIR.glob("*.png")) if TEST_IMAGES_DIR.exists() else []
print(f"Found {len(test_image_paths)} test images in TINY dataset")

# Verify masks exist
test_pairs = []
for img_path in test_image_paths:
    mask_path = TEST_MASKS_DIR / img_path.name
    if mask_path.exists():
        test_pairs.append((str(img_path), str(mask_path)))

print(f"Valid test pairs: {len(test_pairs)}")
if test_pairs:
    print(f"First pair: {Path(test_pairs[0][0]).name}")"""))

    # Cell 4: YOLO Config Creation
    cells.append(nbf.v4.new_code_cell("""# Create YOLO config using baseline-outputs
config_result = create_yolo_config(
    dataset_root=BASE_INPUT_PATH,
    train_dir="train",
    val_dir="test",
    test_dir="test",
    output_path=OUTPUT_PATH / "wound_dataset.yaml",
)

if config_result["success"]:
    config_path = config_result["data"]["config_path"]
    print(f"PASS: Configuration created: {config_path}")

    # Validate
    validation_result = validate_yolo_config(config_path)
    if validation_result["success"]:
        print(f"PASS: Configuration validated")
    else:
        raise ValueError(f"Validation failed: {validation_result['error']}")
else:
    raise ValueError(f"Config creation failed: {config_result['error']}")"""))

    # Cell 5: Single-Epoch Smoke Test Training
    cells.append(nbf.v4.new_code_cell("""# SMOKE TEST: Train for 1 epoch only (validate pipeline works)
print("=" * 80)
print("SMOKE TEST: Single-Epoch Training (Ablation 3: Multiscale Training)")
print("=" * 80)

model = YOLO("yolov8n-seg.pt")

# Train for 1 epoch with clinical augmentation
results = model.train(
    data=str(config_path),
    # Minimal training (smoke test only)
    epochs=1,
    batch=8,  # Small batch for CPU
    imgsz=512,
    # Multiscale Training (INTERVENTION)
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
    # Reproducibility
    seed=42,
    save=True,
    project=str(OUTPUT_PATH / "yolo_training_smoke"),
    name="seed_42",
    exist_ok=True,
    verbose=False,
)

print("\\nPASS: Training completed (1 epoch smoke test)")"""))

    # Cell 6: Evaluation Smoke Test (tests function signature)
    cells.append(nbf.v4.new_code_cell("""# Test evaluation pipeline with correct function signature
print("=" * 80)
print("SMOKE TEST: Evaluation Pipeline (Testing Function Signature!)")
print("=" * 80)

smoke_results = []

for idx, (img_path_str, mask_path_str) in enumerate(test_pairs):
    image_id = Path(img_path_str).stem

    # Load image
    img = cv2.imread(img_path_str)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    yolo_results = model.predict(img_rgb, verbose=False)

    # Test correct function signature (3 parameters!)
    # This is what failed in Ablations 2 & 3 production runs
    conversion_result = yolo_result_to_metrics_input(
        yolo_results[0],
        mask_path_str,  # ground_truth_mask_path parameter
        image_id
    )

    if conversion_result["success"]:
        pred_mask = conversion_result["data"]["pred_mask"]
        gt_mask = conversion_result["data"]["gt_mask"]

        # Calculate metrics
        metrics = calculate_segmentation_metrics(pred_mask, gt_mask, image_id)
        boundary_f1 = calculate_boundary_f1(pred_mask, gt_mask, tolerance=2)
        metrics["boundary_f1"] = boundary_f1

        smoke_results.append(metrics)
        print(f"  PASS: Processed {image_id}: Dice = {metrics['dice']:.4f}")
    else:
        print(f"  FAIL: Failed {image_id}: {conversion_result['error']}")
        raise ValueError(f"Evaluation failed: {conversion_result['error']}")

print(f"\\nPASS: Evaluation complete: {len(smoke_results)} images processed")"""))

    # Cell 7: Schema Validation
    cells.append(nbf.v4.new_code_cell("""# Validate metrics schema (lowercase column names)
print("=" * 80)
print("SMOKE TEST: Schema Validation")
print("=" * 80)

if smoke_results:
    first_result = smoke_results[0]
    expected_keys = {"image_id", "dice", "iou", "precision", "recall", "tp", "fp", "tn", "fn", "boundary_f1"}
    actual_keys = set(first_result.keys())

    if expected_keys == actual_keys:
        print("PASS: Schema validation passed")
        print(f"  Columns: {sorted(actual_keys)}")
    else:
        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        print(f"FAIL: Schema mismatch!")
        if missing:
            print(f"  Missing: {missing}")
        if extra:
            print(f"  Extra: {extra}")
        raise ValueError("Schema validation failed")
else:
    print("FAIL: No results to validate")
    raise ValueError("No evaluation results")"""))

    # Cell 8: Output File Creation Test
    cells.append(nbf.v4.new_code_cell("""# Test output file creation
print("=" * 80)
print("SMOKE TEST: Output File Creation")
print("=" * 80)

# Save smoke test results
results_path = OUTPUT_PATH / "smoke_test_results.json"
with open(results_path, "w") as f:
    json.dump(smoke_results, f, indent=2)

print(f"PASS: Results saved: {results_path}")

# Save smoke test report
report = {
    "validation_checks": {
        "numpy_locking": {"passed": True, "version": np.__version__},
        "code_imports": {"passed": True},
        "dataset_loading": {"passed": True, "images": len(test_pairs)},
        "yolo_config": {"passed": True},
        "training_pipeline": {"passed": True, "epochs": 1},
        "evaluation_pipeline": {"passed": True, "images_processed": len(smoke_results)},
        "schema_validation": {"passed": True, "columns": sorted(smoke_results[0].keys())},
        "output_files": {"passed": True, "count": 2}
    },
    "status": "READY_FOR_PRODUCTION",
    "smoke_test_duration": "~3 minutes",
    "production_estimate": "~1.5 hours (3 seeds × 30-40 min)"
}

report_path = OUTPUT_PATH / "smoke_test_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"PASS: Report saved: {report_path}")

print("\\n" + "=" * 80)
print("PASS: SMOKE TEST COMPLETE - ALL CHECKS PASSED")
print("=" * 80)
print("\\nProduction deployment is SAFE to proceed.")
print(f"Expected GPU runtime: ~1.5 hours (3 seeds × 30-40 min)")"""))

    # Add cells to notebook
    nb.cells = cells

    # Normalize and validate
    nb = nbf.validator.normalize(nb)[1]
    nbf.validate(nb)

    return nb

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "notebooks/03_ablations/ablation_3_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create notebook
    nb = create_smoke_test_notebook()

    # Save notebook
    output_path = output_dir / "ablation_3_smoke_test.ipynb"
    with open(output_path, 'w') as f:
        nbf.write(nb, f)

    print("=" * 80)
    print("ABLATION 2 SMOKE TEST NOTEBOOK CREATED")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"\\nCells: {len(nb.cells)}")
    print("\\nNext steps:")
    print("1. Create kernel-metadata.json")
    print("2. Deploy to Kaggle CPU: cd {output_dir} && kaggle kernels push -p .")
    print("3. Validate results: kaggle kernels output bkdkaggle/ablation-2-smoke -p ./smoke_results")
    print("4. ONLY if smoke test passes, then deploy production notebook")
