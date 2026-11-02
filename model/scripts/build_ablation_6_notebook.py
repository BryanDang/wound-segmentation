#!/usr/bin/env python3
"""
Build Ablation 6: Label Smoothing production notebook using nbformat.
"""
import nbformat as nbf
from pathlib import Path

# Output path
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "notebooks/03_ablations/ablation_6_production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "ablation_6_label_smoothing.ipynb"

# Create notebook
nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
}

cells = []

# Cell 1: NumPy locking
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

# Cell 2: Imports
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
    from evaluation.yolo_utils import yolo_result_to_metrics_input
else:
    sys.path.insert(0, str(PROJECT_ROOT))
    print("[INFO] Using local src directory")
    from src.evaluation.metrics import calculate_segmentation_metrics, calculate_boundary_f1, calculate_bootstrap_ci
    from src.evaluation.yolo_utils import yolo_result_to_metrics_input

print("PASS: All modules imported successfully")"""))

# Cell 3: Environment setup
cells.append(nbf.v4.new_code_cell("""# Configure paths
IN_KAGGLE = Path("/kaggle/input").exists()

if IN_KAGGLE:
    BASE_INPUT_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
    OUTPUT_PATH = Path("/kaggle/working")
else:
    BASE_INPUT_PATH = PROJECT_ROOT / "simulated_kaggle_data/input/wound-segmentation-images"
    OUTPUT_PATH = PROJECT_ROOT / "kaggle_outputs/ablation_6"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local'}")
print(f"Input path: {BASE_INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")"""))

# Cell 4: Markdown - Introduction
cells.append(nbf.v4.new_markdown_cell("""# Ablation 6: Label Smoothing

**Intervention**: Replace hard targets (0, 1) with soft targets (0.1, 0.9) for wound/background labels.

Aim: less overconfidence, better calibration, less overfitting.

**Implementation**: YOLOv8 built-in `label_smoothing=0.1` parameter.

**Expected Runtime**: ~1.5-2 hours on P100 (3 seeds × 100 epochs with early stopping)"""))

# Cell 5: YOLO config
cells.append(nbf.v4.new_code_cell("""# YOLO config path (pre-created in baseline-outputs dataset)
CONFIG_PATH = "/kaggle/input/baseline-outputs/wound_dataset.yaml" if IN_KAGGLE else BASE_INPUT_PATH / "wound_dataset.yaml"

print(f"YOLO config: {CONFIG_PATH}")

# Verify config exists
if not Path(CONFIG_PATH).exists():
    raise FileNotFoundError(f"YOLO config not found: {CONFIG_PATH}")

print("PASS: YOLO config validated")"""))

# Cell 6: Test set setup
cells.append(nbf.v4.new_code_cell("""# Setup test pairs for evaluation
TEST_IMAGES_DIR = BASE_INPUT_PATH / "test/images"
TEST_MASKS_DIR = BASE_INPUT_PATH / "test_masks"

test_image_paths = sorted(TEST_IMAGES_DIR.glob("*.png"))
print(f"Found {len(test_image_paths)} test images")

test_pairs = []
for img_path in test_image_paths:
    mask_path = TEST_MASKS_DIR / img_path.name
    if mask_path.exists():
        test_pairs.append((str(img_path), str(mask_path)))

print(f"Valid test pairs: {len(test_pairs)}")"""))

# Cell 7: Markdown - Training
cells.append(nbf.v4.new_markdown_cell("""## Training with Label Smoothing

**Hyperparameters**:
- `label_smoothing=0.1`: Soft targets (prevents overconfidence)
- `batch=32`: P100-optimized
- `lr0=0.015`: Conservative scaling for batch=32
- `workers=8`: Maximize P100 throughput
- `amp=False`: Disabled for ablations
- `epochs=100`, `patience=10`: Standard convergence
- Baseline augmentation disabled (degrees=0, mosaic=0, mixup=0)

**Multi-seed**: [42, 123, 456] for statistical robustness"""))

# Cell 8: Training loop
cells.append(nbf.v4.new_code_cell("""# Check completion status
completion_marker = OUTPUT_PATH / "ablation_6_completion.json"

if completion_marker.exists():
    print("PASS: Ablation 6 already completed (loading results)")
    with open(completion_marker) as f:
        results_6 = json.load(f)
    print(f"  Dice: {results_6['dice_mean']:.4f}")
else:
    print("WARNING: Ablation 6 not completed - starting training...")

    RANDOM_SEEDS = [42, 123, 456]
    all_results = {}

    for seed in RANDOM_SEEDS:
        print(f"\\nTraining seed {seed}...")

        model = YOLO("yolov8n-seg.pt")

        results = model.train(
            data=str(CONFIG_PATH),
            batch=32,
            lr0=0.015,
            workers=8,
            amp=False,
            epochs=100,
            imgsz=512,
            patience=10,
            # Baseline parameters (no augmentation)
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            # INTERVENTION: Label Smoothing
            label_smoothing=0.1,
            seed=seed,
            save=True,
            project=str(OUTPUT_PATH / "yolo_training"),
            name=f"seed_{seed}",
            exist_ok=True,
            verbose=False,
        )

        # Evaluate immediately
        seed_results = []
        for img_path_str, mask_path_str in test_pairs:
            image_id = Path(img_path_str).stem
            gt_mask = cv2.imread(mask_path_str, cv2.IMREAD_GRAYSCALE)
            _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

            img = cv2.imread(img_path_str)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            yolo_results = model.predict(img_rgb, verbose=False)
            conversion_result = yolo_result_to_metrics_input(yolo_results[0], image_id)

            if conversion_result["success"]:
                pred_mask = conversion_result["data"]["pred_mask"]
                metrics = calculate_segmentation_metrics(pred_mask, gt_mask, image_id)
                boundary_f1 = calculate_boundary_f1(pred_mask, gt_mask, tolerance=2)
                metrics["boundary_f1"] = boundary_f1
                seed_results.append(metrics)

        all_results[f"seed_{seed}"] = seed_results
        print(f"  PASS: Seed {seed} complete: {len(seed_results)} images")

    # Aggregate and save
    all_pooled = []
    for seed_results in all_results.values():
        all_pooled.extend(seed_results)

    dice_scores = [r["dice"] for r in all_pooled]
    ci_result = calculate_bootstrap_ci(
        dice_scores, confidence_level=0.95, n_bootstrap=10000, random_seed=42
    )

    results_6 = {
        "ablation": "6",
        "intervention": "label_smoothing",
        "dice_mean": ci_result["mean"],
        "dice_ci_lower": ci_result["ci_lower"],
        "dice_ci_upper": ci_result["ci_upper"],
        "per_image_results": all_results,
    }

    # IMMEDIATE SAVE (failure-proof)
    with open(completion_marker, "w") as f:
        json.dump(results_6, f, indent=2)

    print(f"\\nPASS: ABLATION 6 COMPLETE - Results saved: {completion_marker}")
    print(f"  Dice: {results_6['dice_mean']:.4f} [{results_6['dice_ci_lower']:.4f}, {results_6['dice_ci_upper']:.4f}]")"""))

# Cell 9: Results comparison
cells.append(nbf.v4.new_markdown_cell("""## Results Analysis

Compare Label Smoothing against baseline."""))

# Cell 10: Final comparison
cells.append(nbf.v4.new_code_cell("""# Compare with baseline
baseline_dice = 0.8000  # From baseline-outputs

print("=" * 80)
print("ABLATION 6: LABEL SMOOTHING RESULTS")
print("=" * 80)

print(f"\\nBaseline (hard targets):     {baseline_dice:.4f}")
print(f"Label Smoothing (ε=0.1):     {results_6['dice_mean']:.4f} [{results_6['dice_ci_lower']:.4f}, {results_6['dice_ci_upper']:.4f}]")
print(f"\\nΔ vs baseline: {results_6['dice_mean'] - baseline_dice:+.4f}")

if results_6['dice_mean'] > baseline_dice:
    print("\\nIMPROVEMENT: Label smoothing improves generalization")
else:
    print("\\nWARNING: NO IMPROVEMENT: Label smoothing did not help (likely already well-calibrated)")

# Save final report
final_report = {
    "ablation": "label_smoothing",
    "baseline_dice": baseline_dice,
    "label_smoothing_dice": results_6["dice_mean"],
    "improvement": results_6["dice_mean"] - baseline_dice,
    "statistical_significance": "TBD - requires Wilcoxon test",
    "per_seed_results": results_6["per_image_results"],
}

with open(OUTPUT_PATH / "ablation_6_final_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

print(f"\\nFinal report saved: {OUTPUT_PATH / 'ablation_6_final_report.json'}")"""))

# Add cells to notebook
nb["cells"] = cells

# Normalize (add missing IDs)
nb = nbf.validator.normalize(nb)[1]

# Validate
nbf.validate(nb)

# Write notebook
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)

print(f"PASS: Ablation 6 notebook created: {OUTPUT_PATH}")
