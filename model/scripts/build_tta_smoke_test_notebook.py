"""
Build TTA Smoke Test Notebook for Kaggle deployment.

Tests Test-Time Augmentation implementation on 5 test images (2-3 min runtime).
"""

import nbformat as nbf
from pathlib import Path

# Notebook metadata
NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0"
    }
}

# Kernel metadata for Kaggle
KERNEL_METADATA = {
    "id": "bkdkaggle/tta-smoke-test",
    "title": "TTA Smoke Test - Ablation 5",
    "code_file": "05_tta_smoke_test.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "false",
    "dataset_sources": [
        "bkdkaggle/baseline-outputs",
        "bkdkaggle/wound-seg-code-v1"
    ],
    "kernel_sources": []
}

# Cell content
CELLS = []

# Cell 0: NumPy version locking
CELLS.append({
    "type": "code",
    "source": """# Lock NumPy to 1.x (prevent ultralytics from upgrading to 2.x)
import subprocess
import numpy as np

print(f"Default NumPy version: {np.__version__}")
default_numpy = np.__version__

# Install ultralytics (will try to upgrade NumPy)
print("\\n" + "="*60)
print("Installing ultralytics...")
print("="*60)
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)

# Force downgrade back to 1.x
print("\\n" + "="*60)
print(f"Forcing NumPy back to {default_numpy}...")
print("="*60)
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall", f"numpy=={default_numpy}"],
    check=True
)

import numpy as np
print(f"\\nFinal NumPy version: {np.__version__}")
assert np.__version__.startswith("1."), f"NumPy 2.x detected: {np.__version__}"
"""
})

# Cell 1: Imports
CELLS.append({
    "type": "code",
    "source": """import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Add code dataset to path
sys.path.append('/kaggle/input/wound-seg-code-v1')

# Import TTA and evaluation utilities
from src.evaluation.tta import TestTimeAugmentation, validate_augmentation_inverses
from src.evaluation.metrics import calculate_segmentation_metrics

print("All imports successful")
"""
})

# Cell 2: Paths
CELLS.append({
    "type": "code",
    "source": """# Dataset paths
BASE_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
TEST_IMAGES_DIR = BASE_PATH / "test_images"
TEST_MASKS_DIR = BASE_PATH / "test_masks"

# Output path
OUTPUT_PATH = Path("/kaggle/working")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Model path (use seed 42 baseline)
MODEL_PATH = "/kaggle/input/baseline-outputs/weights/seed_42/best.pt"

print(f"Test images: {TEST_IMAGES_DIR}")
print(f"Test masks: {TEST_MASKS_DIR}")
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_PATH}")
"""
})

# Cell 3: Validate augmentation transforms
CELLS.append({
    "type": "markdown",
    "source": """## Validation Check 1: Augmentation Transforms

Test that all forward-inverse transforms are perfect inverses (error < 1e-6).
"""
})

CELLS.append({
    "type": "code",
    "source": """print("Validating TTA augmentation transforms...")
print("="*60)

errors = validate_augmentation_inverses(verbose=True)

# Check all errors < 1e-6
all_valid = all(err < 1e-6 for err in errors.values())
assert all_valid, f"Some augmentations have errors >= 1e-6: {errors}"

print("\\n" + "="*60)
print(f"VALIDATION PASSED: All {len(errors)} augmentations are perfect inverses")
print("="*60)
"""
})

# Cell 4: Load baseline model
CELLS.append({
    "type": "markdown",
    "source": """## Load Baseline Model

Load the trained baseline model (seed 42) for TTA testing.
"""
})

CELLS.append({
    "type": "code",
    "source": """print("Loading baseline model...")
model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")
"""
})

# Cell 5: Initialize TTA
CELLS.append({
    "type": "markdown",
    "source": """## Initialize TTA

Use 4 augmentations for smoke test (faster than full 8):
- original
- horizontal_flip
- vertical_flip
- rotate_90
"""
})

CELLS.append({
    "type": "code",
    "source": """# Initialize TTA with subset of augmentations (faster smoke test)
tta = TestTimeAugmentation(
    model=model,
    augmentation_set=["original", "horizontal_flip", "vertical_flip", "rotate_90"],
    aggregation="mean"
)

config = tta.get_config()
print(f"TTA Configuration:")
print(f"  Augmentations: {config['n_augmentations']}")
print(f"  Set: {config['augmentation_set']}")
print(f"  Aggregation: {config['aggregation']}")
"""
})

# Cell 6: Test on 5 images
CELLS.append({
    "type": "markdown",
    "source": """## Validation Check 2: Baseline vs TTA Predictions

Compare baseline (single forward pass) vs TTA (4 augmentations) on 5 test images.

**Expected**:
- TTA Dice ≥ Baseline Dice (no degradation)
- TTA time ≈ 4× baseline time
"""
})

CELLS.append({
    "type": "code",
    "source": """# Select 5 test images
test_images = sorted(list(TEST_IMAGES_DIR.glob("*.png")))[:5]
test_masks = [TEST_MASKS_DIR / img.name for img in test_images]

print(f"Testing on {len(test_images)} images:")
for img in test_images:
    print(f"  {img.name}")

baseline_results = []
tta_results = []
baseline_times = []
tta_times = []

print("\\n" + "="*60)
print("Running predictions...")
print("="*60)

for img_path, mask_path in zip(test_images, test_masks):
    print(f"\\nProcessing: {img_path.name}")

    # Baseline prediction
    start = time.time()
    baseline_result = model.predict(str(img_path), verbose=False)[0]
    baseline_time = time.time() - start
    baseline_times.append(baseline_time)

    # Extract baseline mask
    if baseline_result.masks is not None and len(baseline_result.masks.data) > 0:
        baseline_mask = baseline_result.masks.data[0].cpu().numpy()
        baseline_mask = cv2.resize(baseline_mask, (512, 512), interpolation=cv2.INTER_LINEAR)
        baseline_mask = (baseline_mask > 0.5).astype(np.uint8)
    else:
        baseline_mask = np.zeros((512, 512), dtype=np.uint8)

    # TTA prediction
    start = time.time()
    tta_mask = tta.predict(str(img_path))
    tta_time = time.time() - start
    tta_times.append(tta_time)

    # Load ground truth
    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"  WARNING: Warning: Could not load mask {mask_path.name}")
        continue

    # Calculate metrics
    baseline_metrics = calculate_segmentation_metrics(baseline_mask, gt_mask, img_path.stem)
    tta_metrics = calculate_segmentation_metrics(tta_mask, gt_mask, img_path.stem)

    baseline_results.append(baseline_metrics)
    tta_results.append(tta_metrics)

    print(f"  Baseline: Dice={baseline_metrics['dice']:.4f}, Time={baseline_time:.3f}s")
    print(f"  TTA:      Dice={tta_metrics['dice']:.4f}, Time={tta_time:.3f}s")
    print(f"  Improvement: {(tta_metrics['dice'] - baseline_metrics['dice']):.4f}")
"""
})

# Cell 7: Aggregate results
CELLS.append({
    "type": "markdown",
    "source": """## Results Summary

Aggregate metrics across all test images.
"""
})

CELLS.append({
    "type": "code",
    "source": """import pandas as pd

baseline_df = pd.DataFrame(baseline_results)
tta_df = pd.DataFrame(tta_results)

print("\\n" + "="*60)
print("BASELINE VS TTA COMPARISON (5 images)")
print("="*60)

metrics = ['dice', 'iou', 'precision', 'recall', 'boundary_f1']
comparison = {}

for metric in metrics:
    baseline_mean = baseline_df[metric].mean()
    tta_mean = tta_df[metric].mean()
    improvement = tta_mean - baseline_mean

    comparison[metric] = {
        'baseline': baseline_mean,
        'tta': tta_mean,
        'improvement': improvement
    }

    print(f"\\n{metric.upper():15s}:")
    print(f"  Baseline: {baseline_mean:.4f}")
    print(f"  TTA:      {tta_mean:.4f}")
    print(f"  Δ:        {improvement:+.4f}")

print(f"\\nINFERENCE TIME:")
print(f"  Baseline: {np.mean(baseline_times):.3f}s per image")
print(f"  TTA:      {np.mean(tta_times):.3f}s per image")
print(f"  Multiplier: {np.mean(tta_times)/np.mean(baseline_times):.1f}×")
"""
})

# Cell 8: Validation checks
CELLS.append({
    "type": "markdown",
    "source": """## Validation Check 3: Smoke Test Pass/Fail

Final validation that TTA implementation works correctly.
"""
})

CELLS.append({
    "type": "code",
    "source": """# Validation checks
tta_dice_maintained = tta_df['dice'].mean() >= baseline_df['dice'].mean() - 0.01
inference_time_reasonable = np.mean(tta_times) < 1.0  # <1s per image on P100

smoke_test_report = {
    "validation_checks": {
        "augmentation_transforms": {"passed": all_valid},
        "tta_predictions": {"passed": True},
        "tta_dice_maintained": {"passed": tta_dice_maintained},
        "inference_time_reasonable": {"passed": inference_time_reasonable},
    },
    "metrics": {
        "baseline_dice": float(baseline_df['dice'].mean()),
        "tta_dice": float(tta_df['dice'].mean()),
        "dice_improvement": float(tta_df['dice'].mean() - baseline_df['dice'].mean()),
        "inference_time_multiplier": float(np.mean(tta_times) / np.mean(baseline_times)),
    },
    "configuration": tta.get_config(),
    "smoke_test_status": "PASSED" if all([
        all_valid,
        tta_dice_maintained,
        inference_time_reasonable
    ]) else "FAILED"
}

# Save report
with open(OUTPUT_PATH / 'smoke_test_report.json', 'w') as f:
    json.dump(smoke_test_report, f, indent=2)

print("\\n" + "="*60)
print("SMOKE TEST REPORT")
print("="*60)
print(json.dumps(smoke_test_report, indent=2))

# Assert all checks passed
assert smoke_test_report["smoke_test_status"] == "PASSED", "Smoke test FAILED"

print("\\n" + "="*60)
print("SMOKE TEST PASSED - Ready for production deployment")
print("="*60)
"""
})


def create_notebook():
    """Create the notebook with all cells."""
    nb = nbf.v4.new_notebook()
    nb.metadata = NOTEBOOK_METADATA

    # Add all cells
    for cell_spec in CELLS:
        if cell_spec["type"] == "code":
            nb.cells.append(nbf.v4.new_code_cell(cell_spec["source"]))
        elif cell_spec["type"] == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(cell_spec["source"]))

    # Normalize (adds missing cell IDs)
    nb = nbf.validator.normalize(nb)[1]

    return nb


def main():
    """Build and save the notebook."""
    # Create notebook
    nb = create_notebook()

    # Validate
    nbf.validate(nb)
    print("Notebook validation passed")

    # Save notebook
    output_dir = Path(__file__).parent.parent / "notebooks" / "05_ablation_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook_path = output_dir / "05_tta_smoke_test.ipynb"
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

    print(f"Notebook saved: {notebook_path}")

    # Save kernel metadata
    import json
    metadata_path = output_dir / "kernel-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(KERNEL_METADATA, f, indent=2)

    print(f"Metadata saved: {metadata_path}")

    print(f"\\nReady to deploy:")
    print(f"   cd {output_dir}")
    print(f"   kaggle kernels push -p .")


if __name__ == "__main__":
    main()
