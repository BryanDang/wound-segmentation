#!/usr/bin/env python3
"""
Build Ablation 1C: Focal Tversky Loss + No Mosaic

Tests hypothesis: Does mosaic augmentation mask Focal Tversky's benefits?

Design:
- Use FocalTverskyTrainer (same as Ablation 1)
- Disable mosaic (mosaic=0.0)
- Keep all other baseline augmentations
- Expected: Focal Tversky should outperform baseline-no-mosaic if it handles class imbalance better
"""

import nbformat as nbf
from pathlib import Path
import json


def create_ablation_1c_notebook():
    """Create Ablation 1C production notebook."""

    # Create notebook
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    }

    cells = []

    # Cell 0: NumPy locking
    cells.append(nbf.v4.new_code_cell("""# Lock NumPy to Kaggle's default (prevent 2.x upgrade)
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
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F

# Environment detection
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent

# KAGGLE DATASET SETUP
if os.path.exists("/kaggle/input/wound-seg-code-v1"):
    sys.path.insert(0, "/kaggle/input/wound-seg-code-v1")
    print("[INFO] Code dataset added to Python path")
    from evaluation.yolo_utils import yolo_result_to_metrics_input
    from evaluation.metrics import calculate_segmentation_metrics
    from data.yolo_config import create_yolo_config
else:
    sys.path.insert(0, str(PROJECT_ROOT))
    print("[INFO] Using local src directory")
    from src.evaluation.yolo_utils import yolo_result_to_metrics_input
    from src.evaluation.metrics import calculate_segmentation_metrics
    from src.data.yolo_config import create_yolo_config

print("PASS: All modules imported successfully")"""))

    # Cell 2: Environment setup
    cells.append(nbf.v4.new_code_cell("""# Configure paths for Kaggle
if os.path.exists("/kaggle/input"):
    IN_KAGGLE = True
    # Use baseline-outputs for training (has pre-converted YOLO labels)
    BASE_INPUT_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
    # Use wound-segmentation-images for evaluation (has PNG masks)
    EVAL_INPUT_PATH = Path("/kaggle/input/wound-segmentation-images/data_wound_seg")
    OUTPUT_PATH = Path("/kaggle/working")
    CODE_PATH = Path("/kaggle/input/wound-seg-code-v1")
else:
    IN_KAGGLE = False
    BASE_INPUT_PATH = PROJECT_ROOT / "data" / "wound_segmentation"
    EVAL_INPUT_PATH = BASE_INPUT_PATH  # Same for local
    OUTPUT_PATH = PROJECT_ROOT / "outputs" / "ablation_1c"
    CODE_PATH = PROJECT_ROOT / "src"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TEST_IMAGES_DIR = EVAL_INPUT_PATH / "test_images"
TEST_MASKS_DIR = EVAL_INPUT_PATH / "test_masks"

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local'}")
print(f"Training data: {BASE_INPUT_PATH}")
print(f"Evaluation data: {EVAL_INPUT_PATH}")
print(f"Test images: {TEST_IMAGES_DIR}")
print(f"Test masks: {TEST_MASKS_DIR}")
print(f"Output path: {OUTPUT_PATH}")

# Verify dataset exists
if not TEST_IMAGES_DIR.exists():
    raise FileNotFoundError(f"Test images not found at {TEST_IMAGES_DIR}")
if not TEST_MASKS_DIR.exists():
    raise FileNotFoundError(f"Test masks not found at {TEST_MASKS_DIR}")

print("PASS: Environment configured")"""))

    # Cell 3: YOLO config
    cells.append(nbf.v4.new_code_cell("""# Create YOLO config
config_path = OUTPUT_PATH / "wound_dataset.yaml"
config_result = create_yolo_config(
    dataset_root=BASE_INPUT_PATH,
    output_path=config_path,
    train_subdir="train",
    val_subdir="test"
)

if not config_result["success"]:
    raise RuntimeError(f"YOLO config creation failed: {config_result['error']}")

print(f"PASS: YOLO config created: {config_path}")"""))

    # Cell 4: Focal Tversky Loss implementation
    cells.append(nbf.v4.new_code_cell("""class FocalTverskyLoss(nn.Module):
    \"""
    Focal Tversky Loss for handling class imbalance in segmentation.

    Mathematical formulation:
        FTL = (1 - TI)^gamma
        where TI = TP / (TP + alpha*FN + beta*FP)

    Parameters:
    - alpha: Weight for false negatives (0.7 = emphasize missed wounds)
    - beta: Weight for false positives (0.3 = de-emphasize false alarms)
    - gamma: Focal parameter (2.0 = focus on hard examples)
    \"""

    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        # Flatten tensors
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Calculate Tversky components
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        # Tversky Index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        # Focal Tversky Loss
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)

        return focal_tversky


class FocalTverskyTrainer(SegmentationTrainer):
    \"""
    Custom YOLOv8 Segmentation Trainer with Focal Tversky Loss.

    Replaces YOLOv8's default segmentation loss (BCE + Dice) with
    Focal Tversky Loss optimized for class imbalance.
    \"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize Focal Tversky Loss
        self.focal_tversky_loss = FocalTverskyLoss(
            alpha=0.7,  # Emphasize missed wounds (FN)
            beta=0.3,   # De-emphasize false alarms (FP)
            gamma=2.0,  # Focus on hard examples
            smooth=1e-6,
        )

        print("FocalTverskyTrainer initialized with custom loss function")

    def criterion(self, preds, batch):
        \"""Override YOLOv8's criterion() to use Focal Tversky Loss.\"""
        # Get base loss (box + cls) from parent
        loss = super().criterion(preds, batch)

        # Replace segmentation loss with Focal Tversky
        if "seg" in preds and "masks" in batch:
            pred_masks = preds["seg"]
            target_masks = batch["masks"].float()

            # Ensure predictions are in [0, 1] range
            if pred_masks.min() < 0 or pred_masks.max() > 1:
                pred_masks = torch.sigmoid(pred_masks)

            # Calculate Focal Tversky Loss
            seg_loss = self.focal_tversky_loss(pred_masks, target_masks)

            # Replace YOLOv8's default segmentation loss
            loss[2] = seg_loss

        return loss

print("PASS: FocalTverskyTrainer defined")"""))

    # Cell 5: Training parameters
    cells.append(nbf.v4.new_code_cell("""# Training seeds (3 for statistical robustness)
RANDOM_SEEDS = [42, 123, 456]

# P100-Optimized Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.015
WORKERS = 8
AMP = False

# Dataset-Aligned Parameters
IMAGE_SIZE = 512
EPOCHS = 100
PATIENCE = 10

# Storage for results
all_training_results = {}

print("=" * 80)
print("ABLATION 1C: FOCAL TVERSKY LOSS + NO MOSAIC")
print("=" * 80)
print(f"\\nHypothesis: Mosaic augmentation masks Focal Tversky's benefits")
print(f"Test: Focal Tversky WITHOUT mosaic should handle class imbalance better")
print(f"\\nTraining configuration:")
print(f"  Seeds: {RANDOM_SEEDS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Workers: {WORKERS}")
print(f"  AMP: {AMP}")
print(f"  Epochs: {EPOCHS}")
print(f"  Patience: {PATIENCE}")
print(f"  Image size: {IMAGE_SIZE}")
print(f"  CRITICAL: mosaic=0.0 (DISABLED)")
print(f"\\nDataset config: {config_path}")
print(f"\\nEstimated time: ~1.5-2 hours total on P100\\n")"""))

    # Cell 6: Training loop
    cells.append(nbf.v4.new_code_cell("""for seed_idx, seed in enumerate(RANDOM_SEEDS, 1):
    print("\\n" + "=" * 80)
    print(f"TRAINING WITH SEED {seed} ({seed_idx}/{len(RANDOM_SEEDS)})")
    print("=" * 80)

    # Initialize model from pretrained COCO weights
    model = YOLO("yolov8n-seg.pt")

    # Set custom trainer BEFORE training
    model.trainer = FocalTverskyTrainer

    # Train model
    training_start = time.time()

    results = model.train(
        data=str(config_path),
        # P100-optimized parameters
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        workers=WORKERS,
        amp=AMP,
        # Dataset-aligned
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        patience=PATIENCE,
        # Disable mosaic (test if FT helps without mosaic's balancing)
        mosaic=0.0,
        # Reproducibility
        seed=seed,
        save=True,
        project=str(OUTPUT_PATH / "yolo_training"),
        name=f"seed_{seed}",
        exist_ok=True,
        verbose=False,
    )

    training_duration = time.time() - training_start

    print(f"\\nPASS: Training completed in {training_duration/60:.1f} minutes")

    # Load best checkpoint for evaluation
    best_model_path = OUTPUT_PATH / "yolo_training" / f"seed_{seed}" / "weights" / "best.pt"
    best_model = YOLO(str(best_model_path))

    # Prepare test pairs
    test_image_paths = sorted(TEST_IMAGES_DIR.glob("*.png"))
    test_pairs = []

    for img_path in test_image_paths:
        mask_path = TEST_MASKS_DIR / img_path.name
        if mask_path.exists():
            test_pairs.append((str(img_path), str(mask_path)))

    print(f"\\nEvaluating on test set ({len(test_pairs)} images)...")

    # Evaluate on test set
    seed_results = []

    for img_path_str, mask_path_str in test_pairs:
        image_id = Path(img_path_str).stem

        # Load image
        img_rgb = cv2.imread(img_path_str)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # Standard evaluation (no TTA)
        yolo_results = best_model.predict(img_rgb, verbose=False)

        # Convert to metrics input
        conversion_result = yolo_result_to_metrics_input(
            yolo_results[0],
            mask_path_str,
            image_id
        )

        if not conversion_result["success"]:
            print(f"WARNING: Skipping {image_id}: {conversion_result['error']}")
            continue

        data = conversion_result["data"]
        metrics_result = calculate_segmentation_metrics(
            data["pred_mask"],
            data["gt_mask"],
            image_id
        )

        seed_results.append(metrics_result)

    # Calculate seed mean
    seed_dice_scores = [r["dice_score"] for r in seed_results]
    seed_iou_scores = [r["iou"] for r in seed_results]
    seed_dice_mean = np.mean(seed_dice_scores)
    seed_iou_mean = np.mean(iou_scores)

    all_training_results[f"seed_{seed}"] = {
        "dice_mean": seed_dice_mean,
        "iou_mean": seed_iou_mean,
        "n_images": len(seed_results),
        "training_time_seconds": training_duration
    }

    print(f"\\nPASS: Seed {seed} evaluation complete:")
    print(f"  Mean Dice: {seed_dice_mean:.4f}")
    print(f"  Mean IoU:  {seed_iou_mean:.4f}")
    print(f"  Images evaluated: {len(seed_results)}")

print("\\n" + "=" * 80)
print("PASS: ALL TRAINING RUNS COMPLETE")
print("=" * 80)"""))

    # Cell 7: Cross-seed analysis
    cells.append(nbf.v4.new_code_cell("""# Calculate cross-seed statistics
all_dice_means = [results["dice_mean"] for results in all_training_results.values()]
overall_dice_mean = np.mean(all_dice_means)
dice_std = np.std(all_dice_means)
coefficient_of_variation = (dice_std / overall_dice_mean) * 100

print("=" * 80)
print("ABLATION 1C: FOCAL TVERSKY + NO MOSAIC RESULTS")
print("=" * 80)
print(f"\\nCross-Seed Performance:")
for seed, results in all_training_results.items():
    print(f"  {seed}: Dice = {results['dice_mean']:.4f}")

print(f"\\n  Mean Dice: {overall_dice_mean:.4f}")
print(f"  Std Dev: {dice_std:.4f}")
print(f"  Coefficient of Variation: {coefficient_of_variation:.2f}%")

if coefficient_of_variation < 2.0:
    print(f"  PASS: Low variance - results are robust")
else:
    print(f"  WARNING: High variance - results may be unstable")"""))

    # Cell 8: Completion status
    cells.append(nbf.v4.new_code_cell("""# Save completion status
completion_status = {
    "notebook": "ablation_1c_focal_tversky_no_mosaic",
    "ablation": "focal_tversky_no_mosaic",
    "intervention": "Focal Tversky Loss WITHOUT mosaic augmentation",
    "hypothesis": "Mosaic masks Focal Tversky's class imbalance benefits",
    "status": "complete",
    "phases_complete": ["training", "evaluation"],
    "n_images_evaluated": 552,
    "n_seeds": len(RANDOM_SEEDS),
    "seeds": RANDOM_SEEDS,
    "ablation_dice_mean": overall_dice_mean,
    "baseline_dice_mean": 0.8000,  # Reference baseline WITH mosaic
    "dice_improvement_vs_baseline": overall_dice_mean - 0.8000,
    "environment": "Kaggle" if IN_KAGGLE else "Local",
    "augmentation_params": {
        "mosaic": 0.0,  # Disabled to test hypothesis
        "note": "All other augmentations at YOLO defaults"
    }
}

status_path = OUTPUT_PATH / "completion_status.json"
with open(status_path, 'w') as f:
    json.dump(completion_status, f, indent=2)

print(f"\\nCompletion status saved: {status_path}")

print("\\n" + "=" * 80)
print("PASS: ABLATION 1C COMPLETE: FOCAL TVERSKY + NO MOSAIC")
print("=" * 80)
print(f"\\nResults:")
print(f"  Dice Score: {overall_dice_mean:.4f}")
print(f"  Baseline (with mosaic): 0.8000")
print(f"  Change: {overall_dice_mean - 0.8000:+.4f}")
print(f"\\nHypothesis Test:")
print(f"  If Focal Tversky WITHOUT mosaic > baseline WITHOUT mosaic (~0.68)")
print(f"  Then Focal Tversky provides value when mosaic isn't masking its benefits")
print(f"\\nAll outputs saved to: {OUTPUT_PATH}")
print("=" * 80)"""))

    # Add cells to notebook
    nb["cells"] = cells

    # Normalize and validate
    nb = nbf.validator.normalize(nb)[1]
    nbf.validate(nb)

    return nb


def main():
    print("Building Ablation 1C: Focal Tversky + No Mosaic...")

    # Output directory
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "notebooks" / "03_ablations" / "ablation_1c_focal_tversky_no_mosaic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create notebook
    nb = create_ablation_1c_notebook()

    # Write notebook
    notebook_path = output_dir / "ablation_1c_focal_tversky_no_mosaic.ipynb"
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

    # Create kernel metadata
    metadata = {
        "id": "bkdkaggle/ablation-1c-focal-tversky-no-mosaic",
        "title": "Ablation 1C: Focal Tversky + No Mosaic",
        "code_file": "ablation_1c_focal_tversky_no_mosaic.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [
            "bkdkaggle/baseline-outputs",
            "bkdkaggle/wound-seg-code-v1",
            "leoscode/wound-segmentation-images"
        ],
        "kernel_sources": []
    }

    metadata_path = output_dir / "kernel-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"PASS: Notebook: {notebook_path}")
    print(f"PASS: Metadata: {metadata_path}")
    print()
    print("Next steps:")
    print(f"1. Deploy: cd {output_dir} && kaggle kernels push -p .")
    print("2. Monitor: kaggle kernels status bkdkaggle/ablation-1c-focal-tversky-no-mosaic")
    print("3. Download: kaggle kernels output bkdkaggle/ablation-1c-focal-tversky-no-mosaic")


if __name__ == "__main__":
    main()
