"""
Build Ablation 4c notebook: Hard Negatives Mining (Early Checkpoint Fine-tuning).

Strategy: Load baseline epoch40.pt (seed 42), fine-tune on 513 DDI images for 25 epochs.
"""

import json
import nbformat as nbf
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "03_ablations" / "ablation_4c_early_checkpoint"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create notebook
nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
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

cells = []

# Cell 1: NumPy Locking
cells.append(nbf.v4.new_code_cell("""# Fix: Kaggle's default NumPy 1.26.4 is CORRECT - prevent upgrades to 2.x
# Problem: Ultralytics pulls in NumPy 2.x, but Kaggle's OpenCV 4.12.0 was compiled with NumPy 1.x
# Solution: Install ultralytics, then LOCK NumPy at 1.26.4
import sys

print("Step 1: Record Kaggle's default NumPy version...")
import numpy as np

default_numpy = np.__version__
print(f"   Default NumPy: {default_numpy}")

print("\\nStep 2: Install ultralytics...")
import subprocess

subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)

print("\\nStep 3: LOCK NumPy back to Kaggle's default (prevent 2.x upgrade)...")
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall", f"numpy=={default_numpy}"],
    check=True,
)

print("\\nInstallation complete!")
print("\\nVerifying versions...")
import numpy as np

print(f"   NumPy: {np.__version__}")

from ultralytics import YOLO

print("   PASS: Ultralytics imported successfully")"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Environment detection
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent

# KAGGLE DATASET SETUP: Add code dataset directly to path
if os.path.exists("/kaggle/input/wound-seg-code-v1"):
    sys.path.insert(0, "/kaggle/input/wound-seg-code-v1")
    print("[INFO] Code dataset added to Python path")
    from evaluation.metrics import (
        calculate_bootstrap_ci,
        calculate_boundary_f1,
        calculate_segmentation_metrics,
    )
    from evaluation.yolo_utils import yolo_result_to_metrics_input
    from data.yolo_config import create_yolo_config, validate_yolo_config
else:
    sys.path.insert(0, str(PROJECT_ROOT))
    print("[INFO] Using local src directory")
    from src.evaluation.metrics import (
        calculate_bootstrap_ci,
        calculate_boundary_f1,
        calculate_segmentation_metrics,
    )
    from src.evaluation.yolo_utils import yolo_result_to_metrics_input
    from src.data.yolo_config import create_yolo_config, validate_yolo_config

print("PASS: All modules imported successfully")"""))

# Cell 3: Environment Setup
cells.append(nbf.v4.new_code_cell("""# Detect environment (Kaggle vs local)
IN_KAGGLE = Path("/kaggle/input").exists()

if IN_KAGGLE:
    BASE_INPUT_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
    DDI_IMAGES_DIR = Path("/kaggle/input/ddi-diverse-dermatology/PRIVATE - ddidiversedermatologyimages")
    BASELINE_WEIGHTS_PATH = Path("/kaggle/input/baseline-outputs/yolo_training/seed_42/weights/epoch40.pt")
    OUTPUT_PATH = Path("/kaggle/working")
else:
    BASE_INPUT_PATH = PROJECT_ROOT / "simulated_kaggle_data/input/wound-segmentation-images"
    DDI_IMAGES_DIR = PROJECT_ROOT / "data/ddi_diverse_dermatology"
    BASELINE_WEIGHTS_PATH = PROJECT_ROOT / "kaggle_outputs/baseline_final/yolo_training/seed_42/weights/epoch40.pt"
    OUTPUT_PATH = PROJECT_ROOT / "kaggle_outputs/ablation_4c_early_checkpoint"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local (simulated)'}")
print(f"Input path: {BASE_INPUT_PATH}")
print(f"DDI images: {DDI_IMAGES_DIR}")
print(f"Baseline weights: {BASELINE_WEIGHTS_PATH}")
print(f"Output path: {OUTPUT_PATH}")"""))

# Cell 4: DDI Dataset Preparation
cells.append(nbf.v4.new_markdown_cell("""## DDI Dataset Preparation - Hard Negative Fine-Tuning

**Strategy**: Fine-tune pre-trained baseline model (epoch40.pt) on DDI hard negatives only.

Baseline is 80% Dice with 22.8% FPR. Fine-tuning on dermatology images should mark what is not a wound without losing wound features.

**Training Set**: 513 DDI images (80% of 643) with empty labels (hard negatives).

**Hyperparameters**:
- lr0=0.0015 (10x lower than baseline for fine-tuning)
- epochs=25 (short fine-tuning phase)
- batch=32, workers=8, amp=False (P100-optimized)"""))

cells.append(nbf.v4.new_code_cell("""print("=" * 80)
print("DDI DATASET PREPARATION - FINE-TUNING STRATEGY")
print("=" * 80)

import shutil

# Load DDI split (FIXED seed=42, same as 4a)
ddi_metadata = pd.read_csv(PROJECT_ROOT / "data/ddi_cleaned/ddi_train.csv") if not IN_KAGGLE else None

if IN_KAGGLE:
    # On Kaggle, recreate split from scratch (same seed)
    from sklearn.model_selection import train_test_split
    ddi_image_paths = sorted(DDI_IMAGES_DIR.glob("*.png"))
    ddi_train_paths, _ = train_test_split(ddi_image_paths, test_size=0.2, random_state=42, shuffle=True)
else:
    # Local: use pre-generated split
    ddi_train_paths = [DDI_IMAGES_DIR / row["DDI_file"] for _, row in ddi_metadata.iterrows()]

print(f"DDI training images: {len(ddi_train_paths)}")

# Create DDI-only training dataset
DDI_DATASET_ROOT = OUTPUT_PATH / "ddi_dataset"
(DDI_DATASET_ROOT / "train/images").mkdir(parents=True, exist_ok=True)
(DDI_DATASET_ROOT / "train/labels").mkdir(parents=True, exist_ok=True)
(DDI_DATASET_ROOT / "test/images").mkdir(parents=True, exist_ok=True)
(DDI_DATASET_ROOT / "test/labels").mkdir(parents=True, exist_ok=True)

# Copy DDI training images and create empty labels
for img_path in ddi_train_paths:
    shutil.copy(img_path, DDI_DATASET_ROOT / "train/images" / img_path.name)
    label_path = DDI_DATASET_ROOT / "train/labels" / f"{img_path.stem}.txt"
    label_path.write_text("")  # Empty label = hard negative

print(f"PASS: Created DDI training dataset: {len(ddi_train_paths)} images with empty labels")

# Copy wound test set for evaluation (same as baseline)
baseline_test_imgs = BASE_INPUT_PATH / "test/images"
baseline_test_labels = BASE_INPUT_PATH / "test/labels"

wound_test_count = 0
for img_path in baseline_test_imgs.glob("*.png"):
    shutil.copy(img_path, DDI_DATASET_ROOT / "test/images" / img_path.name)
    wound_test_count += 1

for label_path in baseline_test_labels.glob("*.txt"):
    shutil.copy(label_path, DDI_DATASET_ROOT / "test/labels" / label_path.name)

print(f"PASS: Copied {wound_test_count} wound test images for evaluation")

print("=" * 80)"""))

# Cell 5: YOLO Config
cells.append(nbf.v4.new_code_cell("""# Create YOLO config for DDI fine-tuning
config_result = create_yolo_config(
    dataset_root=DDI_DATASET_ROOT,
    train_dir="train",
    val_dir="test",
    test_dir="test",
    output_path=OUTPUT_PATH / "wound_dataset_4c.yaml",
)

if config_result["success"]:
    config_path = config_result["data"]["config_path"]
    print(f"PASS: Configuration created: {config_path}")
    validation_result = validate_yolo_config(config_path)
    if validation_result["success"]:
        print(f"PASS: Configuration validated successfully")
    else:
        raise ValueError(f"Config validation failed: {validation_result['error']}")
else:
    raise ValueError(f"Config creation failed: {config_result['error']}")"""))

# Cell 6: Load Test Data
cells.append(nbf.v4.new_code_cell("""# Load test data (wound images for evaluation)
TEST_IMAGES_DIR = BASE_INPUT_PATH / "test/images"
TEST_MASKS_DIR = BASE_INPUT_PATH / "test_masks"

test_image_paths = sorted(TEST_IMAGES_DIR.glob("*.png"))
print(f"Found {len(test_image_paths)} test images")

test_pairs = []
for img_path in test_image_paths:
    mask_path = TEST_MASKS_DIR / img_path.name
    if mask_path.exists():
        test_pairs.append((str(img_path), str(mask_path)))
    else:
        print(f"WARNING: Missing mask for {img_path.name}")

print(f"Valid test pairs: {len(test_pairs)}")"""))

# Cell 7: Multi-Seed Fine-Tuning
cells.append(nbf.v4.new_markdown_cell("""## Multi-Seed Fine-Tuning on DDI Hard Negatives

**Process**: Load baseline epoch40.pt for each seed, fine-tune on DDI only.

**Seeds**: 42, 123, 456 (same as baseline for comparison)

**Fine-tuning Hyperparameters**:
- lr0=0.0015 (10x lower than baseline 0.015)
- epochs=25 (short fine-tuning)
- No augmentation (preserve hard negative structure)

**Expected Duration**: ~45 minutes total (~15 min per seed on P100)"""))

cells.append(nbf.v4.new_code_cell("""# Multi-seed fine-tuning
RANDOM_SEEDS = [42, 123, 456]

# Fine-tuning hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0015  # 10x lower than baseline
WORKERS = 8
AMP = False
IMAGE_SIZE = 512
EPOCHS = 25  # Short fine-tuning phase
PATIENCE = 5

all_training_results = {}

print("=" * 80)
print("STARTING ABLATION 4B: FINE-TUNING ON DDI HARD NEGATIVES (EARLY CHECKPOINT)")
print("=" * 80)
print(f"Training configuration:")
print(f"  Seeds: {RANDOM_SEEDS}")
print(f"  Base checkpoint: epoch40.pt (baseline)")
print(f"  Fine-tune dataset: {len(ddi_train_paths)} DDI images")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE} (10x lower than baseline)")
print(f"  Epochs: {EPOCHS}")
print(f"Estimated time: ~45 minutes total (~15 min per seed on P100)\\n")

for seed_idx, seed in enumerate(RANDOM_SEEDS, 1):
    print("\\n" + "=" * 80)
    print(f"FINE-TUNING SEED {seed} ({seed_idx}/{len(RANDOM_SEEDS)})")
    print("=" * 80)

    # Load baseline checkpoint
    if IN_KAGGLE:
        baseline_checkpoint = Path(f"/kaggle/input/baseline-outputs/yolo_training/seed_{seed}/weights/epoch40.pt")
    else:
        baseline_checkpoint = PROJECT_ROOT / f"kaggle_outputs/baseline_final/yolo_training/seed_{seed}/weights/epoch40.pt"

    if not baseline_checkpoint.exists():
        print(f"ERROR: Baseline checkpoint not found: {baseline_checkpoint}")
        continue

    print(f"Loading baseline checkpoint: {baseline_checkpoint}")
    model = YOLO(str(baseline_checkpoint))

    # Fine-tune on DDI hard negatives
    training_start = time.time()

    results = model.train(
        data=str(config_path),
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        workers=WORKERS,
        amp=AMP,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        patience=PATIENCE,
        # No augmentation for hard negatives
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        hsv_v=0.0,
        seed=seed,
        save=True,
        project=str(OUTPUT_PATH / "yolo_training"),
        name=f"seed_{seed}",
        exist_ok=True,
        verbose=False,
    )

    training_duration = time.time() - training_start
    print(f"\\nPASS: Fine-tuning completed in {training_duration/60:.1f} minutes")

    # Evaluate on wound test set
    print(f"\\nEvaluating fine-tuned model on test set ({len(test_pairs)} images)...")

    seed_results = []
    for idx, (img_path_str, mask_path_str) in enumerate(test_pairs):
        image_id = Path(img_path_str).stem

        img = cv2.imread(img_path_str)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_mask = cv2.imread(mask_path_str, cv2.IMREAD_GRAYSCALE)
        _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

        yolo_results = model.predict(img_rgb, verbose=False)

        conversion_result = yolo_result_to_metrics_input(yolo_results[0], image_id)

        if conversion_result["success"]:
            pred_mask = conversion_result["data"]["pred_mask"]
            metrics = calculate_segmentation_metrics(pred_mask, gt_mask, image_id)
            boundary_f1 = calculate_boundary_f1(pred_mask, gt_mask, tolerance=2)
            metrics["boundary_f1"] = boundary_f1
            seed_results.append(metrics)
        else:
            print(f"WARNING: Failed to process {image_id}: {conversion_result['error']}")

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_pairs)} images...")

    all_training_results[f"seed_{seed}"] = seed_results
    print(f"\\nPASS: Seed {seed} complete: {len(seed_results)} images evaluated")

print("\\n" + "=" * 80)
print("ALL FINE-TUNING/EVALUATION RUNS COMPLETE")
print("=" * 80)"""))

# Cell 8: Aggregate Results
cells.append(nbf.v4.new_code_cell("""# Aggregate results across all seeds
print("=" * 80)
print("AGGREGATING RESULTS")
print("=" * 80)

all_finetuned_results = []
for seed_key, seed_results in all_training_results.items():
    all_finetuned_results.extend(seed_results)

print(f"\\nTotal predictions pooled: {len(all_finetuned_results)}")

# Calculate statistics with bootstrap CIs
ft_dice_scores = [r["dice"] for r in all_finetuned_results]
ft_iou_scores = [r["iou"] for r in all_finetuned_results]
ft_precision_scores = [r["precision"] for r in all_finetuned_results]
ft_recall_scores = [r["recall"] for r in all_finetuned_results]
ft_boundary_f1_scores = [r["boundary_f1"] for r in all_finetuned_results]

print(f"\\nCalculating bootstrap confidence intervals...\\n")

ft_summary_stats = {}
baseline_dice = 0.8000  # From baseline-outputs

for metric_name, metric_values in [
    ("dice", ft_dice_scores),
    ("iou", ft_iou_scores),
    ("precision", ft_precision_scores),
    ("recall", ft_recall_scores),
    ("boundary_f1", ft_boundary_f1_scores),
]:
    ci_result = calculate_bootstrap_ci(
        metric_values, confidence_level=0.95, n_bootstrap=10000, random_seed=42
    )
    ft_summary_stats[metric_name] = ci_result

    if metric_name == "dice":
        improvement = ci_result["mean"] - baseline_dice
        print(f"{metric_name.upper()}:")
        print(f"  Fine-tuned:  {ci_result['mean']:.4f} ± {ci_result['std']:.4f}")
        print(f"  95% CI: [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")
        print(f"  Baseline:    {baseline_dice:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({100*improvement/baseline_dice:.1f}%)")
    else:
        print(f"{metric_name.upper()}: {ci_result['mean']:.4f} ± {ci_result['std']:.4f}")
    print()

print("=" * 80)"""))

# Cell 9: Completion Status
cells.append(nbf.v4.new_code_cell("""# Save completion status
completion_status = {
    "notebook": "ablation_4c_early_checkpoint",
    "ablation": "hard_negative_mining_early_checkpoint",
    "status": "complete",
    "strategy": "fine_tune_early_checkpoint",
    "n_images_evaluated": len(test_pairs),
    "n_seeds": len(RANDOM_SEEDS),
    "seeds": RANDOM_SEEDS,
    "dice_mean": ft_summary_stats["dice"]["mean"],
    "baseline_dice": 0.8000,
    "dice_improvement_vs_baseline": ft_summary_stats["dice"]["mean"] - 0.8000,
    "environment": "Kaggle" if IN_KAGGLE else "Local",
}

completion_path = OUTPUT_PATH / "completion_status.json"
with open(completion_path, "w") as f:
    json.dump(completion_status, f, indent=2)

print(f"\\nCompletion status saved: {completion_path}")

print("\\n" + "=" * 80)
print("PASS: ABLATION 4C COMPLETE: DDI HARD NEGATIVES (EARLY CHECKPOINT FINE-TUNING)")
print("=" * 80)
print(f"\\nResults:")
print(f"  Dice Score: {ft_summary_stats['dice']['mean']:.4f}")
print(f"  Baseline:   0.8000")
print(f"  Improvement: {ft_summary_stats['dice']['mean'] - 0.8000:+.4f}")
print(f"\\nAll outputs saved to: {OUTPUT_PATH}")
print("=" * 80)"""))

# Add cells to notebook
nb["cells"] = cells

# Normalize and validate
nb = nbf.validator.normalize(nb)[1]
nbf.validate(nb)

# Write notebook
notebook_path = OUTPUT_DIR / "ablation_4c_early_checkpoint.ipynb"
with open(notebook_path, "w") as f:
    nbf.write(nb, f)

print(f"PASS: Notebook created: {notebook_path}")

# Create kernel metadata
metadata = {
    "id": "bkdkaggle/ablation-4c-early-checkpoint",
    "title": "Ablation 4c: Hard Negatives (Early Checkpoint Fine-tuning)",
    "code_file": "ablation_4c_early_checkpoint_clean.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": [
        "bkdkaggle/baseline-outputs",
        "bkdkaggle/ddi-diverse-dermatology",
        "bkdkaggle/wound-seg-code-v1"
    ],
    "kernel_sources": []
}

metadata_path = OUTPUT_DIR / "kernel-metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"PASS: Metadata created: {metadata_path}")
print("\\nReady for deployment to Kaggle!")
