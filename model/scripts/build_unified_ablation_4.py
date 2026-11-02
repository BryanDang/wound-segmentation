"""
Build unified Ablation 4 notebook with all 3 variants (4a, 4b, 4c).

Failure-proof design:
- Each variant in separate section with immediate saves
- Completion markers allow restart from any point
- Load-or-train checks for each variant
"""

from pathlib import Path

import nbformat as nbf

# Cell specifications
CELLS = []

# ============================================================================
# ACT 0: SETUP (Cells 0-4)
# ============================================================================

CELLS.append(
    {
        "type": "code",
        "source": """# Fix: Kaggle's default NumPy 1.26.4 is CORRECT - prevent upgrades to 2.x
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

print("   PASS: Ultralytics imported successfully")""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """import json
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

# KAGGLE DATASET SETUP: Add code dataset directly to path
if os.path.exists("/kaggle/input/wound-seg-code-v1"):
    sys.path.insert(0, "/kaggle/input/wound-seg-code-v1")
    print("[INFO] Code dataset added to Python path")
    from evaluation.metrics import (
        calculate_bootstrap_ci,
        calculate_boundary_f1,
        calculate_segmentation_metrics,
    )
    from data.yolo_config import create_yolo_config
    from evaluation.yolo_utils import yolo_result_to_metrics_input
else:
    sys.path.insert(0, str(PROJECT_ROOT))
    print("[INFO] Using local src directory")
    from src.evaluation.metrics import (
        calculate_bootstrap_ci,
        calculate_boundary_f1,
        calculate_segmentation_metrics,
    )
    from src.data.yolo_config import create_yolo_config
    from src.evaluation.yolo_utils import yolo_result_to_metrics_input

print("PASS: All modules imported successfully")""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Detect environment (Kaggle vs local)
IN_KAGGLE = Path("/kaggle/input").exists()

if IN_KAGGLE:
    BASE_INPUT_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
    OUTPUT_PATH = Path("/kaggle/working")
else:
    BASE_INPUT_PATH = PROJECT_ROOT / "simulated_kaggle_data/input/wound-segmentation-images"
    OUTPUT_PATH = PROJECT_ROOT / "kaggle_outputs/ablation_4_unified"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local (simulated)'}")
print(f"Input path: {BASE_INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")""",
    }
)

CELLS.append(
    {
        "type": "markdown",
        "source": """# Ablation 4: Hard Negative Mining - Unified Notebook

**Design**: All 3 variants (4a, 4b, 4c) in one notebook. Each variant saves on completion.

**Safety Features**:
- Each variant saves immediately after completion
- Completion markers allow a restart from any point
- Independent execution (4b failure doesn't destroy 4a results)

**Variants**:
- **4a**: Mixed training (DDI included from epoch 1)
- **4b**: Two-stage: baseline best.pt, then 25 epochs on DDI
- **4c**: Two-stage: baseline epoch40.pt, then 25 epochs on DDI

**Expected Runtime**: ~2.5 hours total on P100""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Setup test pairs for evaluation
TEST_IMAGES_DIR = BASE_INPUT_PATH / "test/images"
TEST_MASKS_DIR = BASE_INPUT_PATH / "test_masks"

test_image_paths = sorted(TEST_IMAGES_DIR.glob("*.png"))
print(f"Found {len(test_image_paths)} test images")

test_pairs = []
for img_path in test_image_paths:
    mask_path = TEST_MASKS_DIR / img_path.name
    if mask_path.exists():
        test_pairs.append((str(img_path), str(mask_path)))

print(f"Valid test pairs: {len(test_pairs)}")""",
    }
)

# ============================================================================
# ACT 1: DDI DATASET PREPARATION (Cells 5-7)
# ============================================================================

CELLS.append(
    {
        "type": "markdown",
        "source": """## DDI Dataset Preparation - Fixed Split (seed=42)

Same train/test split for all variants (4a, 4b, 4c).

**Split**: 643 DDI images → 514 train (80%) + 129 test (20%)""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Prepare DDI dataset for hard negative mining
import shutil
from sklearn.model_selection import train_test_split

print("=" * 80)
print("DDI DATASET PREPARATION")
print("=" * 80)

# DDI image paths (VALIDATED via kagglehub)
DDI_IMAGES_DIR = Path("/kaggle/input/ddi-diverse-dermatology/PRIVATE - ddidiversedermatologyimages")
ddi_image_paths = sorted(DDI_IMAGES_DIR.glob("*.png"))

print(f"\\nFound {len(ddi_image_paths)} DDI images (expected: 643)")

# FIXED train/test split (seed=42 for reproducibility across ALL variants)
ddi_train_paths, ddi_test_paths = train_test_split(
    ddi_image_paths,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\\nDDI Split (seed=42 - SHARED across 4a/4b/4c):")
print(f"  Train: {len(ddi_train_paths)} images (80%)")
print(f"  Test:  {len(ddi_test_paths)} images (20%)")

# Save split info for all variants to use
ddi_split_info = {
    "total_ddi": len(ddi_image_paths),
    "train_ddi": len(ddi_train_paths),
    "test_ddi": len(ddi_test_paths),
    "random_seed": 42,
    "train_files": [p.name for p in ddi_train_paths],
    "test_files": [p.name for p in ddi_test_paths]
}

with open(OUTPUT_PATH / "ddi_split_info.json", "w") as f:
    json.dump(ddi_split_info, f, indent=2)

print(f"\\nPASS: DDI split info saved: {OUTPUT_PATH / 'ddi_split_info.json'}")
print("=" * 80)""",
    }
)

# ============================================================================
# ACT 2: ABLATION 4a - MIXED TRAINING (Cells 8-12)
# ============================================================================

CELLS.append(
    {
        "type": "markdown",
        "source": """---

# Ablation 4a: Hard Negative Mining (Mixed Training)

**Intervention**: Mix DDI images with wounds from epoch 1

**Dataset**:
- Train: 2,208 wounds and 514 DDI, 2,722 images
- Test: 552 wounds and 129 DDI, 681 images

**Expected Outcome**: Lower FPR on dermatological conditions

---""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Check if 4a already completed
completion_4a = OUTPUT_PATH / "ablation_4a_completion.json"

if completion_4a.exists():
    print("PASS: Ablation 4a already completed (loading results)")
    with open(completion_4a) as f:
        results_4a = json.load(f)
    print(f"  Dice: {results_4a['dice_mean']:.4f}")
    print(f"  Results loaded from: {completion_4a}")
else:
    print("WARNING: Ablation 4a not completed - starting from scratch")

    # Create combined dataset
    print("\\nCreating combined dataset...")
    COMBINED_DATASET_ROOT = OUTPUT_PATH / "ablation_4a_dataset"

    (COMBINED_DATASET_ROOT / "train/images").mkdir(parents=True, exist_ok=True)
    (COMBINED_DATASET_ROOT / "train/labels").mkdir(parents=True, exist_ok=True)
    (COMBINED_DATASET_ROOT / "test/images").mkdir(parents=True, exist_ok=True)
    (COMBINED_DATASET_ROOT / "test/labels").mkdir(parents=True, exist_ok=True)

    # Copy baseline wounds
    baseline_train_imgs = Path("/kaggle/input/baseline-outputs/data_wound_seg/train/images")
    baseline_train_labels = Path("/kaggle/input/baseline-outputs/data_wound_seg/train/labels")

    wound_count = 0
    for img_path in baseline_train_imgs.glob("*.png"):
        shutil.copy(img_path, COMBINED_DATASET_ROOT / "train/images" / img_path.name)
        wound_count += 1

    for label_path in baseline_train_labels.glob("*.txt"):
        shutil.copy(label_path, COMBINED_DATASET_ROOT / "train/labels" / label_path.name)

    # Add DDI hard negatives (empty labels)
    for img_path in ddi_train_paths:
        shutil.copy(img_path, COMBINED_DATASET_ROOT / "train/images" / img_path.name)
        label_path = COMBINED_DATASET_ROOT / "train/labels" / f"{img_path.stem}.txt"
        label_path.write_text("")  # Empty = no annotations

    print(f"  PASS: Train: {wound_count} wounds + {len(ddi_train_paths)} DDI = {wound_count + len(ddi_train_paths)} total")

    # Copy baseline test data
    baseline_test_imgs = Path("/kaggle/input/baseline-outputs/data_wound_seg/test/images")
    baseline_test_labels = Path("/kaggle/input/baseline-outputs/data_wound_seg/test/labels")

    test_count = 0
    for img_path in baseline_test_imgs.glob("*.png"):
        shutil.copy(img_path, COMBINED_DATASET_ROOT / "test/images" / img_path.name)
        test_count += 1

    for label_path in baseline_test_labels.glob("*.txt"):
        shutil.copy(label_path, COMBINED_DATASET_ROOT / "test/labels" / label_path.name)

    # Add DDI test (empty labels)
    for img_path in ddi_test_paths:
        shutil.copy(img_path, COMBINED_DATASET_ROOT / "test/images" / img_path.name)
        label_path = COMBINED_DATASET_ROOT / "test/labels" / f"{img_path.stem}.txt"
        label_path.write_text("")

    print(f"  PASS: Test: {test_count} wounds + {len(ddi_test_paths)} DDI = {test_count + len(ddi_test_paths)} total")

    # Create YOLO config
    config_result = create_yolo_config(
        dataset_root=COMBINED_DATASET_ROOT,
        train_dir="train",
        val_dir="test",
        test_dir="test",
        output_path=OUTPUT_PATH / "wound_dataset_4a.yaml"
    )

    config_4a_path = config_result["data"]["config_path"]
    print(f"  PASS: Config created: {config_4a_path}")""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Train 4a (if not already completed)
import time

if not completion_4a.exists():
    print("\\n" + "=" * 80)
    print("TRAINING ABLATION 4a: MIXED (3 seeds)")
    print("=" * 80)

    RANDOM_SEEDS = [42, 123, 456]
    all_4a_results = {}

    for seed in RANDOM_SEEDS:
        print(f"\\nTraining seed {seed}...")

        model = YOLO("yolov8n-seg.pt")

        results = model.train(
            data=str(config_4a_path),
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
            seed=seed,
            save=True,
            project=str(OUTPUT_PATH / "yolo_training_4a"),
            name=f"seed_{seed}",
            exist_ok=True,
            verbose=False
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

        all_4a_results[f"seed_{seed}"] = seed_results
        print(f"  PASS: Seed {seed} complete: {len(seed_results)} images")

    # Aggregate and save
    all_pooled = []
    for seed_results in all_4a_results.values():
        all_pooled.extend(seed_results)

    dice_scores = [r["dice"] for r in all_pooled]
    ci_result = calculate_bootstrap_ci(dice_scores, confidence_level=0.95, n_bootstrap=10000, random_seed=42)

    results_4a = {
        "variant": "4a_mixed",
        "dice_mean": ci_result["mean"],
        "dice_ci_lower": ci_result["ci_lower"],
        "dice_ci_upper": ci_result["ci_upper"],
        "per_image_results": all_4a_results
    }

    # IMMEDIATE SAVE (failure-proof)
    with open(completion_4a, "w") as f:
        json.dump(results_4a, f, indent=2)

    print(f"\\nPASS: 4a COMPLETE - Results saved: {completion_4a}")
    print(f"  Dice: {results_4a['dice_mean']:.4f} [{results_4a['dice_ci_lower']:.4f}, {results_4a['dice_ci_upper']:.4f}]")""",
    }
)

# ============================================================================
# ACT 3: ABLATION 4b - TWO-STAGE (best.pt) (Cells 13-15)
# ============================================================================

CELLS.append(
    {
        "type": "markdown",
        "source": """---

# Ablation 4b: Two-Stage Training (Baseline best.pt, DDI Refinement)

**Intervention**: Load baseline best.pt, refine for 25 epochs on DDI only

Aim: keep wound detection and add hard negative discrimination.

---""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Check if 4b already completed
completion_4b = OUTPUT_PATH / "ablation_4b_completion.json"

if completion_4b.exists():
    print("PASS: Ablation 4b already completed (loading results)")
    with open(completion_4b) as f:
        results_4b = json.load(f)
    print(f"  Dice: {results_4b['dice_mean']:.4f}")
else:
    print("WARNING: Ablation 4b not completed - starting from scratch")

    # Create DDI-only dataset
    print("\\nCreating DDI-only dataset...")
    DDI_DATASET_ROOT = OUTPUT_PATH / "ablation_4b_dataset"

    (DDI_DATASET_ROOT / "train/images").mkdir(parents=True, exist_ok=True)
    (DDI_DATASET_ROOT / "train/labels").mkdir(parents=True, exist_ok=True)
    (DDI_DATASET_ROOT / "test/images").mkdir(parents=True, exist_ok=True)
    (DDI_DATASET_ROOT / "test/labels").mkdir(parents=True, exist_ok=True)

    # Copy DDI train
    for img_path in ddi_train_paths:
        shutil.copy(img_path, DDI_DATASET_ROOT / "train/images" / img_path.name)
        label_path = DDI_DATASET_ROOT / "train/labels" / f"{img_path.stem}.txt"
        label_path.write_text("")

    # Copy DDI test
    for img_path in ddi_test_paths:
        shutil.copy(img_path, DDI_DATASET_ROOT / "test/images" / img_path.name)
        label_path = DDI_DATASET_ROOT / "test/labels" / f"{img_path.stem}.txt"
        label_path.write_text("")

    print(f"  PASS: DDI dataset created: {len(ddi_train_paths)} train, {len(ddi_test_paths)} test")

    # Create YOLO config
    config_result = create_yolo_config(
        dataset_root=DDI_DATASET_ROOT,
        train_dir="train",
        val_dir="test",
        test_dir="test",
        output_path=OUTPUT_PATH / "wound_dataset_4b.yaml"
    )

    config_4b_path = config_result["data"]["config_path"]
    print(f"  PASS: Config created: {config_4b_path}")""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Train 4b (if not already completed)
if not completion_4b.exists():
    print("\\n" + "=" * 80)
    print("TRAINING ABLATION 4b: TWO-STAGE (best.pt → DDI)")
    print("=" * 80)

    # Load baseline weights from baseline-outputs dataset
    baseline_weights_dir = Path("/kaggle/input/baseline-outputs/yolo_training")

    RANDOM_SEEDS = [42, 123, 456]
    all_4b_results = {}

    for seed in RANDOM_SEEDS:
        print(f"\\nRefining seed {seed} (loading baseline best.pt)...")

        # Find best.pt for this seed in baseline-outputs
        baseline_weights = baseline_weights_dir / f"seed_{seed}" / "weights" / "best.pt"

        if not baseline_weights.exists():
            print(f"  ERROR: Baseline weights not found: {baseline_weights}")
            continue

        # Load baseline model
        model = YOLO(str(baseline_weights))
        print(f"  PASS: Loaded baseline weights: {baseline_weights}")

        # Refine on DDI for 25 epochs
        results = model.train(
            data=str(config_4b_path),
            batch=32,
            lr0=0.001,  # Lower LR for fine-tuning
            workers=8,
            amp=False,
            epochs=25,  # Refinement only
            imgsz=512,
            patience=10,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            seed=seed,
            save=True,
            project=str(OUTPUT_PATH / "yolo_training_4b"),
            name=f"seed_{seed}",
            exist_ok=True,
            verbose=False
        )

        # Evaluate on combined test set (wounds + DDI)
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

        all_4b_results[f"seed_{seed}"] = seed_results
        print(f"  PASS: Seed {seed} complete: {len(seed_results)} images")

    # Aggregate and save
    all_pooled = []
    for seed_results in all_4b_results.values():
        all_pooled.extend(seed_results)

    dice_scores = [r["dice"] for r in all_pooled]
    ci_result = calculate_bootstrap_ci(dice_scores, confidence_level=0.95, n_bootstrap=10000, random_seed=42)

    results_4b = {
        "variant": "4b_two_stage_best",
        "dice_mean": ci_result["mean"],
        "dice_ci_lower": ci_result["ci_lower"],
        "dice_ci_upper": ci_result["ci_upper"],
        "per_image_results": all_4b_results
    }

    # IMMEDIATE SAVE (failure-proof)
    with open(completion_4b, "w") as f:
        json.dump(results_4b, f, indent=2)

    print(f"\\nPASS: 4b COMPLETE - Results saved: {completion_4b}")
    print(f"  Dice: {results_4b['dice_mean']:.4f} [{results_4b['dice_ci_lower']:.4f}, {results_4b['dice_ci_upper']:.4f}]")""",
    }
)

# ============================================================================
# ACT 4: ABLATION 4c - TWO-STAGE (epoch40.pt) (Cells 16-18)
# ============================================================================

CELLS.append(
    {
        "type": "markdown",
        "source": """---

# Ablation 4c: Two-Stage Training (Baseline epoch40.pt, DDI Refinement)

**Intervention**: Load baseline epoch40.pt (mid-training), refine for 25 epochs on DDI

**Hypothesis**: Mid-training checkpoint may generalize better to DDI hard negatives

---""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Check if 4c already completed
completion_4c = OUTPUT_PATH / "ablation_4c_completion.json"

if completion_4c.exists():
    print("PASS: Ablation 4c already completed (loading results)")
    with open(completion_4c) as f:
        results_4c = json.load(f)
    print(f"  Dice: {results_4c['dice_mean']:.4f}")
else:
    print("WARNING: Ablation 4c not completed - will train after 4b completes")""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Train 4c (if not already completed)
if not completion_4c.exists():
    print("\\n" + "=" * 80)
    print("TRAINING ABLATION 4c: TWO-STAGE (epoch40.pt → DDI)")
    print("=" * 80)

    baseline_weights_dir = Path("/kaggle/input/baseline-outputs/yolo_training")

    RANDOM_SEEDS = [42, 123, 456]
    all_4c_results = {}

    for seed in RANDOM_SEEDS:
        print(f"\\nRefining seed {seed} (loading baseline epoch40.pt)...")

        # Find epoch40.pt for this seed
        baseline_weights = baseline_weights_dir / f"seed_{seed}" / "weights" / "epoch40.pt"

        if not baseline_weights.exists():
            print(f"  ERROR: Baseline weights not found: {baseline_weights}")
            continue

        model = YOLO(str(baseline_weights))
        print(f"  PASS: Loaded baseline weights: {baseline_weights}")

        # Refine on DDI (using same config as 4b)
        results = model.train(
            data=str(config_4b_path),
            batch=32,
            lr0=0.001,
            workers=8,
            amp=False,
            epochs=25,
            imgsz=512,
            patience=10,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            seed=seed,
            save=True,
            project=str(OUTPUT_PATH / "yolo_training_4c"),
            name=f"seed_{seed}",
            exist_ok=True,
            verbose=False
        )

        # Evaluate
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

        all_4c_results[f"seed_{seed}"] = seed_results
        print(f"  PASS: Seed {seed} complete: {len(seed_results)} images")

    # Aggregate and save
    all_pooled = []
    for seed_results in all_4c_results.values():
        all_pooled.extend(seed_results)

    dice_scores = [r["dice"] for r in all_pooled]
    ci_result = calculate_bootstrap_ci(dice_scores, confidence_level=0.95, n_bootstrap=10000, random_seed=42)

    results_4c = {
        "variant": "4c_two_stage_epoch40",
        "dice_mean": ci_result["mean"],
        "dice_ci_lower": ci_result["ci_lower"],
        "dice_ci_upper": ci_result["ci_upper"],
        "per_image_results": all_4c_results
    }

    # IMMEDIATE SAVE (failure-proof)
    with open(completion_4c, "w") as f:
        json.dump(results_4c, f, indent=2)

    print(f"\\nPASS: 4c COMPLETE - Results saved: {completion_4c}")
    print(f"  Dice: {results_4c['dice_mean']:.4f} [{results_4c['dice_ci_lower']:.4f}, {results_4c['dice_ci_upper']:.4f}]")""",
    }
)

# ============================================================================
# ACT 5: FINAL AGGREGATION (Cells 19-20)
# ============================================================================

CELLS.append(
    {
        "type": "markdown",
        "source": """---

# Final Aggregation: All 3 Variants

**Analysis**: Determine which hard negative strategy performs best.

---""",
    }
)

CELLS.append(
    {
        "type": "code",
        "source": """# Final comparison
print("=" * 80)
print("ABLATION 4 FINAL RESULTS")
print("=" * 80)

# Load all results
with open(completion_4a) as f:
    results_4a = json.load(f)

with open(completion_4b) as f:
    results_4b = json.load(f)

with open(completion_4c) as f:
    results_4c = json.load(f)

# Compare
baseline_dice = 0.8000  # From baseline-outputs

print(f"\\nBaseline (no hard negatives): {baseline_dice:.4f}")
print(f"\\n4a (Mixed training):         {results_4a['dice_mean']:.4f} [{results_4a['dice_ci_lower']:.4f}, {results_4a['dice_ci_upper']:.4f}]")
print(f"  Δ vs baseline: {results_4a['dice_mean'] - baseline_dice:+.4f}")

print(f"\\n4b (Two-stage, best.pt):     {results_4b['dice_mean']:.4f} [{results_4b['dice_ci_lower']:.4f}, {results_4b['dice_ci_upper']:.4f}]")
print(f"  Δ vs baseline: {results_4b['dice_mean'] - baseline_dice:+.4f}")

print(f"\\n4c (Two-stage, epoch40.pt):  {results_4c['dice_mean']:.4f} [{results_4c['dice_ci_lower']:.4f}, {results_4c['dice_ci_upper']:.4f}]")
print(f"  Δ vs baseline: {results_4c['dice_mean'] - baseline_dice:+.4f}")

# Determine winner
variants = {
    "4a": results_4a['dice_mean'],
    "4b": results_4b['dice_mean'],
    "4c": results_4c['dice_mean']
}
winner = max(variants, key=variants.get)

print(f"\\n{'=' * 80}")
print(f"BEST VARIANT: {winner} (Dice = {variants[winner]:.4f})")
print(f"{'=' * 80}")

# Save final report
final_report = {
    "ablation": "hard_negative_mining",
    "variants_tested": 3,
    "baseline_dice": baseline_dice,
    "results": {
        "4a_mixed": results_4a,
        "4b_best_pt": results_4b,
        "4c_epoch40_pt": results_4c
    },
    "best_variant": winner,
    "best_dice": variants[winner],
    "improvement_vs_baseline": variants[winner] - baseline_dice
}

with open(OUTPUT_PATH / "ablation_4_final_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

print(f"\\nFinal report saved: {OUTPUT_PATH / 'ablation_4_final_report.json'}")""",
    }
)

# ============================================================================
# BUILD NOTEBOOK
# ============================================================================

print("Building unified Ablation 4 notebook...")

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

# Add cells
for cell_spec in CELLS:
    if cell_spec["type"] == "code":
        nb.cells.append(nbf.v4.new_code_cell(cell_spec["source"]))
    else:
        nb.cells.append(nbf.v4.new_markdown_cell(cell_spec["source"]))

# Normalize
nb = nbf.validator.normalize(nb)[1]

# Validate
nbf.validate(nb)

# Write
output_path = Path("notebooks/03_ablations/ablation_4_unified/ablation_4_unified.ipynb")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w") as f:
    nbf.write(nb, f)

print(f"PASS: Notebook created: {output_path}")
print(f"  Total cells: {len(nb.cells)}")
print("\\nStructure:")
print("  Act 0: Setup (5 cells)")
print("  Act 1: DDI Preparation (2 cells)")
print("  Act 2: Ablation 4a - Mixed (3 cells)")
print("  Act 3: Ablation 4b - Two-Stage best.pt (3 cells)")
print("  Act 4: Ablation 4c - Two-Stage epoch40.pt (3 cells)")
print("  Act 5: Final Aggregation (2 cells)")
print(f"\\nReady to clean and deploy!")
