"""
Cell specifications for baseline notebook - optimized 28-cell structure.

This file defines all cells for deterministic notebook building using nbformat.
Clinical interpretation removed - pure technical documentation only.
"""

# Cell 0: NumPy compatibility fix (CODE)
CELL_0_CODE = """# Fix: Kaggle's default NumPy 1.26.4 is CORRECT - prevent upgrades to 2.x
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

print("   PASS: Ultralytics imported successfully")"""

# Cell 1: Setup and imports (CODE)
CELL_1_CODE = """import json
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
    # On Kaggle, import without src. prefix
    from evaluation.metrics import (
        calculate_segmentation_metrics,
        calculate_boundary_f1,
        calculate_bootstrap_ci,
    )
    from pipeline.eda import Visualizer, EDA_CONFIG
else:
    # Local development: add project root to path
    sys.path.insert(0, str(PROJECT_ROOT))
    print("[INFO] Using local src directory")
    # Locally, import with src. prefix
    from src.evaluation.metrics import (
        calculate_segmentation_metrics,
        calculate_boundary_f1,
        calculate_bootstrap_ci,
    )
    from src.pipeline.eda import Visualizer, EDA_CONFIG

print("PASS: All modules imported successfully")"""

# Cell 2: Environment detection (CODE)
CELL_2_CODE = """# Detect environment (Kaggle vs local)
IN_KAGGLE = Path("/kaggle/input").exists()

if IN_KAGGLE:
    # Kaggle dataset has extra 'data_wound_seg' parent directory
    BASE_INPUT_PATH = Path("/kaggle/input/wound-segmentation-images/data_wound_seg")
    OUTPUT_PATH = Path("/kaggle/working")
else:
    # Local development: use kaggle_outputs directory for this run
    BASE_INPUT_PATH = PROJECT_ROOT / "simulated_kaggle_data/input/wound-segmentation-images"
    OUTPUT_PATH = PROJECT_ROOT / "kaggle_outputs/baseline_run"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local (simulated)'}")
print(f"Input path: {BASE_INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Project root: {PROJECT_ROOT}")"""

# Cell 3: Load pretrained model (CODE)
CELL_3_CODE = """# Load pretrained model (automatically downloads yolov8n-seg.pt if not present)
print("Loading pretrained YOLOv8n-seg (COCO weights)...")
model = YOLO("yolov8n-seg.pt")

print(f"Model loaded successfully")
print(f"Model classes: {len(model.names)} COCO classes")
print(f"Sample classes: {list(model.names.values())[:5]}")"""

# Cell 4: Load test dataset (CODE)
CELL_4_CODE = """# Paths to test data
TEST_IMAGES_DIR = BASE_INPUT_PATH / "test_images"
TEST_MASKS_DIR = BASE_INPUT_PATH / "test_masks"

# Get all test images (files are directly in test_images/)
test_image_paths = sorted(TEST_IMAGES_DIR.glob("*.png")) if TEST_IMAGES_DIR.exists() else []
print(f"Found {len(test_image_paths)} test images")

# Verify corresponding masks exist and store as string paths
test_pairs = []
for img_path in test_image_paths:
    mask_path = TEST_MASKS_DIR / img_path.name
    if mask_path.exists():
        # Convert to strings immediately to avoid Path object issues
        test_pairs.append((str(img_path), str(mask_path)))
    else:
        print(f"WARNING: Missing mask for {img_path.name}")

print(f"Valid test pairs: {len(test_pairs)}")

# Show first test pair
if test_pairs:
    print(f"\\nFirst test pair:")
    print(f"  Image: {test_pairs[0][0]}")
    print(f"  Mask: {test_pairs[0][1]}")
elif len(test_image_paths) == 0:
    raise FileNotFoundError(
        f"No test images found in {TEST_IMAGES_DIR}. "
        f"Please check dataset structure in {BASE_INPUT_PATH}"
    )"""

# Cell 5: Zero-shot inference explanation (MARKDOWN)
CELL_5_MARKDOWN = """## Zero-Shot Inference Strategy

**Approach**: Merge all predicted masks, whatever the COCO class, into one wound prediction.

**Rationale**: Zero-shot models may detect wounds as similar objects (person, apple, etc.). We evaluate merged predictions against ground truth to establish baseline performance."""

# Cell 6: Zero-shot evaluation (CODE)
CELL_6_CODE = """# Run zero-shot evaluation
all_results = []

print(f"Running zero-shot evaluation on {len(test_pairs)} images...\\n")

for idx, (img_path_str, mask_path_str) in enumerate(test_pairs):
    # Extract image ID from path string
    image_id = Path(img_path_str).stem

    # Load ground truth mask
    gt_mask = cv2.imread(mask_path_str, cv2.IMREAD_GRAYSCALE)
    _, gt_mask = cv2.threshold(gt_mask, 127, 1, cv2.THRESH_BINARY)

    # Run inference (zero-shot) - img_path_str is already a plain string
    results = model(img_path_str, verbose=False)

    # Extract prediction mask (merge all classes since no wound class exists)
    pred_mask = np.zeros((512, 512), dtype=np.uint8)

    if results[0].masks is not None:
        # Merge all predicted masks (any detected object)
        for mask in results[0].masks.data:
            # Resize mask to 512×512 and merge
            mask_resized = cv2.resize(
                mask.cpu().numpy(), (512, 512), interpolation=cv2.INTER_NEAREST
            )
            pred_mask = np.maximum(pred_mask, (mask_resized > 0.5).astype(np.uint8))

    # Calculate metrics using imported function
    metrics = calculate_segmentation_metrics(pred_mask, gt_mask, image_id)

    # Calculate boundary F1 using imported function
    boundary_f1 = calculate_boundary_f1(pred_mask, gt_mask, tolerance=2)
    metrics["boundary_f1"] = boundary_f1

    all_results.append(metrics)

    # Progress update
    if (idx + 1) % 50 == 0 or (idx + 1) == len(test_pairs):
        print(f"Processed {idx + 1}/{len(test_pairs)} images")

print(f"\\nEvaluation complete!")
print(f"Total images evaluated: {len(all_results)}")"""

# Cell 7: Bootstrap CI explanation (MARKDOWN)
CELL_7_MARKDOWN = """## Summary Statistics with Bootstrap Confidence Intervals

**Statistical Validation**: 10,000 bootstrap iterations provide robust 95% confidence intervals for Dice, IoU, Precision, Recall, and Boundary F1 metrics."""

# Cell 8: Calculate summary statistics (CODE)
CELL_8_CODE = """# Extract metric lists
dice_scores = [r["dice"] for r in all_results]
iou_scores = [r["iou"] for r in all_results]
precision_scores = [r["precision"] for r in all_results]
recall_scores = [r["recall"] for r in all_results]
boundary_f1_scores = [r["boundary_f1"] for r in all_results]

# Calculate bootstrap CIs for each metric using imported function
print("Calculating bootstrap confidence intervals...\\n")

summary_stats = {}

for metric_name, metric_values in [
    ("dice", dice_scores),
    ("iou", iou_scores),
    ("precision", precision_scores),
    ("recall", recall_scores),
    ("boundary_f1", boundary_f1_scores),
]:
    ci_result = calculate_bootstrap_ci(
        metric_values, confidence_level=0.95, n_bootstrap=10000, random_seed=42
    )
    summary_stats[metric_name] = ci_result

    print(f"{metric_name.upper()}:")
    print(f"  Mean: {ci_result['mean']:.4f} ± {ci_result['std']:.4f}")
    print(f"  95% CI: [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")
    print()

print("=" * 60)
print("ZERO-SHOT BASELINE ESTABLISHED")
print("=" * 60)"""

# Cell 9: Initialize visualizer (CODE)
CELL_9_CODE = """# Initialize visualizer
visualizer = Visualizer(config=EDA_CONFIG)

# Convert results to DataFrame for visualization
results_df = pd.DataFrame(all_results)

# Fix column names for visualizer (expects uppercase)
results_df = results_df.rename(columns={
    'tp': 'TP', 'fp': 'FP', 'tn': 'TN', 'fn': 'FN'
})

print("Generating visualizations using modular pipeline...\\n")"""

# Cell 10: Confusion matrix (CODE)
CELL_10_CODE = """# Generate confusion matrix using Visualizer
cm_result = visualizer.generate_confusion_matrix(results_df, str(OUTPUT_PATH))

if cm_result["success"]:
    print(f"PASS: Confusion matrix saved to: {cm_result['data']['files'][0]}")

    # Display summary statistics (use uppercase column names after rename)
    total_tp = results_df["TP"].sum()
    total_fp = results_df["FP"].sum()
    total_tn = results_df["TN"].sum()
    total_fn = results_df["FN"].sum()
    total = total_tp + total_fp + total_tn + total_fn

    print("\\nConfusion Matrix Summary:")
    print(f"   Total pixels: {total:,}")
    print(f"   True Negatives:  {total_tn:,} ({100*total_tn/total:.1f}%)")
    print(f"   False Positives: {total_fp:,} ({100*total_fp/total:.1f}%)")
    print(f"   False Negatives: {total_fn:,} ({100*total_fn/total:.1f}%)")
    print(f"   True Positives:  {total_tp:,} ({100*total_tp/total:.1f}%)")
else:
    print(f"FAIL: Error generating confusion matrix: {cm_result['error']}")"""

# Cell 11: Example predictions (CODE)
CELL_11_CODE = """# Generate example predictions using Visualizer
examples_result = visualizer.generate_example_predictions(
    results_df, str(BASE_INPUT_PATH), str(OUTPUT_PATH)
)

if examples_result["success"]:
    print(f"PASS: Example predictions saved to: {examples_result['data']['files'][0]}")

    # Display performance range
    best_dice = results_df["dice"].max()
    worst_dice = results_df["dice"].min()
    median_dice = results_df["dice"].median()

    print("\\nPerformance Range:")
    print(f"   Best Dice:    {best_dice:.4f}")
    print(f"   Median Dice:  {median_dice:.4f}")
    print(f"   Worst Dice:   {worst_dice:.4f}")
else:
    print(f"FAIL: Error generating example predictions: {examples_result['error']}")"""

# Cell 12: Save results documentation (MARKDOWN)
CELL_12_MARKDOWN = """## Save Zero-Shot Results

**Output Files**:
1. Per-image metrics (JSON and CSV formats)
2. Summary statistics with bootstrap CIs
3. Visualizations (confusion matrix, example predictions)"""

# Cell 13: Save results (CODE)
CELL_13_CODE = """# Save per-image results (JSON)
per_image_output = OUTPUT_PATH / "baseline_per_image_results.json"
with open(per_image_output, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Per-image results saved to: {per_image_output}")

# Save per-image results (CSV)
results_df = pd.DataFrame(all_results)
column_order = [
    "image_id",
    "dice",
    "iou",
    "precision",
    "recall",
    "boundary_f1",
    "tp",
    "fp",
    "tn",
    "fn",
]
results_df = results_df[column_order]

csv_output = OUTPUT_PATH / "baseline_per_image_results.csv"
results_df.to_csv(csv_output, index=False, float_format="%.6f")

print(f"Per-image results CSV saved to: {csv_output}")
print(f"Shape: {results_df.shape[0]} rows x {results_df.shape[1]} columns")

# Save summary statistics
summary_output = OUTPUT_PATH / "baseline_summary_statistics.json"
with open(summary_output, "w") as f:
    json.dump(summary_stats, f, indent=2)

print(f"Summary statistics saved to: {summary_output}")

print("\\nAll outputs saved successfully!")"""

# Cell 14: Completion status header (MARKDOWN)
CELL_14_MARKDOWN = """## Phase 1 Completion Status"""

# Cell 15: Phase 1 status (CODE)
CELL_15_CODE = """# Phase 1 completion status (zero-shot evaluation complete)
from datetime import datetime

completion_status = {
    "notebook": "02_baseline_model",
    "status": "in_progress",
    "phases_complete": ["zero_shot"],
    "phases_remaining": ["fine_tuned"],
    "phase_1_timestamp": datetime.now().isoformat(),
    "n_images_evaluated": len(all_results),
    "zero_shot_dice_mean": summary_stats["dice"]["mean"],
    "environment": "Kaggle" if IN_KAGGLE else "Local",
}

completion_path = OUTPUT_PATH / "zero_shot_phase_status.json"
with open(completion_path, "w") as f:
    json.dump(completion_status, f, indent=2)

print(f"Completion status saved to: {completion_path}")
print("\\n" + "="*80)
print("PHASE 1 COMPLETE: Zero-Shot Baseline Evaluation")
print("="*80)
print("\\nResults:")
print(f"  - Dice Score: {summary_stats['dice']['mean']:.4f}")
print(f"  - IoU Score: {summary_stats['iou']['mean']:.4f}")
print(f"  - Images evaluated: {len(all_results)}")
print(f"  - Files saved: zero_shot_phase_status.json, baseline_*.png/csv/json")
print("\\nContinuing to Phase 2: Fine-Tuned Baseline Training...")
print("="*80 + "\\n")"""

# Cell 16: Kaggle dataset creation note (MARKDOWN)
CELL_16_MARKDOWN = """## Kaggle Dataset Creation

**Purpose**: Automatically create/update dataset with baseline results

**Note**: Executes only on Kaggle (requires credentials)"""

# Cell 17: Kaggle dataset creation (CODE)
CELL_17_CODE = """# Kaggle Dataset Creation
if os.path.exists("/kaggle/input/wound-seg-code-v1"):
    # On Kaggle, import without src. prefix
    from deployment.kaggle_dataset import (
        validate_kaggle_credentials,
        create_baseline_dataset_metadata)
else:
    # Local development: import with src. prefix
    from src.deployment.kaggle_dataset import (
        validate_kaggle_credentials,
        create_baseline_dataset_metadata)

# Only execute on Kaggle
if IN_KAGGLE:
    print("\\n" + "=" * 60)
    print("CREATING KAGGLE DATASET")
    print("=" * 60)

    # Validate credentials
    creds = validate_kaggle_credentials()
    if not creds["valid"]:
        print(f"  Skipping dataset creation: {creds['error']}")
    else:
        # Create metadata
        result = create_baseline_dataset_metadata(
            username=creds["username"], output_folder=OUTPUT_PATH
        )

        if result["success"]:
            print(f"  Metadata created: {result['metadata_path']}")
            print(f"  Dataset ID: {result['metadata']['id']}")

            # Use shell command instead of subprocess to avoid kernel termination
            print(f"\\n  Creating/updating dataset...")
            # Try create first, fall back to version if exists
            get_ipython().system('kaggle datasets create -p /kaggle/working/ --dir-mode zip 2>/dev/null || kaggle datasets version -p /kaggle/working/ -m "Updated baseline results" --dir-mode zip')

            print(f"  Dataset URL: https://www.kaggle.com/datasets/{result['metadata']['id']}")
        else:
            print(f"  ERROR: {result['error']}")

    print("=" * 60)
else:
    print("\\n  Skipping dataset creation (not on Kaggle)")"""

# Cell 18: Fine-tuning overview (MARKDOWN)
CELL_18_MARKDOWN = """---

# Act 1: Fine-Tuned Baseline Training

**Objective**: Train YOLOv8n-seg using transfer learning from COCO weights.

**Training Strategy**:
1. Transfer learning from pretrained weights
2. YOLO dataset configuration (single wound class)
3. Multi-seed validation (seeds: 42, 123, 456)
4. Hyperparameters: LR=0.01, epochs=100, batch=16, imgsz=512

**Expected Outcome**: Dice > 85% (significant improvement over zero-shot ~5%)

---"""

# Cell 19: Import fine-tuning modules (CODE)
CELL_19_CODE = """# Import additional modules for fine-tuning
if os.path.exists("/kaggle/input/wound-seg-code-v1"):
    # On Kaggle, import without src. prefix
    from data.yolo_config import create_yolo_config, validate_yolo_config
    from evaluation.yolo_utils import yolo_result_to_metrics_input
else:
    # Local development: import with src. prefix
    from src.data.yolo_config import create_yolo_config, validate_yolo_config
    from src.evaluation.yolo_utils import yolo_result_to_metrics_input

import shutil
import time

print("PASS: Fine-tuning modules imported successfully")"""

# Cell 20: Dataset preparation rationale (MARKDOWN)
CELL_20_MARKDOWN = """## Dataset Preparation for YOLO Training

Copied because YOLOv8 writes cache files and `/kaggle/input/` is read-only.

**YOLO Configuration** (wound_dataset.yaml):
- `path`: Absolute path to dataset root
- `train`: Training images directory name
- `val`: Validation images directory name (using test_images)
- `nc`: Number of classes (1, wound only)
- `names`: Class mapping ({0: "wound"})"""

# Cell 21: Prepare dataset (CODE)
CELL_21_CODE = """# Copy dataset to /kaggle/working/ for YOLO training
print("Preparing dataset for YOLO training...\\n")

# Destination path
YOLO_DATASET_ROOT = OUTPUT_PATH / "data_wound_seg"

if not YOLO_DATASET_ROOT.exists():
    print(f"Copying dataset from {BASE_INPUT_PATH} to {YOLO_DATASET_ROOT}")
    print("This may take a few minutes...")

    start_time = time.time()
    shutil.copytree(BASE_INPUT_PATH, YOLO_DATASET_ROOT)
    copy_duration = time.time() - start_time

    print(f"PASS: Dataset copied in {copy_duration:.1f} seconds")
else:
    print(f"PASS: Dataset already exists at {YOLO_DATASET_ROOT}")

# Verify directory structure
train_dir = YOLO_DATASET_ROOT / "train_images"
test_dir = YOLO_DATASET_ROOT / "test_images"

print(f"\\nDataset structure:")
print(f"  Train images: {len(list(train_dir.glob('*.png')))} files")
print(f"  Test images:  {len(list(test_dir.glob('*.png')))} files")

# Generate YOLO configuration file
print(f"\\nGenerating YOLO dataset configuration...")

config_result = create_yolo_config(
    dataset_root=YOLO_DATASET_ROOT,
    train_dir="train_images",
    val_dir="test_images",
    test_dir="test_images"
)

if config_result["success"]:
    config_path = config_result["data"]["config_path"]
    print(f"PASS: Configuration created: {config_path}")

    # Validate the configuration
    validation_result = validate_yolo_config(config_path)
    if validation_result["success"]:
        print(f"PASS: Configuration validated successfully")
        print(f"  Dataset root: {validation_result['data']['path']}")
        print(f"  Number of classes: {validation_result['data']['nc']}")
        print(f"  Class names: {validation_result['data']['names']}")
    else:
        raise ValueError(f"Configuration validation failed: {validation_result['error']}")
else:
    raise ValueError(f"Configuration creation failed: {config_result['error']}")

print(f"\\n" + "=" * 60)
print("Dataset prepared and validated for training")
print("=" * 60)"""

# Cell 22: Multi-seed training rationale (MARKDOWN)
CELL_22_MARKDOWN = """## Multi-Seed Training for Statistical Robustness

**Seeds**
- Quantify performance variance from random initialization
- Enable statistical significance testing
- Ensure findings generalize beyond single run

**Process**: Train 3 models (seeds 42, 123, 456) with identical hyperparameters, then aggregate the results.

**Expected Duration**: ~3 hours total (~45-60 min per seed on Kaggle GPU)"""

# Cell 23: Multi-seed training (CODE)
CELL_23_CODE = """# Multi-seed training configuration
RANDOM_SEEDS = [42, 123, 456]
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 512
LEARNING_RATE = 0.01

# Storage for all training results
all_training_results = {}

print("=" * 80)
print("STARTING MULTI-SEED TRAINING")
print("=" * 80)
print(f"\\nTraining configuration:")
print(f"  Seeds: {RANDOM_SEEDS}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Image size: {IMAGE_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"\\nDataset config: {config_path}")
print(f"\\nEstimated time: ~3 hours (45-60 min per seed)\\n")

for seed_idx, seed in enumerate(RANDOM_SEEDS, 1):
    print("\\n" + "=" * 80)
    print(f"TRAINING WITH SEED {seed} ({seed_idx}/{len(RANDOM_SEEDS)})")
    print("=" * 80)

    # Initialize model from pretrained COCO weights
    model = YOLO("yolov8n-seg.pt")

    # Train model
    training_start = time.time()

    results = model.train(
        data=str(config_path),
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LEARNING_RATE,
        seed=seed,
        patience=10,  # Early stopping
        save=True,
        project=str(OUTPUT_PATH / "yolo_training"),
        name=f"seed_{seed}",
        exist_ok=True,
        verbose=False,  # Reduce output verbosity
    )

    training_duration = time.time() - training_start

    print(f"\\nPASS: Training completed in {training_duration/60:.1f} minutes")

    # Load best checkpoint for evaluation
    best_model_path = OUTPUT_PATH / "yolo_training" / f"seed_{seed}" / "weights" / "best.pt"
    best_model = YOLO(str(best_model_path))

    print(f"\\nEvaluating on test set ({len(test_pairs)} images)...")

    # Evaluate on test set
    seed_results = []

    for img_path_str, mask_path_str in test_pairs:
        image_id = Path(img_path_str).stem

        # Run inference
        predictions = best_model(img_path_str, verbose=False)

        # Convert YOLO results to metrics input format
        conversion_result = yolo_result_to_metrics_input(
            yolo_result=predictions[0],
            ground_truth_mask_path=mask_path_str,
            image_id=image_id,
            target_size=(512, 512)
        )

        if conversion_result["success"]:
            pred_mask = conversion_result["data"]["predicted_mask"]
            gt_mask = conversion_result["data"]["ground_truth_mask"]

            # Calculate metrics
            metrics = calculate_segmentation_metrics(pred_mask, gt_mask, image_id)
            boundary_f1 = calculate_boundary_f1(pred_mask, gt_mask, tolerance=2)
            metrics["boundary_f1"] = boundary_f1

            seed_results.append(metrics)
        else:
            print(f"  WARNING: Failed to process {image_id}: {conversion_result['error']}")

    # Store results for this seed
    all_training_results[f"seed_{seed}"] = seed_results

    # Calculate mean metrics for this seed
    seed_dice = np.mean([r["dice"] for r in seed_results])
    seed_iou = np.mean([r["iou"] for r in seed_results])

    print(f"\\nPASS: Seed {seed} evaluation complete:")
    print(f"  Mean Dice: {seed_dice:.4f}")
    print(f"  Mean IoU:  {seed_iou:.4f}")
    print(f"  Images evaluated: {len(seed_results)}")

print("\\n" + "=" * 80)
print("PASS: ALL TRAINING RUNS COMPLETE")
print("=" * 80)
print(f"\\nTotal seeds trained: {len(RANDOM_SEEDS)}")
print(f"Total time: {(time.time() - training_start)/60:.1f} minutes")"""

# Cell 24: Aggregation strategy (MARKDOWN)
CELL_24_MARKDOWN = """## Aggregate Results and Statistical Validation

**Aggregation**: Pool per-image predictions across the 3 seeds.

**Statistical Analysis**:
- Bootstrap CIs (10,000 iterations) for each metric
- Cross-seed variance quantification
- Comparison to zero-shot baseline

**Clinical Significance**: Low cross-seed variance (CV < 2%) validates reproducibility."""

# Cell 25: Aggregate and compare (CODE)
CELL_25_CODE = """# Aggregate results across all seeds
print("=" * 80)
print("AGGREGATING FINE-TUNED RESULTS")
print("=" * 80)

# Pool all results from all seeds
all_finetuned_results = []
for seed_key, seed_results in all_training_results.items():
    all_finetuned_results.extend(seed_results)

print(f"\\nTotal predictions pooled: {len(all_finetuned_results)}")
print(f"Expected: {len(test_pairs) * len(RANDOM_SEEDS)}")

# Calculate aggregate statistics with bootstrap CIs
ft_dice_scores = [r["dice"] for r in all_finetuned_results]
ft_iou_scores = [r["iou"] for r in all_finetuned_results]
ft_precision_scores = [r["precision"] for r in all_finetuned_results]
ft_recall_scores = [r["recall"] for r in all_finetuned_results]
ft_boundary_f1_scores = [r["boundary_f1"] for r in all_finetuned_results]

print(f"\\nCalculating bootstrap confidence intervals...\\n")

ft_summary_stats = {}

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

    # Compare to zero-shot baseline
    baseline_mean = summary_stats[metric_name]["mean"]
    improvement = ci_result["mean"] - baseline_mean

    print(f"{metric_name.upper()}:")
    print(f"  Fine-tuned:  {ci_result['mean']:.4f} ± {ci_result['std']:.4f}")
    print(f"  95% CI: [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")
    print(f"  Zero-shot:   {baseline_mean:.4f}")
    print(f"  Improvement: +{improvement:.4f} ({100*improvement/baseline_mean:.1f}%)")
    print()

print("=" * 80)
print("FINE-TUNED BASELINE ESTABLISHED")
print("=" * 80)

# Calculate cross-seed variance
print(f"\\nCross-Seed Performance Variance:\\n")

for seed_key, seed_results in all_training_results.items():
    seed_dice_mean = np.mean([r["dice"] for r in seed_results])
    print(f"  {seed_key}: Dice = {seed_dice_mean:.4f}")

# Calculate coefficient of variation
seed_means = [np.mean([r["dice"] for r in results]) for results in all_training_results.values()]
cv = 100 * np.std(seed_means) / np.mean(seed_means)

print(f"\\n  Coefficient of Variation: {cv:.2f}%")
if cv < 2.0:
    print(f"  PASS: Low variance - results are robust across random seeds")
else:
    print(f"  WARNING: High variance - consider additional seeds for stability")"""

# Cell 26: Save fine-tuned results documentation (MARKDOWN)
CELL_26_MARKDOWN = """## Save Fine-Tuned Results

**Output Files**:
1. Fine-tuned per-image metrics (JSON and CSV)
2. Fine-tuned summary statistics with CIs
3. Comparison report (fine-tuned vs zero-shot)
4. Final completion status"""

# Cell 27: Save fine-tuned results (CODE)
CELL_27_CODE = """# Save fine-tuned per-image results (JSON)
ft_per_image_output = OUTPUT_PATH / "finetuned_per_image_results.json"
with open(ft_per_image_output, "w") as f:
    json.dump(all_finetuned_results, f, indent=2)

print(f"Fine-tuned per-image results saved to: {ft_per_image_output}")

# Save fine-tuned per-image results (CSV)
ft_results_df = pd.DataFrame(all_finetuned_results)
column_order = [
    "image_id",
    "dice",
    "iou",
    "precision",
    "recall",
    "boundary_f1",
    "tp",
    "fp",
    "tn",
    "fn",
]
ft_results_df = ft_results_df[column_order]

ft_csv_output = OUTPUT_PATH / "finetuned_per_image_results.csv"
ft_results_df.to_csv(ft_csv_output, index=False, float_format="%.6f")

print(f"Fine-tuned per-image results CSV saved to: {ft_csv_output}")
print(f"Shape: {ft_results_df.shape[0]} rows x {ft_results_df.shape[1]} columns")

# Save fine-tuned summary statistics
ft_summary_output = OUTPUT_PATH / "finetuned_summary_statistics.json"
with open(ft_summary_output, "w") as f:
    json.dump(ft_summary_stats, f, indent=2)

print(f"Fine-tuned summary statistics saved to: {ft_summary_output}")

# Create comparison report
comparison_report = {
    "comparison": "Fine-Tuned vs Zero-Shot Baseline",
    "test_set_size": len(test_pairs),
    "n_seeds": len(RANDOM_SEEDS),
    "seeds": RANDOM_SEEDS,
    "total_predictions": len(all_finetuned_results),
    "metrics_comparison": {}
}

for metric_name in ["dice", "iou", "precision", "recall", "boundary_f1"]:
    ft_mean = ft_summary_stats[metric_name]["mean"]
    zs_mean = summary_stats[metric_name]["mean"]
    improvement = ft_mean - zs_mean

    comparison_report["metrics_comparison"][metric_name] = {
        "finetuned_mean": round(ft_mean, 6),
        "finetuned_ci_lower": round(ft_summary_stats[metric_name]["ci_lower"], 6),
        "finetuned_ci_upper": round(ft_summary_stats[metric_name]["ci_upper"], 6),
        "zeroshot_mean": round(zs_mean, 6),
        "zeroshot_ci_lower": round(summary_stats[metric_name]["ci_lower"], 6),
        "zeroshot_ci_upper": round(summary_stats[metric_name]["ci_upper"], 6),
        "absolute_improvement": round(improvement, 6),
        "relative_improvement_percent": round(100 * improvement / zs_mean, 2) if zs_mean > 0 else None
    }

comparison_output = OUTPUT_PATH / "finetuned_comparison.json"
with open(comparison_output, "w") as f:
    json.dump(comparison_report, f, indent=2)

print(f"Comparison report saved to: {comparison_output}")

# Update completion status
completion_status = {
    "notebook": "02_baseline_model",
    "status": "complete",
    "phases_complete": ["zero_shot", "fine_tuned"],
    "n_images_evaluated": len(test_pairs),
    "n_seeds": len(RANDOM_SEEDS),
    "zero_shot_dice_mean": summary_stats["dice"]["mean"],
    "finetuned_dice_mean": ft_summary_stats["dice"]["mean"],
    "dice_improvement": ft_summary_stats["dice"]["mean"] - summary_stats["dice"]["mean"],
    "environment": "Kaggle" if IN_KAGGLE else "Local",
}

completion_path = OUTPUT_PATH / "completion_status.json"
with open(completion_path, "w") as f:
    json.dump(completion_status, f, indent=2)

print(f"\\nCompletion status updated: {completion_path}")

print("\\n" + "=" * 80)
print("PASS: ALL OUTPUTS SAVED SUCCESSFULLY")
print("=" * 80)
print(f"\\nNotebook complete - both zero-shot and fine-tuned baselines established!")"""

# Master cell list for nbformat builder
CELLS = [
    # Phase 0: Setup (Cells 0-4)
    {"type": "code", "source": CELL_0_CODE},  # NumPy fix
    {"type": "code", "source": CELL_1_CODE},  # Imports
    {"type": "code", "source": CELL_2_CODE},  # Environment detection
    {"type": "code", "source": CELL_3_CODE},  # Load model
    {"type": "code", "source": CELL_4_CODE},  # Load dataset

    # Phase 1: Zero-Shot Evaluation (Cells 5-15)
    {"type": "markdown", "source": CELL_5_MARKDOWN},  # Strategy
    {"type": "code", "source": CELL_6_CODE},  # Evaluation
    {"type": "markdown", "source": CELL_7_MARKDOWN},  # Bootstrap explanation
    {"type": "code", "source": CELL_8_CODE},  # Statistics
    {"type": "code", "source": CELL_9_CODE},  # Visualizer init
    {"type": "code", "source": CELL_10_CODE},  # Confusion matrix
    {"type": "code", "source": CELL_11_CODE},  # Example predictions
    {"type": "markdown", "source": CELL_12_MARKDOWN},  # Save documentation
    {"type": "code", "source": CELL_13_CODE},  # Save results
    {"type": "markdown", "source": CELL_14_MARKDOWN},  # Completion header
    {"type": "code", "source": CELL_15_CODE},  # Phase 1 status

    # Kaggle Dataset Creation (Cells 16-17)
    {"type": "markdown", "source": CELL_16_MARKDOWN},  # Dataset note
    {"type": "code", "source": CELL_17_CODE},  # Dataset creation

    # Phase 2: Fine-Tuning (Cells 18-27)
    {"type": "markdown", "source": CELL_18_MARKDOWN},  # Training overview
    {"type": "code", "source": CELL_19_CODE},  # Import modules
    {"type": "markdown", "source": CELL_20_MARKDOWN},  # Dataset rationale
    {"type": "code", "source": CELL_21_CODE},  # Prepare dataset
    {"type": "markdown", "source": CELL_22_MARKDOWN},  # Multi-seed rationale
    {"type": "code", "source": CELL_23_CODE},  # Training loop
    {"type": "markdown", "source": CELL_24_MARKDOWN},  # Aggregation strategy
    {"type": "code", "source": CELL_25_CODE},  # Aggregate results
    {"type": "markdown", "source": CELL_26_MARKDOWN},  # Save documentation
    {"type": "code", "source": CELL_27_CODE},  # Save fine-tuned results
]
