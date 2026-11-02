"""
Build Ablation 1 Production Notebook

Creates ablation_1_focal_tversky.ipynb by:
1. Starting from baseline notebook structure
2. Inserting FocalTverskyTrainer before training
3. Updating metadata and outputs for Ablation 1
"""

import nbformat as nbf
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_NB_PATH = PROJECT_ROOT / "notebooks/02_baseline/02_baseline_model.ipynb"
OUTPUT_NB_PATH = PROJECT_ROOT / "notebooks/03_ablation_1/ablation_1_focal_tversky.ipynb"

# Read baseline notebook
with open(BASELINE_NB_PATH) as f:
    baseline_nb = nbf.read(f, as_version=4)

# Create new notebook
nb = nbf.v4.new_notebook()
nb.metadata = baseline_nb.metadata.copy()

# Cell specifications for Ablation 1
cells = []

# Keep all cells from baseline up to multi-seed training
# Find the index of the multi-seed training cell
multiseed_idx = None
for idx, cell in enumerate(baseline_nb.cells):
    if "# Multi-seed training configuration" in cell.source:
        multiseed_idx = idx
        break

if multiseed_idx is None:
    raise ValueError("Could not find multi-seed training cell in baseline notebook")

# Add all cells BEFORE multi-seed training
cells.extend(baseline_nb.cells[:multiseed_idx])

# INSERT: FocalTverskyTrainer integration (NEW CELLS)
cells.append(nbf.v4.new_markdown_cell("""---

# Ablation 1: Focal Tversky Loss Integration

**Research Question**: Open: does asymmetric Focal Tversky Loss improve boundary precision beyond 80% Dice.

**Hypothesis**: α=0.7 (FN weight) and β=0.3 (FP weight) will improve performance by 3-5% Dice over baseline.

**Implementation**: Custom YOLOv8 SegmentationTrainer that replaces default loss function.

**Expected Outcome**: Dice 83-85% (baseline: 80.0%)

---"""))

cells.append(nbf.v4.new_code_cell("""# Import FocalTverskyLoss and custom trainer base
if os.path.exists("/kaggle/input/wound-seg-code-v1"):
    from models.focal_tversky import FocalTverskyLoss
else:
    from src.models.focal_tversky import FocalTverskyLoss

from ultralytics.models.yolo.segment import SegmentationTrainer
import torch

print("PASS: Focal Tversky Loss imported successfully")
print(f"  α (FN weight): 0.7")
print(f"  β (FP weight): 0.3")
print(f"  γ (focal parameter): 2.0")"""))

cells.append(nbf.v4.new_code_cell("""# Define custom trainer with Focal Tversky Loss
class FocalTverskyTrainer(SegmentationTrainer):
    \"\"\"
    Custom YOLOv8 Segmentation Trainer with Focal Tversky Loss.

    Replaces YOLOv8's default segmentation loss (BCE + Dice) with
    Focal Tversky Loss optimized for class imbalance (96:4 background:wound).
    \"\"\"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize Focal Tversky Loss
        self.focal_tversky_loss = FocalTverskyLoss(
            alpha=0.7,   # Emphasize missed wounds (FN)
            beta=0.3,    # De-emphasize false alarms (FP)
            gamma=2.0,   # Focus on hard examples
            smooth=1e-6
        )

        print("FocalTverskyTrainer initialized with custom loss function")

    def criterion(self, preds, batch):
        \"\"\"
        Override YOLOv8's criterion() to use Focal Tversky Loss for segmentation.

        Args:
            preds: Model predictions (dict with 'seg' key for segmentation masks)
            batch: Ground truth batch (dict with 'masks' key)

        Returns:
            loss: Total loss (box + cls + seg)
        \"\"\"
        # Get base loss (box + cls) from parent
        loss = super().criterion(preds, batch)

        # Replace segmentation loss with Focal Tversky
        if 'seg' in preds and 'masks' in batch:
            pred_masks = preds['seg']
            target_masks = batch['masks'].float()

            # Ensure predictions are in [0, 1] range (apply sigmoid if needed)
            if pred_masks.min() < 0 or pred_masks.max() > 1:
                pred_masks = torch.sigmoid(pred_masks)

            # Calculate Focal Tversky Loss
            seg_loss = self.focal_tversky_loss(pred_masks, target_masks)

            # Replace YOLOv8's default segmentation loss
            # loss[0] = box loss, loss[1] = cls loss, loss[2] = seg loss
            loss[2] = seg_loss

        return loss

print("PASS: FocalTverskyTrainer defined successfully")
print("  Overrides: criterion() method")
print("  Replaces: Default segmentation loss (BCE+Dice) with Focal Tversky")"""))

# Now add the modified multi-seed training cell
modified_training_cell = nbf.v4.new_code_cell("""# Multi-seed training configuration (P100-Optimized for Experimental Rigor)
RANDOM_SEEDS = [42, 123, 456]

# P100-Optimized Hyperparameters (Use same values for all ablations)
BATCH_SIZE = 32      # 2x speedup, REDUCES gradient noise (better ablation sensitivity)
LEARNING_RATE = 0.015  # Conservative 1.5x scaling for batch=32 (not full 2x)
WORKERS = 8          # Maximize P100 throughput (prevent CPU bottleneck)
AMP = False          # Disabled for ablations (avoid FP16/Focal Tversky interactions)

# Dataset-Aligned Parameters
IMAGE_SIZE = 512     # Match preprocessing
EPOCHS = 100         # Full convergence for fair ablation comparison
PATIENCE = 10        # P100-optimized early stopping (expect stop at 40-60 epochs)

# Storage for all training results
all_training_results = {}

print("=" * 80)
print("STARTING ABLATION 1: FOCAL TVERSKY LOSS (P100-OPTIMIZED)")
print("=" * 80)
print(f"\\nTraining configuration:")
print(f"  Seeds: {RANDOM_SEEDS}")
print(f"  Batch size: {BATCH_SIZE} (P100-optimized, 2x faster)")
print(f"  Learning rate: {LEARNING_RATE} (scaled 1.5x for batch=32)")
print(f"  Workers: {WORKERS} (maximize throughput)")
print(f"  AMP: {AMP} (disabled for ablation safety)")
print(f"  Epochs: {EPOCHS}")
print(f"  Patience: {PATIENCE}")
print(f"  Image size: {IMAGE_SIZE}")
print(f"\\nDataset config: {config_path}")
print(f"\\nEstimated time: ~1.5-2 hours total (~30-40 min per seed on P100)\\n")

for seed_idx, seed in enumerate(RANDOM_SEEDS, 1):
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

        # Reproducibility
        seed=seed,
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
            pred_mask = conversion_result["data"]["pred_mask"]
            gt_mask = conversion_result["data"]["gt_mask"]

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
print(f"Total time: {(time.time() - training_start)/60:.1f} minutes")""")

cells.append(modified_training_cell)

# Add remaining cells from baseline (aggregation, saving, etc.)
# But skip the Kaggle dataset creation cell and modify completion status
for idx in range(multiseed_idx + 1, len(baseline_nb.cells)):
    cell = baseline_nb.cells[idx]

    # Skip Kaggle dataset creation cell
    if "Kaggle Dataset Creation" in cell.source:
        continue

    # Modify completion status cell
    if "completion_status" in cell.source and "notebook" in cell.source:
        modified_cell = nbf.v4.new_code_cell("""# Ablation 1 completion status
completion_status = {
    "notebook": "ablation_1_focal_tversky",
    "ablation": "focal_tversky_loss",
    "status": "complete",
    "phases_complete": ["training", "evaluation"],
    "n_images_evaluated": len(test_pairs),
    "n_seeds": len(RANDOM_SEEDS),
    "seeds": RANDOM_SEEDS,
    "ablation_dice_mean": ft_summary_stats["dice"]["mean"],
    "baseline_dice_mean": 0.8000,  # From baseline-outputs dataset
    "dice_improvement_vs_baseline": ft_summary_stats["dice"]["mean"] - 0.8000,
    "environment": "Kaggle" if IN_KAGGLE else "Local",
}

completion_path = OUTPUT_PATH / "completion_status.json"
with open(completion_path, "w") as f:
    json.dump(completion_status, f, indent=2)

print(f"\\nCompletion status updated: {completion_path}")

print("\\n" + "=" * 80)
print("PASS: ABLATION 1 COMPLETE: FOCAL TVERSKY LOSS")
print("=" * 80)
print(f"\\nResults:")
print(f"  Dice Score: {ft_summary_stats['dice']['mean']:.4f}")
print(f"  Baseline:   0.8000")
print(f"  Improvement: +{ft_summary_stats['dice']['mean'] - 0.8000:.4f}")
print(f"\\nAll outputs saved to: {OUTPUT_PATH}")
print("=" * 80 + "\\n")""")
        cells.append(modified_cell)
    else:
        # Keep other cells as-is
        cells.append(cell)

# Set cells
nb.cells = cells

# Normalize and validate
nb = nbf.validator.normalize(nb)[1]
nbf.validate(nb)

# Write notebook
OUTPUT_NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_NB_PATH, 'w') as f:
    nbf.write(nb, f)

print(f"PASS: Ablation 1 production notebook created: {OUTPUT_NB_PATH}")
print(f"  Total cells: {len(nb.cells)}")
print(f"  Based on: {BASELINE_NB_PATH.name}")
print(f"  New cells: FocalTverskyTrainer integration (3 cells)")
