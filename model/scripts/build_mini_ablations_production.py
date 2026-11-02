#!/usr/bin/env python3
"""
Build Mini-Ablations Production Notebook (GPU)

Runs all 7 mini-ablation configurations sequentially with full 100-epoch training
to isolate which augmentation parameters drive baseline's 80% performance.
"""

import nbformat as nbf
from pathlib import Path
import json


# All 7 mini-ablation configurations
MINI_ABLATIONS = {
    "1_no_mosaic": {
        "name": "No Mosaic",
        "hypothesis": "Mosaic provides critical spatial diversity",
        "expected": "70-75% Dice (significant drop)",
        "params": {"mosaic": 0.0}
    },
    "2_no_color": {
        "name": "No Color Augmentations",
        "hypothesis": "Color augmentations alter diagnostic features",
        "expected": "78-82% Dice",
        "params": {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "erasing": 0.0, "auto_augment": None}
    },
    "3_no_scale": {
        "name": "No Scale",
        "hypothesis": "Scale redundant with mosaic",
        "expected": "78-82% Dice",
        "params": {"scale": 0.0}
    },
    "4_minimal": {
        "name": "Minimal (Mosaic+Flip Only)",
        "hypothesis": "Only essential augmentations needed",
        "expected": "75-80% Dice",
        "params": {"degrees": 0.0, "translate": 0.0, "scale": 0.0, "hsv_h": 0.0, "hsv_s": 0.0,
                   "hsv_v": 0.0, "erasing": 0.0, "auto_augment": None}
    },
    "5_no_translate": {
        "name": "No Translate",
        "hypothesis": "Translate redundant with mosaic",
        "expected": "78-82% Dice",
        "params": {"translate": 0.0}
    },
    "6_clinical_conservative": {
        "name": "Clinical Conservative",
        "hypothesis": "Reduced safe augmentations match baseline",
        "expected": "78-82% Dice",
        "params": {"degrees": 0.0, "scale": 0.3, "hsv_h": 0.0, "hsv_s": 0.0,
                   "hsv_v": 0.2, "erasing": 0.0, "auto_augment": None}
    },
    "7_no_randaugment": {
        "name": "No RandAugment",
        "hypothesis": "Unpredictable transforms hurt clinical color",
        "expected": "78-82% Dice (similar or better)",
        "params": {"auto_augment": None}
    }
}

# Baseline augmentation parameters (YOLO defaults)
BASELINE_PARAMS = {
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "bgr": 0.0,
    "erasing": 0.4,
    "auto_augment": "randaugment",
    "close_mosaic": 10
}


def create_code_cell(source):
    """Helper to create code cells."""
    return nbf.v4.new_code_cell(source)


def create_markdown_cell(source):
    """Helper to create markdown cells."""
    return nbf.v4.new_markdown_cell(source)


def create_mini_ablations_notebook():
    """Create consolidated mini-ablations production notebook."""

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
    cells.append(create_code_cell("""# Lock NumPy to prevent 2.x upgrade
import sys
print("Step 1: Record Kaggle's default NumPy...")
import numpy as np
default_numpy = np.__version__
print(f"   Default: {default_numpy}")

print("\\nStep 2: Install ultralytics...")
import subprocess
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)

print("\\nStep 3: Lock NumPy...")
subprocess.run(["pip", "install", "-q", "--force-reinstall", f"numpy=={default_numpy}"], check=True)

print("\\nPASS: Installation complete")
import numpy as np
from ultralytics import YOLO
print(f"   NumPy: {np.__version__}")"""))

    # Cell 1: Imports
    cells.append(create_code_cell("""import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Environment setup
if os.path.exists("/kaggle/input/wound-seg-code-v1"):
    sys.path.insert(0, "/kaggle/input/wound-seg-code-v1")
    print("[INFO] Using Kaggle code dataset")
    from evaluation.yolo_utils import yolo_result_to_metrics_input
    from evaluation.metrics import calculate_segmentation_metrics
    from data.yolo_config import create_yolo_config
else:
    sys.path.insert(0, str(Path.cwd().parent))
    print("[INFO] Using local src")
    from src.evaluation.yolo_utils import yolo_result_to_metrics_input
    from src.evaluation.metrics import calculate_segmentation_metrics
    from src.data.yolo_config import create_yolo_config

print("PASS: Imports complete")"""))

    # Cell 2: Paths
    cells.append(create_code_cell("""# Configure paths
if os.path.exists("/kaggle/input"):
    # Use baseline-outputs for training (has pre-converted YOLO labels)
    BASE_INPUT_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
    # Use wound-segmentation-images for evaluation (has PNG masks)
    EVAL_INPUT_PATH = Path("/kaggle/input/wound-segmentation-images/data_wound_seg")
    OUTPUT_PATH = Path("/kaggle/working")
else:
    BASE_INPUT_PATH = Path.cwd().parent / "data" / "wound_segmentation"
    EVAL_INPUT_PATH = BASE_INPUT_PATH  # Same for local
    OUTPUT_PATH = Path.cwd().parent / "outputs" / "mini_ablations"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

TEST_IMAGES_DIR = EVAL_INPUT_PATH / "test_images"
TEST_MASKS_DIR = EVAL_INPUT_PATH / "test_masks"

print(f"Training data: {BASE_INPUT_PATH}")
print(f"Evaluation data: {EVAL_INPUT_PATH}")
print(f"Test images: {TEST_IMAGES_DIR}")
print(f"Test masks: {TEST_MASKS_DIR}")
print(f"Output: {OUTPUT_PATH}")

if not TEST_IMAGES_DIR.exists():
    raise FileNotFoundError(f"Test images not found: {TEST_IMAGES_DIR}")
if not TEST_MASKS_DIR.exists():
    raise FileNotFoundError(f"Test masks not found: {TEST_MASKS_DIR}")

print("PASS: Paths configured")"""))

    # Cell 3: YOLO config
    cells.append(create_code_cell("""# Create YOLO config
config_path = OUTPUT_PATH / "wound_dataset.yaml"
config_result = create_yolo_config(
    dataset_root=BASE_INPUT_PATH,
    output_path=config_path,
    train_subdir="train",
    val_subdir="test"
)

if not config_result["success"]:
    raise RuntimeError(f"Config failed: {config_result['error']}")

print(f"PASS: YOLO config: {config_path}")"""))

    # Cell 4: Mini-ablation definitions
    ablation_configs_str = json.dumps(MINI_ABLATIONS, indent=2)
    baseline_params_str = json.dumps(BASELINE_PARAMS, indent=2)

    # Cell 4 setup - define constants first before using in f-strings
    random_seeds = [42, 123, 456]
    n_configs = len(MINI_ABLATIONS)
    n_seeds = len(random_seeds)
    total_runs = n_configs * n_seeds

    cells.append(create_code_cell(f"""# Mini-ablation configurations
MINI_ABLATIONS = {ablation_configs_str}

BASELINE_PARAMS = {baseline_params_str}

# Training parameters
RANDOM_SEEDS = [42, 123, 456]
BATCH_SIZE = 32
LEARNING_RATE = 0.015
WORKERS = 8
AMP = False
IMAGE_SIZE = 512
EPOCHS = 100
PATIENCE = 10

print("=" * 80)
print("MINI-ABLATIONS PRODUCTION RUN")
print("=" * 80)
print(f"\\nConfigurations: {n_configs}")
print(f"Seeds per config: {n_seeds}")
print(f"Total training runs: {total_runs}")
print(f"\\nEstimated time: ~7-8 hours total on P100")
print(f"  (~1 hour per config × 7 configs)")
print("=" * 80)"""))

    # Cell 5: Main training loop
    cells.append(create_code_cell("""# Storage for all results
all_results = {}

# Loop through each mini-ablation config
for ablation_id, config in MINI_ABLATIONS.items():
    print("\\n" + "=" * 80)
    print(f"MINI-ABLATION: {config['name']} ({ablation_id})")
    print("=" * 80)
    print(f"Hypothesis: {config['hypothesis']}")
    print(f"Expected: {config['expected']}")
    print(f"Parameters: {config['params']}")

    ablation_start = time.time()
    ablation_results = {}

    # Train with 3 seeds for statistical robustness
    for seed in RANDOM_SEEDS:
        print(f"\\n--- Seed {seed} ---")

        # Merge baseline params with ablation-specific overrides
        augmentation_params = BASELINE_PARAMS.copy()
        augmentation_params.update(config['params'])

        # Initialize model
        model = YOLO("yolov8n-seg.pt")

        # Train
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
            seed=seed,
            save=True,
            project=str(OUTPUT_PATH / "yolo_training" / ablation_id),
            name=f"seed_{seed}",
            exist_ok=True,
            verbose=False,
            **augmentation_params
        )

        training_duration = time.time() - training_start
        print(f"Training: {training_duration/60:.1f} min")

        # Load best model
        best_model_path = OUTPUT_PATH / "yolo_training" / ablation_id / f"seed_{seed}" / "weights" / "best.pt"
        best_model = YOLO(str(best_model_path))

        # Prepare test pairs
        test_image_paths = sorted(TEST_IMAGES_DIR.glob("*.png"))
        test_pairs = []
        for img_path in test_image_paths:
            mask_path = TEST_MASKS_DIR / img_path.name
            if mask_path.exists():
                test_pairs.append((str(img_path), str(mask_path)))

        print(f"Evaluating {len(test_pairs)} images...")

        # Evaluate
        seed_results = []
        for img_path_str, mask_path_str in test_pairs:
            image_id = Path(img_path_str).stem

            img_rgb = cv2.imread(img_path_str)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

            yolo_results = best_model.predict(img_rgb, verbose=False)

            conversion_result = yolo_result_to_metrics_input(
                yolo_results[0],
                mask_path_str,
                image_id
            )

            if not conversion_result["success"]:
                print(f"WARNING: Skip {image_id}: {conversion_result['error']}")
                continue

            data = conversion_result["data"]
            metrics_result = calculate_segmentation_metrics(
                data["pred_mask"],
                data["gt_mask"],
                image_id
            )

            seed_results.append(metrics_result)

        # Calculate seed statistics
        seed_dice_scores = [r["dice_score"] for r in seed_results]
        seed_dice_mean = np.mean(seed_dice_scores)

        ablation_results[f"seed_{seed}"] = {
            "dice_mean": float(seed_dice_mean),
            "n_images": len(seed_results),
            "training_time_seconds": training_duration
        }

        print(f"PASS: Seed {seed}: Dice = {seed_dice_mean:.4f}")

    # Calculate ablation overall statistics
    all_dice = [r["dice_mean"] for r in ablation_results.values()]
    ablation_dice_mean = np.mean(all_dice)
    ablation_dice_std = np.std(all_dice)

    ablation_duration = time.time() - ablation_start

    all_results[ablation_id] = {
        "name": config["name"],
        "hypothesis": config["hypothesis"],
        "expected": config["expected"],
        "params": config["params"],
        "dice_mean": float(ablation_dice_mean),
        "dice_std": float(ablation_dice_std),
        "seed_results": ablation_results,
        "total_time_seconds": ablation_duration
    }

    print(f"\\nPASS: {config['name']}: Dice = {ablation_dice_mean:.4f} ± {ablation_dice_std:.4f}")
    print(f"   Duration: {ablation_duration/60:.1f} min")

print("\\n" + "=" * 80)
print("PASS: ALL MINI-ABLATIONS COMPLETE")
print("=" * 80)"""))

    # Cell 6: Results summary
    cells.append(create_code_cell("""# Summary table
print("\\n" + "=" * 80)
print("MINI-ABLATIONS RESULTS SUMMARY")
print("=" * 80)
print(f"\\nBaseline (full augmentation): 80.0% Dice\\n")

for ablation_id, result in all_results.items():
    dice = result["dice_mean"]
    std = result["dice_std"]
    diff = dice - 0.8000
    print(f"{ablation_id}: {result['name']}")
    print(f"  Dice: {dice:.4f} ± {std:.4f}")
    print(f"  Change: {diff:+.4f} ({diff*100:+.2f}%)")
    print(f"  Expected: {result['expected']}")
    print()

# Save results
results_path = OUTPUT_PATH / "mini_ablations_results.json"
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"PASS: Results saved: {results_path}")"""))

    # Cell 7: Completion status
    cells.append(create_code_cell("""# Completion status
completion_status = {
    "notebook": "mini_ablations_production",
    "status": "complete",
    "n_configs": len(MINI_ABLATIONS),
    "n_seeds_per_config": len(RANDOM_SEEDS),
    "baseline_dice": 0.8000,
    "results_summary": {
        ablation_id: {
            "name": result["name"],
            "dice_mean": result["dice_mean"],
            "dice_improvement_vs_baseline": result["dice_mean"] - 0.8000
        }
        for ablation_id, result in all_results.items()
    }
}

status_path = OUTPUT_PATH / "completion_status.json"
with open(status_path, 'w') as f:
    json.dump(completion_status, f, indent=2)

print(f"\\nPASS: Completion status: {status_path}")
print("\\n" + "=" * 80)
print("PASS: MINI-ABLATIONS PRODUCTION COMPLETE")
print("=" * 80)"""))

    # Add cells to notebook
    nb["cells"] = cells

    # Normalize and validate
    nb = nbf.validator.normalize(nb)[1]
    nbf.validate(nb)

    return nb


def main():
    print("Building Mini-Ablations Production Notebook...")

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "notebooks" / "03_ablations" / "mini_ablations_production"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create notebook
    nb = create_mini_ablations_notebook()

    # Write notebook
    notebook_path = output_dir / "mini_ablations_production.ipynb"
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

    # Create metadata
    metadata = {
        "id": "bkdkaggle/mini-ablations-production",
        "title": "Mini-Ablations Production (All 7 Configs)",
        "code_file": "mini_ablations_production.ipynb",
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
    print("Configuration:")
    print(f"  - 7 mini-ablation configs")
    print(f"  - 3 seeds per config (42, 123, 456)")
    print(f"  - 100 epochs, P100-optimized hyperparameters")
    print(f"  - Expected runtime: ~7-8 hours on P100 GPU")
    print()
    print("Next steps:")
    print(f"1. Deploy: cd {output_dir} && kaggle kernels push -p .")
    print("2. Monitor: kaggle kernels status bkdkaggle/mini-ablations-production")


if __name__ == "__main__":
    main()
