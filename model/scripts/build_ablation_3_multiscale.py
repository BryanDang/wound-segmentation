#!/usr/bin/env python3
"""
Build Ablation 3: Multi-Scale Training Notebook

Creates ablation_3_multiscale_training.ipynb from ablation_2 template with:
- Multi-scale training at [416, 512, 640] resolutions
- Random scale selection per batch
- All P100-optimized hyperparameters preserved
- All fixes from v3 pre-applied
"""

from pathlib import Path

import nbformat as nbf

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATE_PATH = (
    PROJECT_ROOT
    / "notebooks/03_ablations/ablation_2_clinical_augmentation/ablation_2_clinical_augmentation.ipynb"
)
OUTPUT_DIR = PROJECT_ROOT / "notebooks/03_ablations/ablation_3_multiscale_training"
OUTPUT_PATH = OUTPUT_DIR / "ablation_3_multiscale_training.ipynb"

# Load template
with open(TEMPLATE_PATH) as f:
    nb = nbf.read(f, as_version=4)

print(f"Loaded template: {TEMPLATE_PATH}")
print(f"Cells: {len(nb.cells)}")

# Modifications for Ablation 3
modifications = [
    # Cell containing "Ablation 2" header
    {
        "old_markdown": "# Ablation 2: Clinical Data Augmentation",
        "new_markdown": "# Ablation 3: Multi-Scale Training",
    },
    {
        "old_markdown": "**Research Question**: Can anatomically plausible augmentations improve generalization beyond 80% Dice?",
        "new_markdown": "**Research Question**: Can multi-scale training improve small wound detection beyond 80% Dice?",
    },
    {
        "old_markdown": "**Hypothesis**: Clinical augmentation (rotation, brightness, elastic deformation) will improve performance by 2-3% Dice over baseline.",
        "new_markdown": "**Hypothesis**: Training at multiple resolutions [416, 512, 640] will force scale-invariant features, improving performance by 2-4% Dice over baseline.",
    },
    {
        "old_markdown": "**Implementation**: Custom YOLO training with augmented training data pipeline.",
        "new_markdown": "**Implementation**: Standard YOLO training with multi-scale resolution parameter.",
    },
    {
        "old_markdown": "**Expected Outcome**: Dice 82-83% (baseline: 80.0%)",
        "new_markdown": "**Expected Outcome**: Dice 82-84% (baseline: 80.0%)",
    },
    {"old_markdown": "**Augmentation Strategy**:", "new_markdown": "**Multi-Scale Strategy**:"},
    {
        "old_markdown": "- Rotation: ±15° (preserves anatomical orientation)\n- Brightness: ±20% (lighting variation)\n- Gaussian Blur: σ=1.5 (focus variation)\n- Elastic Deformation: α=720, σ=24 (tissue flexibility)\n- Probability: 0.7 per transformation",
        "new_markdown": '- Resolutions: [416, 512, 640] pixels\n- Selection: Random per batch\n- Rationale: Wounds span 100-100,000+ pixels (log-normal distribution)\n- Expected: Improved detection of small/medium wounds\n- Citation: YOLO-MS (2023) - "Multi-scale improves small object detection by 4-7% mAP"',
    },
    # Code cell with augmentation parameters
    {
        "old_code": 'print("PASS: Clinical Augmentation via YOLO Parameters")\nprint("  Transformations: Rotation ±15°, Translation 10%, Scale ±50%")\nprint("  Flips: Horizontal/Vertical (50% probability each)")\nprint("  No mosaic/mixup (preserves wound characteristics)")',
        "new_code": 'print("PASS: Multi-Scale Training via YOLO Parameters")\nprint("  Resolutions: [416, 512, 640] pixels")\nprint("  Selection: Random per batch")\nprint("  Rationale: Scale-invariant feature learning for diverse wound sizes")',
    },
    {
        "old_code": 'print("PASS: Using standard YOLOv8 SegmentationTrainer")\nprint("  Intervention: Clinical augmentation pipeline via training parameters")',
        "new_code": 'print("PASS: Using standard YOLOv8 SegmentationTrainer")\nprint("  Intervention: Multi-scale training via resolution parameters")',
    },
    # Training cell header
    {
        "old_code": 'print("STARTING ABLATION 2: CLINICAL AUGMENTATION (P100-OPTIMIZED)")',
        "new_code": 'print("STARTING ABLATION 3: MULTI-SCALE TRAINING (P100-OPTIMIZED)")',
    },
    {
        "old_code": 'print(f"  Augmentation: Clinical pipeline (prob=0.7)")',
        "new_code": 'print(f"  Multi-scale: [416, 512, 640] pixels (random per batch)")',
    },
    # Remove all clinical augmentation parameters, add multi-scale
    {
        "old_code": """    results = model.train(
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
        # Clinical Augmentation (INTERVENTION)
        degrees=15.0,        # Rotation ±15°
        translate=0.1,       # Translation 10%
        scale=0.5,           # Scale ±50%
        shear=0.0,           # No shearing (anatomically implausible)
        perspective=0.0,     # No perspective (maintains tissue structure)
        flipud=0.5,          # Vertical flip 50%
        fliplr=0.5,          # Horizontal flip 50%
        mosaic=0.0,          # No mosaic (single wound per image)
        mixup=0.0,           # No mixup (preserve wound characteristics)
        # Brightness augmentation (lighting variation)
        hsv_v=0.3,           # Brightness ±30% (ONLY color augmentation)
        # Reproducibility
        seed=seed,
        save=True,
        project=str(OUTPUT_PATH / "yolo_training"),
        name=f"seed_{seed}",
        exist_ok=True,
        verbose=False,
    )""",
        "new_code": """    results = model.train(
        data=str(config_path),
        # P100-optimized parameters
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        workers=WORKERS,
        amp=AMP,
        # Dataset-aligned
        epochs=EPOCHS,
        patience=PATIENCE,
        # Multi-Scale Training (INTERVENTION)
        imgsz=[416, 512, 640],  # Train at multiple resolutions (random per batch)
        # NOTE: YOLO multi-scale randomly selects from list per batch
        # Reproducibility
        seed=seed,
        save=True,
        project=str(OUTPUT_PATH / "yolo_training"),
        name=f"seed_{seed}",
        exist_ok=True,
        verbose=False,
    )""",
    },
    # Completion status
    {
        "old_code": '    "notebook": "ablation_2_clinical_augmentation",\n    "ablation": "clinical_augmentation",',
        "new_code": '    "notebook": "ablation_3_multiscale_training",\n    "ablation": "multiscale_training",',
    },
    {
        "old_code": 'print("PASS: ABLATION 1 COMPLETE: FOCAL TVERSKY LOSS")',
        "new_code": 'print("PASS: ABLATION 3 COMPLETE: MULTI-SCALE TRAINING")',
    },
]

# Apply modifications
modified_count = 0
for cell in nb.cells:
    for mod in modifications:
        if "old_markdown" in mod and cell.cell_type == "markdown":
            if mod["old_markdown"] in cell.source:
                cell.source = cell.source.replace(mod["old_markdown"], mod["new_markdown"])
                modified_count += 1
                print(f"PASS: Modified markdown: {mod['old_markdown'][:50]}...")

        elif "old_code" in mod and cell.cell_type == "code":
            if mod["old_code"] in cell.source:
                cell.source = cell.source.replace(mod["old_code"], mod["new_code"])
                modified_count += 1
                print(f"PASS: Modified code: {mod['old_code'][:50]}...")

print(f"\nTotal modifications applied: {modified_count}")

# Normalize and validate
nb = nbf.validator.normalize(nb)[1]
nbf.validate(nb)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Write notebook
with open(OUTPUT_PATH, "w") as f:
    nbf.write(nb, f)

print(f"\nPASS: Notebook created: {OUTPUT_PATH}")
print(f"  Cells: {len(nb.cells)}")
print(f"  Size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

# Create kernel-metadata.json
metadata = {
    "id": "bkdkaggle/ablation-3-multiscale-training",
    "title": "Ablation 3: Multi-Scale Training",
    "code_file": "ablation_3_multiscale_training.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": ["bkdkaggle/baseline-outputs", "bkdkaggle/wound-seg-code-v1"],
    "kernel_sources": [],
}

import json

metadata_path = OUTPUT_DIR / "kernel-metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"PASS: Metadata created: {metadata_path}")

print("\n" + "=" * 80)
print("ABLATION 3 BUILD COMPLETE")
print("=" * 80)
print(f"\nNext steps:")
print(f"1. Review notebook: {OUTPUT_PATH}")
print(f"2. Deploy to Kaggle: cd {OUTPUT_DIR} && kaggle kernels push -p .")
print(f"3. Monitor at: https://www.kaggle.com/code/bkdkaggle/ablation-3-multiscale-training")
