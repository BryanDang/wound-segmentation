"""
Build TTA Production Notebook for Kaggle deployment.

Evaluates Test-Time Augmentation on full test set (552 images) with 3 baseline seeds.
Expected runtime: ~20-30 minutes on P100.
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
    "id": "bkdkaggle/ablation-5-tta-production",
    "title": "Ablation 5: Test-Time Augmentation (Production)",
    "code_file": "05_tta_production.ipynb",
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

# Cell 0: NumPy locking
CELLS.append({
    "type": "code",
    "source": """# Lock NumPy to 1.x (prevent ultralytics from upgrading to 2.x)
import subprocess
import numpy as np

print(f"Default NumPy version: {np.__version__}")
default_numpy = np.__version__

subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall", f"numpy=={default_numpy}"],
    check=True
)

import numpy as np
print(f"Final NumPy version: {np.__version__}")
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
import pandas as pd
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

sys.path.append('/kaggle/input/wound-seg-code-v1')
from src.evaluation.tta import TestTimeAugmentation
from src.evaluation.metrics import calculate_segmentation_metrics

print("All imports successful")
"""
})

# Cell 2: Configuration
CELLS.append({
    "type": "markdown",
    "source": """## Configuration

Ablation 5: Test-Time Augmentation

**Hypothesis**: TTA ensemble improves robustness, achieving +1-2% Dice improvement.

**Configuration**:
- Augmentations: 8 (original, h/v flip, 4 rotations, 2 compositions)
- Aggregation: mean (threshold at 0.5)
- Models: All 3 baseline seeds (42, 123, 456)
- Test set: 552 images
"""
})

CELLS.append({
    "type": "code",
    "source": """ABLATION_CONFIG = {
    "name": "ablation_5_tta",
    "augmentation_set": [
        "original",
        "horizontal_flip",
        "vertical_flip",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "hflip_rotate_90",
        "hflip_rotate_270",
    ],
    "aggregation": "mean",
    "seeds": [42, 123, 456],
}

print("Ablation 5 Configuration:")
print(json.dumps(ABLATION_CONFIG, indent=2))
"""
})

# Cell 3: Paths
CELLS.append({
    "type": "code",
    "source": """# Dataset paths
BASE_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
TEST_IMAGES_DIR = BASE_PATH / "test_images"
TEST_MASKS_DIR = BASE_PATH / "test_masks"

# Model paths
MODEL_PATHS = {
    42: "/kaggle/input/baseline-outputs/weights/seed_42/best.pt",
    123: "/kaggle/input/baseline-outputs/weights/seed_123/best.pt",
    456: "/kaggle/input/baseline-outputs/weights/seed_456/best.pt",
}

# Output path
OUTPUT_PATH = Path("/kaggle/working/ablation_5")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"Test images: {TEST_IMAGES_DIR}")
print(f"Test masks: {TEST_MASKS_DIR}")
print(f"Output: {OUTPUT_PATH}")
"""
})

# Cell 4: Load test set
CELLS.append({
    "type": "code",
    "source": """# Load all test images
test_images = sorted(list(TEST_IMAGES_DIR.glob("*.png")))
test_masks = [TEST_MASKS_DIR / img.name for img in test_images]

print(f"Test set: {len(test_images)} images")
assert len(test_images) == 552, f"Expected 552 images, got {len(test_images)}"
"""
})

# Cell 5: Evaluate TTA for each seed
CELLS.append({
    "type": "markdown",
    "source": """## Evaluation Loop

For each seed (42, 123, 456):
1. Load baseline model
2. Evaluate baseline (single forward pass)
3. Evaluate TTA (8 augmentations)
4. Save per-image metrics
"""
})

CELLS.append({
    "type": "code",
    "source": """all_results = {}

for seed in ABLATION_CONFIG["seeds"]:
    print(f"\\n" + "="*60)
    print(f"SEED {seed}")
    print("="*60)

    # Create seed output directory
    seed_dir = OUTPUT_PATH / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(MODEL_PATHS[seed])
    print(f"Model loaded: {MODEL_PATHS[seed]}")

    # Initialize TTA
    tta = TestTimeAugmentation(
        model=model,
        augmentation_set=ABLATION_CONFIG["augmentation_set"],
        aggregation=ABLATION_CONFIG["aggregation"]
    )
    print(f"TTA initialized: {len(ABLATION_CONFIG['augmentation_set'])} augmentations")

    # Storage for results
    baseline_results = []
    tta_results = []
    baseline_times = []
    tta_times = []

    # Process all test images
    print(f"\\nProcessing {len(test_images)} test images...")

    for i, (img_path, mask_path) in enumerate(zip(test_images, test_masks)):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(test_images)} images...")

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

        # Calculate metrics
        baseline_metrics = calculate_segmentation_metrics(baseline_mask, gt_mask, img_path.stem)
        tta_metrics = calculate_segmentation_metrics(tta_mask, gt_mask, img_path.stem)

        baseline_results.append(baseline_metrics)
        tta_results.append(tta_metrics)

    # Save results
    baseline_df = pd.DataFrame(baseline_results)
    tta_df = pd.DataFrame(tta_results)

    baseline_df.to_csv(seed_dir / "per_image_metrics_baseline.csv", index=False)
    tta_df.to_csv(seed_dir / "per_image_metrics_tta.csv", index=False)

    # Summary statistics
    summary = {
        "seed": seed,
        "n_images": len(test_images),
        "baseline": {
            "dice_mean": float(baseline_df['dice'].mean()),
            "dice_std": float(baseline_df['dice'].std()),
            "precision_mean": float(baseline_df['precision'].mean()),
            "recall_mean": float(baseline_df['recall'].mean()),
            "boundary_f1_mean": float(baseline_df['boundary_f1'].mean()),
            "time_per_image_sec": float(np.mean(baseline_times)),
        },
        "tta": {
            "dice_mean": float(tta_df['dice'].mean()),
            "dice_std": float(tta_df['dice'].std()),
            "precision_mean": float(tta_df['precision'].mean()),
            "recall_mean": float(tta_df['recall'].mean()),
            "boundary_f1_mean": float(tta_df['boundary_f1'].mean()),
            "time_per_image_sec": float(np.mean(tta_times)),
        },
        "improvement": {
            "dice": float(tta_df['dice'].mean() - baseline_df['dice'].mean()),
            "precision": float(tta_df['precision'].mean() - baseline_df['precision'].mean()),
            "recall": float(tta_df['recall'].mean() - baseline_df['recall'].mean()),
            "boundary_f1": float(tta_df['boundary_f1'].mean() - baseline_df['boundary_f1'].mean()),
        }
    }

    with open(seed_dir / "summary_statistics.json", 'w') as f:
        json.dump(summary, f, indent=2)

    all_results[seed] = summary

    print(f"\\nSeed {seed} complete:")
    print(f"   Baseline Dice: {summary['baseline']['dice_mean']:.4f}")
    print(f"   TTA Dice:      {summary['tta']['dice_mean']:.4f}")
    print(f"   Improvement:   {summary['improvement']['dice']:+.4f}")
"""
})

# Cell 6: Aggregate results across seeds
CELLS.append({
    "type": "markdown",
    "source": """## Aggregate Results

Combine results from all 3 seeds for statistical analysis.
"""
})

CELLS.append({
    "type": "code",
    "source": """# Load all per-image results
all_baseline_dfs = []
all_tta_dfs = []

for seed in ABLATION_CONFIG["seeds"]:
    seed_dir = OUTPUT_PATH / f"seed_{seed}"

    baseline_df = pd.read_csv(seed_dir / "per_image_metrics_baseline.csv")
    baseline_df['seed'] = seed

    tta_df = pd.read_csv(seed_dir / "per_image_metrics_tta.csv")
    tta_df['seed'] = seed

    all_baseline_dfs.append(baseline_df)
    all_tta_dfs.append(tta_df)

# Combine
combined_baseline = pd.concat(all_baseline_dfs, ignore_index=True)
combined_tta = pd.concat(all_tta_dfs, ignore_index=True)

# Save combined results
combined_baseline.to_csv(OUTPUT_PATH / "all_baseline_results.csv", index=False)
combined_tta.to_csv(OUTPUT_PATH / "all_tta_results.csv", index=False)

print(f"Combined results: {len(combined_baseline)} rows (552 images × 3 seeds)")
"""
})

# Cell 7: Statistical testing
CELLS.append({
    "type": "markdown",
    "source": """## Statistical Testing

Wilcoxon signed-rank test (paired) to test if TTA significantly improves Dice.

**Null hypothesis**: TTA Dice = Baseline Dice
**Alternative**: TTA Dice > Baseline Dice
**Significance level**: α = 0.0056 (Bonferroni-corrected for 9 ablations)
"""
})

CELLS.append({
    "type": "code",
    "source": """# Wilcoxon signed-rank test (paired)
baseline_dice = combined_baseline['dice'].values
tta_dice = combined_tta['dice'].values

stat, p_value = wilcoxon(baseline_dice, tta_dice, alternative='less')  # baseline < tta

print("="*60)
print("STATISTICAL TESTING")
print("="*60)
print(f"Test: Wilcoxon signed-rank (paired)")
print(f"Null hypothesis: TTA Dice = Baseline Dice")
print(f"Alternative: TTA Dice > Baseline Dice")
print(f"")
print(f"Baseline Dice: {baseline_dice.mean():.4f} ± {baseline_dice.std():.4f}")
print(f"TTA Dice:      {tta_dice.mean():.4f} ± {tta_dice.std():.4f}")
print(f"Improvement:   {(tta_dice.mean() - baseline_dice.mean()):.4f}")
print(f"")
print(f"p-value: {p_value:.6f}")
print(f"Bonferroni-corrected α: 0.0056")
print(f"Significant: {'YES' if p_value < 0.0056 else 'NO'}")

# Cohen's d effect size
mean_diff = tta_dice.mean() - baseline_dice.mean()
pooled_std = np.sqrt((baseline_dice.std()**2 + tta_dice.std()**2) / 2)
cohens_d = mean_diff / pooled_std

print(f"")
print(f"Effect size (Cohen's d): {cohens_d:.3f}")
print(f"Interpretation: ", end="")
if cohens_d > 0.8:
    print("Large effect")
elif cohens_d > 0.5:
    print("Medium effect WARNING: ")
else:
    print("Small effect")

# Save statistical analysis
statistical_analysis = {
    "test": "wilcoxon_signed_rank",
    "n_samples": len(baseline_dice),
    "baseline_dice_mean": float(baseline_dice.mean()),
    "baseline_dice_std": float(baseline_dice.std()),
    "tta_dice_mean": float(tta_dice.mean()),
    "tta_dice_std": float(tta_dice.std()),
    "improvement": float(mean_diff),
    "p_value": float(p_value),
    "alpha_bonferroni": 0.0056,
    "significant": bool(p_value < 0.0056),
    "cohens_d": float(cohens_d),
    "effect_size_interpretation": "large" if cohens_d > 0.8 else ("medium" if cohens_d > 0.5 else "small")
}

with open(OUTPUT_PATH / "statistical_analysis.json", 'w') as f:
    json.dump(statistical_analysis, f, indent=2)
"""
})

# Cell 8: Visualizations
CELLS.append({
    "type": "markdown",
    "source": """## Visualizations

Compare baseline vs TTA performance.
"""
})

CELLS.append({
    "type": "code",
    "source": """# Box plots comparing baseline vs TTA
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Baseline vs TTA Performance (552 images × 3 seeds)', fontsize=14, fontweight='bold')

metrics_to_plot = ['dice', 'precision', 'recall', 'boundary_f1']
titles = ['Dice Score', 'Precision', 'Recall', 'Boundary F1']

for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
    data_to_plot = [
        combined_baseline[metric].values,
        combined_tta[metric].values
    ]

    bp = ax.boxplot(data_to_plot, labels=['Baseline', 'TTA'], patch_artist=True)

    # Color boxes
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')

    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)

    # Add mean values
    baseline_mean = combined_baseline[metric].mean()
    tta_mean = combined_tta[metric].mean()
    improvement = tta_mean - baseline_mean

    ax.text(0.5, 0.95, f'Baseline: {baseline_mean:.3f}\\nTTA: {tta_mean:.3f}\\nΔ: {improvement:+.3f}',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'tta_vs_baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved: tta_vs_baseline_comparison.png")
"""
})

# Cell 9: Completion status
CELLS.append({
    "type": "markdown",
    "source": """## Completion Status

Final report for Ablation 5 (TTA).
"""
})

CELLS.append({
    "type": "code",
    "source": """completion_status = {
    "ablation": "5_test_time_augmentation",
    "status": "COMPLETE",
    "timestamp": pd.Timestamp.now().isoformat(),
    "configuration": ABLATION_CONFIG,
    "results": {
        "baseline_dice_mean": float(baseline_dice.mean()),
        "tta_dice_mean": float(tta_dice.mean()),
        "improvement": float(tta_dice.mean() - baseline_dice.mean()),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.0056),
        "cohens_d": float(cohens_d),
    },
    "decision": "ADOPT" if (p_value < 0.0056 and cohens_d > 0.8 and mean_diff >= 0.015) else (
        "MARGINAL" if (p_value < 0.01 and cohens_d > 0.5 and mean_diff >= 0.005) else "REJECT"
    ),
    "outputs": {
        "per_image_metrics": "seed_*/per_image_metrics_*.csv",
        "summary_statistics": "seed_*/summary_statistics.json",
        "statistical_analysis": "statistical_analysis.json",
        "visualization": "tta_vs_baseline_comparison.png",
    }
}

with open(OUTPUT_PATH / "completion_status.json", 'w') as f:
    json.dump(completion_status, f, indent=2)

print("="*60)
print("ABLATION 5 COMPLETION STATUS")
print("="*60)
print(json.dumps(completion_status, indent=2))

print("\\n" + "="*60)
print("ABLATION 5 COMPLETE")
print("="*60)
"""
})


def create_notebook():
    """Create the notebook with all cells."""
    nb = nbf.v4.new_notebook()
    nb.metadata = NOTEBOOK_METADATA

    for cell_spec in CELLS:
        if cell_spec["type"] == "code":
            nb.cells.append(nbf.v4.new_code_cell(cell_spec["source"]))
        elif cell_spec["type"] == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(cell_spec["source"]))

    nb = nbf.validator.normalize(nb)[1]
    return nb


def main():
    """Build and save the notebook."""
    nb = create_notebook()
    nbf.validate(nb)
    print("Notebook validation passed")

    output_dir = Path(__file__).parent.parent / "notebooks" / "05_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook_path = output_dir / "05_tta_production.ipynb"
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook saved: {notebook_path}")

    import json
    metadata_path = output_dir / "kernel-metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(KERNEL_METADATA, f, indent=2)
    print(f"Metadata saved: {metadata_path}")

    print(f"\\nWARNING: WAIT for smoke test to pass before deploying!")
    print(f"\\nWhen ready:")
    print(f"   cd {output_dir}")
    print(f"   kaggle kernels push -p .")


if __name__ == "__main__":
    main()
