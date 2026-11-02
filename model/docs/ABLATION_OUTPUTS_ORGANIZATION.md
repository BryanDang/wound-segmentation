# Ablation Outputs: Kaggle Directory Structure

Reference. Last updated 2025-10-16.

## Standard layout

```
/kaggle/working/
├── checkpoints/              # model weights (3 seeds x 6.45MB)
├── training_logs/            # logs and CSV metrics
├── evaluation/               # per-image and aggregated test results
├── visualizations/           # plots and figures
└── completion_status.json    # execution metadata
```

## Files

checkpoints/
- `seed_{seed}_best.pt`: best checkpoint per seed, 6.45MB each.
- `seed_{seed}_epoch40.pt`: optional early checkpoint.

training_logs/
- `seed_{seed}_training.log`: full training output.
- `seed_{seed}_metrics.csv`: per-epoch epoch, train_loss, val_loss, val_dice.

evaluation/
- `seed_{seed}_per_image_results.csv`: 552 rows; image_id, dice_score, precision, recall, tp, fp, tn, fn.
- `aggregated_results.json`: multi-seed mean, std, CI.
- `statistical_analysis.json`: Wilcoxon test, Cohen's d.

visualizations/
- `training_curves.png`: loss and Dice over epochs.
- `confusion_matrix.png`: aggregated confusion matrix.
- `example_predictions.png`: best, median, worst predictions.
- `per_seed_comparison.png`: Dice bars with CI error bars.

completion_status.json:
```json
{
  "status": "complete",
  "ablation_name": "focal_tversky",
  "seeds_completed": [42, 123, 456],
  "total_training_time_hours": 1.8,
  "intervention_config": {...}
}
```

## Per-ablation additions

Ablation 1 (Focal Tversky):
- `intervention_config.json`: alpha, beta, gamma.
- `loss_comparison.png`: Focal Tversky vs Dice loss.

Ablation 2 (clinical augmentation):
- `augmentation_config.json`: rotation, flips, brightness.
- `augmented_samples.png`.

Ablation 3 (multi-scale training):
- `scale_config.json`: [416, 512, 640].
- `scale_analysis.json`: per-scale performance.
- `scale_distribution.png`: wound size histogram.

Ablation 4 (hard negative mining):
- `ddi_evaluation/`: DDI test results.
- `per_condition_fpr.json`: FPR by condition.
- `worst_false_positives.png`.

## Download

```bash
kaggle kernels output bkdkaggle/ablation-1-focal-tversky -p /tmp/ablation_1/

mkdir -p kaggle_outputs/ablation_1_focal_tversky/
cp -r /tmp/ablation_1/* kaggle_outputs/ablation_1_focal_tversky/

required_files=(
    "completion_status.json"
    "checkpoints/seed_42_best.pt"
    "evaluation/aggregated_results.json"
)
for file in "${required_files[@]}"; do
    [ -f "kaggle_outputs/ablation_1_focal_tversky/$file" ] && echo "FOUND: $file" || echo "MISSING: $file"
done
```

## Rules

- Prefix files with `seed_{seed}`.
- Write `completion_status.json`.
- Keep output under 100MB per ablation.
- Do not copy test images to /kaggle/working/ (500MB).
- No temporary or intermediate files.
- No spaces in filenames.

## Cross-ablation comparison

```python
import json
import pandas as pd
from pathlib import Path

comparison_data = []
for ablation_dir in Path("kaggle_outputs").glob("ablation_*"):
    with open(ablation_dir / "evaluation/aggregated_results.json") as f:
        results = json.load(f)
    with open(ablation_dir / "completion_status.json") as f:
        status = json.load(f)

    comparison_data.append({
        "ablation": status["ablation_name"],
        "dice_mean": results["dice"]["mean"],
        "dice_ci_95": results["dice"]["ci_95"],
        "training_time_hours": status["total_training_time_hours"]
    })

df = pd.DataFrame(comparison_data).sort_values("dice_mean", ascending=False)
print(df.to_markdown(index=False))
```
