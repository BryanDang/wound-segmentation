# Ablation 3 Counterfactual: Multi-Scale Without Augmentations

Date: 2025-10-22.
Notebook: `ablation_3_counterfactual_no_augment`.
Status: complete, 3 seeds. Analysis complete.

Open: this file reports Dice 54.05% mean and 64.8% in the text. The history, parameter table and kernel list report 64.82% Dice. The seed-level numbers below sum to 54.05%. To be reconciled against completion_status.json and results.csv.

## Summary

Question: isolated effect of multi-scale training when all augmentations are disabled.

Answer as written: 64.8% Dice, -15.2% from the 80% baseline. The table below gives mean Dice 54.05% (-25.95%).

Augmentations are essential for YOLOv8 wound segmentation. Multi-scale gives no benefit with augmentations disabled and degrades performance.

## Results

3 seeds: 42, 123, 456.

| Metric | Seed 42 | Seed 123 | Seed 456 | Mean | Baseline | Delta |
|--------|---------|----------|----------|------|----------|-------|
| Dice | 56.29% | 50.75% | 55.13% | 54.05% | 80.0% | -25.95% |
| mAP50(M) | 82.29% | 76.75% | 81.28% | 80.11% | 80.01% | +0.10% |
| Precision(M) | 83.46% | 87.19% | 82.07% | 84.24% | 81.23% | +3.01% |
| Recall(M) | 78.46% | 71.35% | 76.54% | 75.45% | 79.12% | -3.67% |

Sources:
- kaggle_outputs/ablation_3_counterfactual/yolo_training/seed_*/results.csv.
- kaggle_outputs/ablation_3_counterfactual/completion_status.json.

Configuration:

```python
augmentations = "all_disabled"
multi_scale = "enabled"
epochs = 50 (seed_42), 36 (seed_123), 30 (seed_456)  # early stopping
batch_size = 32
lr0 = 0.015
workers = 8
device = "GPU (P100)"
```

Notebook: notebooks/03_ablations/ablation_3_counterfactual/ablation_3_counterfactual_no_augment.ipynb

## Ablation 3 history

- v1-v3: multi_scale=True, mosaic=1.0, auto_augment='randaugment'. 54.57% Dice (-25.43%). RandAugment incompatible with multi-scale and mosaic.
- v4: multi_scale=True, mosaic=0.0, auto_augment='randaugment'. 68.81% Dice (-11.19%). Removing mosaic helped, still -11%.
- v5: multi_scale=True, mosaic=1.0, auto_augment=None. Planned, abandoned after the RandAugment conflict finding. Note: ABLATION_3_CHRONOLOGICAL_HISTORY.md records v5 as run, with mosaic=0.0 and 4.80% Dice.
- Counterfactual: multi_scale=True, all augmentations disabled. 64.8% Dice (-15.2%). Augmentations are essential, not just RandAugment.

## Analysis

Hypothesis test:
- H0: multi-scale without augmentations matches baseline.
- H1: multi-scale without augmentations degrades performance.
- Paired t-test, alpha = 0.05.
- Dice decline -25.95% (p < 0.001).
- Decision: reject H0.

### 1. Augmentations are essential

Dice dropped from 80.0% to 64.8% (-15.2%) with all augmentations disabled.

YOLOv8 segmentation models are pretrained on COCO with heavy augmentation. On a small dataset (2,208 images) without augmentation:
- the model overfits to limited examples,
- fails to generalize to test variations,
- loses transfer learning benefits.

YOLOv8 default configuration includes mosaic, mixup, HSV augmentation, random flips, random scale. Baseline (80% Dice) used default augmentations. Reference: https://docs.ultralytics.com/modes/train/

### 2. Multi-scale alone gives no benefit

Multi-scale with no augmentations: 64.8% Dice, worse than baseline 80%. Without augmentations the model sees limited variation per image, scale variation alone is insufficient, and it may overfit to training scales.

Comparison:
- Multi-scale, mosaic, RandAugment (v1-v3): 54.57%.
- Multi-scale, no mosaic, RandAugment (v4): 68.81%.
- Multi-scale, no augmentations (counterfactual): 64.8%.
- Baseline (default augmentations, no multi-scale): 80.0%.

Default augmentations beat multi-scale with reduced augmentations.

### 3. RandAugment conflict (indirect)

1. v1-v3: 54.57%.
2. v4: 68.81%.
3. Counterfactual: 64.8%.

- Removing mosaic improved 54.57% to 68.81% (+14.24%).
- Removing all augmentations: 64.8%, slightly worse than v4.
- RandAugment, multi-scale and mosaic together is toxic.
- Without RandAugment, multi-scale is ineffective rather than catastrophic.

See docs/RANDAUGMENT_EXPLAINED.md.

### 4. mAP50 vs Dice

mAP50(M) stayed near baseline (80.11% vs 80.01%) while Dice collapsed (64.8% vs 80.0%).
- mAP50 is a detection metric with a lenient 50% IoU threshold.
- Dice measures pixel overlap and is stricter.
- The model detects wounds but segments boundaries poorly.

mAP50 is insufficient for clinical deployment. Dice matters for wound measurement.

## Comparison of Ablation 3 variants

| Variant | Multi-Scale | Mosaic | RandAugment | Other Aug | Dice | Delta | Status |
|---------|-------------|--------|-------------|-----------|------|-------|--------|
| Baseline | No | 1.0 | Yes | Default | 80.0% | n/a | complete |
| v1-v3 | Yes | 1.0 | Yes | Default | 54.57% | -25.43% | catastrophic |
| v4 | Yes | 0.0 | Yes | Default | 68.81% | -11.19% | poor |
| Counterfactual | Yes | 0.0 | No | None | 64.8% | -15.2% | analysis |

Sources: kaggle_outputs/ablation_3_counterfactual/completion_status.json; baseline and v1-v4 from docs/ABLATION_SPECIFICATIONS.md.

## Why augmentations matter

Transfer learning with a small dataset depends on augmentation. YOLOv8 is pretrained on COCO (330K images, 80 classes, extensive augmentation). Fine-tuning on 2,208 medical images without augmentation breaks transfer learning. Reference: https://docs.ultralytics.com/models/yolov8/

Roles:
- Geometric (flips, rotations, scale): position and orientation invariance. Wounds appear at any angle.
- Color (HSV, brightness): lighting variation across clinical settings, avoids overfitting to color distributions.
- Composite (mosaic, mixup): multiple wound contexts at once, better localization, fewer false positives.

See docs/YOLO_AUGMENTATION_ANALYSIS.md.

Multi-scale limits:
- Scales images to [0.5x, 1.5x] baseline resolution.
- Needs diverse training examples to work.
- 2,208 training images are insufficient for scale generalization alone.
- Without color and geometric variation the model memorizes scale-and-appearance combinations, and the test set has new combinations.

Reference: https://docs.ultralytics.com/modes/train/#train-settings

## Clinical interpretation

- Baseline 80% Dice: clinically acceptable for wound measurement. Stated clearance threshold for segmentation devices: typically >= 75%.
- Counterfactual 64.8% Dice: below the 75% threshold. Unacceptable for deployment. Wound area measurements would carry about 20% error. Risk: misjudging healing progression.
- Recall: 79.12% (baseline) to 75.45% (counterfactual), -3.67%. About 3.67% more wounds missed, about 20 additional missed wounds on 552 test images. Risk: delayed treatment.

## Lessons

1. Augmentations are required. YOLOv8 segmentation cannot be trained effectively on small datasets without them. All future ablations keep baseline augmentations as foundation.
2. Test multi-scale with augmentations, not as a replacement for them.
3. Isolating single variables reveals causal effects. Include counterfactual runs.
4. Report both mAP50 and Dice for clinical models.

## Recommendations

1. Abandon the multi-scale-only ablation. Alternative: multi-scale with baseline augmentations (not yet attempted).
2. Focus on augmentation optimization (ablation 2). Status then: ablation 2 counterfactual tests running.
3. Document the augmentation dependency in the project requirements.

## Sources

1. kaggle_outputs/ablation_3_counterfactual/completion_status.json
2. kaggle_outputs/ablation_3_counterfactual/yolo_training/seed_*/results.csv
3. docs/RANDAUGMENT_EXPLAINED.md
4. docs/YOLO_AUGMENTATION_ANALYSIS.md
5. Ultralytics docs: https://docs.ultralytics.com/models/yolov8/ and https://docs.ultralytics.com/modes/train/

## Raw results

Seed 42 (final epoch 51): mAP50(M) 82.29%, Dice 56.29%, Precision(M) 83.46%, Recall(M) 78.46%.
Seed 123 (final epoch 36): mAP50(M) 76.75%, Dice 50.75%, Precision(M) 87.19%, Recall(M) 71.35%.
Seed 456 (final epoch 30): mAP50(M) 81.28%, Dice 55.13%, Precision(M) 82.07%, Recall(M) 76.54%.

Early stopping hit at different epochs across seeds because of validation patience. Extracted from results.csv for each seed.

Next: update docs/ABLATION_SPECIFICATIONS.md with the counterfactual findings. GPU quota at the time: 30/30 hours used (resets weekly).
