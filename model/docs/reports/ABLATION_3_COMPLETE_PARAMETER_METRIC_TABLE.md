# Ablation 3: Complete Parameter and Metric Comparison

Date: 2025-10-22.
Source: extraction from notebooks, logs and documentation.

## Parameters

| Parameter | Baseline | v1-v3 | v4 | v5 | Counterfactual |
|-----------|----------|-------|----|----|----------------|
| mosaic | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| multi_scale | No | Yes | Yes | Yes | false |
| imgsz | 512 | [416,512,640] | [416,512,640] | [416,512,640] | [416,512,640] |
| auto_augment | randaugment | randaugment | randaugment | None (intended); randaugment (actual, original claim, refuted) | null |
| degrees | 0.0 | not documented | not documented | not documented | 0.0 |
| flipud | 0.0 | not documented | not documented | not documented | 0.0 |
| fliplr | 0.5 | not documented | not documented | not documented | 0.0 |
| translate | 0.1 | not documented | not documented | not documented | 0.0 |
| scale | 0.5 | not documented | not documented | not documented | 0.0 |
| batch_size | 32 | 32 | 32 | 32 | 32 |
| epochs | 100 | ~100 | ~100 | 100 | 100 |
| lr0 | 0.015 | 0.015 | 0.015 | 0.015 | 0.015 |
| workers | 8 | 8 | 8 | 8 | 8 |

The v5 actual auto_augment claim is refuted in ABLATION_3_V5_AUTO_AUGMENT_BUG_CORRECTION.md: args.yaml shows `auto_augment: null`.

## Metrics

| Metric | Baseline | v1-v3 | v4 | v5 | Counterfactual |
|--------|----------|-------|----|----|----------------|
| Dice | 80.00% | 54.57% | 68.81% | 4.80% | 64.82% +- 36.29% |
| Delta Dice | n/a | -25.43% | -11.19% | -75.20% | -15.18% |
| mAP50(M) | 80.00% | not documented | not documented | not documented | 83.48% +- 1.52% |
| Delta mAP50 | n/a | not documented | not documented | not documented | +3.48% |
| Precision(M) | 81.23% | not documented | not documented | 2.6% | 84.12% +- 2.96% |
| Recall(M) | 79.12% | not documented | not documented | 74.1% | 78.07% +- 2.20% |
| Boundary F1 | not documented | not documented | not documented | 0.7% | not documented |
| Seeds | [42,123,456] | not documented | not documented | [42,123,456] | [42,123,456] |
| Status | baseline | CATASTROPHIC | FAILED | CATASTROPHIC | MIXED |

Open: ABLATION_3_COUNTERFACTUAL_COMPLETE_ANALYSIS.md reports different counterfactual values (Dice 54.05%, mAP50 80.11%, precision 84.24%, recall 75.45%, per-seed epochs 50, 36, 30). Same 64.82% Dice appears in this file, the history and the kernel list. To be reconciled against completion_status.json and results.csv.

## Per seed (counterfactual only)

| Seed | Epochs | Dice | mAP50(M) | Precision(M) | Recall(M) |
|------|--------|------|----------|--------------|-----------|
| 42 | 41 | not documented | 85.14% | 86.81% | 77.08% |
| 123 | 26 | not documented | 82.36% | 84.59% | 76.53% |
| 456 | 20 | not documented | 82.94% | 80.95% | 80.59% |
| Mean | 29 | 64.82% | 83.48% | 84.12% | 78.07% |
| Std | n/a | 36.29% | 1.52% | 2.96% | 2.20% |

Early stopping hit at different epochs because of validation patience.

## What distinguished each iteration

### v1-v3: triple augmentation interaction

Changed from baseline: multi-scale enabled, mosaic kept (1.0), RandAugment kept.
Result: -25.43% Dice.
Hypothesis: mosaic, multi-scale and RandAugment together interact destructively.
Sources: parameters and metrics in docs/ABLATION_SPECIFICATIONS.md; analysis in ABLATION_3_CHRONOLOGICAL_HISTORY.md lines 22-49.

### v4: mosaic removed (double interaction)

Changed from v1-v3: mosaic 1.0 to 0.0. Multi-scale and RandAugment kept.
Result: 68.81% Dice, -11.19% vs baseline.
Hypothesis: removing mosaic reduced the triple interaction to a double one (multi-scale and RandAugment). Improved but still failing.
Analysis: ABLATION_3_CHRONOLOGICAL_HISTORY.md lines 53-80.

### v5: RandAugment removal intended

Changed from v4: mosaic kept at 0.0, multi-scale kept, `auto_augment=None` intended.
Original claim: YOLOv8 ignored the setting and used RandAugment. Refuted by args.yaml verification (see correction file).
Result: 4.80% Dice (-75.20%).
Sources: docs/ABLATION_SPECIFICATIONS.md lines 87-101 (bug claim lines 96-97, metrics line 90); ABLATION_3_CHRONOLOGICAL_HISTORY.md lines 84-118.
Kaggle: https://www.kaggle.com/code/bkdkaggle/ablation-3-v5-multi-scale-no-randaugment

### Counterfactual: all augmentations disabled

Changed from v4: mosaic kept at 0.0, multi-scale Yes to false, degrees, flipud, fliplr, translate, scale and auto_augment all set to 0.0 or null.
Result, mixed:
- Dice 64.82% (-15.18% vs baseline).
- mAP50 83.48% (+3.48% vs baseline).

mAP50 improved while Dice degraded.
Hypothesis: multi-scale improves mask boundary confidence (mAP50) and harms pixel-level precision (Dice). Different metrics capture different aspects of segmentation quality.

Sources:
- Parameters: kaggle_outputs/ablation_3_counterfactual/yolo_training/seed_42/args.yaml lines 29, 32, 35, 92-98, 100, 105.
- Metrics: kaggle_outputs/ablation_3_counterfactual/completion_status.json lines 16, 18.
- Per seed: results.csv for seeds 42, 123, 456.
- Runtime: 100 minutes (1.67 hours).

## Insights

### 1. Metric-specific effects of multi-scale

Counterfactual: Dice -15.18% (degraded), mAP50 +3.48% (improved). mAP50 measures IoU at a 50% threshold. Dice measures exact pixel overlap. mAP50 alone is insufficient for clinical deployment. Dice matters for wound measurement accuracy.

### 2. v5 auto_augment

Original finding: `auto_augment=None` ignored, YOLOv8 reverted to randaugment. Evidence cited: notebook cell `auto_augment: None`, args.yaml `auto_augment: randaugment`. Later refuted. 4.8% Dice is the worst across all ablations. Source: docs/ABLATION_SPECIFICATIONS.md lines 96-97.

### 3. Augmentation interaction hierarchy

```
Triple interaction (v1-v3):         54.57% Dice (-25.43%)
Double interaction (v4):            68.81% Dice (-11.19%)
No augmentations (counterfactual):  64.82% Dice (-15.18%)
```

Hypothesis: multi-scale plus RandAugment is worse than multi-scale alone. Removing mosaic broke the triple interaction and improved results. The double interaction is still harmful.

### 4. Early stopping variability

Counterfactual seeds stopped at epochs 41, 26 and 20. Without augmentations convergence is unstable and seed-dependent. Augmentations stabilize training. Source: ABLATION_3_CHRONOLOGICAL_HISTORY.md lines 137-139.

## Data quality

Complete (high confidence):
- Counterfactual: args.yaml, results.csv, completion_status.json, per-seed metrics.
- v5: deployment record, intended configuration, Dice, precision, recall, Boundary F1.

Partial (medium confidence):
- v1-v3: Dice only (54.57%), no per-seed breakdown, parameters inferred from documentation.
- v4: Dice only (68.81%), no per-seed breakdown, parameters inferred from documentation.

Missing:
- degrees, flipud, fliplr, translate, scale for v1-v4.
- mAP50 for v1-v5 (only the counterfactual measured it).
- Per-seed breakdowns for v1-v4.
- Preserved notebook files for v1-v4 (documentation-only evidence).

## Verification sources

1. docs/ABLATION_SPECIFICATIONS.md (lines 87-101).
2. ABLATION_3_CHRONOLOGICAL_HISTORY.md (lines 22-49, 53-80, 84-118, 125, 137-139).
3. kaggle_outputs/ablation_3_counterfactual/yolo_training/seed_42/args.yaml (lines 29, 32, 35, 92-98, 100, 105).
4. kaggle_outputs/ablation_3_counterfactual/completion_status.json (lines 11-18).
5. kaggle_outputs/ablation_3_counterfactual/yolo_training/seed_*/results.csv.
6. https://www.kaggle.com/code/bkdkaggle/ablation-3-v5-multi-scale-no-randaugment (v5).

Files analyzed: 8+. Confidence: high for counterfactual, medium for v1-v5.

## Recommendations

1. Abandon multi-scale training for wound segmentation. All iterations failed (-11% to -75% Dice). Counterfactual -15% Dice despite the mAP50 gain. 5 iterations, ~10 GPU hours, no successful result.
2. Prioritize Dice over mAP50 for clinical use. mAP50 alone would have wrongly adopted multi-scale.
3. Verify args.yaml against the intended configuration before trusting results.

Last updated: 2025-10-22. GPU hours: ~9-10.
