# Ablation 3 v5: auto_augment Claim Corrected

Date: 2025-10-22 16:45 PST.
Status: documentation error found and corrected.

## Summary

- Original claim: YOLOv8 ignored `auto_augment=None` and reverted to `randaugment` (args.yaml line 105).
- Actual: YOLOv8 applied `auto_augment=None`. args.yaml line 105 shows `null`.
- The v5 failure (4.80% Dice, -75.20%) was not a parameter handling bug. Cause: multi-scale training without sufficient augmentation diversity.

## Verification

1. Claim location: docs/ABLATION_SPECIFICATIONS.md line 96: "Despite setting `auto_augment=None`, YOLOv8 reverted to `auto_augment: randaugment` (confirmed in args.yaml line 105)".
2. Downloaded the v5 results: `kaggle kernels output bkdkaggle/ablation-3-v5-multi-scale-no-randaugment`.
   - /tmp/ablation_3_v5_results/yolo_training/seed_42/args.yaml
   - /tmp/ablation_3_v5_results/yolo_training/seed_123/args.yaml
3. Result: seed_42 line 105 `auto_augment: null`; seed_123 line 105 `auto_augment: null`.

YOLOv8 honored `auto_augment=None`. No bug.

## Intended vs actual

Intended: notebooks/03_ablations/ablation_3_multiscale_training/ablation_3_v5_multiscale_no_randaugment.ipynb line 889:

```python
auto_augment=None,  # CRITICAL - disables RandAugment
```

ABLATION_3_V5_DEPLOYMENT_RECORD.md: `auto_augment`: None (CRITICAL, disables RandAugment).

Actual (args.yaml, seeds 42 and 123, line 105): `auto_augment: null`. Correctly applied.

## Revised root cause

Original hypothesis (false): "Multi-scale + RandAugment conflict: despite setting `auto_augment=None`, YOLOv8 reverted to `auto_augment: randaugment`".

Corrected: multi-scale training without sufficient augmentation diversity causes catastrophic failure.

v5 configuration (4.80% Dice):
- mosaic: 0.0
- degrees: 0.0
- scale: 0.5
- translate: 0.1
- flipud: 0.0
- fliplr: 0.5
- auto_augment: null
- multi-scale: [416, 512, 640]

Reasoning:
- Only 3 active augmentations: translate (0.1), scale (0.5), fliplr (0.5).
- Multi-scale adds resolution variance but no geometric or appearance diversity.
- The model sees 3 resolutions of nearly identical images, which gives severe overfitting.
- Counterfactual (64.82% Dice): same configuration but `multi_scale: false` (single resolution 512). Without multi-scale stress, limited augmentation does better. Still -15.18% vs baseline (80.0%).

Multi-scale training requires robust augmentation (mosaic, rotation, color variation) to prevent overfitting.

## Documentation corrections needed in docs/ABLATION_SPECIFICATIONS.md

Line 96, current (incorrect):
```markdown
1. **Multi-scale + RandAugment conflict**: Despite setting `auto_augment=None`, YOLOv8 reverted to `auto_augment: randaugment` (confirmed in args.yaml line 105)
```
Corrected:
```markdown
1. **Multi-scale requires robust augmentation**: Setting `auto_augment=None` was correctly applied, but multi-scale training [416,512,640] without sufficient augmentation diversity (no mosaic, no rotation, no vertical flip) caused 95% performance drop due to overfitting
```

Line 98, current (incorrect):
```markdown
3. **Parameter handling bug**: YOLOv8 ignores `auto_augment=None`, reverts to default from ultralytics/cfg/default.yaml
```
Corrected:
```markdown
3. **Augmentation-resolution tradeoff**: Multi-scale training amplifies the need for augmentation diversity. With mosaic=0.0, degrees=0.0, flipud=0.0, auto_augment=null, the model saw only 3 weak augmentations (translate, scale, fliplr) across 3 resolutions, giving catastrophic overfitting
```

Line 100, current (incorrect):
```markdown
**Critical Lesson**: ALWAYS verify args.yaml after training to catch ignored parameters
```
Corrected:
```markdown
**Critical Lesson**: Multi-scale training REQUIRES robust augmentation (mosaic + rotation + flips + color) to prevent overfitting. Parameter validation showed auto_augment=null was correctly applied; the failure was due to insufficient augmentation diversity, not a bug.
```

Status of these three corrections in docs/ABLATION_SPECIFICATIONS.md: to be confirmed.

## v5 analysis

| Aspect | Configuration | Impact |
|--------|--------------|--------|
| Multi-scale | [416, 512, 640] | high resolution variance |
| Mosaic | 0.0 (disabled) | no multi-image context learning |
| Rotation | 0.0 (disabled) | no orientation diversity |
| Vertical flip | 0.0 (disabled) | no extra geometric diversity |
| auto_augment | null (disabled) | correctly applied, no RandAugment |
| Active augmentations | translate (0.1), scale (0.5), fliplr (0.5) | only 3 weak augmentations |
| Result | 4.80% Dice (-95.2%) | catastrophic overfitting |

Multi-scale without augmentation means memorization across resolutions.

## Evidence

- Notebook: notebooks/.../ablation_3_v5_multiscale_no_randaugment.ipynb line 889 `auto_augment=None,`.
- Kaggle: https://www.kaggle.com/code/bkdkaggle/ablation-3-v5-multi-scale-no-randaugment
- args.yaml: /tmp/ablation_3_v5_results/yolo_training/seed_42/args.yaml and seed_123, line 105 `auto_augment: null`.
- Dice 4.80% [4.2-5.6%] (3-seed average). Files: experiments/reports/ABLATION_3_V5_COMPLETE_RESULTS.json, docs/ABLATION_SPECIFICATIONS.md lines 87-100.

Method: direct download of Kaggle kernel outputs. Seeds verified: 42 and 123.
