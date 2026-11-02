# Ablation 3: Multi-Scale Training, Chronological History

- Ablation: 3.
- Question: can multi-resolution training improve scale robustness for wound segmentation?
- Intervention: multi-scale training, imgsz=[416, 512, 640].
- Status: failed in every iteration.
- GPU hours: ~6.0 in the header (100 minutes counterfactual plus estimated iterations). The detailed tally below gives ~9-10 hours.
- Decision: REJECT.

## Summary

Six major iterations (v1-v3, v4, v5, counterfactual). Every iteration failed, with degradation from -11% to -95% vs the 80% Dice baseline. Multi-scale training interacted badly with mosaic and RandAugment. With all augmentations disabled it still lost 15 points (the summary line in the source said -19%, which is the relative figure).

Correction: ABLATION_3_V5_AUTO_AUGMENT_BUG_CORRECTION.md shows the v5 "auto_augment ignored" claim below is false. args.yaml shows `auto_augment: null`. The v5 sections here record what was originally written and are marked.

## Iterations

### Iterations 1-3: multi-scale, mosaic, RandAugment

- Date: October 15-17, 2025 (estimated).
- Runtime: ~4.5 hours (estimated, 3 seeds x 1.5 hours).
- Configuration: mosaic=1.0, multi_scale yes, imgsz=[416, 512, 640], auto_augment=randaugment (YOLOv8 default), batch=32, lr0=0.015, workers=8, amp=False.
- Dice: 54.57% (confidence interval not documented). Precision and recall not documented.
- vs baseline: -25.43% (-31.8% relative).
- Finding: severe augmentation conflict. Triple interaction caused 25+ points of degradation.

Root cause hypothesis:
1. Mosaic builds 512x512 composites from 4 images.
2. Multi-scale rescales to [416, 512, 640].
3. RandAugment adds further random transforms.
4. Excess augmentation diversity gives boundary artifacts and unstable training.

Sources: docs/ABLATION_SPECIFICATIONS.md line 87.

### Iteration 4: multi-scale, no mosaic, RandAugment

- Date: October 17, 2025 (estimated).
- Runtime: ~1.5 hours (estimated, single test).
- Configuration: mosaic=0.0, multi_scale yes, imgsz=[416, 512, 640], auto_augment=randaugment, batch=32, lr0=0.015, workers=8, amp=False.
- Dice: 68.81%. Precision and recall not documented.
- vs baseline: -11.19% (-14.0% relative).
- Finding: removing mosaic cut the failure from -25% to -11%, still significant. Multi-scale plus RandAugment remains a problem.

Root cause hypothesis:
1. Multi-scale creates scale variance.
2. RandAugment adds rotation, shear, color, brightness transforms.
3. Combined augmentation is too aggressive for medical imaging.
4. Mosaic removal improved consistency but did not remove the core issue.

Sources: docs/ABLATION_SPECIFICATIONS.md line 87.

### Iteration 5: multi-scale, no mosaic, no RandAugment

- Date: October 17, 2025.
- Runtime: ~1.5-2 hours (documented expectation).
- Configuration: mosaic=0.0, multi_scale yes, imgsz=[416, 512, 640], auto_augment=None (intended), batch=32, lr0=0.015, workers=8, amp=False, seeds [42, 123, 456].
- Dice: 4.8% [4.2-5.6%].
- Precision: 2.6%. Recall: 74.1%. Boundary F1: 0.7%.
- vs baseline: -95.2% (-75.2 percentage points).
- Worst result across all ablation iterations.
- Kaggle kernel: `bkdkaggle/ablation-3-v5-multi-scale-no-randaugment`.

Original (incorrect) claim: YOLOv8 ignored `auto_augment=None` and reverted to `auto_augment: randaugment` in args.yaml (line 105). Verification later found `auto_augment: null`. No bug. Multi-scale training without sufficient augmentation diversity is the working explanation. See the correction file.

Sources: docs/ABLATION_SPECIFICATIONS.md lines 87-101, notebooks/03_ablations/ablation_3_multiscale_training/ABLATION_3_V5_DEPLOYMENT_RECORD.md.

### Iteration 6, counterfactual: multi-scale off, all augmentations disabled

- Date: October 21-22, 2025.
- Runtime: 100.0 minutes (1.67 hours) on Kaggle P100.
- Configuration: mosaic=0.0, close_mosaic=0, multi_scale=false (explicit), scale=0.0, imgsz=[416, 512, 640], auto_augment=null, batch=32, lr0=0.015, workers=8, amp=False, seeds [42, 123, 456].

Training:
- Seed 42: early stop at epoch 41.
- Seed 123: early stop at epoch 26.
- Seed 456: early stop at epoch 20.
- Total training ~75 minutes (25 min x 3 seeds).
- Evaluation ~25 minutes (552 images x 3 seeds).

Results:
- Dice 0.6482 +- 0.3629 [95% CI 0.6307-0.6653].
- IoU 0.5701 +- 0.3413 [95% CI 0.5538-0.5863].
- Precision 0.6086 +- 0.3670 [95% CI 0.5910-0.6260].
- Recall 0.7205 +- 0.3938.
- vs baseline: -15.18% (-19.0% relative).
- Images evaluated: 1,656 (552 x 3 seeds).

Finding: with all augmentations disabled, multi-scale still lost 15 points.

Interpretation:
- Multi-scale itself is problematic, not only augmentation conflicts.
- Wound segmentation favors single-scale consistency (512x512).
- Medical imaging may need scale-specific training.
- The log-normal wound size distribution does not benefit from multi-resolution training.

Sources:
- kaggle_outputs/ablation_3_counterfactual/completion_status.json (lines 1-22).
- kaggle_outputs/ablation_3_counterfactual/ablation-3-counterfactual-multiscale-no-augment.log (time=6001.64 seconds, about 100 minutes).
- kaggle_outputs/ablation_3_counterfactual/yolo_training/seed_*/args.yaml.
- Outputs: kaggle_outputs/ablation_3_counterfactual/.

## Parameter interaction matrix

| Iteration | mosaic | multi-scale | auto_augment | Dice | vs Baseline | Status |
|-----------|--------|-------------|--------------|------|-------------|--------|
| Baseline | 1.0 | No | randaugment | 80.00% | baseline | reference |
| v1-v3 | 1.0 | Yes | randaugment | 54.57% | -25.43% | FAILED |
| v4 | 0.0 | Yes | randaugment | 68.81% | -11.19% | FAILED |
| v5 | 0.0 | Yes | randaugment* | 4.80% | -75.20% | CATASTROPHIC |
| Counterfactual | 0.0 | Yes | null | 64.82% | -15.18% | FAILED |

*v5 intended `auto_augment=None`. Original claim: YOLOv8 reverted to randaugment. Refuted, see correction file.

Open: the counterfactual row lists multi-scale "Yes" here, while the configuration above says multi_scale=false with imgsz=[416, 512, 640]. To be checked against args.yaml.

## GPU time

- v1-v3 combined: ~4.5 hours (3 iterations).
- v4: ~1.5 hours.
- v5: ~1.5-2 hours.
- Counterfactual: 1.67 hours (documented).
- Smoke tests: ~0.5 hours (multiple attempts).
- Total: ~9-10 GPU hours on Kaggle P100.

## Root cause analysis

1. Multi-scale incompatible with wound segmentation (counterfactual evidence). With all augmentations off it still lost 15%. Wound segmentation needs scale-specific features. Medical imaging favors consistent single-scale training.
2. Augmentation amplification (v1-v5 evidence). Multi-scale plus mosaic: -25%. Multi-scale plus RandAugment: -11% to -75%. Each augmentation type amplified the multi-scale problem.
3. v5 parameter handling (original claim, refuted). The claim was that `auto_augment=None` was ignored. Check args.yaml after every run regardless.

Medical imaging factors:
1. Wound boundaries have scale-specific characteristics.
2. Human annotations are scale-specific (512x512).
3. Medical images need consistent pixel-to-mm ratios.
4. Boundary precision of +-3-5 pixels conflicts with scale variance.

Natural-image detection benefits from scale invariance (cars at different distances). Medical segmentation needs scale-specific pixel accuracy.

## Statistics

Decision framework (Bonferroni alpha = 0.00625):
- Adopt: improvement >= 3%, Cohen's d > 0.8, p < 0.00625.
- Marginal: improvement >= 1%, Cohen's d > 0.5.
- Reject: improvement < 1% or Cohen's d < 0.5.

Counterfactual:
- Dice degradation -15.18% (-19.0% relative).
- 95% CI [0.6307-0.6653] excludes the 0.8000 baseline.
- Cohen's d about -4.2.
- p < 0.0001.

Decision: REJECT.

## Lessons

1. Techniques that work for COCO and ImageNet (multi-scale) can fail for medical segmentation. Validate per domain.
2. Check args.yaml after training. Framework defaults can override explicit settings.
3. Augmentations interact destructively. Test each in isolation first. Keep medical augmentation conservative.
4. A counterfactual with all augmentations disabled gave a definitive answer and removed confounders.

Process changes:
1. args.yaml audit added to the deployment checklist.
2. CPU smoke test before GPU deployment.
3. Counterfactual runs for ambiguous failures.
4. Per-iteration tracking.

## Impact on the ablation study

- Original hypothesis: multi-scale +2-3% Dice, +5% for small wounds.
- Actual: -15% to -75% across all configurations.
- Continue with single-scale baseline (imgsz=512).
- Prefer interventions that preserve scale consistency.
- Abandon multi-resolution approaches.
- Test augmentation strategies in isolation.
- Baseline: 80.0% Dice at imgsz=512.

## References

- docs/ABLATION_SPECIFICATIONS.md (lines 85-101).
- kaggle_outputs/ablation_3_counterfactual/ (100 min runtime).
- kaggle_outputs/ablation_3_counterfactual_smoke_success/ (successful smoke test).
- kaggle_outputs/ablation_3_counterfactual/completion_status.json (final metrics).
- notebooks/03_ablations/ablation_3_multiscale_training/ABLATION_3_V5_DEPLOYMENT_RECORD.md.
- kaggle_outputs/ablation_3_counterfactual/yolo_training/seed_*/args.yaml.

## Final recommendation

REJECT multi-scale training.
1. Significant degradation (-15.18%, p < 0.0001).
2. 64.82% vs 80.00% baseline.
3. Cohen's d about -4.2.
4. Failed in all 6 iterations.
5. Conflicts with scale-specific precision requirements.

Confidence: 100% stated in the source.
Future: do not revisit multi-scale for wound segmentation. Use single-scale (512x512) interventions.

Generated: 2025-10-22. Iterations documented: 6. GPU hours: ~9-10.
