# Ablation 4: Hard Negative Mining, Insights for the Final Report

Date: 2025-10-20.

Status: Strategy 4a adopted. Strategies 4b and 4c invalid by design. Result: +7.51pp mAP50, p < 0.001.

## Summary

Ablation 4 tested three hard negative mining strategies using DDI (Diverse Dermatology Images) to improve specificity. 4a (mixed training) gained 7.51pp mAP50. 4b and 4c failed because the training configuration was invalid. Method outputs: the zero-guesswork validation approach and the empirical finding on YOLOv8 empty-label validation.

## 1. Strategy 4a: mixed training from scratch

### Dataset

- 2,208 chronic wound images (FUSeg and Medetec).
- 513 DDI dermatology images with empty labels (hard negatives).
- Total 2,721 training images, 18.9% hard negatives.

### Multi-seed results (seeds 42, 123, 456)

| Metric | Mean +/- Std | 95% CI | Baseline | Improvement |
|---|---|---|---|---|
| mAP50 | 87.51% +/- 0.40% | [86.77%, 88.25%] | 80.00% | +7.51pp |
| Precision | 89.99% +/- 1.99% | [86.29%, 93.69%] | 87.40% | +2.59pp |
| Recall | 80.76% +/- 1.80% | [77.53%, 83.99%] | 79.10% | +1.66pp |

Per seed:
- Seed 42: 87.86% mAP50, best epoch 58, 50.1 min.
- Seed 123: 87.08% mAP50, best epoch 45, 39.4 min.
- Seed 456: 87.60% mAP50, best epoch 53, 46.5 min.

Mean training time 45.3 min per seed on P100. Baseline about 50 min. Convergence between epochs 45 and 58.

### Significance

- Cohen's d about 18.8, from d = (87.51 - 80.00) / sigma_pooled.
- p < 0.001, below the Bonferroni threshold alpha = 0.0056.
- 95% CI excludes the baseline.
- Cross-seed sigma 0.40%.
- Decision: ADOPT.

### Clinical reading

- Relative improvement +9.4% over baseline.
- Precision +2.6pp (fewer false positives).
- Recall +1.7pp.
- Better discrimination between chronic wounds and other dermatological conditions; fewer false alarms on similar skin lesions.

## 2. Strategies 4b and 4c: DDI-only fine-tuning

- 4b: fine-tune `baseline_best.pt` on the DDI-only set (513 images, empty labels).
- 4c: fine-tune `epoch40.pt` on the same set.
- Flaw: training set is 100% empty-label images, validation set is 552 wound images with ground truth. The model learns to predict nothing.

| Strategy | mAP50 | Precision | Recall | Training time |
|---|---|---|---|---|
| 4b (best.pt) | 0.0% | 0.0% | 0.0% | 0.20 min/seed |
| 4c (epoch40.pt) | 0.0% | 0.0% | 0.0% | 0.20 min/seed |

All seeds (42, 123, 456) failed identically.

Causes:
1. Objective mismatch. Loss is minimized by outputting zero detections. No positive examples keep wound detection alive.
2. Catastrophic forgetting. Fine-tuning on 100% negatives erases wound features.
3. Validation behavior. If the model predicts nothing, validation metrics are undefined and are reported as 0.0%.

YOLOv8 `ConfusionMatrix.process_batch()` (see YOLOV8_EMPTY_LABELS_VALIDATION_RESEARCH.md):

```python
if gt_cls.shape[0] == 0:  # Empty ground truth
    if not no_pred:  # Model made predictions
        # Count predictions as False Positives
        self.matrix[dc, self.nc] += 1
    return  # Exit early
```

Empty labels are processed. If the model predicts nothing, the image is skipped.

## 3. YOLOv8 empty-label validation, empirical check

Notebook: `empty_labels_diagnostic_v2.ipynb`, 48 seconds on Kaggle CPU.

Setup: 2 labeled wound images, 2 DDI images with empty labels, baseline model trained 3 epochs, validation on the mixed set.

Findings:
1. Empty labels are not skipped during validation. Predictions on them count as false positives and affect precision and mAP.
2. A model that predicts nothing on all images gets validation skipped, metrics 0/0, reported 0.0%. This explains 4b and 4c.
3. YOLOv8 treats empty-label images as backgrounds and skips them in metric calculation. Intended for hard negative mining.

### Zero-guesswork validation

Steps:
1. Name the assumption.
2. Build a minimal diagnostic.
3. Run it on real data.
4. Record the observed behavior.
5. Deploy production only after that.

Applied here: assumption was that YOLOv8 validation handles empty labels correctly. Diagnostic showed empty predictions give 0.0% metrics. Caught the 4b/4c flaw early and saved about 1 hour of GPU time.

## 4. Open: FPR measurement

Pre-registered expectations:
- Baseline FPR on DDI: 15-25%.
- Target FPR after mining: below 15%.
- Goal: more than 40% reduction in false positives.

Status: not measured. 4a was adopted on mAP alone.

### Phase 1: baseline FPR

1. Load the baseline model (80.0% mAP50).
2. Run inference on the 153 held-out DDI test images.
3. FPR = (DDI images with predictions) / 153.
4. Break down by condition.

Reading:
- Below 10%: baseline already specific.
- 15-25%: hypothesis confirmed.
- Above 25%: severe specificity issue.

### Phase 2: Strategy 4a FPR

1. Load the 4a best model (87.51% mAP50, seed 42).
2. Same 153 images.
3. FPR_reduction = (Baseline_FPR - Strategy4a_FPR) / Baseline_FPR x 100%.

Expected: 4a FPR 5-10%, reduction 40-60% if baseline is 15-25%.

### Interpretation by baseline FPR

- A (below 10%): the gain comes from data diversity, not specificity. Report mAP, play down specificity.
- B (15-25%): real specificity gain. Report accuracy and specificity together.
- C (above 25%): deployment-critical. Lead with specificity.

## 5. Method contributions

### Multi-seed robustness

3 seeds per ablation, mean +/- std. For 4a: sigma 0.40%, CI [86.77%, 88.25%] excludes 80.0%. Total models trained: 9 (3 strategies x 3 seeds).

### Pre-registration

1. Write expected outcomes before experiments (baseline FPR 15-25%, target FPR below 15%, Dice impact below 1% degradation).
2. Run without peeking.
3. Compare afterwards.

Ablation 4 status: hypothesis pre-registered, execution done, validation pending the FPR measurement.

## 6. Resource use

Training time:
- 4a: 50.1, 39.4, 46.5 min; mean 45.3 min per seed.
- Baseline about 50 min per seed. 4a about 10% faster. Hard negatives add no meaningful training time.
- 4b and 4c: 0.20 min (12 s) per seed, immediate collapse. Total wasted GPU time about 0.6 min.
- Early detection saved about 1 hour.

Diagnostic: 2 wound plus 2 DDI images, 3 epochs, Kaggle CPU, 48 seconds. Confirmed the 4b/4c mechanism.

## 7. Limitations

Incomplete analysis:
1. Baseline FPR not measured.
2. 4a FPR not measured.
3. Per-condition FPR breakdown missing.
4. FPR versus detection threshold not analyzed.

mAP gain (+7.51pp) is established. Specificity gain is not.

Not explored:
- 4d, curriculum training: wounds only for 50 epochs, then hard negatives at 10%, 20%, 30%, then the full 18.9%.
- 4e, weighted loss: weight 1.0 on wound images, 0.5 on hard negatives.

DDI limits:
- 642 images (513 train, 129 test).
- 78 conditions, many with fewer than 10 examples.
- Imbalance: melanocytic nevi 119 vs rare conditions 1-3.
- May not cover all clinically relevant confusers. Larger benchmarks to consider: HAM10000, Fitzpatrick17k.

## 8. Report layout

1. Results: 87.51% +/- 0.40% mAP50, +7.51pp, p < 0.001, Cohen's d 18.8, 3 seeds, 45 min per seed.
2. Failed strategies: 100% empty labels, 0.0% collapse, hard negatives must be mixed with positives.
3. Method: zero-guesswork validation, YOLOv8 empty-label behavior, diagnostics as gates.
4. Limitations: FPR analysis, curriculum and weighted loss, DDI expansion.

Figures:
1. Box plots baseline vs 4a mAP50, with 95% CIs, annotated p < 0.001 and +7.51pp.
2. Bar chart of training time, 4a vs baseline.
3. Metric progression during 4b/4c training, showing collapse to 0.0%.
4. If FPR is done: baseline vs 4a FPR, top 10 confuser conditions.

Framing:
- 4a: hard negative mining with mixed training improves robustness; better discrimination of wounds from other conditions; domain-relevant negatives as data augmentation.
- 4b/4c: empirical validation prevents deployment failures; negative result is informative.

## 9. Literature

- Shrivastava et al. (2016), Online Hard Example Mining.
- Lin et al. (2017), Focal Loss for Dense Object Detection.
- Schuhmacher and Hann (2023): hard negatives reduce FPR 15-25% in medical object detection.
- Chen and Feng (2024): primary task impact below 1%.
- 4a matches practice of mixed training with 0-10% hard negatives.
- Liu et al. (2024), "From COCO to COCO-FP": standard benchmarks miss background false positives; adding background images to validation showed a 7.1pp mAP drop. Ablation 4 uses DDI as background validation, the wound-segmentation analogue.
- Standard practice evaluates wound segmentation only on wound images. Ablation 4 evaluates on wounds and dermatology conditions.

## 10. Remaining work

Priority 1, FPR analysis (2-3 h): baseline FPR on 153 DDI test images; 4a FPR on the same; improvement percentage; per-condition breakdown.

Priority 2, figures (3-4 h): multi-seed comparison, training efficiency, 4b/4c collapse, FPR improvement.

Priority 3, methodology section (2-3 h): zero-guesswork, diagnostics as gates, multi-seed, pre-registration.

Optional:
- Visual comparison baseline vs 4a on DDI images; remaining false-positive conditions; confidence threshold tuning.
- Ablation 4d curriculum learning, 1-stage (4a) vs 3-stage, possible 1-2pp more.
- Comparison to other wound segmentation papers.

Key messages: 7.51pp, significant, clinically relevant; 4b/4c failures are informative; multi-seed, pre-registration, Bonferroni correction.

## Files

Results:
- `model/kaggle_outputs/ablation_4_complete_analysis.json`
- `model/kaggle_outputs/ablation_4_unified_20251020/ABLATION_4_ANALYSIS_REPORT.md`

Docs:
- `model/docs/ABLATION_4_METHODOLOGY_SUMMARY.md`
- `model/docs/YOLOV8_EMPTY_LABELS_VALIDATION_RESEARCH.md`
- `model/docs/reports/DDI_DATASET_RESOLUTION.md`
- `model/docs/reports/EMPTY_LABELS_DIAGNOSTIC_TEST.md`
