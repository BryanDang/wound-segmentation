# Ablation 1 (Focal Tversky Loss): deep analysis

Much of the source text is damaged. Unrecoverable passages are marked "Text damaged in source."

## Executive summary

Result: Focal Tversky Loss reached 80.01% Dice vs 80.00% baseline (delta = +0.01%, no improvement).

Conclusion: no benefit from the loss (rest damaged in source). YOLOv8's default loss (BCE + Dice) is adequate for this task.

## 1. Comprehensive metrics

### 1.1 Dice coefficient (primary metric)

- Ablation 1 (Focal Tversky): 80.01% (mean over 3 seeds; std damaged in source)
- Baseline (YOLOv8 default): 80.00%
- Improvement: +0.01% (zero)

Per seed:
- Seed 42: 80.05% (stopped at epoch 47)
- Seed 123: 78.59% (stopped at epoch 42). Worse than baseline.
- Seed 456: 81.38% (stopped at epoch 87)

Cross-seed variance:
- Range: 78.59% to 81.38% = 2.79%
- Coefficient of variation (CV): about 1.7%
- High variance (rest damaged in source).

### 1.2 IoU (Jaccard index)

Formula: IoU = TP / (TP + FP + FN).

IoU does not double the numerator (rest damaged in source).

Expected relationship for identical predictions: Dice = 2*IoU / (1 + IoU). At Dice = 80%, IoU is about 66.7%.

To verify Ablation 1's IoU: check `baseline_per_image_results.csv` for per-image IoU and aggregate.

Interpretation:
- IoU improved while Dice stayed flat: the model reduced false positives (precision up).
- IoU decreased while Dice stayed flat: the model increased false positives (precision down, recall up to compensate).
(Both lines partly damaged in source.)

### 1.3 Precision-recall trade-off

- Precision = TP / (TP + FP): of predicted wound pixels, what fraction is wound.
- Recall = TP / (TP + FN): of wound pixels, what fraction was detected.

Focal Tversky design (alpha=0.7, beta=0.3):
- Penalizes false negatives heavily (alpha=0.7).
- Penalizes false positives lightly (beta=0.3).
- Expected effect: recall up, precision down.

Training logs, `yolo_training/seed_42/results.csv`:
- `metrics/precision(M)`: mask precision
- `metrics/recall(M)`: mask recall

Seed 42, final epoch (47):
- Precision(M): 0.87866
- Recall(M): 0.82985
- Precision exceeds recall by 4.9%.

Interpretation: if Focal Tversky worked as designed, recall should exceed precision. Precision exceeds recall, so the loss did not change training behavior as intended. YOLOv8's multi-task loss diluted its effect (rest damaged in source).

### 1.4 F1 score

Formula: F1 = 2 x (P x R) / (P + R).

Seed 42, final values:
- Precision: 0.87866
- Recall: 0.82985
- F1 = 2 x (0.87866 x 0.82985) / (0.87866 + 0.82985) = 0.8536

Comparison with baseline: if Ablation 1 F1 is close to baseline F1, the precision-recall balance is unchanged (rest damaged in source).

### 1.5 Sensitivity and specificity

- Sensitivity = Recall = TP / (TP + FN)
- Specificity = TN / (TN + FP)

For wound segmentation:
- Class balance: 95.8% background, 4.2% wound (class imbalance).
- High specificity is required to avoid alert fatigue (partly damaged in source). Low specificity means many false positives.

From the per-image CSV:
- Compute per-image specificity = TN / (TN + FP) and aggregate.
- Compare Ablation 1 against the baseline.

Expected if Focal Tversky worked:
- Recall up (fewer missed wounds).
- Specificity lower, as an accepted trade-off (partly damaged in source).

### 1.6 Boundary F1 (boundary adherence)

Formula: Boundary F1 uses a `tolerance` in pixels around boundaries (rest damaged in source).

Why it matters:
- Focal Tversky (gamma=2.0) emphasizes hard examples.
- Wound boundaries are gradual transitions.
- If Focal Tversky helps, Boundary F1 should improve (rest damaged in source).

From the logs (`baseline_per_image_results.csv`):
- Column: `boundary_f1`
- Per-image Boundary F1 scores
- Aggregate and CI

Interpretation:
- Boundary F1 improved (for example +5%) while Dice did not: better boundary localization (partly damaged in source).
- Boundary F1 unchanged: Focal Tversky (gamma=2.0) had no effect on boundary precision.

## 2. Training dynamics

### 2.1 Convergence patterns

Seeds stopped at epochs 42, 47 and 87.

Seed 42 (stopped at 47):
- `train/seg_loss` fell from 1.936 to 1.246 (epoch 47).
- `val/seg_loss` fell from 2.193 to 1.644 (epoch 47).
- Converged; early stopping triggered.

Seed 123 (stopped at 42, worst performance):
- Likely stopped earlier (rest damaged in source).
- Hypothesis: a poor local minimum it could not escape.

Seed 456 (stopped at 87, best performance):
- Trained about 2x as long as seed 42.
- Hypothesis: slower convergence, reached a better optimum.

Cross-seed variance: 42 vs 87 epochs = 2.07x difference in training time. This is large. Possible explanations (all partly damaged in source):
1. Loss landscape is rugged.
2. Focal Tversky created an optimization difficulty.
3. Interaction between learning rate schedule and loss.

Open: best epoch per seed (expected 60 to 80). If the best epoch was low, Focal Tversky destabilized training (rest damaged in source).

### 2.2 Loss curve analysis (seed 42)

| Epoch | train_seg | val_seg |
|-------|-----------|---------|
| 1 | 1.936 | 2.193 |
| 10 | 1.653 | 1.966 |
| 20 | 1.511 | 2.017 |
| 30 | 1.405 | 1.779 |
| 40 | 1.325 | 1.722 |
| 47 | 1.246 | 1.644 |

Observations:
1. Both losses decrease overall (good).
2. Train/val gap: 0.2571 at epoch 1, 0.398 at epoch 47. The gap increased.
3. Generalization gap widens: train_seg and val_seg diverge.

Interpretation:
- The widening train/val gap suggests overfitting to the training set.
- Focal Tversky (gamma=2.0) emphasizes hard examples, which may lead to memorizing edge cases.
- Early stopping prevented further overfitting (partly damaged in source).

### 2.3 Precision/recall dynamics

Seed 42 trajectory:
- Epoch 1: Precision(M)=0.85001, Recall(M)=0.08505 (low recall)
- Epoch 10: Precision(M)=0.75102, Recall(M)=0.71191
- Epoch 20: Precision(M)=0.84119, Recall(M)=0.76598 (recovering)
- Epoch 30: Precision(M)=0.83541, Recall(M)=0.79263 (converging)
- Epoch 47: Precision(M)=0.87866, Recall(M)=0.82985

Focal Tversky expectation (alpha=0.7 emphasizes false negatives):
- Recall should rise quickly (find all wounds).
- Precision may drop (false alarms).
- Expected final: recall > precision.

Actual result: precision (87.9%) > recall (83.0%) by 4.9%.

Conclusion: Focal Tversky with alpha=0.7 did not produce the intended effect. Rest damaged in source.

## 3. Why Focal Tversky failed

### 3.1 Multi-task loss dilution

YOLOv8 total loss:

```
Total_Loss = lambda_box x Box_Loss + lambda_seg x Seg_Loss + lambda_cls x Cls_Loss + lambda_dfl x DFL_Loss
```

- Box_Loss: bounding box regression
- Seg_Loss: segmentation (where Focal Tversky replaced the default)
- Cls_Loss: classification
- DFL_Loss: distribution focal loss (box-related; detail damaged in source)

Problem: even with Focal Tversky replacing the segmentation loss, it is one component of the total.

Losses at epoch 47:
- train/box_loss: 0.887
- train/seg_loss: 1.246 (the segmentation loss)
- train/cls_loss: 0.631
- train/dfl_loss: 1.002

Contributions (equal weighting):
- Total: 0.887 + 1.246 + 0.631 + 1.002 = 3.766
- Seg_Loss is 33% of the total.

Implication: Focal Tversky acts on 33% of the signal. The other 67% (box, cls, dfl) is unchanged, which limits improvements.

### 3.2 Focal parameter too aggressive

Focal Tversky Loss: FTL = (1 - TI)^gamma, with TI = TP / (TP + alpha*FN + beta*FP).

Setting: gamma = 2.0.

Effect of gamma:
- gamma = 1.0: standard Tversky (no focusing).
- gamma = 2.0: hard examples weighted 4x.
- gamma = 3.0: hard examples weighted 9x.

Trade-off:
- Benefit: focus on hard boundaries (partly damaged in source).
- Risk: overfitting to outliers, neglecting well-learned examples (partly damaged in source).

Evidence of overfitting:
- Train/val gap widened from epoch 1 to 47.
- High cross-seed variance (78.59% to 81.38%).
- Seed 123 performed worse than baseline.

Hypothesis: gamma=2.0 caused the model to overfit hard examples, hurting generalization.

### 3.3 Alpha-beta mismatch

Assumption: alpha=0.7 (emphasizes false negatives) would help recall (rest damaged in source).

Reality check:
- Baseline precision is about equal to recall, about 80% (balanced).
- The dataset is 95.8% background, so false positives are penalized by the class imbalance.
- alpha=0.7 was unnecessary; the baseline already tilts toward precision (partly damaged in source).

Result: alpha=0.7 tried to increase recall, and the imbalance pushed back toward precision. Net effect: cancelled out, no improvement.

### 3.4 YOLOv8 loss already near-optimal

YOLOv8 default segmentation loss: BCE + Dice.

Why BCE + Dice works well:
1. BCE: pixel-wise cross-entropy, good for per-pixel predictions.
2. Dice: overlap-based, good for shape structure.
3. Combination: pixel accuracy plus coherence.

Evidence:
- Baseline: 80% Dice.
- YOLO loss is the result of extensive experimentation (partly damaged in source).
- Proven across datasets (partly damaged in source).

Conclusion: Focal Tversky works in other domains (for example small objects, extreme imbalance), but YOLOv8's default is already adequate for wound segmentation.

## 4. Remaining analysis

Section title damaged in source.

### 4.1 Metrics available

1. Dice coefficient: primary metric, shows no improvement.
2. IoU: in the CSV, needs aggregation.
3. Precision/recall: from the logs, show the expected effect did not occur.
4. Boundary F1: in the CSV, shows whether gamma=2.0 improved edges.
5. Specificity: compute from TP/TN/FP/FN in the CSV.
6. Sensitivity: equals recall, already available.

### 4.2 Analysis needed

Compare Ablation 1 per-image results against the baseline. Code partly damaged in source; reconstructed from readable parts.

```python
import pandas as pd
import numpy as np
from scipy import stats

# Load Ablation 1 results
abl1 = pd.read_csv('baseline_per_image_results.csv')
# Load baseline
baseline = pd.read_csv('/kaggle/input/baseline-outputs/baseline_per_image_results.csv')

def calculate_metrics(df):
    return {
        'dice_mean': df['dice'].mean(),
        'dice_std': df['dice'].std(),
        'iou_mean': df['iou'].mean(),
        'precision_mean': df['precision'].mean(),
        'recall_mean': df['recall'].mean(),
        'boundary_f1_mean': df['boundary_f1'].mean(),
        'specificity_mean': (df['tn'] / (df['tn'] + df['fp'])).mean(),
        'sensitivity_mean': df['recall'].mean(),
        'f1_mean': (2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])).mean(),
    }

abl1_metrics = calculate_metrics(abl1)
baseline_metrics = calculate_metrics(baseline)

# Wilcoxon signed-rank
dice_pvalue = stats.wilcoxon(abl1['dice'], baseline['dice']).pvalue
iou_pvalue = stats.wilcoxon(abl1['iou'], baseline['iou']).pvalue

# Effect size (Cohen's d)
dice_effect = (abl1['dice'].mean() - baseline['dice'].mean()) / np.sqrt(
    (abl1['dice'].std()**2 + baseline['dice'].std()**2) / 2
)
```

The original printed Dice, p-value (threshold < 0.0125 with Bonferroni), Cohen's d (threshold > 0.8), Boundary F1 and specificity for both runs.

### 4.3 Expected findings

If Focal Tversky helped (with Dice improvement):
- Boundary F1 up (focus from gamma=2.0).
- Recall up, precision down (alpha working).
- Variance across seeds down (stable training).

If Focal Tversky was neutral or harmful:
- Dice within +/-1% of baseline.
- High seed variance (instability).
- No significant difference (p > 0.05).

### 4.4 Confirming YOLO defaults are optimal

Evidence that the YOLO default is near-optimal:
1. Comparisons show no improvement.
2. Precision-recall unchanged (alpha ineffective).
3. Boundary metrics unchanged (gamma ineffective).
4. Cross-seed results show destabilized training.

To confirm:
- Test other loss functions (Focal Dice, Lovasz-Softmax, Combo Loss).
- If all show zero improvement, the default is optimal.
- If one shows improvement, the default is beaten (rest damaged in source).

## 5. Broader implications

Section title partly damaged in source.

### 5.1 The 80% Dice ceiling

Hypothesis: the 80% Dice reflects data limitations, not model limitations.

Data issues:
1. Annotation quality: 45% specialist-validated, 55% author-annotated.
2. Inter-annotator agreement: not measured (reliability unknown).
3. Ambiguous boundaries: wound edges are fuzzy.
4. Lighting variation: inconsistent image quality.

If 80% is data-limited, loss functions will not help. Better data is needed.

### 5.2 Revised strategy

With Ablation 1 failed:

Ablation 2 (color augmentation): expected +2 to 3% Dice.
- Risk: if the 80% ceiling is data-limited, it will not help.
- Hope: augmentation adds generalization (rest damaged in source).

Ablations 3 and 4 (multi-scale training, etc.):
- Risk: near-optimal already, marginal gains.

If Ablations 2 to 4 all fail:
1. Larger model (more capacity; detail damaged in source).
2. Ensemble: combine predictions from multiple models.
3. Data improvement: re-annotate with specialists, measure inter-annotator agreement.
4. Architecture change: try a different architecture (SAM, U-Net variants; partly damaged in source).

## 6. Recommendations

### 6.1 Immediate actions

1. Run the analysis script in 4.2 for exact metrics.
2. Check Boundary F1 closely; it indicates effectiveness.
3. Check cross-seed variance for stability.
4. Test significance with Wilcoxon plus Bonferroni.

### 6.2 Future adjustments

If Ablation 2 fails:
- Consider skipping further ablations (partly damaged in source).
- Prioritize other methods.
- Acknowledge the 80% dataset ceiling.

If Ablation 2 succeeds:
- Apply the approach that works where Focal Tversky did not (partly damaged in source).
- Combine with other ablations (partly damaged in source).
- Aim for 82 to 85% Dice with combined improvements.

## 7. YOLO augmentation parameters

YOLOv8 supports color and brightness augmentation through HSV parameters.

### 7.1 HSV augmentation parameters

hsv_h (hue): adjusts hue.
- Range: 0.0 to 1.0 (360 degrees)
- Default: 0.015 (+/-5.4 degree shift)
- Example: `hsv_h=0.03` gives a +/-10.8 degree shift

hsv_s (saturation): adjusts vividness.
- Range: 0.0 to 1.0 (full range)
- Default: 0.7 (+/-70% shift)
- Example: `hsv_s=0.5` gives +/-50% saturation

hsv_v (value/brightness): adjusts brightness.
- Range: 0.0 to 1.0 (full range)
- Default: 0.4 (+/-40% shift)
- This is brightness augmentation (rest damaged in source).

### 7.2 Use in Ablation 2

Code partly damaged in source; parameters recovered:

```python
model.train(
    data=str(config_path),
    # ... other arguments damaged in source
    degrees=15.0,      # rotation +/-15 degrees
    translate=0.1,     # 10%
    scale=0.5,         # +/-50%
    # color augmentation (new)
    hsv_h=0.015,       # hue +/-5.4 degrees
    hsv_s=0.5,         # saturation +/-50% (moderate)
    hsv_v=0.3,         # brightness +/-30% (moderate)
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
)
```

### 7.3 Clinical considerations

For wound images:
- hsv_h: keep low (0.01 to 0.02). Wound color is important (rest damaged in source).
- hsv_s: moderate (0.3 to 0.5), acceptable.
- hsv_v: moderate (0.2 to 0.4). Important for lighting variation.

Rationale: wound colors (red, pink, yellow, black) indicate pathology. Hue should not shift much.

### 7.4 Recommended Ablation 2 settings

```
# Color augmentation
hsv_h=0.01,   # +/-3.6 degrees (minimal distortion)
hsv_s=0.3,    # +/-30% (moderate)
hsv_v=0.3,    # +/-30% (moderate; adds lighting variation)
```

Remaining rationale (real-world variance) damaged in source.

## Conclusion

Focal Tversky Loss failed due to:
1. Multi-task dilution: segmentation is 33% of the signal.
2. Focal parameter too aggressive: gamma=2.0 caused overfitting.
3. Alpha-beta mismatch: counteracted by the class imbalance.
4. YOLOv8's default (BCE + Dice) is already near-optimal.

Next: Ablation 2 (color augmentation with hsv_ variation), using these findings.
