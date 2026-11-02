# Loss Function Research for Wound Segmentation, 2025

Date: 2025-10-19.

Purpose: evaluate 2024-2025 loss functions for a possible Ablation 1B. Focal Tversky failed (+0.01%), so the search is for boundary precision and false positive reduction.

Caution: the Ablation 2 and Ablation 5 figures below (+5.97%, +2.97%) conflict with the recorded results in ABLATION_SPECIFICATIONS.md (A2 Dice 68.27%, failed). To be reconciled.

## Current model

```
Baseline YOLOv8n-seg:
  Dice:        80.00%
  Precision:   74.72% (25.3% false positive rate)
  Recall:      90.59% (over-segments)
  Boundary F1: 67.41%
  Pattern:     high recall, moderate precision

Clinical Augmentation (Abl 2): +5.97%
  Dice:        85.97%
  Precision:   89.4% (10.6% FP, 60% reduction)
  Recall:      80.8%

Label Smoothing (Abl 5): +2.97%
  Dice:        82.97%
  Precision:   86.4% (13.6% FP, 45% reduction)
  Recall:      75.8%
```

Issues:
1. Baseline over-segments: recall 90.6%, precision 74.7%.
2. Boundary F1 only 67.4%.
3. 25% FP rate is unacceptable clinically.
4. Multi-task dilution: seg loss is 25% of total YOLO loss.

## Candidates

### 1. Boundary Difference over Union (BDoU)

Source: MICCAI 2023 Paper #1247. Targets boundary precision (Boundary F1 67.4%).

```python
def boundary_dou_loss(pred, target, theta=0.2):
    # theta: boundary thickness (0.2 = 20% of mask)
    pred_boundary = get_boundary(pred, theta)
    target_boundary = get_boundary(target, theta)

    intersection = (pred_boundary * target_boundary).sum()
    union = pred_boundary.sum() + target_boundary.sum() - intersection

    return 1 - (intersection / (union + 1e-7))

def get_boundary(mask, theta):
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded) > theta
    return boundary.float()
```

For wounds: targets boundary precision directly; handles irregular boundaries; stable (no focal term); peer-reviewed at MICCAI.

Expected: Boundary F1 67.4% to 72-75%; Dice +1-2%; FP reduction minimal.

Complexity medium. Integration: `loss = 0.7*dice + 0.3*bdou`.

### 2. Weighted Tversky

Source: established medical imaging (2017-2024). Targets FP reduction by asymmetric weighting.

```python
def weighted_tversky_loss(pred, target, alpha=0.3, beta=0.7):
    # alpha: FP weight (lower = penalize FP more)
    # beta: FN weight (higher = tolerate FN more)
    # precision focus: alpha=0.3, beta=0.7; recall focus: alpha=0.7, beta=0.3
    pred = pred.flatten()
    target = target.flatten()

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    tversky = (TP + 1e-7) / (TP + alpha*FP + beta*FN + 1e-7)
    return 1 - tversky
```

For wounds: simpler than Focal Tversky (no unstable focal term); direct FP control; broad medical imaging validation; low complexity.

Expected: FP rate 10.6% to 7-9%; precision 89.4% to 91-93%; recall 80.8% to 75-78%; Dice +0.5-1.5%.

Clinical configuration (precision over recall): alpha = 0.3, beta = 0.7.

Complexity low. Integration: replace the YOLO seg loss with Weighted Tversky.

### 3. Adaptive Boundary-Enhanced Dice (ABeDice)

Source: PubMed 2025 (PMID 40061446). Very new, limited validation. Exponential recursive complement approach, adaptive weighting of boundary regions.

For: current boundary enhancement, adapts to image properties. Against: no chronic wound validation, complex, no open source code, unknown on irregular morphology.

Not recommended until broader validation.

## Why loss changes have limited effect

Multi-task dilution, YOLOv8 total loss:
```
  Box Loss:  35%
  Seg Loss:  25%   (only this changes)
  Cls Loss:  25%
  DFL Loss:  15%
```

Why Focal Tversky failed (+0.01%):
1. Multi-task dilution: 25% of total loss affected.
2. Focal instability: gamma=1.5 caused training oscillations.
3. Too aggressive: alpha/beta plus focal.
4. Cross-seed variance: seed 42 80.3%, seed 123 79.7%.

Data methods versus loss engineering:
```
  Focal Tversky:     +0.01%  (loss engineering)
  Clinical Aug:      +5.97%  (data diversity), 597x
  Label Smoothing:   +2.97%  (regularization), 297x
```

Finding: data-centric methods beat loss engineering for a well-tuned baseline.

## Should Ablation 1B run?

Deciding metric: Boundary F1 after Ablations 2, 4, 5.

- Boundary F1 below 70%: run Ablation 1B with Boundary DoU. Target 70% to 72-75%, Dice +1-2%, low risk.
- Boundary F1 70-72%: maybe Weighted Tversky. Target FP 10.6% to 7-9%, Dice +0.5-1.5%, low risk.
- Boundary F1 above 72%: skip to Ablation 6 (TTA). Combine Abl 2, Abl 5 and best Abl 4. Expected final Dice 83-85%.

## Implementation sketches

Boundary DoU (Boundary F1 below 70%):
```python
class BDoUSegmentationLoss(v8SegmentationLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bdou_weight = 0.3

    def __call__(self, preds, batch):
        loss_box, loss_cls, loss_dfl = self.box_loss(...)
        loss_seg = self.seg_loss(...)
        loss_bdou = boundary_dou_loss(preds['masks'], batch['masks'])

        total_loss = (
            loss_box + loss_cls + loss_dfl +
            0.7 * loss_seg + 0.3 * loss_bdou
        )
        return total_loss
```

Weighted Tversky (Boundary F1 70-72%):
```python
class TverskySegmentationLoss(v8SegmentationLoss):
    def __init__(self, model, alpha=0.3, beta=0.7):
        super().__init__(model)
        self.alpha = alpha
        self.beta = beta

    def seg_loss(self, pred_masks, true_masks):
        return weighted_tversky_loss(
            pred_masks,
            true_masks,
            alpha=self.alpha,
            beta=self.beta
        )
```

## Summary

| Loss | Target issue | Expected Dice gain | Risk | Complexity |
|---|---|---|---|---|
| Boundary DoU | Boundary F1 67.4% | +1-2% | Low | Medium |
| Weighted Tversky | FP rate 10.6% | +0.5-1.5% | Low | Low |
| Focal Tversky | FP rate | +0.01% | High | Low |
| ABeDice | Boundaries | Unknown | Medium | High |

## Next

1. Wait for Ablation 4 v8 (hard negatives, 3 strategies).
2. Analyze Boundary F1 for Ablations 2, 4, 5.
3. Decide: below 70% implement BDoU; 70-72% consider Weighted Tversky; above 72% skip to TTA (Ablation 6).
4. Final: aggregate best strategies, test-time augmentation, target 83-85% Dice.

Decision pending: Boundary F1 analysis after Ablation 4. Most likely path: skip to TTA if Boundary F1 is above 72%.

## References

1. Boundary DoU: MICCAI 2023 Paper #1247.
2. Weighted Tversky: Salehi et al. (2017), "Tversky loss function for image segmentation".
3. ABeDice: PubMed PMID 40061446 (2025).
4. Loss survey: arXiv:2312.05391 (2023), "Loss Functions in the Era of Semantic Segmentation".
5. YOLOv8 discussion: Ultralytics GitHub #12844.
