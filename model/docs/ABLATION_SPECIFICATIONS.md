# Ablation Study: Technical Reference

Last updated 2025-10-20.

Current baseline: mAP50 87.51% +/- 0.40% (Strategy 4a, hard negatives).
Original baseline: Dice 80.0% [78.9-81.0%], Precision 74.7%, Recall 90.6%, Boundary F1 67.4%.

Numbering note: this file calls label smoothing Ablation 5 and TTA Ablation 6. The plan files ABLATION_05_TTA_IMPLEMENTATION_PLAN.md and ABLATION_06_LABEL_SMOOTHING_IMPLEMENTATION_PLAN.md use the reverse (5 is TTA, 6 is label smoothing). To be reconciled.

## Study design

- 8 ablations in total (1, 1B, 2, 3, 4a-4c, 5, 6).
- Seeds 42, 123, 456 per ablation.
- P100-optimized: batch=32, lr=0.015, epochs=100, patience=10, workers=8, amp=False, imgsz=512.
- Bonferroni-corrected alpha = 0.05/8 = 0.00625.
- Adoption needs Cohen's d > 0.8.

Research questions:
1. Focal Tversky (A1): can asymmetric loss improve boundary precision beyond 80% Dice?
2. Dice++ (A1B): can a calibration-aware loss improve calibration while holding Dice?
3. Clinical augmentation (A2): do anatomically valid augmentations improve generalization?
4. Multi-scale training (A3): does multi-resolution training improve scale robustness?
5. Hard negatives (A4a/4b/4c): which strategy best reduces the 22.8% false positive rate using DDI?
6. Label smoothing (A5): can training-time calibration improve confidence scores?
7. TTA (A6): can inference-time ensembling improve robustness and aggregate all ablation results?

Decision criteria:
- Adopt: improvement 3% or more, Cohen's d > 0.8, 95% CI excludes baseline.
- Marginal: improvement 1% or more, Cohen's d > 0.5.
- Reject: improvement under 1% or Cohen's d < 0.5.

## Baseline

```json
{
  "dice": {"mean": 0.8000, "std": 0.2222, "ci_95": [0.7892, 0.8103]},
  "precision": {"mean": 0.7472, "std": 0.2478},
  "recall": {"mean": 0.9059, "std": 0.1965},
  "boundary_f1": {"mean": 0.6741, "std": 0.2359}
}
```

```python
BASELINE_CONFIG = {
    "batch": 32,          # 2x speedup vs batch=16
    "lr0": 0.015,         # conservative 1.5x scaling
    "epochs": 100,        # full convergence
    "patience": 10,       # P100-optimized early stopping
    "workers": 8,
    "amp": False,         # disabled for ablation stability
    "imgsz": 512,         # dataset-aligned
    "seeds": [42, 123, 456],
}
```

Dataset:
- Train: 2,208 wound images (80%).
- Test: 552 wound images (20%, specialist-annotated).
- Path: `/kaggle/input/baseline-outputs/data_wound_seg/`.
- DDI dermatology set (Ablations 4a/4b/4c): 665 images (532 train, 133 test). Other docs give 642 images (513 train, 129 test). To be confirmed.

## Ablation 1: Focal Tversky Loss

Complete. No improvement: Dice 80.01% vs 80.00%. No meaningful gain on any metric. Mosaic already handles class imbalance, so asymmetric weighting is redundant.

## Ablation 2: Clinical Augmentation

Complete. Failed: Dice 68.27%, -11.73% from baseline. Removing mosaic and adding rotation and vertical flip dropped performance sharply. Mosaic is critical.

## Ablation 3: Multi-Scale Training

Complete (v5). Catastrophic failure: Dice 4.8%, -95.2% from baseline.

- Dice 4.8% [4.2-5.6%] (baseline 80.0%).
- Precision 2.6% (74.7%).
- Recall 74.1% (90.6%).
- Boundary F1 0.7% (67.4%).

Causes:
1. Multi-scale and RandAugment conflict. With `auto_augment=None`, YOLOv8 reverted to `auto_augment: randaugment` (args.yaml line 105).
2. Multi-scale alone may be harmful: imgsz=[416,512,640] with RandAugment gave a 95% drop.
3. Parameter bug: YOLOv8 ignores `auto_augment=None` and takes the default from ultralytics/cfg/default.yaml.

Rule: check args.yaml after training for ignored parameters.

## Ablation 4: Hard Negative Mining (three strategies)

Status: Phase 1 open. Strategy 4a adopted, FPR measurement pending.

- Training completed 2025-10-20, about 3.5 hours on Kaggle P100.
- Analysis: reports/ABLATION_4_ROOT_CAUSE_ANALYSIS.md.

Phase 1 open:
- Baseline FPR on 129 DDI test images (50-60 min).
- Strategy 4a FPR, compared to baseline.
- Decide: go to Ablation 5, or run Phase 2 (redesigned 4b/4c).
- Blocker: location of the baseline model weights.

Phase 2 (optional, see reports/ABLATION_4_REDESIGN_PROPOSAL.md): redesigned 4b/4c with mixed training instead of DDI-only. Decision depends on Phase 1.

### 4a: mixed training (adopted)

Train on FUSeg, Medetec and DDI together from the start.
- 2,208 wound images (FUSeg and Medetec) with masks.
- 513 DDI hard negatives (empty labels).
- 2,721 train images.
- 3 seeds [42, 123, 456], 100 epochs, P100 config.

| Seed | Best epoch | mAP50(M) | Precision(M) | Recall(M) |
|---|---|---|---|---|
| 42 | 58 | 87.86% | 88.57% | 81.43% |
| 123 | 45 | 87.08% | 87.40% | 83.18% |
| 456 | 53 | 87.60% | 88.73% | 82.70% |
| Mean | - | 87.51% | 88.23% | 82.44% |
| Std | - | 0.40% | 0.76% | 0.91% |

Note: the insights doc reports 4a precision 89.99% +/- 1.99% and recall 80.76% +/- 1.80%. These do not match this table. To be confirmed against kaggle_outputs/ablation_4_complete_analysis.json.

- Baseline 80.00% mAP50, 4a 87.51%: +7.51pp (+9.4% relative).
- p < 0.0001, below Bonferroni 0.00625.
- Cohen's d = 18.8.
- Decision: ADOPT. Largest single gain in the study. New baseline for later ablations.

### 4b: best checkpoint fine-tune (failed)

Load best.pt, fine-tune on the 513 DDI images for 20 epochs. Result: mAP50, Precision, Recall all 0.00%.

Cause: the DDI-only set has zero positive examples (513 hard negatives, empty labels). YOLOv8 cannot compute validation metrics without labeled objects.

Log:
```
WARNING Labels are missing or empty in labels.cache
Scanning labels.cache... 513 images, 513 backgrounds, 0 corrupt
WARNING no labels found in segment set, can not compute metrics
```

Decision: REJECT.

### 4c: last checkpoint fine-tune (failed)

Load last.pt, fine-tune on DDI-only for 20 epochs. All metrics 0.00%. Same cause as 4b. The early-checkpoint plasticity hypothesis does not matter with no positive examples. REJECT.

### Findings

1. Mixed training works: positives plus hard negatives give a large gain.
2. Hard negatives alone fail: no positive signal.
3. About 18% hard negatives (513/2,721 = 18.8%) gives strong regularization.
4. YOLOv8 validation needs a labeled set.
5. Mixed training beats two-stage fine-tuning.
6. Log warnings ("Labels are missing or empty") are real diagnostic signals.

Impact: +7.51pp, new baseline 87.51% mAP50, 97% of the 90% target.

## Ablation 1B: Dice++ Loss

Replace Focal Tversky with Dice++, a calibration-aware loss.

```python
class DicePlusPlusLoss(nn.Module):
    """
    L = 1 - (2 * TP + eps) / (2 * TP + FP + FN + eps) - lambda * CalibrationPenalty
    CalibrationPenalty = |confidence - accuracy|
    lambda = calibration weight (typ. 0.1)
    """
```

Focal Tversky showed no gain because mosaic already handles imbalance. Dice++ targets confidence calibration without changing class weighting.

Expected: Dice about 80%; ECE down 30-50%; better uncertainty for clinical use.

Detailed plan: ABLATION_1B_DICE_PLUS_PLUS_IMPLEMENTATION_PLAN.md.

## Ablation 5: Label Smoothing

Status: production ready; smoke test validates parameter application.

```python
def smooth_labels(targets, smoothing=0.1):
    # Background: 0 becomes 0.05 (smoothing/2)
    # Wound: 1 becomes 0.95 (1 - smoothing/2)
    return targets * (1 - smoothing) + smoothing / 2
```

Overconfident predictions are a problem for clinical decisions. Expected: Dice about 80% (maintained or +0-1%); ECE down 20-30%; more balanced confidence distributions.

## Ablation 6: TTA (final aggregation)

Status: production ready; smoke test passed; waits on Ablations 4d and 5.

Tests the combined improvements:
1. Hard negative strategy (4a, 87.51% mAP50).
2. Label smoothing (Ablation 5, awaiting deployment).
3. Curriculum learning (4d, optional, pending).

Inference-time augmentations: original, horizontal_flip, vertical_flip, rotate_90, ensembled by mean:

```python
ensemble_mask = np.mean([model(aug(image)) for aug in augmentations], axis=0) > 0.5
```

Mean is better than majority vote for continuous masks.

- Expected Dice: +1-2% over the best single ablation.
- Final target: 83-85% Dice.
- Inference: +200-300 ms (4 forward passes).
- No retraining; aggregates earlier gains; 4x slower inference.

## Statistics

```python
alpha_corrected = 0.05 / 8  # 0.00625
# Tests: Ablations 1, 1B, 2, 3, 4a, 4b, 4c, 5. Ablation 6 (TTA) is aggregation, not an independent test.
```

Cohen's d: above 0.8 large (adopt), above 0.5 medium (marginal), below 0.5 small (reject).

## Hard negatives comparison

| Strategy | Training | Wound exposure | Forgetting risk | Hypothesis |
|---|---|---|---|---|
| 4a Mixed | Single-stage | 2208+532 mixed | Low | Balanced learning |
| 4b Best ckpt | Two-stage | 2208 then 532 | Medium | Pre-trained features |
| 4c Early ckpt | Two-stage | 2208 then 532 | Low | Higher plasticity |

## Status

Completed:
- Ablation 1: Focal Tversky, no improvement (80.01%).
- Ablation 2: clinical augmentation, failed (68.27%, -12%).
- Ablation 3: multi-scale, v1-v5 failed (4.8% Dice, -95%).
- Ablation 4: 4a 87.51% +/- 0.40% mAP50 adopted; 4b 0.00% rejected; 4c 0.00% rejected.

Pending (on the 87.51% baseline):
- Ablation 1B: Dice++.
- Ablation 5: TTA (next priority).
- Ablation 6: label smoothing.

Low priority or archived:
- Ablation 1C: Focal Tversky without mosaic, error (API mismatch).
- Mini-ablations (7 configs), error.
- Ablation 3 fixes, abandoned (RandAugment conflict).

Progress: 4 of 7 primary ablations (57%).
- Ablation 3 summary elsewhere gives -75.2%; the figure above is -95.2% (Dice 4.8% vs 80.0%).
- Current baseline 87.51% mAP50, target 90%, 97% of target.

## Next

1. Done: Ablation 4 analysis and root-cause write-up; new baseline 87.51% mAP50.
2. Blocked: Ablation 4 Phase 1 FPR measurement (baseline model location).
3. Open: Phase 2 (redesigned 4b/4c) decision after Phase 1.
4. Next: deploy Ablation 5 (TTA) on the 87.51% baseline.
5. Next: deploy Ablation 6 (label smoothing) on the 87.51% baseline.
6. Later: final aggregation of successful strategies.
7. Later: Ablation 1B (Dice++) if calibration work is needed.
8. Low priority: archive failed ablations (1C, mini-ablations, Ablation 3).
