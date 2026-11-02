# Ablation 4 Redesign Proposal

Date: 2025-10-20.
Status: proposal.
Purpose: fix the flaw in the original 4b/4c design and test a meaningful hypothesis.

## Original design flaw

Original strategies 4b and 4c:
- Training: 100% DDI images (empty labels).
- Validation: 100% DDI images (empty labels).
- Result: metrics cannot be computed, all 0.00%.

No labeled examples means no validation metrics, no model selection, and a failed experiment.

## Redesign: mixed training with checkpoint comparison

Hypothesis: starting from a pretrained checkpoint improves hard negative mining compared to training from scratch.
Sub-hypothesis: checkpoint plasticity (early vs converged) affects adaptation to hard negatives.

### Baseline FPR measurement (new)

```python
Model: baseline_best.pt (no DDI training)
Test set: 129 DDI images (dermatology conditions)
Measurement: false positive rate (FPR)

for each DDI image:
    prediction = baseline_model.predict(image)
    if prediction.has_wound_detected():
        false_positive_count += 1

baseline_fpr = (false_positive_count / 129) * 100
```

Expected: 15-25% FPR (pre-registered hypothesis from methodology).

### Strategy 4a: mixed training from scratch (done)

```python
Starting point: YOLOv8n-seg pretrained weights (COCO)
Training dataset: 2,208 wound images (labeled) + 513 DDI images (empty labels)
Validation dataset: 552 wound images (labeled) + 129 DDI images (empty labels, skipped)
Epochs: 100
LR: 0.015
Seeds: [42, 123, 456]
```

Collected: wound detection 87.51% +- 0.40% mAP50. Not yet measured: FPR on DDI.

### Strategy 4b (redesigned): mixed training from best checkpoint

```python
Starting point: baseline_best.pt (converged model, ~epoch 50)
Training dataset: 2,208 wound images (labeled) + 513 DDI images (empty labels)  # changed
Validation dataset: 552 wound images (labeled) + 129 DDI images (empty, skipped)  # changed
Epochs: 25 (fine-tuning)
LR: 0.0015 (10x reduced to prevent catastrophic forgetting)
Seeds: [42, 123, 456]
```

Changes from the original: includes wound images (validation metrics computable), mixed training, both wound performance and FPR measurable.

Metrics to collect: mAP50 on the 552 wound test set, FPR reduction on the 129 DDI test set, time to convergence (should be shorter than 4a).

### Strategy 4c (redesigned): mixed training from plastic checkpoint

```python
Starting point: epoch40.pt (early checkpoint, more plastic)
Training dataset: 2,208 wound images (labeled) + 513 DDI images (empty labels)  # changed
Validation dataset: 552 wound images (labeled) + 129 DDI images (empty, skipped)  # changed
Epochs: 25
LR: 0.0015
Seeds: [42, 123, 456]
```

Hypothesis: an earlier checkpoint adapts better to hard negatives than a converged one.
Metrics: mAP50 on 552 wounds, FPR reduction on 129 DDI, convergence rate vs 4b.

## Scientific value

For:
1. Tests whether checkpoint fine-tuning beats from-scratch for hard negative mining: 4a (scratch, 100 epochs), 4b (best checkpoint, 25 epochs), 4c (plastic checkpoint, 25 epochs).
2. Efficiency: if 4b/4c match 4a in 25% of the time. 4a runtime ~1.5-2 hours (100 epochs), 4b/4c ~30-45 minutes (25 epochs). Potential 75% time savings.
3. Plasticity: foundation model literature (2024) suggests early checkpoints adapt better. 4c > 4b supports it. 4b > 4c suggests convergence stability matters more.
4. FPR analysis: baseline FPR expected 15-25%. Measure 4a/4b/4c actual improvement. Deployment readiness: FPR < 15%?
5. Validates mixed training as the method and gives a reusable pattern.

Against:
1. Compute: 2-3 GPU hours for 4b and 4c (3 seeds x 25 epochs x 2 strategies).
2. Diminishing returns: 4a already gave +7.51%. The question is "faster", not "better". 4b/4c may only match 4a.
3. Remaining ablations (5, 6, 7) compete for time.
4. Redesign departs from the pre-registered experiment. Counter: the original was flawed (100% empty labels). This is a corrected design.

## Expected outcomes

1. 4b/4c match 4a (~87% mAP50, similar FPR reduction): fine-tuning is faster (25 vs 100 epochs), 75% time savings, adopt checkpoint fine-tuning. Value: high.
2. 4a outperforms 4b/4c (4a 87.51%, 4b/4c ~85-86%): from-scratch learns better discrimination, confirms 4a. Value: medium.
3. 4c outperforms 4b in FPR reduction: supports the plasticity theory, guidance for transfer learning. Value: medium-high.
4. 4b/4c exceed 4a (>87.51% mAP50, better FPR reduction): fine-tuning superior, re-adopt 4b or 4c as baseline. Value: very high.

## Resources

| Task | Time | GPU |
|------|------|-----|
| Baseline FPR (129 DDI images) | ~5 min | none (CPU) |
| 4a FPR (3 seeds, 129 DDI images) | ~15 min | none (CPU) |
| 4b training (best checkpoint, mixed dataset, 3 seeds x 25 epochs) | 30-45 min | P100 |
| 4c training (epoch40 checkpoint, mixed dataset, 3 seeds x 25 epochs) | 30-45 min | P100 |
| 4b FPR (3 seeds) | ~15 min | none |
| 4c FPR (3 seeds) | ~15 min | none |

Total new compute: GPU ~1.5 hours, CPU ~50 minutes, wall time ~2 hours.

## Checklist

### Phase 1: FPR measurement (low cost, high value; do regardless of the 4b/4c decision)

1. Baseline FPR:
   - [ ] Load baseline_best.pt.
   - [ ] Run inference on 129 DDI test images.
   - [ ] Count false positives (any wound detection on a DDI image).
   - [ ] baseline_fpr = (fp_count / 129) * 100.
   - [ ] Per-condition FPR breakdown.
2. Strategy 4a FPR:
   - [ ] Load 4a models (seeds 42, 123, 456).
   - [ ] Inference on 129 DDI test images.
   - [ ] FPR per seed, mean +- std.
   - [ ] FPR reduction vs baseline.
3. Analysis:
   - [ ] Compare baseline vs 4a FPR.
   - [ ] Which dermatology conditions improved most.
   - [ ] Whether FPR < 15% (clinical deployment threshold).

Time: 30-60 minutes. No training required.

### Phase 2: redesigned 4b/4c (optional, depends on phase 1)

Decision criteria:
- Baseline FPR < 10%: skip (4a already excellent).
- Baseline FPR 10-20%: consider (moderate improvement possible).
- Baseline FPR > 20%: recommended (significant problem).

Steps:
1. Dataset:
   - [ ] Create mixed_training_dataset.yaml (2,208 wounds + 513 DDI).
   - [ ] Validate structure (labeled wounds + empty DDI).
   - [ ] 5-epoch smoke test to verify metrics compute.
2. 4b training:
   - [ ] Load baseline_best.pt.
   - [ ] 25 epochs, LR=0.0015, seeds [42, 123, 456].
   - [ ] Wound mAP50 should stay about 87%.
   - [ ] Save best models per seed.
3. 4c training:
   - [ ] Load epoch40.pt.
   - [ ] 25 epochs, LR=0.0015, seeds [42, 123, 456].
   - [ ] Wound mAP50 should stay about 87%.
   - [ ] Save best models per seed.
4. FPR:
   - [ ] Run 4b and 4c models on 129 DDI test images.
   - [ ] Compare baseline, 4a, 4b, 4c.
5. Analysis:
   - [ ] Wound mAP50 across 4a, 4b, 4c.
   - [ ] FPR reduction across 4a, 4b, 4c.
   - [ ] Training time: 4a (100 epochs) vs 4b/4c (25 epochs).
   - [ ] Recommendation: from-scratch vs fine-tuning.

Time: 2-3 hours.

## Recommendation

Phase 1: do it. 30-60 minutes, no GPU training, answers whether hard negative mining reduced FPR (currently unknown), gives a deployment readiness metric.

Phase 2: lean toward skip.
- If baseline FPR > 20% and efficiency gains matter: run.
- If baseline FPR 10-20%: consider. Factor: how many future ablations need hard negatives.
- If baseline FPR < 10%: skip.
- If compute budget is tight: skip.

Reasons to skip:
1. 4a already succeeded (+7.51%).
2. Ablations 5-7 (TTA, label smoothing, CRF) test new factors.
3. The efficiency question (25 vs 100 epochs) is secondary.
4. 4b/4c unlikely to exceed 4a.
5. ~2 hours saved for ablations 5-7.

Exceptions to run 4b/4c:
- Phase 1 shows baseline FPR > 25%.
- Many more hard negative ablations expected.
- Publishing on checkpoint plasticity for hard negatives.

## Alternative: document as a lesson

Instead of rerunning 4b/4c, document:
1. Error: original 4b/4c design flaw (100% empty labels).
2. Corrected design.
3. Future guidance: hard negative fine-tuning practice.
4. Open research question: does checkpoint fine-tuning beat from-scratch? (unanswered, low priority).

Preserves the lesson, uses no compute, keeps effort on ablations 5-7.

## Summary

The redesign is sound and better than the original 4b/4c. Phase 1 (FPR measurement): do. Phase 2 (4b/4c redesign): probably skip, focus on ablations 5-7. Both decisions are defensible. Decision open.
