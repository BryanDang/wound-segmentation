# Ablation 4: Hard Negative Mining, Key Findings

Date: 2025-10-20.
Status at the time: incomplete, decision open.

Later reports (ABLATION_4_ROOT_CAUSE_ANALYSIS.md, ABLATION_4_EMPTY_LABELS_CONCLUSIVE_ANSWER.md) explain the missing 4b/4c metrics: DDI-only datasets have 100% empty labels and YOLOv8 cannot compute metrics on them. They also give 4a as 87.51% +- 0.40%, versus 87.43% here.

## Summary

- Strategy 4a (mixed training): 87.43% mAP50, +7.4 points over baseline.
- Strategies 4b and 4c: trained, validation metrics missing.
- Open: continue without 4b/4c data, or rerun with validation.

## Strategy 4a: mixed training on combined dataset

- Mask mAP50: 87.43% +- 0.35%.
- Improvement: +7.43 points over the 80% baseline.
- Seeds: 42 (87.80%), 123 (87.10%), 456 (87.40%).
- Training time: ~56 minutes per seed.
- Significance: very high, consistent across seeds.
- Precision 89.67% (few false positives). Recall 81.40% (catches most wounds). Slightly conservative, precision over recall.

## Strategies 4b and 4c: finetuning

- Models trained (2.3-2.4 min per seed).
- No validation metrics in the logs. No comparison to 4a or baseline possible.
- 4b/4c might be better than 4a (faster, potential gain) or worse (smaller dataset, overfitting). Unknown.

Cause as assessed at the time: finetuning runs were very short (2.3-2.4 min) and validation may not have triggered. Logs show "Results saved" for 4b/4c, no validation metrics in standard format, completion status shows all runs finished, no validation tables. Likely: validation disabled or early stopping prevented it.

## Options

### Option 1: accept 4a and move on (fastest)
- Adopt 4a, skip 4b/4c evaluation. Time 0 hours. Saves ~3 hours of GPU time. Proceed to ablation 5 (TTA).
- Cons: might miss a better strategy. Incomplete experiment. Unknown whether finetuning on DDI helps.
- Use when time-constrained.

### Option 2: diagnostic validation (recommended)
- Download trained 4b/4c weights and run standalone validation. ~30 minutes setup plus 10 minutes validation.
- Pros: no retraining, quick answer, ~10 minutes GPU.
- Cons: weights download (~100MB), manual validation script, slightly less rigorous than a rerun.
- Reasoning: models already trained, validation is cheap. If 4b/4c are close to 4a, option 1 is validated. If significantly better, option 3 is worthwhile.
- Steps:
  1. Check whether model weights are in the Kaggle kernel output.
  2. If not, use existing 4a models as proxy (best.pt checkpoints).
  3. Run validation on the test set.
  4. Compare 4a, 4b, 4c.
  5. Decide ADOPT, MARGINAL or REJECT with complete data.

### Option 3: complete rerun (most thorough)
- Rerun 4b and 4c with forced validation. ~15 minutes GPU, ~1 hour total including setup.
- Pros: complete data, full statistical comparison.
- Cons: notebook changes, another Kaggle deployment cycle.
- Use when completeness is the priority.

## Timeline impact

- Option 1: none, proceed to ablation 5.
- Option 2: +30 minutes.
- Option 3: +1-2 hours, one session delay.

## Questions before deciding

1. Are model weights available in the kernel output? Yes: option 2 is easy. No: option 1 or 3.
2. Completeness vs timeline? Timeline: option 1. Balance: option 2. Rigor: option 3.
3. Is 4b/4c suspected to be better? Yes: option 2 or 3. No: option 1.

## Diagnostic validation sketch

```python
from ultralytics import YOLO

model_4a = YOLO("yolo_training_4a/seed_42/weights/best.pt")
model_4b = YOLO("yolo_training_4b/seed_42/weights/best.pt")
model_4c = YOLO("yolo_training_4c/seed_42/weights/best.pt")

metrics_4a = model_4a.val(data="dataset.yaml", split="test")
metrics_4b = model_4b.val(data="dataset.yaml", split="test")
metrics_4c = model_4c.val(data="dataset.yaml", split="test")

print(f"4a: {metrics_4a.seg.map50:.4f}")
print(f"4b: {metrics_4b.seg.map50:.4f}")
print(f"4c: {metrics_4c.seg.map50:.4f}")
```

~3 minutes per model, ~10 minutes total.

## Decision status

Open between options 1, 2 and 3. Default: option 2.
Confidence: high for 4a, unknown for 4b/4c.
Created: 2025-10-20.
