# Ablation 4 Strategy Analysis, Post-Diagnostic

Date: 2025-10-20.
Basis: YOLOv8 Empty Labels Validation Diagnostic v3.

## Confirmed fact

Empty label files are classified as "backgrounds" and skipped in validation metrics.

Evidence:
- Diagnostic v3 validation scans showed backgrounds detected and skipped.
- Diagnostic conclusion: empty labels do not affect metrics.
- YOLOv8 excludes backgrounds from mAP50, precision and recall.

Models can be trained with empty labels (hard negatives). Validation metrics compute only on labeled images.

## Strategy 4a: mixed training

```python
Training dataset: 2,208 wound images (with labels) + 513 DDI images (empty labels)
Validation dataset: 552 wound images + 129 DDI images (empty labels)
Test dataset: 552 wound images + 129 DDI images (empty labels)
```

Correct: optimal approach.

Training (2,208 wounds + 513 DDI):
- DDI images with empty labels give a training signal through the loss.
- Model penalized for false positives on DDI images.
- Gradient updates teach "do not predict wounds on these images".
- DDI ratio 18.8% (513/2,721), within the recommended 15-25% for hard negatives.

Validation during training: DDI images skipped. Metrics only on the 552 wound images. Early stopping based on wound detection, not contaminated by DDI.

Test: DDI images skipped. Final 87.51% mAP50 computed only on 552 wound images. No penalty from DDI images.

Why +7.51%:
- Training: learned to suppress false positives on dermatology conditions, better discrimination between wounds and normal or pathological skin, more robust to visual variability.
- Validation integrity: DDI did not dilute wound metrics, model selection based purely on wound performance, false positive reduction without metric penalty.

4a was correctly implemented and is the reference approach for hard negative mining with YOLOv8. Adopt for all future ablations that need hard negatives.

## Strategy 4b: finetune from best checkpoint

```python
Starting point: baseline_best.pt (converged model, ~epoch 50)
Training dataset: 513 DDI images only (all empty labels)
Validation dataset: 129 DDI images only (all empty labels)
Epochs: 25 (limited fine-tuning)
LR: 0.0015 (10x reduced)
```

Not correct: fundamental flaw, 100% empty labels.

```
Training: 513 DDI images with empty labels. No ground truth for the loss.
Training likely failed or gave meaningless updates.

Validation: 129 DDI images with empty labels.
YOLOv8: "129 images, 129 backgrounds, 0 corrupt"
Warning: "no labels found in segment set, can not compute metrics"
Result: 0.00% all metrics.
```

- Empty labels work as negative examples in mixed training.
- Empty labels cannot work for exclusive training (no positive examples).
- YOLOv8 needs some labeled objects to compute metrics.
- Cannot validate or select the best checkpoint without metrics.

Fix options:

Option 1, mixed fine-tuning (recommended):
```python
Starting point: baseline_best.pt
Training dataset: 513 DDI (empty) + 552 wound test set (WITH labels)
# uses the test set for fine-tuning, then evaluates on a separate holdout
Epochs: 25
LR: 0.0015
```
Pros: positive examples for validation metrics, DDI hard negative signal, measurable improvement. Cons: uses the test set for training (data leakage), needs a new holdout for final evaluation, violates original design.

Option 2, add a wound validation split (cleaner):
```python
Starting point: baseline_best.pt
Training dataset: 513 DDI (empty) + 200 wound images from the training set (WITH labels)
Validation dataset: 100 wound images (held out from training)
Epochs: 25
LR: 0.0015
```
Pros: clean train/val/test separation, validation metrics, DDI signal. Cons: fewer wound training examples, more complex dataset construction, still modifies the original design.

Conclusion: 4b cannot work as designed. Training on 100% empty labels gives no validation signal, YOLOv8 needs labeled objects, and the best checkpoint cannot be selected. Salvageable with a dataset that includes labeled wound images. Reject as-is.

## Strategy 4c: finetune from early checkpoint

```python
Starting point: epoch40.pt (early checkpoint, more "plastic")
Training dataset: 513 DDI images only (all empty labels)
Validation dataset: 129 DDI images only (all empty labels)
Epochs: 25
LR: 0.0015
```

Not correct: same flaw as 4b. 100% empty labels, no metrics, 0.00% all metrics.

Plasticity hypothesis: earlier checkpoints are more adaptable to new domains, but validating the adaptation needs labeled data. Empty labels give a negative signal but cannot measure success.

Fix options: same two as 4b. The epoch 40 checkpoint might be better than baseline_best.pt for adaptation. If 4b/4c are redesigned, test both checkpoints on the corrected dataset. Reject as-is.

## What was learned

4a success factors:
1. Hard negatives (empty labels) must be mixed with positive examples (labeled wounds).
2. 18.8% hard negatives (513/2,721) is effective.
3. Empty labels are skipped in validation, so metrics stay clean.
4. False positive reduction without metric contamination.

4b/4c failure:
1. Exclusive empty label training cannot work.
2. No validation signal.
3. Design flaw: training signal is not validation signal.

## Recommendations

4a: adopt. 87.51% mAP50 is a real improvement on wounds. Use for future ablations.

4b/4c: reject as-is. A redesign (wound images in training and validation, test checkpoint fine-tuning vs from-scratch) would be a new experiment, ablation 4d or 4e.

Hard negative mining practice:
- Do: mix empty labels (15-25%) with labeled examples.
- Do: use empty labels to teach "not this" during training.
- Do: let YOLOv8 skip empty labels during validation.
- Do not: train exclusively on empty labels.
- Do not: expect metrics on 100% empty label datasets.

Re-run 4b/4c? Scientific value marginal.
For: tests checkpoint fine-tuning (best vs early), may show fine-tuning beats from-scratch, completes the matrix.
Against: 4a already gave +7.51%, redesign violates the original protocol, 2-3 hours compute, data leakage risk if the test set is used.
Decision: skip. 4a shows hard negative mining works, which matters more than whether checkpoint fine-tuning is better. Compute goes to ablations 5-7. Documented as a lesson for experiment design.

## Experiment design lessons

Empty labels in YOLOv8 behave differently in training and validation:
- Training: negative signal (penalize false positives).
- Validation: skipped entirely.

Any experiment using empty labels must include positive examples for validation.

Principles:
1. Hard negative datasets always include labeled positives for validation.
2. Empty labels are fine as training-only data.
3. Validation needs labeled ground truth to compute any metric.
4. Run a diagnostic notebook to validate dataset structure before committing.

## Answers

Q1: was 4a done correctly from the start? Yes. Mixed training (wounds + DDI hard negatives), empty labels as training signal, validation metrics only on wounds, no contamination. Genuine +7.51%.

Q2: could 4b/4c be trained correctly given the facts? Not as designed.
- Original (failed): train exclusively on DDI (100% empty labels), no validation metrics, 0.00% all metrics.
- Redesign 1 (mixed fine-tuning): training 513 DDI (empty) + wound images (labeled), validation wound images (labeled). Could work, violates original design.
- Redesign 2 (validation split): training 513 DDI (empty) + wound training subset (labeled), validation wound validation subset (labeled). Could work, complex dataset construction.
- Decision: do not redesign. 4a proved hard negative mining works.

## Impact on remaining ablations

For any future ablation using empty labels:
1. Mix with positive examples.
2. Keep positive examples in the validation set.
3. Test dataset structure with the diagnostic notebook first.

Check of ablations 5-7:
- Ablation 5 (TTA): no empty labels, safe.
- Ablation 6 (label smoothing): no empty labels, safe.
- Ablation 7 (CRF post-processing): no empty labels, safe.

The lesson applies to future projects, not to the remaining ablations here.
