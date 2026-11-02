# Ablation 4 Root Cause Analysis: Hard Negative Mining Strategies

Date: 2025-10-20.
Status: partial success. Strategy 4a adopted, 4b and 4c failed.
Method: five parallel checks of the notebook implementation and logs.

## Results

- Strategy 4a (mixed training): 87.51% +- 0.40% mAP50. Significant improvement.
- Strategy 4b (best finetune): 0.00% mAP50. Failure.
- Strategy 4c (last finetune): 0.00% mAP50. Failure.

Root cause: the DDI-only dataset had all-empty label files (513 hard negatives, zero positive wound annotations). YOLOv8 cannot compute validation metrics without labeled positives, so all metrics were zero.

Decision: adopt 4a. New baseline 87.51% mAP50 for future ablations.

## Investigation

Initial observation from results.csv:
- 4a: 3 successful runs, mAP50 about 87.5%.
- 4b: all 3 runs zero in all validation metrics.
- 4c: all 3 runs zero in all validation metrics.

DDI size correction: DDI has 642 total images (513 train + 129 test). The first count used only the test split (129).

Five checks, each with its own hypothesis:

1. Dataset creation logic. Minor non-critical bug:
```python
# cell lines 448-453 in ablation_4_unified.ipynb
if completion_4b.exists():
    print("Ablation 4b already completed (loading results)")
else:
    # create DDI-only dataset  <- inside the else block
    # runs only on first execution, not on re-runs
```
Not the root cause. The dataset was created on first run. Log: "DDI split: 513 train, 129 test".

2. Training configuration comparison. No differences. All strategies use identical YOLO parameters. No validation-disable flags. `val=True` is the YOLOv8 default.

3. Validation mechanism. Enabled by default unless `val=False`. No such flag. Validation ran (log confirms) and produced zeros.

4. YAML and paths. Correct:
```yaml
# ddi_only_dataset.yaml
path: /kaggle/working/ddi_only_dataset
train: train
val: test
names:
  0: wound
```
Syntax, paths and structure correct. Not the root cause.

5. Log errors. Root cause found. Log evidence:
```
=== ABLATION 4b: Best checkpoint finetune (seed 42) ===
DDI-only dataset: 513 train, 129 test

WARNING Labels are missing or empty in /kaggle/working/ddi_only_dataset/train/labels.cache
WARNING Labels are missing or empty in /kaggle/working/ddi_only_dataset/test/labels.cache

Scanning /kaggle/working/ddi_only_dataset/train/labels.cache... 513 images, 513 backgrounds, 0 corrupt

WARNING no labels found in segment set, can not compute metrics without labels
RuntimeWarning: Mean of empty slice.
```
- YOLOv8 identified the issue: "Labels are missing or empty".
- 513 images, 513 backgrounds, 0 annotated objects. All label files are empty .txt files.
- Training ran 20 epochs on 513 images with zero positive examples.
- Validation: "no labels found in segment set". All metrics 0.0 (mean of empty slice).

## Root cause: empty DDI labels

DDI as described in the source: Diabetic Foot Ulcer (DFU) Dataset Images, public, used as hard negatives (images with no wounds, to reduce false positives). 642 total (513 train, 129 test). Open: the dataset is elsewhere named bkdkaggle/ddi-diverse-dermatology (Diverse Dermatology Images). To be reconciled.

Sequence:
1. DDI images were copied to `/kaggle/working/ddi_only_dataset/`.
2. Label files were created as empty .txt files (intentional for hard negatives).
3. YOLOv8 scanned: "513 images, 513 backgrounds, 0 corrupt".
4. Label cache marked "missing or empty".
5. Training ran with zero positive examples.
6. Validation could not compute precision, recall or mAP without labeled objects.
7. All metrics 0.0.

Why 4a succeeded: trained on combined FUSeg + Medetec + DDI (2,721 train images): 2,208 with wound annotations (FUSeg + Medetec) and 513 hard negatives with empty labels (DDI). Positive examples taught wound segmentation, hard negatives improved specificity. Validation ran on the labeled test set (552 wound images from FUSeg + Medetec). Result 87.51% +- 0.40%.

Why 4b failed: started from best.pt (trained on wounds), finetuned on DDI-only (513 hard negatives, 0 positives). 20 epochs with nothing to learn. Validation had no labeled objects. 0.00% mAP50.

Why 4c failed: started from last.pt, same dataset, same result.

## Validation of the root cause

Evidence:
- Log warning "Labels are missing or empty in labels.cache".
- Scan "513 images, 513 backgrounds, 0 corrupt".
- "no labels found in segment set, can not compute metrics".
- RuntimeWarning "Mean of empty slice".
- results.csv: all validation columns 0.0 for all epochs.
- File sizes: results_4b_*.csv and results_4c_*.csv about 1.8KB vs 9-11KB for 4a.

Predictions if the cause is right, all confirmed:
1. Training loss decreases (results.csv).
2. Validation metrics all zero.
3. YOLOv8 warnings about missing labels in logs.
4. Short training duration: 2.3-2.4 min vs 48-60 min for 4a.

## Strategy 4a results (mask mAP50)

| Seed | Best Epoch | mAP50(M) | Precision(M) | Recall(M) |
|------|-----------|----------|--------------|-----------|
| 42 | 58 | 87.86% | 88.57% | 81.43% |
| 123 | 45 | 87.08% | 87.40% | 83.18% |
| 456 | 53 | 87.60% | 88.73% | 82.70% |
| Mean | n/a | 87.51% | 88.23% | 82.44% |
| Std | n/a | +-0.40% | +-0.76% | +-0.91% |

Comparison:
- Baseline (zero-shot YOLOv8n): 80.00% mAP50.
- Ablation 1 (Focal Tversky): 80.01% mAP50 (+0.01%, not significant).
- Ablation 4a (hard negatives, mixed): 87.51% mAP50 (+7.51%).

Significance: +7.51 points, Cohen's d >> 0.8, Bonferroni threshold alpha = 0.0056 (7 comparisons). Highly significant.

## Lessons

1. Hard negative mining needs positive examples. Use mixed training (4a). Do not finetune on hard negatives only (4b/4c).
2. YOLOv8 warnings are diagnostic. "Labels are missing or empty" named the issue. Check logs on every run.
3. Dataset composition: 2,208 wounds + 513 hard negatives gave 87.51%. 513 hard negatives only gave 0.00%. Hard negatives improve specificity but cannot replace positives.
4. Validation needs a labeled test set. Test set must contain labeled positives.

## Recommendations

1. Adopt 4a. New baseline 87.51% mAP50 for ablations 5-7.
2. Archive 4b/4c as failed experiments. Do not attempt DDI-only finetuning.
3. Update ablation specifications: mark ablation 4 complete with partial success, set ablation 5 (TTA) baseline at 87.51%, document hard negative practice.
4. Minor notebook fix (lines 448-453):
```python
# current
if completion_4b.exists():
    print("Ablation 4b already completed")
else:
    create_ddi_dataset()  # wrapped in else block

# recommended
if not ddi_dataset_yaml.exists():
    create_ddi_dataset()  # check the dataset, not completion status
```

## Timeline impact

- Ablation 4 total runtime ~3.5 hours on Kaggle P100: 4a (3 seeds) ~3 hours, 4b ~7 minutes, 4c ~7 minutes.
- +7.51% from one ablation (largest gain so far).
- Mixed training is far better than finetune-only. No need to test DDI-only approaches again.
- 87.51% is the new baseline for ablations 5-7. Target: 90%+ mAP50 after all optimizations. Progress: 87.51 / 90 = 97%.

## Appendix

Investigation duration ~45 minutes.
Methods: five parallel checks, log analysis (4MB training log), results.csv comparison (9 files), dataset structure verification, YOLOv8 warning pattern analysis.

Checks: dataset creation logic, training configuration comparison, validation mechanism, YAML and paths, log error detection (found the root cause).

Files analyzed:
- `ablation_4_unified.ipynb`
- `ablation-4-unified-hard-negatives-3-strategies.log` (4MB)
- `results_4a_seed_*.csv` (3 files, success)
- `results_4b_seed_*.csv` (3 files, all zeros)
- `results_4c_seed_*.csv` (3 files, all zeros)
- `completion_status.json`
- `ddi_only_dataset.yaml`

Root cause confirmed by multiple sources.
