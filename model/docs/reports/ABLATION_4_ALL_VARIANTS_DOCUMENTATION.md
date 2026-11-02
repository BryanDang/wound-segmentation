# Ablation 4: Variant Documentation

Date: 2025-10-22.
Scope: all ablation 4 notebooks and directories. Method: file system analysis and kernel metadata extraction. 9 directories.

Ablation 4 tests hard negative mining with the DDI dermatology dataset (665 images: 532 train, 133 test, as written here). Open: other reports give 642 (513 train, 129 test), 643 and 859 DDI images. To be reconciled.

Directories: 9.
- Production: 4_unified, 4a, 4d_production, 4b and 4c.
- Smoke tests: 4_smoke, 4d_smoke, 4a_mixed.
- Deprecated: 4_TBD (empty), 4a_hard_negatives_mixed (superseded).

## Production notebooks

### 1. ablation_4_unified/ (adopted, strategy 4a)

- Location: `notebooks/03_ablations/ablation_4_unified/`.
- Status: complete. 87.51% mAP50, +7.51% vs baseline.
- Kernel: [bkdkaggle/ablation-4-unified-hard-negatives-3-strategies](https://www.kaggle.com/code/bkdkaggle/ablation-4-unified-hard-negatives-3-strategies).

Three strategies in one notebook:
- 4a, mixed training (all wound and DDI images from the start): adopted.
- 4b, best checkpoint (100% DDI-only, resume from best baseline checkpoint): failed. Metrics cannot be computed on 100% empty labels.
- 4c, early checkpoint (100% DDI-only, resume from early baseline checkpoint): failed. Same reason.

Files:
- `ablation_4_unified_hard_negatives.ipynb` (14KB): production notebook (GPU, 100 epochs).
- `ablation_4_unified.ipynb` (32KB): earlier version with all 3 strategies.
- `ablation_4_unified_clean.ipynb` (31KB): cleaned version.
- `kernel-metadata.json`: Kaggle deployment configuration.

Strategy 4a results:
```
Seed 42:  87.86% mAP50, 88.57% Precision, 81.43% Recall
Seed 123: 87.08% mAP50, 87.40% Precision, 83.18% Recall
Seed 456: 87.60% mAP50, 88.73% Precision, 82.70% Recall
Mean:     87.51% +- 0.40% mAP50
```

Mixed training (wounds and DDI from the start) is far better than two-stage finetuning on DDI-only.

### 2. ablation_4a_hard_negatives_mixed/ (superseded by 4_unified)

- Location: `notebooks/03_ablations/ablation_4a_hard_negatives_mixed/`.
- Kernel: [bkdkaggle/ablation-4a-hard-negatives-mixed](https://www.kaggle.com/code/bkdkaggle/ablation-4a-hard-negatives-mixed).
- Files: `ablation_4a_hard_negatives_mixed.ipynb` (50KB, original), `ablation_4a_hard_negatives_mixed_clean.ipynb` (49KB), `kernel-metadata.json`.
- Same strategy 4a implementation as 4_unified, created before consolidation. Results identical. Use `ablation_4_unified/`.

### 3. ablation_4b_best_checkpoint/ (failed)

- Location: `notebooks/03_ablations/ablation_4b_best_checkpoint/`.
- Strategy: finetune the baseline's best checkpoint on 100% DDI-only data.
- Why failed: YOLOv8 validation classifies 100% empty-label datasets as "backgrounds" and skips them. mAP50, precision and recall cannot be computed with zero labeled images.
- Files: `ablation_4b_best_checkpoint.ipynb` (19KB), `ablation_4b_best_checkpoint_clean.ipynb` (19KB), `kernel-metadata.json`.
- Lesson: training on pure hard negatives gives gradient updates, but validation needs some labeled images.

### 4. ablation_4c_early_checkpoint/ (failed)

- Location: `notebooks/03_ablations/ablation_4c_early_checkpoint/`.
- Strategy: finetune the baseline's early checkpoint (epoch 10) on 100% DDI-only data.
- Why failed: same as 4b.
- Files: `ablation_4c_early_checkpoint.ipynb` (19KB), `ablation_4c_early_checkpoint_clean.ipynb` (19KB), `kernel-metadata.json`.
- Rationale: an early checkpoint overfits less, so it should finetune better. The metric limitation blocked the test.

### 5. ablation_4d_production/ (ready, not deployed)

- Location: `notebooks/03_ablations/ablation_4d_production/`.
- Strategy: curriculum learning, two approaches.
  - 4d.2: progressive hard negatives (DDI curriculum only, wounds constant).
  - 4d.3: bilateral curriculum (wounds and DDI both on curriculum).

Files:
- `ablation_4d_curriculum_learning.ipynb` (27KB): production notebook.
  - Uses `train_images` for hardness scoring (correct).
  - ultralytics >= 8.3.40 (fixes parameter control bugs).
  - 100 epochs per strategy (33+33+34 split into 3 phases).
  - Seeds [42, 123, 456]. GPU-enabled.
- `ablation_4d_minimal_smoke.ipynb` (10KB): smoke test, 2-3 min, CPU.
- `kernel-metadata.json`: production deployment config.
- `kernel-metadata-minimal.json`: smoke test config.
- `PRODUCTION_CONFIGURATION.md`: configuration documentation.
- `archive/`: contains the incorrect `ablation_4d_curriculum_learning_production.ipynb` (used test_images).

Hardness stratification:
```python
# stratify images by wound size into 3 difficulty tiers
easy_images = wound_sizes < 33rd_percentile
medium_images = 33rd_percentile <= wound_sizes < 67th_percentile
hard_images = wound_sizes >= 67th_percentile

# training curriculum (3 phases)
Phase 1 (33 epochs): easy images only
Phase 2 (33 epochs): easy + medium images
Phase 3 (34 epochs): all images
```

Expected runtime: 20-24 hours (both strategies, 3 seeds each).

## Smoke test notebooks

### 6. ablation_4_smoke/
- Location: `notebooks/03_ablations/ablation_4_smoke/`.
- Purpose: fast validation of the unified approach (3 strategies).
- Files: `ablation_4_smoke_test.ipynb` (13KB), `ablation_4_smoke_test_clean.ipynb` (15KB), `kernel-metadata.json`.
- Configuration: CPU only, 10 epochs, limited images.

### 7. ablation_4d_smoke/
- Location: `notebooks/03_ablations/ablation_4d_smoke/`.
- Purpose: smoke test of curriculum learning.
- Files: `ablation_4d_curriculum_learning.ipynb` (26KB, full smoke test, 3 epochs per phase), `ablation_4d_curriculum_learning_production.ipynb` (19KB, older), `ablation_4d_minimal_smoke.ipynb` (10KB, 2-3 min validation), `kernel-metadata.json`, `kernel-metadata-minimal.json`.
- Configuration: CPU or GPU, 3 epochs per phase.

### 8. ablation_4a_mixed/ (duplicate smoke test)
- Location: `notebooks/03_ablations/ablation_4a_mixed/`.
- Same as `ablation_4a_hard_negatives_mixed/`. Created during exploratory development.
- Files: `ablation_4a_hard_negatives_mixed.ipynb` (12KB), `kernel-metadata.json`.

## Deprecated

### 9. ablation_4_TBD/
- Location: `notebooks/03_ablations/ablation_4_TBD/`.
- Empty placeholder (0 files). Can be deleted or reused.

## Summary table

| Directory | Status | Purpose | Kaggle Kernel | Result |
|-----------|--------|---------|---------------|--------|
| ablation_4_unified | adopted | Strategy 4a, mixed training | [link](https://www.kaggle.com/code/bkdkaggle/ablation-4-unified-hard-negatives-3-strategies) | 87.51% mAP50 |
| ablation_4a_hard_negatives_mixed | superseded | Same as 4_unified | [link](https://www.kaggle.com/code/bkdkaggle/ablation-4a-hard-negatives-mixed) | Same as 4_unified |
| ablation_4b_best_checkpoint | failed | DDI-only finetuning (best ckpt) | n/a | Cannot compute metrics |
| ablation_4c_early_checkpoint | failed | DDI-only finetuning (early ckpt) | n/a | Cannot compute metrics |
| ablation_4d_production | ready | Curriculum learning (2 strategies) | not deployed | Pending deployment |
| ablation_4_smoke | validation | Smoke test, unified approach | n/a | Validation only |
| ablation_4d_smoke | validation | Smoke test, curriculum | n/a | Validation only |
| ablation_4a_mixed | duplicate | Duplicate of 4a | n/a | Duplicate |
| ablation_4_TBD | empty | Placeholder | n/a | Empty |

## Findings

Worked:
1. Mixed training (4a): wound images plus DDI hard negatives from the start gave the largest single improvement in the study (+7.51 points).
2. Empty labels in training give a valid negative signal through the loss and penalize false positives on background images.
3. Stable across 3 seeds (87.51% +- 0.40%).

Failed:
1. 100% DDI-only finetuning (4b/4c): YOLOv8 cannot compute validation metrics on 100% empty labels. Validation needs some labeled images.
2. Two-stage approach (finetune on hard negatives after baseline): impossible because of that limitation.

Pending:
1. Curriculum learning (4d): progressive hard negatives, not yet tested. Expected to reduce false positive rate further.
2. FPR measurement: needs a separate background-only test set. Standard mAP50 does not capture it.

## Next

Immediate:
1. Deploy ablation 4d (4d.2 and 4d.3) on Kaggle.
2. Measure baseline FPR on the 129 DDI test images.
3. Compare 4a vs 4d.

Later:
1. Ablation 5: label smoothing (production notebook ready).
2. Ablation 6: test-time augmentation (final aggregation ablation).
3. Integrate the best hard negative strategy into the final model.

## Related reports

- ABLATION_4_ROOT_CAUSE_ANALYSIS.md
- ABLATION_4_REDESIGN_PROPOSAL.md
- ABLATION_4_KEY_FINDINGS.md
- ABLATION_4_EMPTY_LABELS_CONCLUSIVE_ANSWER.md
