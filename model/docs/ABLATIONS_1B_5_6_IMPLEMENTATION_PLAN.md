# Ablations 1b, 5, 6: Implementation Plan

Date: 2025-10-15. Status: design complete, not implemented.

This is the first draft covering all three. The per-ablation plans dated 2025-10-16 are more detailed and differ in design and paths. Where they disagree, the per-ablation plans are current:

- ABLATION_1B_DICE_PLUS_PLUS_IMPLEMENTATION_PLAN.md
- ABLATION_05_TTA_IMPLEMENTATION_PLAN.md
- ABLATION_06_LABEL_SMOOTHING_IMPLEMENTATION_PLAN.md

Differences worth keeping from this draft: it uses YOLO's built-in `label_smoothing` parameter for Ablation 6 (the 06 plan builds a custom smoothed BCE loss), and it uses 8 D4 transforms with Wilcoxon on the baseline model for Ablation 5.

## Workflow for every ablation

1. Research and specification.
2. Code implementation, if needed.
3. Unit tests.
4. Diagnostic notebook.
5. CPU smoke test on Kaggle.
6. Production notebook on Kaggle GPU.

Notebooks are built with nbformat.

## Ablation 1b: Dice++ Loss

Question: can a calibration-aware loss match Focal Tversky performance with better confidence calibration?

Hypothesis: Dice 83-85% with lower ECE than Focal Tversky.

Independent of Ablation 1. Does not need Focal Tversky to be rerun.

### Phase 1: research and specification (30 min)

Open items for the literature pass:
- Original Dice++ paper (year, authors, venue).
- PyTorch implementation examples.
- Hyperparameter values (alpha, beta, gamma).
- Expected calibration improvement.
- Exact formulation; how overconfidence is penalized; typical values for medical segmentation; any published wound segmentation results with Dice++.

Loss contract:

```
L = 1 - (2 * TP + eps) / (2 * TP + FP + FN + eps) - lambda * CalibrationPenalty
CalibrationPenalty = |confidence - accuracy|
lambda = 0.1 (lambda_calib)
eps = 1e-6
```

- Input: predictions (B, C, H, W) logits or probabilities; targets (B, C, H, W) binary masks.
- Output: scalar loss.
- Pure function, no side effects.
- Validation: same shape for predictions and targets, values in [0, 1] if probabilities, batch size > 0.

Integration point: custom trainer subclassing `SegmentationTrainer`; `set_model_attributes()` calls super, then sets `self.model.loss_fcn = DicePlusPlusLoss(lambda_calib=0.1)`.

### Phase 2: code (2 h)

- `src/models/dicepp_loss.py`: `DicePlusPlusLoss(nn.Module)` with hyperparameters in `__init__`, forward with Dice term, calibration penalty, combined return, docstrings with formulas, validation checks.
- `src/training/dicepp_trainer.py`: `DicePPTrainer` subclass of ultralytics `SegmentationTrainer`, overrides `set_model_attributes()`, adds training validation hooks.
- `tests/unit/models/test_dicepp_loss.py`:
  1. test_loss_shape_validation
  2. test_loss_value_range (loss in [0, 2])
  3. test_perfect_prediction (loss 0)
  4. test_worst_prediction (loss 2)
  5. test_calibration_penalty
  6. test_overconfident_penalty
  7. test_batch_processing
  8. test_gradient_flow
- Run: `python3 -m pytest tests/unit/models/test_dicepp_loss.py -v`

### Phase 3: diagnostic notebook (1 h)

Builder: `scripts/build_ablation_1b_diagnostic.py`. Checks:
1. DicePlusPlusLoss imports from the code dataset.
2. Custom trainer initializes.
3. Loss computed on TINY dataset (2 images).
4. One training epoch runs.
5. Metrics schema validated (lowercase columns).
6. Output files created.

Deploy (CPU, 3-minute runtime):

```bash
python3 scripts/build_ablation_1b_diagnostic.py
cd notebooks/03_ablations/ablation_1b_diagnostic/
kaggle kernels push -p .
kaggle kernels output bkdkaggle/ablation-1b-diagnostic -p ./diagnostic_results
cat diagnostic_results/diagnostic_report.json
```

Expected report: `validation_checks` with loss_import, trainer_init, training_epoch (loss_value 0.342), schema_validation (columns tp, fp, tn, fn), output_files (count 5), all passed; status `READY_FOR_PRODUCTION`.

### Phase 4: production notebook (2 h)

Builder: `scripts/build_ablation_1b.py`. Cells:
- 0: NumPy locking and ultralytics install.
- 1: imports (code dataset, custom loss and trainer).
- 2: environment detection.
- 3-5: dataset setup from baseline-outputs.
- 6: multi-seed training loop, seeds 42, 123, 456; DicePPTrainer, train, evaluate on test set, save per seed.
- 7-9: aggregate results, bootstrap CIs.
- 10: visualizations.
- 11: save outputs and completion status.
- 12: two-pass markdown commentary.

Settings: completion markers, P100-optimized (batch=32, lr=0.015, workers=8), schema validation, logging.

Commentary text: YOLOv8n-seg with Dice++ replacing the default loss. lambda_calib 0.1, eps 1e-6, all else as baseline. Expected Dice about 83-85%, ECE below 0.05 (Focal Tversky ECE about 0.08). Both losses target imbalanced segmentation; Focal Tversky emphasizes hard examples via the focal parameter, Dice++ adds an explicit calibration penalty. Independent experiments.

Deploy:

```bash
python3 scripts/build_ablation_1b.py
python3 scripts/clean_notebook.py notebooks/03_ablations/ablation_1b/ablation_1b.ipynb
cd notebooks/03_ablations/ablation_1b/
kaggle kernels push -p .
```

Runtime about 1.5-2 hours (3 seeds x 30-40 min).

Post-run:

```bash
kaggle kernels output bkdkaggle/ablation-1b-dicepp-loss -p ./ablation_1b_results
cat ablation_1b_results/completion_status.json
```

Expected: ablation_dice_mean 0.834, calibration_ece 0.047, baseline_dice_mean 0.800, dice_improvement 0.034.

## Ablation 5: Test-Time Augmentation

Question: can TTA improve prediction robustness without retraining?

Hypothesis: D4 symmetry group ensemble improves Dice by 1-2%.

Post-hoc. Runs on an already trained model (baseline or best ablation).

### Phase 1: research and specification (30 min)

Open items: TTA for medical segmentation (2023-2024 papers); which D4 transforms; aggregation method (mean, max, voting); inference overhead (8x expected); medical imaging practice.

D4 transforms (8):
1. Identity
2. Rotate 90 clockwise
3. Rotate 180
4. Rotate 270 clockwise
5. Horizontal flip
6. Vertical flip
7. Diagonal flip (main)
8. Anti-diagonal flip

Aggregation: mean of predicted probabilities, `final_mask = np.mean([pred1, ..., pred8], axis=0)`.

Contract: `predict_with_tta(model: YOLO, image: np.ndarray (H, W, 3) RGB, transformations=D4_GROUP, aggregation="mean")` returns dict `{success, data: {pred_mask (H, W), confidence, transformations_used, individual_predictions}, error}`. Runs inference 8 times; may use GPU memory for batching. Image must be (H, W, 3) RGB, model loaded, all transforms reversible.

### Phase 2: code (2 h)

`src/evaluation/tta.py`:
1. Transform functions: rotate_90, rotate_180, rotate_270, flip_horizontal, flip_vertical, flip_diagonal, flip_antidiagonal.
2. Inverse transforms to map predictions back.
3. `predict_with_tta()`: transform, infer, inverse, aggregate by mean.
4. `show_tta_predictions(individual_preds)` for analysis.

`tests/unit/evaluation/test_tta.py`:
1. test_d4_transformations_reversible
2. test_tta_improves_dice (TTA >= baseline)
3. test_tta_preserves_shape
4. test_individual_predictions_stored (all 8)
5. test_aggregation_methods (mean, max, voting)
6. test_inference_time_overhead (8x)
7. test_confidence_scores

Run: `python3 -m pytest tests/unit/evaluation/test_tta.py -v`

### Phase 3: diagnostic notebook (1 h)

Builder: `scripts/build_ablation_5_diagnostic.py`. Checks: TTA module imports from the code dataset; D4 transforms compute; model inference runs 8 times; predictions aggregate; visualization generates; runtime about 2 min on TINY.

```bash
python3 scripts/build_ablation_5_diagnostic.py
cd notebooks/03_ablations/ablation_5_diagnostic/
kaggle kernels push -p .
kaggle kernels output bkdkaggle/ablation-5-tta-diagnostic -p ./diagnostic_results
```

CPU, 3-minute runtime.

### Phase 4: production notebook (2 h)

Cells:
- 0: NumPy locking, ultralytics.
- 1: imports (code dataset, TTA module).
- 2: environment detection.
- 3: load model, `YOLO("/kaggle/input/baseline-outputs/yolo_training/seed_42/weights/best.pt")` (best model from Ablation 1, or baseline).
- 4: TTA inference loop over test pairs with `predict_with_tta`, metrics via `calculate_segmentation_metrics`, plus `tta_confidence`.
- 5: baseline inference (single transform) on the same pairs.
- 6: paired Wilcoxon, `wilcoxon(tta_dice_scores, baseline_dice_scores)`, significant at p < 0.05.
- 7-9: visualizations, save outputs.

Commentary: 8 geometric transforms (rotations 0, 90, 180, 270; flips horizontal, vertical, diagonal, anti-diagonal), mean aggregation. Cost: 8x inference time (about 800 ms vs about 100 ms per image). Expected Dice about 81-82% (baseline 80%), improved boundary precision on ambiguous wound boundaries.

```bash
python3 scripts/build_ablation_5.py
python3 scripts/clean_notebook.py notebooks/03_ablations/ablation_5/ablation_5_tta.ipynb
cd notebooks/03_ablations/ablation_5/
kaggle kernels push -p .
```

Runtime about 30-40 minutes (552 images x 8 transforms x about 200 ms).

## Ablation 6: Label Smoothing

Question: can label smoothing reduce overconfidence without losing accuracy?

Hypothesis: epsilon 0.1 keeps Dice at about 80% and lowers ECE.

Training-time. Retrain from scratch.

### Phase 1: research and specification (30 min)

Open items: label smoothing for segmentation (2023-2024); epsilon range 0.05-0.15; YOLO built-in parameter; best epsilon for imbalanced segmentation; effect on the rare wound class; size of calibration gain; wound segmentation papers using it.

Mechanism: hard labels [0, 1] become soft labels [eps, 1-eps], [0.1, 0.9] at eps 0.1. Slight accuracy cost, better calibration.

YOLO call:

```python
model.train(
    data="wound_dataset.yaml",
    label_smoothing=0.1,  # only intervention
    mixup=0.0,            # disabled
    # all other params as baseline
)
```

### Phase 2: code (30 min)

No custom code. Set `label_smoothing` in `model.train()` and keep `mixup=0.0`.

Optional `tests/unit/training/test_label_smoothing.py`: parameter accepts epsilon in [0, 1]; mixup disabled (0.0); one-epoch smoke training completes. Run with `python3 -m pytest tests/unit/training/test_label_smoothing.py -v`.

### Phase 3: diagnostic notebook (1 h)

Builder: `scripts/build_ablation_6_diagnostic.py`. Checks: YOLO accepts `label_smoothing`; mixup disabled; one epoch on TINY; loss values not diverging; metrics schema; output files.

```bash
python3 scripts/build_ablation_6_diagnostic.py
cd notebooks/03_ablations/ablation_6_diagnostic/
kaggle kernels push -p .
kaggle kernels output bkdkaggle/ablation-6-label-smoothing-diagnostic -p ./diagnostic_results
```

CPU, 3-minute runtime.

### Phase 4: production notebook (2 h)

Same cells as baseline up to training. Per seed in RANDOM_SEEDS: `YOLO("yolov8n-seg.pt")` then `model.train` with batch=32, lr0=0.015, workers=8, amp=False, epochs=100, imgsz=512, patience=10, label_smoothing=0.1, mixup=0.0, seed=seed, save=True, project=`OUTPUT_PATH / "yolo_training"`, name=`seed_{seed}`. Later cells: evaluation, aggregation, baseline comparison, ECE, visualizations, save outputs.

Commentary: soft targets (0.1, 0.9) instead of (0, 1). Epsilon 0.1; mixup 0.0. Expected Dice about 79-81%, ECE about 0.05 (baseline about 0.08). Mixup excluded: vanilla mixup creates anatomically impossible training scenarios for medical segmentation (2022-2024 literature).

```bash
python3 scripts/build_ablation_6.py
python3 scripts/clean_notebook.py notebooks/03_ablations/ablation_6/ablation_6_label_smoothing.ipynb
cd notebooks/03_ablations/ablation_6/
kaggle kernels push -p .
```

Runtime about 1.5-2 hours (3 seeds x 30-40 min).

## Universal steps for ablation X

1. Research, 30 min to 1 h: library docs, recent papers (2023-2024), findings recorded in the plan.
2. Code, 0-2 h: module in `src/`, docstrings with contracts, unit tests, `pytest tests/unit/module/test_file.py -v`.
3. Diagnostic, 1 h: `python3 scripts/build_ablation_X_diagnostic.py`, push from `notebooks/03_ablations/ablation_X_diagnostic/`, fetch with `kaggle kernels output bkdkaggle/ablation-X-diagnostic -p ./diagnostic_results`, read `diagnostic_report.json`. Proceed only if all checks pass.
4. Production, 2 h: `python3 scripts/build_ablation_X.py`, `jupyter nbconvert --to notebook --execute ablation_X.ipynb`, `python3 scripts/clean_notebook.py ablation_X.ipynb`, push from `notebooks/03_ablations/ablation_X/`.
5. Post-run, 15 min: `kaggle kernels output bkdkaggle/ablation-X -p ./ablation_X_results`, check `completion_status.json` and `summary_statistics.json`.

## Timeline

| Ablation | Research | Code + tests | Diagnostic | Production | Total |
|---|---|---|---|---|---|
| 1b (Dice++) | 30 min | 2 h | 1 h | 2 h | 5.5 h |
| 5 (TTA) | 30 min | 2 h | 1 h | 2 h | 5.5 h |
| 6 (Label Smoothing) | 30 min | 30 min | 1 h | 2 h | 4 h |
| Total | | | | | 15 h |

Development time only. Kaggle GPU runtime adds 1.5-2 hours per training ablation.

## Success criteria

Ablation 1b:
- Dice 83-85% (matches Focal Tversky).
- ECE below 0.05.
- Training completes; all 3 seeds complete.

Ablation 5:
- Dice +1-2% over baseline.
- p < 0.05 (Wilcoxon).
- Inference about 8x slower.
- Visualization shows individual predictions.

Ablation 6:
- Dice 79-81%.
- ECE below 0.06.
- mixup=0.0 confirmed in logs.
- No divergence.

## Open items

1. Smoke tests for Ablations 2 and 3 (to prevent a repeat of the signature mismatch failure).
2. Deploy fixed Ablations 2 and 3 to Kaggle GPU.
3. Implement Ablations 1b, 5, 6.
4. Update the PRD with the final ablation specifications.

## References

To be filled after the literature review: Dice++ paper, TTA, label smoothing, YOLO documentation, medical imaging literature 2023-2024.
