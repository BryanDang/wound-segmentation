# Ablation 1B: Dice++ Loss, TDD Implementation Plan

Created 2025-10-16. Status: design. Estimate 6-8 hours (2 h design, 2 h implementation, 2 h smoke test, 2 h production). The timeline table below sums to 9 hours with debugging.

Supersedes the Ablation 1b section of ABLATIONS_1B_5_6_IMPLEMENTATION_PLAN.md (different paths and design).

## Hypothesis

Dice++ gives Dice near baseline (about 80%) with better calibrated confidence (lower ECE). It addresses calibration, not class imbalance.

Reason: Focal Tversky failed (80.01% vs 80.00%) because mosaic already handles imbalance. Overconfident predictions are dangerous clinically, so accurate uncertainty is needed.

## Formula

```
L = 1 - Dice(pred, target) - lambda * CalibrationPenalty

Dice(pred, target) = (2 * TP + eps) / (2 * TP + FP + FN + eps)
CalibrationPenalty = |confidence - accuracy|
lambda = 0.1 (typical)
eps = 1e-7
```

Per-image calibration:
```python
confidence = pred.max()
accuracy = (pred.argmax() == target).float().mean()
calibration_penalty = abs(confidence - accuracy)
```

## Phase 1: unit tests (2 h)

File `tests/unit/models/test_dice_plus_plus_loss.py`:
1. Dice component matches baseline Dice.
2. Calibration penalty computation.
3. Combined loss (Dice + lambda * calibration).
4. Gradient flow (differentiable).
5. Edge cases: all correct, all wrong, perfect calibration.
6. Lambda sensitivity: 0.0, 0.1, 0.5, 1.0.

Acceptance: all pass, 100% coverage of the loss, no NaN/inf in gradients.

## Phase 2: loss (2 h)

File `src/models/dice_plus_plus_loss.py`:

```python
class DicePlusPlusLoss(nn.Module):
    """
    Args:
        lambda_cal: weight for calibration penalty (default 0.1)
        smooth: smoothing constant for Dice (default 1e-7)
    Returns:
        loss: combined Dice + calibration loss
        metrics: dict with 'dice_component' and 'calibration_penalty'
    """
    def __init__(self, lambda_cal=0.1, smooth=1e-7): ...

    def forward(self, pred, target):
        # pred: (B, C, H, W) probabilities (post-softmax)
        # target: (B, H, W) labels
        # TODO
        ...

    def _compute_dice(self, pred, target): ...
    def _compute_calibration_penalty(self, pred, target): ...
```

Notes:
- Multi-class (background and wound); per-class Dice, macro-averaged.
- Calibration penalty per image, averaged over the batch.
- Return loss and components for monitoring.

## Phase 3: YOLOv8 integration (1 h)

File `src/models/dice_plus_plus_trainer.py`: `DicePlusPlusTrainer(model_path, lambda_cal=0.1)` holds `YOLO(model_path)` and `DicePlusPlusLoss`; `train(data_yaml, **kwargs)` overrides YOLO's default loss (TODO).

YOLOv8 does not expose loss customization. Options:
1. Preferred: subclass YOLO's `DetectionTrainer` and override the `loss` property.
2. Alternative: `add_callback` for custom loss logging (monitor only).

## Phase 4: smoke test notebook (2 h)

File `notebooks/01b_ablation_smoke/01b_dice_plus_plus_smoke_test.ipynb`. Kaggle CPU, TINY dataset (2-5 images; `tiny-wound-seg-smoke-test`: 2 test, 4 train), 2-3 minutes.

Checks:
1. Loss computes without errors.
2. Dice component matches a manual calculation.
3. Calibration penalty in [0, 1].
4. Combined loss decreases during training on tiny data.
5. No NaN/inf.
6. Visual check of predictions after training.
7. Dice and calibration terms logged over time.

Cells:
1. NumPy locking: record `np.__version__`, `pip install -q ultralytics`, then `pip install -q --force-reinstall numpy==<default>`.
2. `sys.path.append('/kaggle/input/wound-seg-code-v1')`; import `DicePlusPlusLoss`, `DicePlusPlusTrainer`.
3. Load TINY dataset.
4. `DicePlusPlusLoss(lambda_cal=0.1)`.
5. Dummy test: `torch.rand(1, 2, 512, 512)` predictions, `torch.randint(0, 2, (1, 512, 512))` targets; print loss, dice_component, calibration_penalty.
6. Train 5 epochs: `DicePlusPlusTrainer("yolov8n-seg.pt", lambda_cal=0.1).train(data="wound_dataset.yaml", epochs=5, batch=2, imgsz=512, patience=999)`.
7. Assert final loss below initial loss.
8. Write `/kaggle/working/smoke_test_report.json` with `validation_checks` (loss_computation, dice_component, calibration_penalty, loss_decrease, no_nan_inf) and `smoke_test_status` "PASSED".

Deploy:
```bash
cd notebooks/01b_ablation_smoke
kaggle kernels push -p .
```

## Phase 5: production notebook (2 h)

File `notebooks/01b_ablation/01b_dice_plus_plus.ipynb`. Deploy only after the smoke test passes.

```python
ABLATION_1B_CONFIG = {
    "name": "ablation_1b_dice_plus_plus",
    "loss": "dice_plus_plus",
    "lambda_cal": 0.1,

    # P100-optimized, same as baseline
    "batch": 32,
    "lr0": 0.015,
    "epochs": 100,
    "patience": 10,
    "workers": 8,
    "amp": False,
    "imgsz": 512,

    "seeds": [42, 123, 456],
}
```

Extra metrics beyond Dice/IoU/Precision/Recall:
- `compute_ece(predictions, targets, n_bins=10)`: Expected Calibration Error; bins predictions, compares average confidence with average accuracy per bin.
- `compute_mce(predictions, targets, n_bins=10)`: Maximum Calibration Error; worst bin.
- `plot_reliability_diagram(predictions, targets, n_bins=10)`: perfect calibration is the diagonal.

Output:
```
/kaggle/working/ablation_1b/
├── seed_42/
│   ├── best.pt
│   ├── last.pt
│   ├── training_metrics.csv
│   ├── calibration_metrics.json  # ECE, MCE
│   └── reliability_diagram.png
├── seed_123/
├── seed_456/
├── aggregated_results.csv
├── calibration_comparison.json    # baseline vs Ablation 1B
└── completion_status.json
```

## Success criteria

Primary:
- Dice about 80% (no regression).
- ECE below 0.02 (30-50% reduction from baseline).
- p < 0.0056 (Bonferroni, 9 ablations).
- Cohen's d > 0.5 for ECE improvement.

Secondary:
- Precision/Recall maintained.
- Boundary F1 maintained or slightly better.
- Training time about 1.5-2 h on P100.
- No inference cost (loss only affects training).

Decision:
- Adopt: ECE improvement at least 30%, Dice at least 79.5%.
- Marginal: ECE improvement at least 15%, Dice maintained.
- Reject: ECE improvement under 15% or Dice regression over 0.5%.

## Risks

1. YOLO loss override difficult. Fallback: post-hoc temperature scaling (simpler, still improves calibration).
2. Calibration penalty too strong. Test lambda in {0.05, 0.1, 0.2} in the smoke test.
3. Dice regression. If Dice drops over 1%, reduce lambda or switch to post-hoc calibration.

## Timeline

| Phase | Task | Duration | Depends on |
|---|---|---|---|
| 1 | Unit tests | 2 h | none |
| 2 | Dice++ loss | 2 h | 1 |
| 3 | YOLO integration | 1 h | 2 |
| 4 | Smoke test notebook | 2 h | 3 |
| 5 | Production notebook | 2 h | 4 (smoke passes) |

Total 9 hours with debugging.

## Status

Done:
1. `tests/unit/models/test_dice_plus_plus_loss.py`
2. `src/models/dice_plus_plus_loss.py`
3. `src/models/dice_plus_plus_trainer.py`
4. `notebooks/01b_ablation_smoke/01b_dice_plus_plus_smoke_test.ipynb`

Open:
5. Deploy the smoke test to Kaggle CPU (2-3 min).
6. If it passes, build the production notebook.
7. Deploy production to Kaggle P100 (1.5-2 h).

Note: `src/models/` is missing from the repo snapshot (see COMPONENTS.md).

## References

- Kumar et al., "Post hoc calibration of medical segmentation models" (2025).
- Mask-TS Net (2024): temperature scaling for polyp segmentation.
- Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017).
