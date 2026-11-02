# Ablation 6: Label Smoothing, TDD Implementation Plan

Created 2025-10-16. Status: design. Estimate 5-7 hours (1.5 h design, 2 h implementation, 1.5 h smoke test, 2 h production). The timeline table gives 6.5 hours with debugging.

Supersedes the Ablation 6 section of ABLATIONS_1B_5_6_IMPLEMENTATION_PLAN.md, which used YOLO's built-in `label_smoothing` parameter. This plan builds a custom smoothed BCE loss.

## Hypothesis

Label smoothing improves calibration (ECE down 20-30%) and keeps Dice near 80%.

- Targets overconfidence.
- Training-time only; no inference cost (unlike TTA).
- Simpler than Dice++; no custom loss function needed in principle.
- Widely used for calibration in medical AI.

## Math

```
Hard labels: y = 0 (background) or 1 (wound)
Soft labels: y_smooth = y * (1 - eps) + eps / C
  eps = smoothing parameter (typically 0.1)
  C = number of classes (2)

Binary:
  Background: 0 becomes 0.05  (eps/2)
  Wound:      1 becomes 0.95  (1 - eps/2)
```

Instead of "100% wound", the pixel is "95% wound, 5% uncertain".

Cross-entropy:
```
standard:  loss = -[y * log(p) + (1-y) * log(1-p)]
smoothed:  y_smooth = y * (1 - eps) + eps/2
           loss = -[y_smooth * log(p) + (1-y_smooth) * log(1-p)]
```

## Phase 1: unit tests (1 h)

File `tests/unit/data/test_label_smoothing.py`:
1. Transform: 0 to 0.05, 1 to 0.95.
2. Epsilon range [0, 0.3].
3. Batch processing.
4. eps = 0 recovers hard labels.
5. Numerical stability (no NaN/inf).
6. Shape preserved.

Acceptance: all pass; stable for eps in [0, 0.3]; batch matches per-image.

## Phase 2: implementation (2 h)

File `src/data/label_smoothing.py`:

```python
class LabelSmoothing:
    """
    Args:
        epsilon: smoothing parameter (default 0.1)
                 background 0 becomes eps/2; foreground 1 becomes 1 - eps/2
    Example:
        smoother = LabelSmoothing(epsilon=0.1)
        smoother(np.array([[0, 1], [1, 0]]))  # [[0.05, 0.95], [0.95, 0.05]]
    """
    def __init__(self, epsilon: float = 0.1):
        if not 0 <= epsilon <= 0.3:
            raise ValueError(f"epsilon must be in [0, 0.3], got {epsilon}")
        self.epsilon = epsilon

    def __call__(self, labels):   # (H, W) or (B, H, W), np.ndarray or torch.Tensor
        # dispatch to _smooth_torch / _smooth_numpy; TypeError otherwise
        ...

    # both: labels * (1 - self.epsilon) + self.epsilon / 2
    def get_config(self) -> dict:  # {"epsilon": self.epsilon}
        ...
```

File `src/models/smoothed_loss_trainer.py`:

```python
class SmoothedBCELoss(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        # self.smoother = LabelSmoothing(epsilon); self.bce = nn.BCEWithLogitsLoss()
        ...
    def forward(self, pred, target):
        # pred: (B, 1, H, W) logits; target: (B, H, W) labels in [0, 1]
        # smooth target, unsqueeze(1) to match pred, return BCE
        ...

class LabelSmoothingTrainer:
    def __init__(self, model_path: str, epsilon: float = 0.1): ...
    def train(self, data_yaml: str, **kwargs):
        # TODO: override YOLO's default loss with SmoothedBCELoss.
        # YOLOv8 does not expose loss customization; may need callbacks
        # or a DetectionTrainer subclass.
        ...
```

Notes:
- Smoothing applies to ground truth masks before the loss.
- Works with any loss (BCE, Dice, Focal).
- Epsilon typically 0.1. Test 0.05, 0.1, 0.15 in the smoke test.

## Phase 3: smoke test notebook (1.5 h)

File `notebooks/06_ablation_smoke/06_label_smoothing_smoke_test.ipynb`. TINY dataset (4 train images, `tiny-wound-seg-smoke-test`), 5 epochs, 2-3 minutes.

Checks:
1. Smoothing applies (visual check).
2. Training converges with smoothed labels.
3. Loss decreases over epochs.
4. Predictions are still binary.
5. No NaN/inf.
6. Training time vs baseline.
7. Epsilon values 0.05, 0.1, 0.15.

Cells:
1. NumPy locking.
2. `sys.path.append('/kaggle/input/wound-seg-code-v1')`; import `LabelSmoothing`, `LabelSmoothingTrainer`.
3. Transform test on a 3x4 dummy mask; assert background 0.05 and wound 0.95 at eps 0.1.
4. Loop eps in [0.05, 0.1, 0.15] and print background and wound values.
5. `LabelSmoothingTrainer("yolov8n-seg.pt", epsilon=0.1).train(data="wound_dataset.yaml", epochs=5, batch=2, imgsz=512, patience=999)`.
6. Assert final loss below initial loss.
7. Predict on the first PNG in `/kaggle/input/tiny-wound-seg-smoke-test/test_images`; print unique values, min, max of the predicted mask (no detection is expected for some images).
8. Write `/kaggle/working/smoke_test_report.json`: validation_checks (label_smoothing_transform, training_convergence, no_nan_inf, predictions_valid), epsilon_tested [0.05, 0.1, 0.15], final_epsilon 0.1, loss_decrease (initial, final, decrease_pct), smoke_test_status "PASSED".

Deploy:
```bash
cd notebooks/06_ablation_smoke
kaggle kernels push -p .
```

Runtime 2-3 minutes on P100.

## Phase 4: production notebook (2 h)

File `notebooks/06_ablation/06_label_smoothing.ipynb`. Deploy only after the smoke test passes.

```python
ABLATION_6_CONFIG = {
    "name": "ablation_6_label_smoothing",
    "epsilon": 0.1,

    # P100-optimized, same as baseline
    "batch": 32,
    "lr0": 0.015,
    "epochs": 100,
    "patience": 10,
    "workers": 8,
    "amp": False,
    "imgsz": 512,

    "seeds": [42, 123, 456],
    "epsilon_variants": [0.05, 0.1, 0.15],  # optional
}
```

Evaluation:
- For seeds 42, 123, 456: `LabelSmoothingTrainer("yolov8n-seg.pt", epsilon=0.1).train(data="wound_dataset.yaml", epochs=100, batch=32, lr0=0.015, patience=10, workers=8, imgsz=512, seed=seed, project="/kaggle/working/ablation_6", name=f"seed_{seed}")`; evaluate on the test set with `calculate_segmentation_metrics`.
- Per seed: ECE and MCE from `ablation_6/seed_{seed}/best.pt` over the test set.
- Compare with baseline ECE by paired t-test (`scipy.stats.ttest_rel`).

Output:
```
/kaggle/working/ablation_6/
├── seed_42/
│   ├── best.pt
│   ├── last.pt
│   ├── training_metrics.csv
│   ├── per_image_metrics.csv
│   ├── calibration_metrics.json  # ECE, MCE
│   └── reliability_diagram.png
├── seed_123/
├── seed_456/
├── aggregated_results.csv
├── calibration_comparison.json    # baseline vs Ablation 6
├── statistical_analysis.json      # paired t-test
└── completion_status.json
```

## Success criteria

Primary:
- Dice about 80% (no regression).
- ECE below 0.02 (20-30% reduction from baseline).
- p < 0.0056 (Bonferroni).
- Cohen's d > 0.5 for ECE.

Secondary:
- Precision/Recall and Boundary F1 maintained.
- Training about 1.5-2 h on P100 (same as baseline).
- No inference cost.

Decision:
- Adopt: ECE improvement at least 20%, Dice at least 79.5%.
- Marginal: ECE improvement at least 10%, Dice maintained.
- Reject: ECE improvement under 10% or Dice regression over 0.5%.

## Advantages

1. Simple: transform labels before the loss.
2. No inference overhead.
3. Validated in medical imaging literature.
4. Clear probabilistic reading (label uncertainty).
5. Combines with TTA and better augmentation.

## Risks

1. Dice regression. Test eps in {0.05, 0.1, 0.15} in the smoke test; use the best Dice/ECE trade-off.
2. YOLO loss override difficult. Fallback: implement smoothing as data preprocessing (transform masks before YOLO).
3. ECE improvement under 10%. Still some value for clinical deployment.

## Timeline

| Phase | Task | Duration | Depends on |
|---|---|---|---|
| 1 | Unit tests | 1 h | none |
| 2 | Label smoothing | 2 h | 1 |
| 3 | Smoke test notebook | 1.5 h | 2 |
| 4 | Production notebook | 2 h | 3 (smoke passes) |

Total 6.5 hours with debugging.

## Status

Done:
1. `tests/unit/data/test_label_smoothing.py`
2. `src/data/label_smoothing.py`
3. `src/models/smoothed_loss_trainer.py`
4. `notebooks/06_ablation_smoke/06_label_smoothing_smoke_test.ipynb`

Open:
5. Deploy the smoke test to Kaggle P100 (2-3 min).
6. If it passes, build the production notebook.
7. Deploy production to Kaggle P100 (1.5-2 h).

Note: `src/models/` is missing from the repo snapshot (see COMPONENTS.md).

## References

- Muller and Kornblith, "When Does Label Smoothing Help?" (NeurIPS 2019).
- Kumar et al., "Calibration of medical imaging classifiers" (Medical Image Analysis 2019).
- Pereyra et al., "Regularizing Neural Networks by Penalizing Confident Output Distributions" (ICLR 2017).
