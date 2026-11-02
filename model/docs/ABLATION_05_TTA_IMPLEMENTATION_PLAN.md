# Ablation 5: Test-Time Augmentation, TDD Implementation Plan

Created 2025-10-16. Status: design. Estimate 4-6 hours (1.5 h design, 1.5 h implementation, 1 h smoke test, 2 h production). The timeline table gives 5.5 hours with debugging.

Supersedes the Ablation 5 section of ABLATIONS_1B_5_6_IMPLEMENTATION_PLAN.md (different transform set and paths).

## Hypothesis

Ensembling predictions over augmented copies of each image gives +1-2% Dice without retraining.

- No retraining; works on the existing baseline model.
- Standard in medical imaging competitions.
- Low risk: should maintain or improve performance.
- Fast iteration.

## Math

```
Image I, augmentation set A = {aug_1, ..., aug_n}

For each aug_i:
  1. I' = aug_i(I)
  2. P' = model(I')
  3. P  = inverse(aug_i)(P')

P_ensemble = MajorityVote([P_1, ..., P_n])  or  Mean([P_1, ..., P_n])
```

Set (8 forward passes per image):
```python
AUGMENTATION_SET = [
    "original",
    "horizontal_flip",
    "vertical_flip",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "hflip_rotate_90",
    "hflip_rotate_270",
]
```

Rationale: wounds have no preferred orientation for flips; rotations cover all 90-degree orientations cheaply; compositions cover the symmetry transforms. Random rotations (+/-5 degrees) excluded as too expensive.

## Phase 1: unit tests (1 h)

File `tests/unit/evaluation/test_tta.py`:
1. Forward transforms apply.
2. Inverse transforms apply.
3. Forward-inverse identity, aug^-1(aug(I)) = I.
4. Majority vote aggregation.
5. Mean aggregation.
6. End-to-end TTA pipeline.
7. Batch processing.

Acceptance: all pass; forward-inverse error below 1e-6; ensemble output is a valid probability distribution.

## Phase 2: implementation (1.5 h)

File `src/evaluation/tta.py`.

```python
class TestTimeAugmentation:
    """
    Args:
        model: YOLOv8 model instance
        augmentation_set: list of augmentation names
        aggregation: 'majority_vote' or 'mean'
    Returns:
        ensembled prediction mask (H, W)
    """

    AUGMENTATIONS = {
        "original": (lambda x: x, lambda x: x),
        "horizontal_flip": (lambda x: torch.flip(x, dims=[-1]),
                            lambda x: torch.flip(x, dims=[-1])),
        "vertical_flip": (lambda x: torch.flip(x, dims=[-2]),
                          lambda x: torch.flip(x, dims=[-2])),
        "rotate_90": (lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
                      lambda x: torch.rot90(x, k=-1, dims=[-2, -1])),
        "rotate_180": (lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
                       lambda x: torch.rot90(x, k=-2, dims=[-2, -1])),
        "rotate_270": (lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
                       lambda x: torch.rot90(x, k=-3, dims=[-2, -1])),
        "hflip_rotate_90": (
            lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1]),
            lambda x: torch.flip(torch.rot90(x, k=-1, dims=[-2, -1]), dims=[-1])),
        "hflip_rotate_270": (
            lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=3, dims=[-2, -1]),
            lambda x: torch.flip(torch.rot90(x, k=-3, dims=[-2, -1]), dims=[-1])),
    }

    def __init__(self, model, augmentation_set=None, aggregation="mean"):
        # validates every name in augmentation_set against AUGMENTATIONS
        ...

    def predict(self, image_path: str) -> np.ndarray:
        # load RGB, preprocess to (C, H, W);
        # for each augmentation: forward, model.predict(verbose=False)[0],
        # extract binary mask, inverse transform;
        # mean: np.mean(predictions, axis=0) > 0.5
        # majority_vote: np.sum(predictions, axis=0) > len(predictions) / 2
        # return uint8 mask
        ...

    def _preprocess(self, image): ...      # TODO
    def _extract_mask(self, yolo_result): ...  # TODO, use existing yolo_utils
    def predict_batch(self, image_paths): ...
```

Notes:
- Torch operations, not PIL, for efficiency.
- Each entry is (forward_fn, inverse_fn); inverse undoes forward exactly.
- Both aggregations supported.
- Reuse `yolo_utils.extract_mask_from_yolo_result()`.

## Phase 3: smoke test notebook (1 h)

File `notebooks/05_ablation_smoke/05_tta_smoke_test.ipynb`. 5 test images from baseline-outputs, 2-3 minutes.

Checks:
1. All augmentations apply.
2. All inverse transforms undo forward transforms.
3. TTA predictions are valid binary masks.
4. TTA Dice at least baseline Dice.
5. Inference time about 8x baseline.
6. Visual comparison, baseline vs TTA.

Cells:
1. NumPy locking (same pattern as the other plans).
2. `sys.path.append('/kaggle/input/wound-seg-code-v1')`; import `TestTimeAugmentation`, `extract_mask_from_yolo_result`, `calculate_segmentation_metrics`.
3. `YOLO('/kaggle/input/baseline-outputs/weights/seed_42/best.pt')`.
4. `TestTimeAugmentation(model, augmentation_set=["original", "horizontal_flip", "vertical_flip", "rotate_90"], aggregation="mean")`.
5. Forward-inverse identity on `torch.rand(3, 512, 512)`; assert max error below 1e-6 for each augmentation.
6. First 5 PNGs of `/kaggle/input/baseline-outputs/data_wound_seg/test_images`, masks from `test_masks`; per image baseline and TTA prediction, timing, metrics via `calculate_segmentation_metrics`.
7. DataFrames of baseline and TTA metrics: mean +/- std Dice, time per image, slowdown factor.
8. Write `/kaggle/working/smoke_test_report.json`: validation_checks (augmentation_transforms, tta_predictions, tta_dice_maintained = TTA mean Dice at least baseline minus 0.01, inference_time_reasonable = under 1.0 s per image on P100), baseline_dice, tta_dice, dice_improvement, inference_time_multiplier, smoke_test_status "PASSED".

Deploy:
```bash
cd notebooks/05_ablation_smoke
kaggle kernels push -p .
```

Runtime 2-3 minutes on Kaggle P100 (5 images x 8 augmentations x about 30 ms, about 1.2 s total).

## Phase 4: production notebook (2 h)

File `notebooks/05_ablation/05_tta.ipynb`. Deploy only after the smoke test passes.

```python
ABLATION_5_CONFIG = {
    "name": "ablation_5_tta",
    "model": "baseline_best.pt",
    "augmentation_set": [
        "original", "horizontal_flip", "vertical_flip",
        "rotate_90", "rotate_180", "rotate_270",
        "hflip_rotate_90", "hflip_rotate_270",
    ],
    "aggregation": "mean",
    "test_aggregations": ["mean", "majority_vote"],
    "seeds": [42, 123, 456],
    "test_images": "/kaggle/input/baseline-outputs/data_wound_seg/test_images",
    "test_masks": "/kaggle/input/baseline-outputs/data_wound_seg/test_masks",
}
```

Evaluation: for seeds 42, 123, 456 load `/kaggle/input/baseline-outputs/weights/seed_{seed}/best.pt`, run baseline and TTA on all 552 test images, compute metrics, save per-image results. Aggregate 3 seeds x 552 images. Wilcoxon on baseline vs TTA Dice, significance at alpha 0.0056.

Output:
```
/kaggle/working/ablation_5/
├── seed_42/
│   ├── per_image_metrics_baseline.csv
│   ├── per_image_metrics_tta.csv
│   └── summary_statistics.json
├── seed_123/
├── seed_456/
├── aggregated_results.csv
├── statistical_analysis.json
├── tta_vs_baseline_comparison.png
└── completion_status.json
```

## Success criteria

Primary:
- Dice +1-2% (target 81-82%).
- p < 0.0056 (Bonferroni).
- Cohen's d > 0.5.

Secondary:
- Precision and recall maintained or improved.
- Boundary F1 may improve (smoother boundaries).
- Inference under 1 s per image.

Decision:
- Adopt: Dice +1.5%, p < 0.0056, d > 0.8.
- Marginal: Dice +0.5%, p < 0.01, d > 0.5.
- Reject: Dice under +0.5% or inference over 2 s per image.

## Advantages

1. No retraining.
2. Should not degrade performance.
3. Deterministic.
4. Ensemble of known transforms.
5. Can be switched on or off by time budget.

## Risks

1. Inference too slow. Test 4 augmentations instead of 8.
2. Improvement under 0.5%. Still useful for high-stakes predictions.
3. Memory (8 predictions in RAM). Process one augmentation at a time.

## Timeline

| Phase | Task | Duration | Depends on |
|---|---|---|---|
| 1 | Unit tests | 1 h | none |
| 2 | TTA implementation | 1.5 h | 1 |
| 3 | Smoke test notebook | 1 h | 2 |
| 4 | Production notebook | 2 h | 3 (smoke passes) |

Total 5.5 hours with debugging.

## Status

Done:
1. `tests/unit/evaluation/test_tta.py`
2. `src/evaluation/tta.py`
3. `notebooks/05_ablation_smoke/05_tta_smoke_test.ipynb`

Open:
4. Deploy the smoke test to Kaggle P100 (2-3 min).
5. If it passes, build the production notebook.
6. Deploy production to Kaggle P100 (about 20-30 min for 552 images).

## References

- Lyksborg et al., "An Ensemble of 2D Convolutional Neural Networks for Tumor Segmentation" (2015).
- Wang et al., "Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation" (NeurIPS 2019).
- Kaggle medical imaging competitions: TTA is standard, 1-3% typical.
