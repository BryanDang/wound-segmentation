# Ablation 1 (Focal Tversky Loss): unit test learnings

Date: 2025-10-15.
Notebook: [experiments/ablation_1_unit_tests.ipynb](ablation_1_unit_tests.ipynb)
Kaggle kernel: [focal-tversky-loss-unit-tests](https://www.kaggle.com/code/bkdkaggle/focal-tversky-loss-unit-tests)

## Result

Environment: Kaggle (PyTorch 2.6.0+cu124, CPU-only).
Tests: 7. Passed: 7 (100%). Failed: 0.

Individual results:

1. `test_perfect_prediction`: PASS. Loss 0.000000 (expected about 0.0, TI=1.0).
2. `test_worst_prediction`: PASS. Loss 1.000000 (expected about 1.0, TI approaches 0).
3. `test_tversky_index_calculation`: PASS. Loss 0.500000 (expected 0.5, TP=1, FP=1, FN=1). Manually validated: TI = 1 / (1 + 0.7x1 + 0.3x1) = 0.5.
4. `test_parameter_validation`: PASS. Checks alpha + beta = 1.0. Raises AssertionError on invalid parameters.
5. `test_gradient_flow`: PASS. Max gradient 0.0120. No NaN or Inf gradients. Backward pass works.
6. `test_asymmetric_weighting`: PASS. FN loss 0.2593 > FP loss 0.1304, as expected for alpha=0.7. Missed wound pixels are penalized more heavily.
7. `test_yolo_compatibility`: PASS. Loss 0.250042 on 640x640 input (YOLOv8-sized). Loss bounded in [0, 1].

## Validated

- Tversky index: TI = TP / (TP + alpha*FN + beta*FP)
- Focal modulation: FTL = (1 - TI)^gamma
- Asymmetric weighting: alpha=0.7 (FN), beta=0.3 (FP)
- Gradient flow: gradients computed correctly, no NaN or Inf, max magnitude 0.012
- YOLOv8 integration: handles 640x640 tensors, loss bounded in [0, 1]

## Lessons

### L1: code-as-dataset import path (critical)

Problem: `ModuleNotFoundError` for `src.models.focal_tversky`.

Cause:
- The versioned local directory is `src/`.
- The Kaggle upload dropped the `src/` prefix.
- Local structure: `src/models/focal_tversky.py`.
- Kaggle structure: `/kaggle/input/wound-seg-code-v1/models/focal_tversky.py`.

Fix:
```python
# correct
sys.path.append('/kaggle/input/wound-seg-code-v1')
from models.focal_tversky import FocalTverskyLoss

# wrong
from src.models.focal_tversky import FocalTverskyLoss
```

Mnemonic: text damaged in source.

### L2: validation before production

A production run would have failed on this import error. The unit test run on Kaggle caught it first. Remaining details (PyTorch version, CUDA, environment checks, GPU runs saved): text damaged in source.

### L3: Kaggle directory debugging

Technique: print the contents of the code dataset directory.

```python
print("Directory contents:")
for item in os.listdir('/kaggle/input/wound-seg-code-v1'):
    print(f"- {item}")
```

The listing showed `models` directly under the dataset root, alongside `__init__.py`, `visualization` and `evaluation`. This revealed the missing `src/` prefix immediately.

## Next (phase 2)

Production notebooks:

1. Diagnostic notebook (`experiments/ablation_1_diagnostic.ipynb`): tiny test (2 to 5 images), validate the data pipeline, check for gradient problems.
2. Production notebook (3 seeds): full 2,208 images, seeds [42, 123, 456], 100 epochs, patience=10. Expected improvement: Dice +3 to 5% from 80.0%.

## References

- Focal Tversky loss paper (2019): https://arxiv.org/abs/1810.07842
