# Ablation 4d v10 Failure Analysis

Date: 2025-10-22.
Notebook: `ablation_4d_curriculum_learning.ipynb` (v10).
Status: failed with ModuleNotFoundError.
Runtime: 17 seconds (failed at the first cell).
Minimal smoke test: https://www.kaggle.com/code/bkdkaggle/ablation-4d-minimal-smoke-test-2-3-min

## Finding

The v10 failure was a missing package, not PyTorch 2.6. The PyTorch 2.6 hypothesis was never exercised.

Error from the v10 logs:

```python
ModuleNotFoundError: No module named 'ultralytics'
```

- Location: import statement, `from ultralytics import YOLO` (line 6 of the cell after the install cell).
- Time: 17 seconds into execution.
- Environment: Kaggle with `enable_internet: true` in metadata.

Sequence:
1. Cell 1 (line 1): `!pip install -q torch==2.4.1 torchvision==0.19.1 ultralytics==8.0.196 'numpy<2.0'`.
2. Cell 2 (line 6): `from ultralytics import YOLO` raised ModuleNotFoundError.

Root cause: the pip install in cell 1 did not complete before the import in cell 2.

Notes on execution order:
- Cell IDs: cell 1 `wd1mnr2ac3`, cell 2 `f0e81eeb`. Cells with IDs are not guaranteed to run in visual order.
- Papermill ran the cells. The pip install may have failed silently or not finished.
- Cell 2 depends on the side effects of cell 1. No check that cell 1 succeeded. No error handling for a failed pip install.

## Assumptions and status

| Assumption | Status | Confidence | Needs test |
|------------|--------|------------|------------|
| PyTorch 2.6 caused the AttributeError | invalidated (never reached PyTorch code) | n/a | n/a |
| PyTorch 2.4 fixes AttributeError | untested | 60% | yes |
| Baseline model loads with 2.4 | untested | 70% | yes |
| Inference works on DDI images | untested | 80% | yes |
| Inference works on wound images | untested | 90% | yes |
| `.boxes.conf` exists for segmentation | untested | 40% | yes |
| Hardness scoring logic is correct | untested | 50% | yes |
| DDI dataset paths are correct | untested | 85% | yes |
| Dice calculation works | untested | 95% | yes |

Validated: 0 of 8. Confidence in curriculum learning: 60%, too low for production.

Chain of dependent assumptions: packages install, then PyTorch 2.4 fixes AttributeError, model loads, inference works, `.boxes.conf` exists for segmentation, hardness scoring is correct, curriculum learning works. Assumption zero (packages install) failed.

Open questions on result structure:
- Does `results[0].boxes.conf` exist for segmentation models?
- Is `results[0].masks.data[0]` the correct way to extract masks?
- Does confidence scoring work on DDI images?
- Does the path `/kaggle/input/ddi-diverse-dermatology` exist?
- Do segmentation models have `.boxes`, and does `.boxes.conf` hold confidence scores?

Known: the 'Segment' has no 'detect' AttributeError was seen in earlier experiments. It could be a version issue. Unvalidated.

## Problems

1. No package installation validation.

Broken:
```python
# Cell 1
!pip install -q torch==2.4.1 ...
# Cell 2
from ultralytics import YOLO  # assumes cell 1 succeeded
```

Correct:
```python
# Cell 1
!pip install -q torch==2.4.1 torchvision==0.19.1 ultralytics==8.0.196 'numpy<2.0'
import importlib.util
assert importlib.util.find_spec('ultralytics') is not None, "ultralytics not installed"
assert importlib.util.find_spec('torch') is not None, "torch not installed"
print("Packages installed successfully")
# Cell 2
from ultralytics import YOLO
```

2. No progressive validation. The notebook did everything at once: load baseline model, score 129 DDI images, score 552 wound images, stratify both datasets, train 3 curriculum phases. Better order: package imports, model loading, inference on 1 image, scoring on 1 image, then scale up.
3. Assumption cascade. One failure anywhere breaks everything and the location is unknown.

## Minimal smoke test (5 tests, 2-3 minutes, CPU)

Notebook: `ablation_4d_minimal_smoke.ipynb`, in `notebooks/03_ablations/ablation_4d_smoke/` with `kernel-metadata.json`. Cost: $0 (CPU tier). Deployed 2025-10-22 (v1).

Test 1, package installation (10 s):
```python
!pip install -q torch==2.4.1 torchvision==0.19.1 ultralytics==8.0.196 'numpy<2.0'
import importlib.util
assert importlib.util.find_spec('ultralytics') is not None
assert importlib.util.find_spec('torch') is not None
import torch
import ultralytics
print(f"PyTorch: {torch.__version__}")
print(f"Ultralytics: {ultralytics.__version__}")
```

Test 2, model loading with PyTorch 2.4 (20 s):
```python
from ultralytics import YOLO
model = YOLO('/kaggle/input/baseline-outputs/yolo_training/seed_42/weights/best.pt')
print(f"Model loaded: {model.model}")
```

Test 3, inference on a DDI image (30 s):
```python
import os
ddi_images = os.listdir('/kaggle/input/ddi-diverse-dermatology')
test_img = f'/kaggle/input/ddi-diverse-dermatology/{ddi_images[0]}'
results = model(test_img, verbose=False)
print(f"Results type: {type(results[0])}")
print(f"Has boxes: {results[0].boxes is not None}")
if results[0].boxes is not None:
    print(f"Boxes type: {type(results[0].boxes)}")
    print(f"Has conf attr: {hasattr(results[0].boxes, 'conf')}")
    if hasattr(results[0].boxes, 'conf'):
        print(f"Confidence shape: {results[0].boxes.conf.shape}")
        print(f"Max confidence: {results[0].boxes.conf.max()}")
```

Test 4, inference on a wound image (30 s):
```python
import os
wound_images = os.listdir('/kaggle/input/baseline-outputs/data_wound_seg/test_images')
test_img = f'/kaggle/input/baseline-outputs/data_wound_seg/test_images/{wound_images[0]}'
results = model(test_img, verbose=False)
print(f"Has masks: {results[0].masks is not None}")
if results[0].masks is not None:
    print(f"Masks type: {type(results[0].masks)}")
    print(f"Number of masks: {len(results[0].masks)}")
    print(f"Mask shape: {results[0].masks.data[0].shape}")
```

Test 5, Dice calculation (30 s):
```python
from PIL import Image
import numpy as np
mask_name = wound_images[0]
mask_path = f'/kaggle/input/baseline-outputs/data_wound_seg/test_masks/{mask_name}'
gt_mask = np.array(Image.open(mask_path).convert('L'))
gt_mask = (gt_mask > 127).astype(np.uint8)
pred_mask_data = results[0].masks.data[0].cpu().numpy()
import cv2
pred_mask = cv2.resize(pred_mask_data, (gt_mask.shape[1], gt_mask.shape[0]))
pred_mask = (pred_mask > 0.5).astype(np.uint8)
intersection = np.logical_and(pred_mask, gt_mask).sum()
union = pred_mask.sum() + gt_mask.sum()
dice = 2.0 * intersection / union if union > 0 else 1.0
print(f"Dice score: {dice:.4f}")
```

Success: all 5 pass (packages install, model loads, DDI inference, wound inference, Dice calculation). On any failure: record the exact error, update the assumption table, fix the root cause, redeploy the minimal test.

## Plan after the minimal test

- Phase 2, full smoke test: `ablation_4d_curriculum_learning.ipynb`, SMOKE_TEST=True, 15-20 minutes, both strategies (4d.2 and 4d.3), 9 epochs total (3 per curriculum phase).
- Phase 3, production: `ablation_4d_curriculum_learning_production.ipynb`, ~3 hours on P100, both strategies, 100 epochs per strategy.
- Estimated time to production-ready: ~4 hours if all tests pass. Production roughly 1 day after smoke validation.

## Confidence

| Metric | Before minimal test | Expected after |
|--------|--------------------|----------------|
| PyTorch 2.4 fixes error / compatibility | 60% | 95% |
| Hardness scoring works / scoring logic | 40% | 95% |
| Inference pipeline | n/a | 95% |
| Curriculum logic | 80% | n/a |
| Overall readiness | 60% | 90% |

Expected overall after full smoke test: 98%. After production run: 100%.

Known: v10 failed because ultralytics was not installed. PyTorch 2.6 was not the issue. `enable_internet` was true. Notebook structure appears sound pending validation.

## Lessons

1. Test primitives first: package install, model load, single-image inference, single calculation, then scale up.
2. Validate side effects. When cell B depends on cell A, assert completion:
```python
# Cell A
!pip install package
assert importlib.util.find_spec('package') is not None
# Cell B
import package
```
3. Do not diagnose errors not yet seen. Time went into a PyTorch 2.6 AttributeError theory without confirming it occurs with 2.4, that it is a version issue, or the actual message.
4. Progressive complexity: 1 image, then 10, then full dataset. 1 test, then 5, then full pipeline. 3 epochs, then 9, then 100.
5. Deploy to P100 only after CPU validation. A 2-3 minute test catches errors that cost 30 minutes on GPU.

## Timeline

- v10 deployment: failed after 17 seconds.
- 2025-10-22: root cause identified, minimal smoke test created and deployed.
- Expected results 2-3 minutes after deployment.
