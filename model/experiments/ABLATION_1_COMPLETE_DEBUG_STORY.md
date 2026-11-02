# Ablation 1 (Focal Tversky Loss): deployment debug history, v1 to v6

Date: 14, 2025 (rest of date damaged in source).
Status: v6 running.
Time spent: about 3 hours.
Deployments: 6 versions.
Bugs fixed: 3.

## Summary

Ablation 1 failed on 5 Kaggle deployments and ran on the 6th. Root cause: the Kaggle dataset structure was never validated before deployment. The baseline-outputs dataset has a hybrid flat/nested layout that was not documented.

Rule: validate the Kaggle dataset structure before every deployment.

## Bug 1: wrong dataset path (v1 to v2)

Error: `FileNotFoundError` for `/kaggle/input/wound-segmentation-images/data_wound_seg/test_images`.

Cause: the notebook expected the `wound-segmentation-images` dataset. `kernel-metadata.json` attached `baseline-outputs`.

```python
# v1
BASE_INPUT_PATH = Path("/kaggle/input/wound-segmentation-images/data_wound_seg")
# v2 onward
BASE_INPUT_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
```

Deployed: v2. Result: failed on bug 2.

## Bug 2: mask directory (v2 to v4)

Error: `Valid pairs: 0` (found 5520 masks).

Cause: baseline-outputs holds both nested and flat structures.
- Nested (training): `test/images/`, `test/labels/`
- Flat (evaluation): `test_masks/`, `train_masks/`
- Missing: `test/masks/` does not exist.

The notebook assumed `test/masks/`.

Debug process:
1. v2 to v4 reported missing mask files (example: "fusc_0001.png").
2. Inspected the real structure with `kagglehub.dataset_download('bkdkaggle/baseline-outputs')`, then `Path / "data_wound_seg"`, checking `test_masks` with a `*.png` glob (552 files) and whether `test/masks` exists (it does not).
3. Found the hybrid structure: flat masks, nested images and labels.

```python
# v2 to v4
TEST_IMAGES_DIR = BASE_INPUT_PATH / "test/images"   # nested
TEST_MASKS_DIR = BASE_INPUT_PATH / "test/masks"     # does not exist
# v5 onward
TEST_IMAGES_DIR = BASE_INPUT_PATH / "test/images"   # nested (images)
TEST_MASKS_DIR = BASE_INPUT_PATH / "test_masks"     # flat (masks, underscore)
```

Deployed: v5. Result: zero-shot evaluation succeeded. YOLO training failed on bug 3.

## Bug 3: YOLO config written to read-only input (v5)

Error: `ValueError: Failed to write file: [Errno 30] Read-only file system: '/kaggle/input/baseline-outputs/data_wound_seg/wound_dataset.yaml'`

Cause: `create_yolo_config()` writes `dataset_root/wound_dataset.yaml` by default. `/kaggle/input/` is read-only.

```python
# v5
create_yolo_config(
    dataset_root=YOLO_DATASET_ROOT,   # /kaggle/input/... (read-only)
    train_dir="train", val_dir="test", test_dir="test",
)
# v6
create_yolo_config(
    dataset_root=YOLO_DATASET_ROOT,
    train_dir="train", val_dir="test", test_dir="test",
    output_path=OUTPUT_PATH / "wound_dataset.yaml",   # /kaggle/working/ (writable)
)
```

Deployed: v6. Result: training started.

## Version history

| Version | Status | Issue | Next |
|---------|--------|-------|------|
| v1 | error | dataset path mismatch | v2: fix BASE_INPUT_PATH |
| v2 | error | Valid pairs: 0 (mask path) | v3: attempted extension fix |
| v3 | error | Valid pairs: 0 (mask path) | v4: attempted fix |
| v4 | error | Valid pairs: 0, wrong path | v5: inspected with kagglehub, found hybrid structure |
| v5 | error | zero-shot OK, YOLO config write failed | v6: add output_path |
| v6 | running | all three bugs fixed | monitor training |

Text damaged in source for part of the v3 and v4 rows.

## Lessons

1. Dataset structure was assumed. `test/masks/` was assumed to exist while only `test/images/` and `test/labels/` did. Validate the structure of any Kaggle dataset before deploying.
2. The quality check did not catch this. The smoke test ran on simulated data, so it could not surface real dataset-structure errors, and new Kaggle datasets were not covered. Smoke tests must use the exact validated structure.
3. `/kaggle/input/` is read-only. Write all generated files (YOLO config, labels, outputs) to `/kaggle/working/`.

## Workflow going forward

For any Kaggle deployment:
1. Validate the dataset structure first.
2. Build the smoke test from the exact validated structure.
3. Run the smoke test on Kaggle (CPU, about 2 minutes).
4. Deploy production once the smoke test passes.

## Time analysis

Spent:
- 5 failed deployments at about 10 minutes each: about 50 minutes.
- Debug time: 2+ hours.
- Total: about 3 hours.

With validation first:
- Dataset validation: 5 minutes.
- Smoke test creation: 10 minutes.
- Smoke test on Kaggle: 2 minutes.
- Production deployment would have worked first time.
- Total: 17 minutes.

Savings figure (stated as a 9.4x improvement): text damaged in source.

## Current state (v6)

- Valid pairs: 5520.
- Zero-shot evaluation completed.
- YOLO training started. Expected runtime 1.5 to 2 hours on P100.

Expected output:
- Dice improvement of +3 to 5% over the 80% baseline.
- Final Dice 83 to 85%.
- Validated across 3 seeds.

## Final code (v6)

```python
BASE_INPUT_PATH = Path("/kaggle/input/baseline-outputs/data_wound_seg")
OUTPUT_PATH = Path("/kaggle/working")
# hybrid structure
TEST_IMAGES_DIR = BASE_INPUT_PATH / "test/images"   # nested (training)
TEST_MASKS_DIR = BASE_INPUT_PATH / "test_masks"     # flat (underscore)
# YOLO config written to the writable location
create_yolo_config(
    dataset_root=YOLO_DATASET_ROOT,
    train_dir="train", val_dir="test", test_dir="test",
    output_path=OUTPUT_PATH / "wound_dataset.yaml",
)
```

## Next

1. Monitor v6 (about 1.5 to 2 hours).
2. Apply the same fixes to Ablations 2, 3 and 4.
3. Validate dataset structure for all notebooks.
