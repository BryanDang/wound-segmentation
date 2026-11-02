# DDI Dataset Path Resolution

Date: 2025-10-19.
Status: resolved.

## Issue

Ablation 4 deployment failed with a `FileNotFoundError` for DDI dataset files. The Kaggle dataset `bkdkaggle/ddi-diverse-dermatology` had images but lacked the cleaned CSV metadata files.

## Fix

Dataset `bkdkaggle/ddi-diverse-dermatology`, version created 2025-10-19 23:17:47. Files added:
- `ddi_train.csv` (22KB): 656 wound images metadata.
- `ddi_test.csv` (7KB): 203 wound images metadata.
- `ddi_metadata_cleaned.csv` (29KB): complete metadata for all DDI images.

Structure:
```
/kaggle/input/ddi-diverse-dermatology/
  PRIVATE - ddidiversedermatologyimages/
    000001.png
    000002.png
    ... (643+ dermatology images)
  ddi_train.csv               (added)
  ddi_test.csv                (added)
  ddi_metadata_cleaned.csv    (added)
```

Kernel metadata (`kernel-metadata.json`):
```json
{
  "id": "bkdkaggle/ablation-4a-hard-negatives-mixed",
  "dataset_sources": [
    "bkdkaggle/baseline-outputs",
    "bkdkaggle/ddi-diverse-dermatology",
    "bkdkaggle/wound-seg-code-v1"
  ]
}
```

Code paths in the notebooks:
```python
DDI_IMAGES_DIR = Path("/kaggle/input/ddi-diverse-dermatology/PRIVATE - ddidiversedermatologyimages")
DDI_TRAIN_CSV = "/kaggle/input/ddi-diverse-dermatology/ddi_train.csv"
DDI_TEST_CSV = "/kaggle/input/ddi-diverse-dermatology/ddi_test.csv"
```

## Current implementation

Ablation 4 notebooks:
1. Use dataset path `bkdkaggle/ddi-diverse-dermatology`.
2. Load images directly from the image directory.
3. Split train/test programmatically with `train_test_split(random_state=42)`.
4. Do not need the CSV files.

The CSV files were added for reproducibility reference, metadata tracking, optional use in variant experiments, and dataset documentation.

## Checks done

- Dataset exists on Kaggle: `bkdkaggle/ddi-diverse-dermatology`.
- CSV files uploaded.
- Kernel metadata uses the correct dataset path.
- Notebooks use the correct image directory path.
- No references to the old incorrect path (`diabetic-foot-ulcer-dfu-dataset`).
- Dataset structure documented.

## Files

Local:
- `/tmp/ddi_dataset_verification.md`: verification notes.
- `data/ddi_cleaned/ddi_train.csv`: train split metadata.
- `data/ddi_cleaned/ddi_test.csv`: test split metadata.
- `data/ddi_cleaned/ddi_metadata_cleaned.csv`: complete metadata.

Kaggle: `bkdkaggle/ddi-diverse-dermatology`, version updated with CSV files.

Notebooks (already correct):
- `notebooks/03_ablations/ablation_4a_hard_negatives_mixed/`
- `notebooks/03_ablations/ablation_4b_best_checkpoint/`
- `notebooks/03_ablations/ablation_4c_early_checkpoint/`
- `notebooks/03_ablations/ablation_4_smoke/`

## Status

Notebooks deploy as-is. Optional: verify the smoke test on Kaggle. Then run the production ablation 4 variants on GPU.

## Expected behavior on Kaggle

1. Find 643 DDI images in `/kaggle/input/ddi-diverse-dermatology/PRIVATE - ddidiversedermatologyimages/`.
2. Split with `train_test_split(test_size=0.2, random_state=42)`: 514 train, 129 test.
3. Combine with baseline wound images: about 2,722 train images.
4. Create the YOLO dataset with empty labels for DDI images (hard negatives).
5. Train multi-seed models (seeds 42, 123, 456).

## Dataset composition

DDI images (from `ddi_train.csv` and `ddi_test.csv`):
- Total: 859 images after cleaning.
- Train: 656 wound images (skin lesions).
- Test: 203 wound images.
- Purpose: hard negative mining (non-chronic-wound dermatology images).

Open: 643 images in the directory, 514+129 used by the notebooks, 859 in the CSVs. The notebooks use the directory, not the CSVs. Other reports give 642 (513+129) and 665 (532+133). To be reconciled.

Combined ablation 4 dataset:
- Train: about 2,722 images (2,208 wound + about 514 DDI).
- Test: about 681 images (552 wound + about 129 DDI).
- All DDI images have empty label files.

## Troubleshooting

1. Verify the dataset is available:
```bash
kaggle datasets files bkdkaggle/ddi-diverse-dermatology | grep -E "csv|png" | head -20
```

2. Check the image count:
```python
DDI_DIR = Path("/kaggle/input/ddi-diverse-dermatology/PRIVATE - ddidiversedermatologyimages")
print(f"Images found: {len(list(DDI_DIR.glob('*.png')))}")
```

3. Verify the CSV files exist:
```python
import os
csv_files = [
    "/kaggle/input/ddi-diverse-dermatology/ddi_train.csv",
    "/kaggle/input/ddi-diverse-dermatology/ddi_test.csv"
]
for f in csv_files:
    print(f"{f}: {os.path.exists(f)}")
```
