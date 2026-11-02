"""
Kaggle paths for Ablation 4, checked with kagglehub.

Dataset structure:

baseline-outputs/
├── data_wound_seg/
│   ├── train/
│   │   ├── images/     # 2,208 .png files
│   │   └── labels/     # 2,208 .txt files (YOLO format)
│   ├── test/
│   │   ├── images/     # 552 .png files
│   │   └── labels/     # 552 .txt files (YOLO format)
│   ├── train_images/   # (duplicate of train/images/)
│   ├── train_masks/    # PNG masks for reference
│   ├── test_images/    # (duplicate of test/images/)
│   └── test_masks/     # PNG masks for evaluation (552 .png)

ddi-diverse-dermatology/
└── PRIVATE - ddidiversedermatologyimages/
    └── *.png           # 643 .png files (dermatological conditions)
"""

KAGGLE_PATHS = {
    # Baseline wound dataset
    "baseline_root": "/kaggle/input/baseline-outputs/data_wound_seg",
    "baseline_train_imgs": "/kaggle/input/baseline-outputs/data_wound_seg/train/images",
    "baseline_train_labels": "/kaggle/input/baseline-outputs/data_wound_seg/train/labels",
    "baseline_test_imgs": "/kaggle/input/baseline-outputs/data_wound_seg/test/images",
    "baseline_test_labels": "/kaggle/input/baseline-outputs/data_wound_seg/test/labels",
    "baseline_test_masks": "/kaggle/input/baseline-outputs/data_wound_seg/test_masks",  # PNG for evaluation
    # DDI dataset
    "ddi_root": "/kaggle/input/ddi-diverse-dermatology",
    "ddi_images": "/kaggle/input/ddi-diverse-dermatology/PRIVATE - ddidiversedermatologyimages",
    # Code dataset
    "code_dataset": "/kaggle/input/wound-seg-code-v1",
    # Output paths
    "output_root": "/kaggle/working",
}

# Expected counts (for validation)
EXPECTED_COUNTS = {
    "baseline_train": 2208,
    "baseline_test": 552,
    "ddi_total": 643,  # Actual count is 643, not 665
    "ddi_train_80pct": 514,  # 643 * 0.8
    "ddi_test_20pct": 129,  # 643 * 0.2
}

# DDI has 643 images, not 665. Ablation specs to be updated.
