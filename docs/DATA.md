# Data

Images and datasets in this repo come from public Kaggle datasets. Only small samples and derived label files are committed. Full images stay on Kaggle.

## Source collections (wound images)

2,760 chronic-wound images from three public sources, resized to 512 by 512. Split: 2,208 train, 552 test. Details in `model/docs/EXPERIMENT-PROTOCOL.md`, section 3.1.

| Source | Images | Link |
|---|---|---|
| Medetec Wound Database | 374 | None in the repo. |
| Wang et al. 2020 (FUSeg) | 1,210 | None in the repo. |
| Oota et al. 2023 (WSNet) | 1,176 | None in the repo. |

License: to be confirmed. (Medetec, Wang et al. 2020, Oota et al. 2023; none stated in the repo.)

Citation text in the repo: Oota, S.R., et al. "WSNet: Towards An Effective Method for Wound Image Segmentation." Full reference to be confirmed.

## Hard negatives

DDI (Diverse Dermatology Images), skin images with empty labels, used in Ablation 4.

- Kaggle dataset: `bkdkaggle/ddi-diverse-dermatology`
- Mounted at: `/kaggle/input/ddi-diverse-dermatology`
- Counts: 642 or 665 images depending on the document. Open. Split 513 train, 129 test in the runs.
- Cleaned summary: `model/data/ddi_cleaned/ddi_dataset_summary.txt`. YOLO labels: `model/data/ddi_yolo_labels/`.

License: to be confirmed.

## Kaggle datasets used by the notebooks

| Kaggle id | Role | License |
|---|---|---|
| `bkdkaggle/baseline-outputs` | Wound images and masks in YOLO layout (`data_wound_seg/`). Single source for all ablations. Layout in `model/docs/BASELINE_OUTPUTS_HYBRID_STRUCTURE.md`. | License: to be confirmed. |
| `wound-segmentation-images` | Original source layout, reference only. | License: to be confirmed. |
| `bkdkaggle/ddi-diverse-dermatology` | DDI hard negatives. | License: to be confirmed. |
| `bkdkaggle/wound-seg-code-v1` | Pipeline code from `model/src`. Metadata in `model/src/dataset-metadata.json`. | MIT (stated in the metadata). |
| `bkdkaggle/tiny-wound-seg-smoke-test` | 3 train and 3 test images for smoke runs. Metadata in `model/kaggle_notebooks/TINY_SMOKE_TEST/dataset-metadata.json`. | CC0-1.0 (stated in the metadata). |
| `bkdkaggle/wound-segmentation-eda-results` | EDA outputs. Metadata in `model/kaggle_outputs/dataset-metadata.json`. | CC0-1.0 (stated in the metadata). |

## In the repo

- `model/simulated_kaggle_data/`: 3 sample images and masks used by tests. Two of the five train images and no test images are present.
- `app/Wound kit Final images/`: demo images for the app. Dummy data, generated images and a screenshot. Source and license to be confirmed.
- `model/assets/confusion_matrix.png`: output of a run.
