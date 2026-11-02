# Reusable components: model

Paths are relative to `model/`.

## Contents
1. Summary
2. Library code (src/)
3. Scripts (scripts/)
4. Notebook families (notebooks/)
5. Interface applications with heavy computation
6. Not reusable or not built

## 1. Summary

- Best reuse: `src/evaluation/statistical_tests.py` and `src/evaluation/metrics.py` (plain NumPy and SciPy, no model needed), `src/data/yolo_config.py`, `src/data/preprocessing.py`.
- Everything is Python. Nothing converted to Rust.
- CPU test run without torch: 173 passed, 10 failed, 34 skipped, 1 collection error (see EXPERIMENT-PROTOCOL.md, "How to run and test").
- Reuse levels: High = works on any segmentation project with little change. Medium = needs edits for another dataset or tool. Low = tied to this project's Kaggle setup.

## 2. Library code (src/)

| Component | File | Purpose | Reuse | Outside dependencies | Tests |
|---|---|---|---|---|---|
| Segmentation metrics | `src/evaluation/metrics.py` | Dice, IoU, precision, recall per image; boundary F1; bootstrap confidence interval | High | numpy, opencv | `tests/unit/evaluation/test_metrics.py` (passes) |
| Statistical tests | `src/evaluation/statistical_tests.py` | Wilcoxon signed-rank, Cohen's d, Bonferroni correction, combined comparison report with ADOPT / CONDITIONAL / REJECT call (rule in the function docstring, line 648) | High | numpy, pandas, scipy; imports `src.utils.io` | `tests/unit/evaluation/test_statistical_tests.py` (passes) |
| YOLO config builder | `src/data/yolo_config.py` | Writes and checks the dataset YAML YOLO reads | High | none (standard library) | `tests/unit/data/test_yolo_config.py` (passes) |
| Mask to YOLO label | `src/data/preprocessing.py` | Binary mask to YOLO polygon label; empty labels for hard negatives; train/test split for DDI skin images | Medium | opencv, pandas | `tests/unit/data/test_baseline_dataset_format.py`, `tests/integration/test_yolo_label_creation.py` |
| Clinical augmentation | `src/data/clinical_augmentation.py` | Rotation, brightness, blur, elastic deformation applied to image and mask together | Medium | numpy, opencv, scipy | `tests/unit/data/test_clinical_augmentation.py` (passes). Not used in final runs; Ablation 2 used YOLO's built-in augmentation settings. |
| Test-time augmentation | `src/evaluation/tta.py` | Runs a model on flipped and rotated copies and merges masks; checks each transform can be undone | Medium | torch, numpy | `tests/unit/evaluation/test_tta.py` (needs torch, did not run) |
| YOLO result helpers | `src/evaluation/yolo_utils.py` | Ultralytics result object to a mask for the metrics code | Medium | numpy | `tests/unit/evaluation/test_yolo_utils.py` (passes) |
| EDA pipeline | `src/pipeline/eda.py` (1,435 lines) | Per-image statistics, summary tables, plots, report for an image and mask folder | Medium | pandas, numpy, opencv, matplotlib, seaborn | `tests/unit/pipeline/*`, `tests/unit/visualization/*`, `tests/integration/test_full_pipeline.py` (8 fail: sample images missing from the snapshot) |
| Kaggle dataset metadata | `src/deployment/kaggle_dataset.py` | Checks Kaggle credentials; builds metadata for a code dataset | Low | kaggle CLI (not imported here) | `tests/unit/deployment/test_kaggle_dataset.py`, `tests/integration/scripts/test_kaggle_dataset_creation.py` |
| Environment detector | `src/utils/environment.py` | Returns "kaggle" or "local" from one folder check (9 lines) | Low | none | `tests/unit/utils/test_environment_detection.py` (1 fails: test expects a dict; 8 skipped as "not yet implemented") |

## 3. Scripts (scripts/)

No tests cover the scripts (unverified: no test imports them).

| Component | File | Purpose | Reuse | Outside dependencies |
|---|---|---|---|---|
| Notebook builders | `scripts/build_*.py` (23 files) | Each writes one Kaggle notebook cell by cell. Newest: `build_ablation_4_unified.py`, `build_tta_production_notebook.py`, `build_mini_ablations_production.py`. | Medium | nbformat |
| Notebook cleaner | `scripts/clean_notebook.py` | Strips outputs and cell ids before upload | High | nbformat |
| Safe deploy | `scripts/safe_notebook_deploy.py` | Checks a notebook and kernel metadata, then pushes to Kaggle | Medium | kaggle CLI |
| Metadata checker | `scripts/validate_kernel_metadata.py` | Catches common kernel-metadata mistakes before a push | Medium | none |
| Dataset manager | `scripts/manage_kaggle_datasets.py` | Create and version Kaggle datasets | Low | kaggle CLI |
| DDI preparation | `scripts/prepare_ddi_dataset.py` | Builds the DDI hard-negative split and empty labels | Low | opencv, pandas |

## 4. Notebook families (notebooks/)

Each is a Kaggle notebook plus `kernel-metadata.json`. Full runs need a GPU and the pretrained weights `yolov8n-seg.pt`.

| Family | Folder | Reuse | Note |
|---|---|---|---|
| EDA | `notebooks/01_eda` | Medium | Calls `src/pipeline/eda.py` |
| Baseline and zero-shot | `notebooks/02_baseline` | Medium | Trains YOLOv8n-seg, 3 seeds |
| Ablations 1 to 6, mini-ablations | `notebooks/03_ablations/<name>` | Low | One changed setting per notebook. `*_smoke*` = tiny CPU check, `*_production` = full run, `*_clean` = outputs stripped. |
| DDI diagnostic | `notebooks/03_ddi_diagnostic` | Low | Checks the empty-label (hard negative) behavior of YOLO |

## 5. Interface applications with heavy computation (Rust question)

- None qualify. `web_demo/` is a README only; the planned in-browser TensorFlow.js app was never built (`web_demo/README.md`, garbled text, unverified).
- Training notebooks do heavy computation inside Ultralytics and PyTorch on a Kaggle GPU. No project code loop to convert. Open: decide whether anything should move to Rust. No change made.

## 6. Not reusable or not built

- `src/training/`, `src/visualization/plots.py`, `src/deployment/deployment.py`: stubs of 0 to 1 lines. The package docstrings promise a Trainer, config, checkpoints and quantization that do not exist.
- `src/models/` and `tests/unit/models/`: missing from the snapshot (original `.gitignore` rule `models/`). The Focal Tversky loss code was probably there (unverified).
- Model weights: left out of the snapshot.
