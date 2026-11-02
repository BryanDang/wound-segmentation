# Experimental Protocol: wound segmentation ablation study

Steps, data, settings and measures that make a result repeatable. Paths are relative to `model/`. "Unverified" means the claim could not be checked in the files that survived the snapshot; the evidence looked at is named.

## Contents
1. Start here
2. What it is
3. What is built
   - 3.1 Data
   - 3.2 Fixed settings
   - 3.3 Measures and decision rules
   - 3.4 Procedure
   - 3.5 Experiment register
   - 3.6 Ablation results
   - 3.7 Where the results disagree
   - 3.8 Code and notebooks
4. TBD
5. How to run and test
6. Open questions

## 1. Start here

1. Study that trains a small YOLOv8 model to outline chronic wounds and tests, one change at a time ("ablation"), which training changes help.
2. Start at 3.6 (results) and 3.7 (numbers conflict).
3. Full runs need a Kaggle GPU (notebooks in `notebooks/`). Tests run on CPU with `python -m pytest` (section 5).
4. Broken: wound images and all model weights are not in the snapshot; `src/models/` is missing; the "80 percent baseline" has no raw file behind it; 10 tests fail and 1 cannot load here.
5. First: decide the official baseline number (question 1 in section 6), then finish the unfinished ablations (section 4, items 1 to 6).

## 2. What it is

Capstone project. Goal: a YOLOv8n-seg model (smallest YOLOv8 variant with pixel masks) for wound outlines that could later run in a web browser.

Design: train a baseline, change one thing at a time, three seeds each (42, 123, 456), compare to the baseline with a significance test (`docs/ABLATION_SPECIFICATIONS.md`, "Study design").

Research questions, one per ablation:
1. A1 Focal Tversky loss: does a loss that punishes missed wound pixels more improve outlines?
2. A1B Dice++ loss: does a calibration-aware loss improve confidence scores?
3. A2 clinical augmentation: do rotation and vertical flip help?
4. A3 multi-scale training: does training at several image sizes help?
5. A4 hard negatives: does adding skin images with no wound (DDI) reduce false alarms? Three ways: 4a mixed, 4b and 4c two-stage.
6. A5 label smoothing: does it improve confidence scores?
7. A6 test-time augmentation (TTA): does averaging predictions on flipped copies help?

Wound app: described as using dummy data. In the snapshot the app is a README only (`web_demo/README.md`), so no app data exists to check (unverified).

## 3. What is built

### 3.1 Data

2,760 chronic-wound images from three public sources, resized to 512 by 512.

| Source | Images |
|---|---|
| Medetec Wound Database | 374 |
| Wang et al. 2020 | 1,210 |
| Oota et al. 2023 | 1,176 |

- Split: 2,208 train, 552 test. Test images are all specialist-annotated, from Wang et al.
- Kaggle path used by every run: `/kaggle/input/baseline-outputs/data_wound_seg` with `train/` and `test/` (`wound_dataset.yaml`). One class: wound.
- The YAML sets `val: test`. Early stopping and model selection use the same 552 images as the final score. Best-epoch results are slightly optimistic (question 5).
- Hard-negative data (A4): DDI, skin images with no wound. Two counts in the docs: 642 images, 513 train and 129 test (`docs/reports/ABLATION_4_ROOT_CAUSE_ANALYSIS.md`), and 665 images, 532 train and 133 test (`docs/ABLATION_SPECIFICATIONS.md`, "Dataset"). Unverified which is right. Labels are empty text files; `data/ddi_yolo_labels/train/` has 489 in the snapshot.
- In the snapshot: 3 sample images in `simulated_kaggle_data/`, a tiny smoke dataset in `kaggle_notebooks/TINY_SMOKE_TEST/` (notebook and metadata only), label files, and `kaggle_outputs/ablation_2_counterfactual_complete/data_wound_seg/correspondence_table.xlsx` (maps files to sources). Real images were left out.
- Smoke tests use dummy data (`data/TINY_TEST`-style folders of a few images). They check that code runs, not that the model is good.

### 3.2 Fixed settings

Same for every run unless the register in 3.5 says otherwise. Source: `kaggle_outputs/ablation_1_focal_tversky/training_logs/yolo_training/seed_42/args.yaml`.

| Setting | Value |
|---|---|
| Model | `yolov8n-seg.pt` (pretrained), Ultralytics 8.0.196 pinned in `docs/reports/ABLATION_4D_V10_FAILURE_ANALYSIS.md` |
| Epochs, patience | 100, 10 |
| Batch, image size, workers | 32, 512, 8 |
| Learning rate lr0 | 0.015 |
| amp | false |
| Seeds | 42, 123, 456 |
| Default augmentation | mosaic 1.0, close_mosaic 10, degrees 0, translate 0.1, scale 0.5, flipud 0, fliplr 0.5, hsv_v 0.4, auto_augment randaugment |
| Hardware | Kaggle P100 GPU, about 1.5 hours per seed |

RandAugment is on by default in YOLOv8 and was on in all early runs. Found 2025-10-17.

### 3.3 Measures and decision rules

- Per image on the 552 test images: Dice, IoU, precision, recall, boundary F1 (`src/evaluation/metrics.py`, `calculate_segmentation_metrics`, `calculate_boundary_f1`). Mean with bootstrap 95% interval (`calculate_bootstrap_ci`).
- Training log metric: mask mAP50, written to `results.csv` by Ultralytics as `metrics/mAP50(M)`.
- Dice and mask mAP50 are different measures. The docs mix them (3.7).
- Significance: Wilcoxon signed-rank on paired per-image scores, Cohen's d, Bonferroni: alpha 0.05 / 8 comparisons = 0.00625 (`src/evaluation/statistical_tests.py`; `docs/ABLATION_SPECIFICATIONS.md`, "Statistics").
- Rule in the docs: Adopt = gain of at least 3 points, d above 0.8, 95% interval excludes baseline. Marginal = gain of at least 1 point, d above 0.5. Reject = below that.
- Rule in code: ADOPT / CONDITIONAL / REJECT from d and corrected p only, no point threshold (`src/evaluation/statistical_tests.py`, line 648). The two rules differ (question 6).

### 3.4 Procedure

1. Build the code dataset and upload to Kaggle (`scripts/manage_kaggle_datasets.py`).
2. Generate the notebook with its builder (`scripts/build_<ablation>.py`). Strip outputs with `scripts/clean_notebook.py`.
3. Check `kernel-metadata.json` with `scripts/validate_kernel_metadata.py`.
4. Smoke test first: tiny data, CPU, 3 to 17 minutes (`notebooks/03_ablations/<ablation>_smoke`).
5. Production notebook: 3 seeds, settings from 3.2 plus the one change.
6. Download outputs to `kaggle_outputs/<ablation>/`. Check `args.yaml` to confirm the change was applied.
7. Compute the measures in 3.3, compare to baseline, apply the decision rule, record in 3.6.
8. Counterfactual run: put one setting back to its baseline value to test whether it caused a change (A2 and A3 have one each).

### 3.5 Experiment register

| ID | Change from baseline | Folder under `notebooks/` | State |
|---|---|---|---|
| Zero-shot | none; pretrained, no training | `02_baseline` | done |
| Baseline | fine-tuned, 3.2 | `02_baseline` | done per docs; no raw file |
| A1 | Focal Tversky (alpha 0.7, beta 0.3, gamma 2.0) | `03_ablations/ablation_1_focal_tversky` | done |
| A1B | Dice++ loss | `03_ablations/ablation_1b_smoke` | plan and smoke notebook only |
| A1C | Focal Tversky, no mosaic | `03_ablations/ablation_1c_focal_tversky_no_mosaic` | error: API mismatch |
| A2 | mosaic 0, degrees 15, flipud 0.5, hsv_v 0.3 | `03_ablations/ablation_2_clinical_augmentation` | done |
| A2-CF | A2 with mosaic back to 1.0 | `03_ablations/ablation_2_counterfactual_*` | trained; raw results not in snapshot |
| A3 v1 to v5 | sizes [416, 512, 640], mosaic or RandAugment toggled | `03_ablations/ablation_3_multiscale_training`, `ablation_3_v6_multiscale` | done, failed |
| A3-CF | multi-scale, all augmentation off | `03_ablations/ablation_3_counterfactual_no_augment` | done |
| A4a | train on wound + DDI together | `03_ablations/ablation_4a_hard_negatives_mixed`, `ablation_4_unified` | done |
| A4b, A4c | train on wounds, then fine-tune on DDI only | `03_ablations/ablation_4b_*`, `ablation_4c_*` | done, failed |
| A4d | curriculum learning | `03_ablations/ablation_4d_*` | not finished |
| A5 | label smoothing | `03_ablations/ablation_5_label_smoothing` | smoke passed only |
| A6 | test-time augmentation | `03_ablations/ablation_6_production` | smoke passed only |
| Mini | 7 augmentation switches | `03_ablations/mini_ablations*` | smoke runs only |

### 3.6 Ablation results

"Raw" = computed from a file in the snapshot. "Docs only" = a report states it, no raw file here. Raw numbers are best epoch per seed, mask mAP50, percent.

| ID | Result | Evidence | Check |
|---|---|---|---|
| Zero-shot | Dice 4.87 (95% CI 4.21 to 5.59), precision 2.6, recall 76.7, boundary F1 0.7, n = 552 | `kaggle_outputs/baseline_run/baseline_summary_statistics.json` | raw |
| Baseline | Dice 80.0 (CI 78.9 to 81.0), precision 74.7, recall 90.6, boundary F1 67.4; false positives 22.8, false negatives 6.6 | `docs/ABLATION_SPECIFICATIONS.md`, "Baseline" | docs only; unverified |
| A1 | Dice 80.01 vs 80.00: no gain. Raw mask mAP50 86.77 (sd 0.46); seeds 42/123/456 = 86.68 / 86.36 / 87.27; stopped at epochs 57 / 52 / 97 | `kaggle_outputs/ablation_1_focal_tversky/training_logs/yolo_training/seed_*/results.csv` | mAP50 raw; Dice docs only |
| A2 | Dice 68.27 (-11.73), or 62.74 (-17.26) in another report | `docs/reports/ABLATION_2_COUNTERFACTUAL_COMPLETE_ANALYSIS.md` | docs only; reports disagree |
| A2-CF | mask mAP50 86.78 (seed 42 only; "pending final download"); older counterfactual with translate and scale on: 85.99 (sd 1.25) | `docs/reports/ABLATION_2_COMPLETE_PARAMETER_METRIC_TABLE.md`; `kaggle_outputs/ablation_2_counterfactual_complete/completion_status.json` has no metrics | docs only; unverified |
| A3 v1 to v3 | Dice 54.57 (-25.43) | `docs/reports/ABLATION_3_CHRONOLOGICAL_HISTORY.md` | docs only |
| A3 v4 | Dice 68.81 (-11.19) | same | docs only |
| A3 v5 | Dice 4.80, precision 2.6, recall 74.1 | `docs/reports/ABLATION_3_COMPLETE_PARAMETER_METRIC_TABLE.md` | docs only |
| A3-CF | mask mAP50 83.48 (sd 1.47); seeds = 85.14 / 82.36 / 82.94; Dice 64.82 | `kaggle_outputs/ablation_3_counterfactual/yolo_training/seed_*/results.csv`; `completion_status.json` | mAP50 and Dice raw; another report says Dice 54.05 |
| A4a | mask mAP50 87.51 (sd 0.40); seeds = 87.86 / 87.08 / 87.60; precision 89.99; recall 80.76 | `kaggle_outputs/ablation_4_complete_analysis.json`; `kaggle_outputs/results_4a_seed_*.csv` | raw |
| A4b, A4c | 0.00 on every metric, all seeds | same JSON; `kaggle_outputs/results_4b_seed_*.csv`, `results_4c_seed_*.csv` | raw |
| Mini | all 7 smoke runs "success" in 14 to 37 seconds; no accuracy recorded | `kaggle_outputs/mini_ablations_smoke/mini_ablations_smoke_test_results.json` | raw; smoke only |

Supported by the evidence:
- A4b and A4c fail because the DDI-only data has zero wound examples, so YOLO cannot compute any metric (log line "no labels found in segment set, can not compute metrics", `docs/ABLATION_SPECIFICATIONS.md`, "4b"; `docs/reports/ABLATION_4_EMPTY_LABELS_CONCLUSIVE_ANSWER.md`). Empty label files do not break validation.
- A4a trains stably across seeds (sd 0.40, raw).
- Multi-scale with no augmentation is 3.3 points of mask mAP50 below A1 (83.48 vs 86.77, raw, same test set).
- Mosaic matters: runs with mosaic off scored far lower in the reports (A2, A3 v4). Docs only.

### 3.7 Where the results disagree

- Headline "A4a gains 7.5 points over the 80.0 baseline" compares mask mAP50 (87.51) with a number labelled Dice (80.0). On the same measure and test split, A1 (close to baseline) scores 86.77 raw, so A4a is ahead by about 0.7 points, not 7.5. Significance claims built on 7.5 points (Cohen's d 18.8 to 21.2 in `docs/ABLATION_SPECIFICATIONS.md` and `docs/reports/ABLATION_4_RESULTS_SUMMARY.txt`) are unverified.
- No raw file shows the 80.0 baseline. The only baseline file is the zero-shot run (Dice 4.87). `kaggle_outputs/ablation_1_focal_tversky/raw_outputs/baseline_per_image_results.csv` is byte-identical to it.
- Original A2: Dice 68.27 vs 62.74. A3-CF: Dice 64.82 (`completion_status.json`) vs 54.05 (`docs/reports/ABLATION_3_COUNTERFACTUAL_COMPLETE_ANALYSIS.md`).
- A3 v5: one report said YOLO ignored `auto_augment=None`; a later report says it was applied and the first claim was wrong (`docs/reports/ABLATION_3_V5_AUTO_AUGMENT_BUG_CORRECTION.md`). `docs/ABLATION_SPECIFICATIONS.md` still carries the old claim.
- 4a precision and recall per seed differ between `docs/ABLATION_SPECIFICATIONS.md` and the JSON. The JSON is used above.
- Text corruption: `README.md`, `web_demo/README.md` and `kaggle_outputs/ablation_1_focal_tversky/analysis/ABLATION_1_DEEP_SCIENTIFIC_ANALYSIS.md` have letters missing. Do not trust these three files.

### 3.8 Code and notebooks

Reusable pieces and tests: `docs/COMPONENTS.md`. Reports behind each ablation: `docs/reports/`. Plans: `docs/ABLATIONS_1B_5_6_IMPLEMENTATION_PLAN.md` and the per-ablation plans.

## 4. TBD

Each item: what, state, where work stopped, done-when.

1. Baseline raw results. State: the 80.0 baseline exists only in reports. Stopped at `kaggle_outputs/baseline_run/` (zero-shot only). Done-when: per-image CSV and summary JSON from the trained baseline (3 seeds, 552 test images) are in `kaggle_outputs/baseline_trained/`, and 3.6 shows Dice and mask mAP50 from them.
2. Same-measure comparison table. State: results mix Dice and mAP50. Stopped at 3.7. Done-when: 3.6 reports every finished ablation on both measures from raw files, with `comprehensive_comparison_report` output saved beside them.
3. A4 Phase 1: false-positive rate on DDI test images. State: not run; blocker was the baseline model weights location (`docs/ABLATION_SPECIFICATIONS.md`, "Ablation 4"). Done-when: a CSV gives the false-positive rate on DDI test images for baseline and A4a, 3 seeds each.
4. A4 redesign (mixed 4b/4c). State: proposal only (`docs/reports/ABLATION_4_REDESIGN_PROPOSAL.md`). Done-when: `results.csv` for both, 3 seeds, non-zero mask mAP50, compared with A4a.
5. A4d curriculum learning. State: v10 failed with `ModuleNotFoundError: ultralytics`; a minimal smoke test was deployed; no later result (`docs/reports/ABLATION_4D_V10_FAILURE_ANALYSIS.md`). Stopped at `notebooks/03_ablations/ablation_4d_smoke`. Done-when: production run gives `results.csv` for 3 seeds.
6. A1B, A5, A6 production runs. State: plans and smoke passes only. Stopped at `notebooks/03_ablations/ablation_1b_smoke`, `ablation_5_label_smoothing`, `ablation_6_production`. Done-when: each has 3-seed results in `kaggle_outputs/`.
7. A2-CF and A1C raw results. State: A2-CF has only a status file; A1C errored. Done-when: `results.csv` for 3 seeds in `kaggle_outputs/ablation_2_counterfactual_complete/`; A1C runs or is marked dropped in 3.5.
8. Mini-ablations (7 augmentation switches). State: smoke only. Stopped at `notebooks/03_ablations/mini_ablations_production`. Done-when: production results for all 7.
9. Separate validation split. State: validation equals test (`wound_dataset.yaml`). Done-when: train / validation / test split defined and the baseline rerun.
10. Final report and statistics. State: draft notes in `docs/ABLATION_4_INSIGHTS_FOR_FINAL_REPORT.md`. Done-when: one document gives every ablation's adopt / conditional / reject call from `comprehensive_comparison_report` at Bonferroni alpha 0.00625.
11. Browser deployment. State: quantization and TensorFlow.js planned; `web_demo/` has only a README; `src/deployment/deployment.py` and `src/training/trainer.py` are stubs. Done-when: an exported model loads in a page under `web_demo/` and returns a mask on the dummy data.
12. Restore missing code. State: `src/models/` and `tests/unit/models/` were dropped by the old `.gitignore`. Done-when: both restored and `python -m pytest` collects them.
13. Tests marked "not yet implemented". State: 34 skipped, in `tests/unit/data/test_data_loader.py`, `test_dataset_integrity.py`, `tests/unit/utils/test_environment_detection.py`. Done-when: loader, validator and environment detector exist and the skips are gone.

## 5. How to run and test

- Tests (CPU, about 30 seconds): `python -m pytest` from the project root. Single test: `python -m pytest tests/unit/evaluation/test_metrics.py::TestBootstrapCI::test_single_value`.
- Installed for this run: pytest, numpy, pandas, opencv-python-headless, scipy, matplotlib, scikit-learn, seaborn, openpyxl, pillow, pyyaml. `tests/unit/evaluation/test_tta.py` also needs torch.
- Result with those packages on Python 3.13 (numpy 2.5; `requirements.txt` asks for numpy below 2.0), no torch, no ultralytics: 173 passed, 10 failed, 34 skipped, 1 collection error.

| Cause | Count | Tests |
|---|---|---|
| Sample images left out of the snapshot ("Could not find image fusc_0001.png", "should have 4 training images") | 8 | `tests/integration/test_full_pipeline.py` (5), `tests/unit/pipeline/test_eda_contract_refactored.py` (2), `tests/unit/pipeline/test_eda_pipeline_visuals.py::test_generate_example_predictions` |
| Running as root, read-only folder still writable | 1 | `tests/unit/data/test_baseline_dataset_format.py::TestDatasetFormat::test_readonly_output_directory` |
| `detect_environment()` returns a string; test indexed it like a dict. Fixed: the test now asserts the string | 1 | `tests/unit/utils/test_environment_detection.py::TestEnvironmentDetection::test_detect_kaggle_environment` |
| torch not installed | 1 error | `tests/unit/evaluation/test_tta.py` |
| Skipped as "not yet implemented" | 34 | TBD item 13 |

- Before the fix, every test failed to import because `src/__init__.py` imported the missing `src/models`.
- Not run: any training, any Kaggle step, any notebook. No model downloaded. CI: none found (unverified).
- Full runs: upload the code dataset, push the notebook with the Kaggle CLI (needs a Kaggle login; credentials come from environment variables, never from the repository).

## 6. Open questions

1. Official baseline: Dice 80.0, or mask mAP50 from a trained baseline run? The A4a "+7.5 points" result depends on it. Baseline `results.csv` or per-image file: to be located.
2. Keep both `kaggle_outputs/ablation_4_unified_20251020` and `ablation_4_unified_hard_negatives`? They differ only by an analysis report and `yolo_training_4a/`.
3. DDI size: 642 images or 665?
4. Validation split from test (TBD item 9) changes every number.
5. Official decision rule: point-gain rule in the docs, or p-value and effect-size rule in the code?
6. Restore `src/models/`. The `.gitignore` line `models/` hid it.
7. "Wound app uses dummy data": covers `web_demo/` only, or also the smoke datasets? Treated as `web_demo/` and smoke data.
