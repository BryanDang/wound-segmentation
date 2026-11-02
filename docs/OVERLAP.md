# Overlap

Three projects were merged here: `app/`, `model/`, `benchmark/`. This note lists what repeated, what was removed, what stayed, and what is unique.

## Removed

Dead copies:
- `model/attic/` (19 files). Byte copies or older drafts of live files: old kaggle_outputs copies, old notebook numbering (`05_ablation`, `05_ablation_smoke`), a smaller draft of ablation 4a, a corrupted backup of the ablation specifications.
- `app/public/` (`index.html`, `app.js`). Older hosted copy of the app. Current app is `app/index.html` plus `app/ios-camera.js`.
- `model/data_wound_seg/correspondence_table.xlsx`. Same bytes as the copy under `model/kaggle_outputs/ablation_2_counterfactual_complete/data_wound_seg/`.
- `model/kaggle_notebooks/TINY_SMOKE_TEST/data_wound_seg/`. Label files identical to `model/kaggle_outputs/ablation_3_counterfactual_smoke_success/`.
- `benchmark/act-1.ipynb`. Only the intro cell, identical to the first cell of `act-1-clean.ipynb`.

Setup files with no use in the project: two files under `model/`.

Firebase files with no use (the app calls no Firebase):
- `app/firebase.json`, `app/firestore.indexes.json`. The deny-all rules stay.

Stale logs, task lists and status reports, in `docs/`, `model/docs/`, `model/docs/reports/`, `model/experiments/` and one under `model/notebooks/03_ablations/`. Facts that still mattered are in the live design and result docs.

One-off patch and deploy scripts already applied, in `model/scripts/` and `model/experiments/`.

## Kept, still repeated

- `*_clean.ipynb` twins in `model/notebooks/` (12 pairs). The clean copy is the one pushed to Kaggle. The other copy may hold outputs. To be confirmed which to keep.
- `model/kaggle_outputs/ablation_4_unified_20251020/` and `ablation_4_unified_hard_negatives/`. 131 files identical. The second adds `yolo_training_4a/`, the first adds the analysis report and json.
- `model/kaggle_outputs/baseline_run/baseline_per_image_results.csv` equals the copy in `ablation_1_focal_tversky/raw_outputs/`.
- `model/scripts/build_ablation_4_unified.py` and `build_unified_ablation_4.py`. Overlapping builders. To be confirmed.
- `model/scripts/build_*.py`, `*_cells.py`. Generate notebooks that are also committed. Notebook text repeats the script text.
- `model/src/wound_dataset.yaml` and `model/wound_dataset.yaml`. Different content, same role.
- Overlapping ablation 4 reports in `model/docs/reports/`. None is a strict subset.
- `model/docs/ABLATIONS_1B_5_6_IMPLEMENTATION_PLAN.md`. Earlier draft of the per-ablation plans.
- Smoke notebooks under `model/notebooks/03_ablations/*_smoke/` repeat the production notebook structure.

## Unique to each part

`app/`:
- Browser app: `index.html`, `ios-camera.js`.
- Exported models: `QATbest.pt`, `QATbest.onnx`, `QATbest_int8.tflite`.
- Demo images, hosting header file `vercel.json`, deny-all Firebase rules.

`model/`:
- Package `src/` and tests (`tests/`).
- Ablation notebooks, experiment notes, Kaggle outputs, DDI labels.
- Design and result docs in `docs/`.

`benchmark/`:
- Three notebooks (Act 1 domain adaptation, Act 2 factor analysis, Act 3 TinyML web deployment).
- `validate_notebook.py`.

## Not checked

- No model file in `app/` is shown to come from a specific training run in `model/`. Open.
- Voice pass on notebook markdown and on the notebook-building scripts is partial. Prose in notebook cells is not rewritten.
