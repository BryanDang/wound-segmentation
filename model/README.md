# Wound Segmentation Model

YOLOv8n-seg wound segmentation: training and evaluation pipeline, Kaggle notebooks for a baseline and a set of ablations, tests, and result reports. License: MIT (`LICENSE`). Citation: `CITATION.cff`.

Status: capstone work in progress. Tests run on CPU. Notebooks not re-run here.

## Layout

- `src/`: package. `data/` (preprocessing, clinical augmentation, YOLO config), `evaluation/` (metrics, statistical tests, TTA), `pipeline/` (EDA), `utils/`, `visualization/`. `deployment/` and `training/` are placeholders.
- `tests/`: unit and integration tests.
- `notebooks/`: Kaggle notebooks. `01_eda`, `02_baseline`, `03_ddi_diagnostic`, `03_ablations/` (ablations 1 to 6, smoke and diagnostic variants, mini ablations).
- `scripts/`: notebook builders and Kaggle deployment helpers.
- `experiments/`: diagnostics and notes from the Ablation 1 debugging.
- `kaggle_outputs/`: result files from runs.
- `docs/`: design, method and result docs. Index in `docs/README.md`.
- `simulated_kaggle_data/`: small sample for local runs.
- `web_demo/`: README only.
- `data/`: DDI labels and summary.

Data sources are in `../docs/DATA.md`.

## Run it

Python 3.11. CPU only, no GPU needed for the tests.

    cd model
    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q

`requirements-kaggle.txt` holds the pins that match the Kaggle notebook image.

Test counts (CPU, current library versions from `requirements.txt`, running as root):
- Before: 191 passed, 10 failed, 34 skipped, 0 errors.
- After: 193 passed, 0 failed, 43 skipped, 0 errors.

The 10 former failures:
- 8 need sample images that are not in the repo (5 in `tests/integration/test_full_pipeline.py`, 2 in `tests/unit/pipeline/test_eda_contract_refactored.py`, 1 in `tests/unit/pipeline/test_eda_pipeline_visuals.py`). Now skipped with the reason "Sample images not in the repo".
- `test_readonly_output_directory` expects a permission error. Root ignores directory permissions. Skipped when running as root.
- `test_detect_kaggle_environment` expected a dict. `detect_environment()` returns "kaggle" or "local", and the Kaggle notebook uses it as a string. The test was wrong and now asserts the string. A matching test for "local" was added (193 passed includes both).

The 34 earlier skips are placeholders for a data loader, a dataset validator and an `environment_detector` module that are not implemented. They skip on ImportError and exercise no existing code. The 43 skips are these 34, the 8 missing sample images and 1 root-permission test.

Not run: the Kaggle notebooks. They need Kaggle datasets and a GPU.

## Security

`pip-audit` on `requirements.txt`: no known vulnerabilities. On `requirements-kaggle.txt`: scikit-learn 1.2.2 was flagged, now 1.5.2.
