# Medical AI: Wound Segmentation Benchmark

Domain adaptation study: performance gain from fine-tuning YOLOv8 for clinical wound segmentation.

Status: notebooks validated for syntax only. Not executed here.

## Notebooks

- Act 1: domain adaptation for wound segmentation (`act-1-clean.ipynb`).
- Act 2: factor analysis for model performance (`act2_factor_analysis.ipynb`).
- Act 3: TinyML web deployment pipeline (`act3_tinyml_web_deployment.ipynb`).

## Run it

    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    python validate_notebook.py act-1-clean.ipynb
    python validate_notebook.py act2_factor_analysis.ipynb
    python validate_notebook.py act3_tinyml_web_deployment.ipynb

`validate_notebook.py` parses the JSON and compiles every code cell. It does not execute the notebooks. Executing them needs Kaggle data and, for training, a GPU. Not run here.

No pytest suite exists. Counts of notebooks that validate:
- Before: 4 passed, 0 failed (included `act-1.ipynb`, a markdown-only duplicate of the first cell of `act-1-clean.ipynb`).
- After: 3 passed, 0 failed.

`requirements.txt` installs on CPU (Python 3.11). `pip check` reports no broken requirements. `pip-audit` before: 72 findings (Pillow 9.5.0, scikit-learn 1.3.0, pytest 7.4.0). After the version bumps: none.

No GitHub Actions workflow is in this repo. CI is not set up.
