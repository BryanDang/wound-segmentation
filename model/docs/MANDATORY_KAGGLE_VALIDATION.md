# Kaggle Kernel Validation Before Every Push

Blocking step before any `kaggle kernels push`.

## Problem

Kernels fail when `enable_internet: "false"` and the notebook runs `pip install ultralytics`.

Failures:
- Ablation 4 v1 (2025-10-17): DNS failure, internet disabled.
- Ablation 6 v1 (2025-10-17): DNS failure, internet disabled.
- Earlier repeats of the same error.

Cost: wasted GPU hours and debugging time.

## Protocol

```bash
cd notebooks/<notebook_dir>

# 1. Clean notebook
python3 ../../scripts/clean_notebook.py <notebook>.ipynb

# 2. Validate metadata (exits with error on failure)
python3 ../../scripts/validate_kernel_metadata.py kernel-metadata.json

# 3. Deploy only if validation passed
kaggle kernels push -p .
```

## What `scripts/validate_kernel_metadata.py` checks

1. Internet (critical): a notebook with `pip install` must have `enable_internet: "true"`. Blocks otherwise.
2. GPU: YOLO training notebooks should have `enable_gpu: "true"`. Warns if missing.
3. Datasets: training notebooks must have `bkdkaggle/wound-seg-code-v1`; should have `bkdkaggle/baseline-outputs`.
4. Files: notebook file exists; `kernel-metadata.json` structure.

## kernel-metadata.json template for training notebooks

```json
{
  "id": "bkdkaggle/<kernel-slug>",
  "title": "<Human Readable Title>",
  "code_file": "<notebook_name>_clean.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "true",        // required for YOLO training
  "enable_internet": "true",   // required for pip install
  "dataset_sources": [
    "bkdkaggle/baseline-outputs",
    "bkdkaggle/wound-seg-code-v1"
  ],
  "kernel_sources": [],
  "competition_sources": []
}
```

Booleans must be strings: `"true"`, not `true`.

## Common mistakes

1. Internet disabled for training: `"enable_internet": "false"` fails at pip install.
2. Boolean instead of string: `"enable_internet": true`. Kaggle requires strings.
3. Missing dataset: `dataset_sources` with only `bkdkaggle/baseline-outputs` lacks `wound-seg-code-v1`.

## Command

```bash
python3 scripts/validate_kernel_metadata.py <path-to-kernel-metadata.json>
```

Success output:
```
Validating: notebooks/03_ablations/ablation_3/kernel-metadata.json
============================================================
Kernel ID exists: bkdkaggle/ablation-3-multi-scale-training
Internet enabled (needed for pip install)
GPU enabled (detected YOLO/model usage)

Dataset Sources (2):
   - bkdkaggle/baseline-outputs
   - bkdkaggle/wound-seg-code-v1
2 dataset(s) configured
Notebook exists: ablation_3_multiscale_training_clean.ipynb

============================================================
VALIDATION RESULTS
============================================================

VALIDATION PASSED - Safe to deploy
```

Failure output:
```
============================================================
VALIDATION RESULTS
============================================================

ERRORS (1):
   CRITICAL: Notebook contains 'pip install' but enable_internet=false
   Set enable_internet='true' in metadata

VALIDATION FAILED - Fix errors before deploying
```

## Pre-push order

1. Read `kernel-metadata.json`.
2. If the notebook contains `pip install`, confirm `enable_internet: "true"`. Fix before deploying if false.
3. Confirm required dataset sources.
4. Run `python3 scripts/validate_kernel_metadata.py kernel-metadata.json`.
5. Deploy.

Validator: `scripts/validate_kernel_metadata.py`.
