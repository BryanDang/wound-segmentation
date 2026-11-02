"""
Build single comprehensive smoke test for all 7 mini-ablations.

Runs all 7 augmentation configurations sequentially in one notebook:
- 1 epoch training per configuration
- CPU execution (no GPU quota used)
- TINY dataset (4 train images)
- ~21 minutes total (7 configs × 3 min each)
"""

import nbformat as nbf
from pathlib import Path

# All 7 mini-ablation configurations
MINI_ABLATIONS = {
    "1_no_mosaic": {
        "name": "No Mosaic",
        "hypothesis": "Mosaic provides critical spatial diversity",
        "expected": "70-75% Dice (significant drop)",
        "params": {"mosaic": 0.0}
    },
    "7_no_randaugment": {
        "name": "No RandAugment",
        "hypothesis": "Unpredictable transforms hurt clinical color",
        "expected": "78-82% Dice (similar or better)",
        "params": {"auto_augment": None}
    },
    "2_no_color": {
        "name": "No Color Augmentations",
        "hypothesis": "Color augmentations alter diagnostic features",
        "expected": "78-82% Dice",
        "params": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "erasing": 0.0,
            "auto_augment": None
        }
    },
    "4_minimal": {
        "name": "Minimal (Mosaic+Flip Only)",
        "hypothesis": "Only essential augmentations needed",
        "expected": "75-80% Dice",
        "params": {
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "erasing": 0.0,
            "auto_augment": None
        }
    },
    "5_no_translate": {
        "name": "No Translate",
        "hypothesis": "Translate redundant with mosaic",
        "expected": "78-82% Dice",
        "params": {"translate": 0.0}
    },
    "6_clinical_conservative": {
        "name": "Clinical Conservative",
        "hypothesis": "Reduced safe augmentations match baseline",
        "expected": "78-82% Dice",
        "params": {
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.3,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.2,
            "erasing": 0.0,
            "auto_augment": None
        }
    },
    "3_no_scale": {
        "name": "No Scale",
        "hypothesis": "Scale redundant with mosaic",
        "expected": "78-82% Dice",
        "params": {"scale": 0.0}
    }
}

# Baseline defaults (what parameters inherit if not specified)
BASELINE_PARAMS = {
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "bgr": 0.0,
    "erasing": 0.4,
    "auto_augment": "randaugment",
    "close_mosaic": 10,
}

SMOKE_TEST_PARAMS = {
    "epochs": 1,
    "batch": 4,
    "imgsz": 512,
    "patience": 10,
    "seed": 42,
}


def create_markdown_cell(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def create_code_cell(code: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(code)


def build_comprehensive_smoke_test() -> nbf.NotebookNode:
    """Build single notebook testing all 7 mini-ablations."""

    cells = []

    # Cell 0: Title
    cells.append(create_markdown_cell(f"""# Mini-Ablations Comprehensive Smoke Test (CPU)

**Purpose**: Tests all 7 augmentation configurations in one notebook

**Configurations**:
{chr(10).join(f"- **{config['name']}**: {config['hypothesis']}" for config in MINI_ABLATIONS.values())}

**Smoke Test Parameters**:
- Dataset: TINY (4 train images, 2 test images)
- Training: 1 epoch per config, batch=4 (CPU-compatible)
- Expected Runtime: ~21 minutes (7 configs × 3 min each)
- Purpose: Validate configurations before GPU deployment

**Reference**: `docs/BASELINE_VS_CLINICAL_AUGMENTATION_COMPARISON.md`. Not in the repo.
"""))

    # Cell 1: NumPy locking
    cells.append(create_code_cell("""# NumPy Version Locking
import numpy as np
import subprocess

default_numpy = np.__version__
print(f"Default NumPy: {default_numpy}")

subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall", f"numpy=={default_numpy}"],
    check=True
)

print(f"Locked NumPy: {np.__version__}")
"""))

    # Cell 2: Imports
    cells.append(create_code_cell("""import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import json
import time

print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
"""))

    # Cell 3: Environment setup
    cells.append(create_code_cell("""# Environment Setup
IN_KAGGLE = Path('/kaggle/input').exists()

if IN_KAGGLE:
    BASE_INPUT_PATH = Path('/kaggle/input')
    OUTPUT_PATH = Path('/kaggle/working')
    TINY_PATH = BASE_INPUT_PATH / 'tiny-wound-seg-smoke-test'
else:
    print("ERROR: Must run on Kaggle")
    sys.exit(1)

print(f"Output path: {OUTPUT_PATH}")
print(f"TINY path: {TINY_PATH}")
"""))

    # Cell 4: YOLO config
    cells.append(create_code_cell("""# YOLO Configuration
config_content = f'''
path: {TINY_PATH}
train: train/images
val: test/images

names:
  0: background
  1: wound
'''

config_path = OUTPUT_PATH / 'wound_dataset.yaml'
config_path.write_text(config_content.strip())

print(f"YOLO config created: {config_path}")
"""))

    # Cell 5: Mini-ablation configurations
    ablation_configs_str = "{\n"
    for ablation_id, config in MINI_ABLATIONS.items():
        ablation_configs_str += f'    "{ablation_id}": {{\n'
        ablation_configs_str += f'        "name": "{config["name"]}",\n'
        ablation_configs_str += f'        "hypothesis": "{config["hypothesis"]}",\n'
        ablation_configs_str += f'        "expected": "{config["expected"]}",\n'
        ablation_configs_str += f'        "params": {config["params"]}\n'
        ablation_configs_str += "    },\n"
    ablation_configs_str += "}"

    cells.append(create_code_cell(f"""# Mini-Ablation Configurations
MINI_ABLATIONS = {ablation_configs_str}

BASELINE_PARAMS = {BASELINE_PARAMS}

SMOKE_TEST_PARAMS = {SMOKE_TEST_PARAMS}

print(f"Testing {{len(MINI_ABLATIONS)}} mini-ablation configurations")
print(f"Smoke test: {{SMOKE_TEST_PARAMS['epochs']}} epoch, batch={{SMOKE_TEST_PARAMS['batch']}}")
"""))

    # Cell 6: Training loop
    cells.append(create_code_cell("""# Train All Mini-Ablations
results_summary = {}

for ablation_id, config in MINI_ABLATIONS.items():
    print("\\n" + "=" * 80)
    print(f"MINI-ABLATION: {config['name']}")
    print("=" * 80)
    print(f"Hypothesis: {config['hypothesis']}")
    print(f"Expected: {config['expected']}")

    # Merge baseline params with ablation-specific overrides
    augmentation_params = BASELINE_PARAMS.copy()
    augmentation_params.update(config['params'])

    print("\\nAugmentation Parameters:")
    for key, value in sorted(augmentation_params.items()):
        if value != BASELINE_PARAMS.get(key):
            print(f"  {key}: {value} ← CHANGED")
        else:
            print(f"  {key}: {value}")

    # Initialize model
    model = YOLO('yolov8n-seg.pt')

    # Train
    print(f"\\nTraining...")
    start_time = time.time()

    try:
        results = model.train(
            data=str(config_path),
            batch=SMOKE_TEST_PARAMS['batch'],
            epochs=SMOKE_TEST_PARAMS['epochs'],
            imgsz=SMOKE_TEST_PARAMS['imgsz'],
            patience=SMOKE_TEST_PARAMS['patience'],
            seed=SMOKE_TEST_PARAMS['seed'],
            **augmentation_params,
            save=True,
            project=str(OUTPUT_PATH / "mini_ablations"),
            name=ablation_id,
            exist_ok=True,
            verbose=False
        )

        elapsed = time.time() - start_time
        status = "success"
        error_msg = None

        print(f"PASS: Training completed in {elapsed/60:.1f} minutes")

    except Exception as e:
        elapsed = time.time() - start_time
        status = "failed"
        error_msg = str(e)
        print(f"FAIL: Training failed: {error_msg}")

    # Store results
    results_summary[ablation_id] = {
        "name": config['name'],
        "hypothesis": config['hypothesis'],
        "expected": config['expected'],
        "augmentation_params": augmentation_params,
        "status": status,
        "error": error_msg,
        "training_time_seconds": elapsed
    }

print("\\n" + "=" * 80)
print("ALL MINI-ABLATIONS COMPLETE")
print("=" * 80)
"""))

    # Cell 7: Save results
    cells.append(create_code_cell("""# Save Smoke Test Results
output_file = OUTPUT_PATH / "mini_ablations_smoke_test_results.json"
with open(output_file, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"Results saved: {output_file}")

# Summary
print("\\nSummary:")
for ablation_id, result in results_summary.items():
    status_icon = "PASS" if result['status'] == "success" else "FAIL"
    print(f"  {status_icon} {result['name']}: {result['status']}")

# Completion status
completion = {
    "status": "complete",
    "n_configs_tested": len(MINI_ABLATIONS),
    "n_successful": sum(1 for r in results_summary.values() if r['status'] == 'success'),
    "n_failed": sum(1 for r in results_summary.values() if r['status'] == 'failed')
}

completion_file = OUTPUT_PATH / "completion_status.json"
with open(completion_file, 'w') as f:
    json.dump(completion, f, indent=2)

print(f"\\nCompletion status: {completion_file}")
print(f"Successful: {completion['n_successful']}/{completion['n_configs_tested']}")
"""))

    # Create notebook
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    }
    nb["cells"] = cells

    # Normalize and validate
    nb = nbf.validator.normalize(nb)[1]
    nbf.validate(nb)

    return nb


def main():
    """Generate comprehensive smoke test notebook."""

    output_dir = Path(__file__).parent.parent / "notebooks" / "03_ablations" / "mini_ablations_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building comprehensive mini-ablations smoke test...")
    print(f"Output directory: {output_dir}")

    # Build notebook
    nb = build_comprehensive_smoke_test()

    # Write notebook
    notebook_path = output_dir / "mini_ablations_smoke_test.ipynb"
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

    # Create kernel metadata
    metadata = {
        "id": "bkdkaggle/mini-ablations-comprehensive-smoke-test",
        "title": "Mini Ablations Comprehensive Smoke Test",
        "code_file": "mini_ablations_smoke_test.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "bkdkaggle/tiny-wound-seg-smoke-test"
        ],
        "kernel_sources": []
    }

    metadata_path = output_dir / "kernel-metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"PASS: Notebook: {notebook_path}")
    print(f"PASS: Metadata: {metadata_path}")
    print()
    print("Next steps:")
    print(f"1. Deploy: cd {output_dir} && kaggle kernels push -p .")
    print("2. Monitor: kaggle kernels status bkdkaggle/mini-ablations-comprehensive-smoke-test")
    print("3. Download: kaggle kernels output bkdkaggle/mini-ablations-comprehensive-smoke-test")


if __name__ == "__main__":
    main()
