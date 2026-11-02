"""
Build CPU smoke test notebooks for 7 augmentation mini-ablations.

Each smoke test validates a specific augmentation configuration with TINY dataset:
- CPU execution (no GPU quota used)
- 1 epoch training
- batch=4 for CPU compatibility
- ~3 minutes runtime per test

"""

import nbformat as nbf
from pathlib import Path

# Mini-ablation configurations
MINI_ABLATIONS = {
    "1_no_mosaic": {
        "name": "Mini-Ablation 1: No Mosaic",
        "description": "Test if mosaic is critical (expect 70-75% Dice)",
        "hypothesis": "Mosaic provides 5x more spatial diversity than translate",
        "params": {
            "mosaic": 0.0,  # ONLY CHANGE - disable mosaic
            # All other YOLO defaults inherited
        }
    },
    "7_no_randaugment": {
        "name": "Mini-Ablation 7: No RandAugment",
        "description": "Test if RandAugment harmful (expect 78-82% Dice)",
        "hypothesis": "Unpredictable transforms (Solarize, Posterize) hurt clinical color fidelity",
        "params": {
            "auto_augment": None,  # ONLY CHANGE - disable RandAugment
            # Keep hsv_h, hsv_s, hsv_v, erasing (test RandAugment separately)
        }
    },
    "2_no_color": {
        "name": "Mini-Ablation 2: No Color Augmentations",
        "description": "Test if ALL color augmentations harmful (expect 78-82% Dice)",
        "hypothesis": "Color augmentations alter diagnostic features (erythema, tissue viability)",
        "params": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "erasing": 0.0,
            "auto_augment": None,
            # Keep spatial augmentations (mosaic, fliplr, translate, scale)
        }
    },
    "4_minimal": {
        "name": "Mini-Ablation 4: Minimal (Mosaic+Flip Only)",
        "description": "Establish lower bound (expect 75-80% Dice)",
        "hypothesis": "Only essential augmentations needed",
        "params": {
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,  # KEPT
            "mosaic": 1.0,  # KEPT
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "erasing": 0.0,
            "auto_augment": None,
        }
    },
    "5_no_translate": {
        "name": "Mini-Ablation 5: No Translate",
        "description": "Test if translate redundant with mosaic (expect 78-82% Dice)",
        "hypothesis": "Mosaic (256px) >> translate (51px) for spatial diversity",
        "params": {
            "translate": 0.0,  # ONLY CHANGE - disable translate
            # Keep mosaic and all other defaults
        }
    },
    "6_clinical_conservative": {
        "name": "Mini-Ablation 6: Clinical Conservative",
        "description": "Validate clinically-justified configuration (expect 78-82% Dice)",
        "hypothesis": "Reduced, safe augmentations match baseline performance",
        "params": {
            "degrees": 0.0,
            "translate": 0.1,  # KEPT (camera positioning)
            "scale": 0.3,  # REDUCED from 0.5 (±30% instead of ±50%)
            "flipud": 0.0,
            "fliplr": 0.5,  # KEPT
            "mosaic": 1.0,  # KEPT
            "hsv_h": 0.0,  # DISABLED (diagnostic color)
            "hsv_s": 0.0,  # DISABLED (tissue viability)
            "hsv_v": 0.2,  # REDUCED from 0.4 (lighting only)
            "erasing": 0.0,  # DISABLED (may hide features)
            "auto_augment": None,  # DISABLED
        }
    },
    "3_no_scale": {
        "name": "Mini-Ablation 3: No Scale",
        "description": "Test if scale redundant with mosaic (expect 78-82% Dice)",
        "hypothesis": "Mosaic provides sufficient scale diversity via 4-image combination",
        "params": {
            "scale": 0.0,  # ONLY CHANGE - disable scale
            # Keep all other defaults including mosaic
        }
    }
}

# Baseline configuration for reference
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

# Smoke test hyperparameters (CPU-compatible)
SMOKE_TEST_PARAMS = {
    "epochs": 1,
    "batch": 4,  # CPU-friendly
    "imgsz": 512,
    "patience": 10,
    "seed": 42,  # Single seed for smoke test
}


def create_markdown_cell(text: str) -> nbf.NotebookNode:
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(text)


def create_code_cell(code: str) -> nbf.NotebookNode:
    """Create a code cell."""
    return nbf.v4.new_code_cell(code)


def build_augmentation_params_cell(ablation_config: dict) -> str:
    """Build cell that sets augmentation parameters."""
    params = ablation_config["params"]

    # Start with all YOLO defaults
    all_params = BASELINE_PARAMS.copy()

    # Override with ablation-specific changes
    all_params.update(params)

    # Format as Python code
    code = "# Augmentation Configuration\n"
    code += f"# {ablation_config['name']}\n"
    code += f"# Hypothesis: {ablation_config['hypothesis']}\n\n"
    code += "AUGMENTATION_PARAMS = {\n"

    for key, value in sorted(all_params.items()):
        if isinstance(value, str):
            code += f'    "{key}": "{value}",\n'
        elif value is None:
            code += f'    "{key}": None,\n'
        else:
            code += f'    "{key}": {value},\n'

    code += "}\n\n"
    code += "print('Augmentation Configuration:')\n"
    code += "for key, value in sorted(AUGMENTATION_PARAMS.items()):\n"
    code += "    print(f'  {key}: {value}')\n"

    return code


def build_training_cell(ablation_name: str) -> str:
    """Build training cell with augmentation parameters."""
    code = """# Training with Augmentation Configuration
import time

start_time = time.time()

results = model.train(
    data=str(config_path),
    # Smoke test parameters (CPU-compatible)
    batch=SMOKE_TEST_PARAMS['batch'],
    epochs=SMOKE_TEST_PARAMS['epochs'],
    imgsz=SMOKE_TEST_PARAMS['imgsz'],
    patience=SMOKE_TEST_PARAMS['patience'],
    seed=SMOKE_TEST_PARAMS['seed'],
    # Augmentation parameters (mini-ablation specific)
    **AUGMENTATION_PARAMS,
    # Output configuration
    save=True,
    project=str(OUTPUT_PATH / "yolo_training"),
    name="smoke_test",
    exist_ok=True,
    verbose=True,
)

elapsed = time.time() - start_time
print(f"\\nTraining completed in {elapsed/60:.1f} minutes")
"""
    return code


def build_smoke_test_notebook(ablation_id: str, config: dict) -> nbf.NotebookNode:
    """Build complete smoke test notebook for mini-ablation."""

    cells = []

    # Cell 0: Title and description
    cells.append(create_markdown_cell(f"""# {config['name']} - Smoke Test (CPU)

**Ablation ID**: `{ablation_id}`

**Description**: {config['description']}

**Hypothesis**: {config['hypothesis']}

**Smoke Test Configuration**:
- Dataset: TINY (2 test images, 4 train images)
- Training: 1 epoch, batch=4 (CPU-compatible)
- Expected Runtime: ~3 minutes
- Purpose: check the augmentation configuration before GPU deployment

Mini-ablation specifications: `docs/ABLATION_SPECIFICATIONS.md`.
"""))

    # Cell 1: NumPy version locking
    cells.append(create_code_cell("""# NumPy Version Locking
import numpy as np
import subprocess

default_numpy = np.__version__
print(f"Default NumPy: {default_numpy}")

# Install ultralytics (may try to upgrade NumPy)
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)

# Force reinstall to prevent 2.x upgrade
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
    CODE_PATH = BASE_INPUT_PATH / 'wound-seg-code-v1'
    BASELINE_PATH = BASE_INPUT_PATH / 'baseline-outputs'
    TINY_PATH = BASE_INPUT_PATH / 'tiny-wound-seg-smoke-test'
else:
    print("ERROR: Must run on Kaggle with required datasets attached")
    sys.exit(1)

sys.path.append(str(CODE_PATH))

print(f"Environment: {'Kaggle' if IN_KAGGLE else 'Local'}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Code path: {CODE_PATH}")
print(f"Baseline path: {BASELINE_PATH}")
print(f"TINY path: {TINY_PATH}")
"""))

    # Cell 4: Smoke test parameters and ablation ID
    cells.append(create_code_cell(f"""# Smoke Test Parameters
ablation_id = "{ablation_id}"

SMOKE_TEST_PARAMS = {{
    'epochs': {SMOKE_TEST_PARAMS['epochs']},
    'batch': {SMOKE_TEST_PARAMS['batch']},
    'imgsz': {SMOKE_TEST_PARAMS['imgsz']},
    'patience': {SMOKE_TEST_PARAMS['patience']},
    'seed': {SMOKE_TEST_PARAMS['seed']},
}}

print(f'Ablation ID: {{ablation_id}}')
print('\\nSmoke Test Configuration:')
for key, value in SMOKE_TEST_PARAMS.items():
    print(f'  {{key}}: {{value}}')
"""))

    # Cell 5: Augmentation parameters (ablation-specific)
    cells.append(create_code_cell(build_augmentation_params_cell(config)))

    # Cell 6: YOLO config creation
    cells.append(create_code_cell("""# YOLO Configuration
config_content = f'''
path: {TINY_PATH}
train: train_images
val: test_images

names:
  0: background
  1: wound
'''

config_path = OUTPUT_PATH / 'wound_dataset.yaml'
config_path.write_text(config_content.strip())

print(f"YOLO config created: {config_path}")
print("\\nConfig content:")
print(config_content)
"""))

    # Cell 7: Model initialization
    cells.append(create_code_cell("""# Model Initialization
model = YOLO('yolov8n-seg.pt')
print(f"Model: {model.model_name}")
print(f"Task: {model.task}")
"""))

    # Cell 8: Training with augmentation
    cells.append(create_code_cell(build_training_cell(config['name'])))

    # Cell 9: Smoke test validation
    cells.append(create_code_cell("""# Smoke Test Validation
import json

validation = {
    "ablation_id": ablation_id,
    "status": "passed" if results else "failed",
    "augmentation_params": AUGMENTATION_PARAMS,
    "smoke_test_params": SMOKE_TEST_PARAMS,
    "training_epochs": results.epoch if results else 0,
}

output_file = OUTPUT_PATH / f"{ablation_id}_smoke_test_validation.json"
output_file.write_text(json.dumps(validation, indent=2))

print("\\nSmoke Test Validation:")
print(json.dumps(validation, indent=2))
"""))

    # Cell 10: Completion status
    cells.append(create_code_cell("""# Completion Status
completion = {
    "status": "complete",
    "ablation_id": ablation_id,
    "smoke_test": True,
}

completion_file = OUTPUT_PATH / "completion_status.json"
completion_file.write_text(json.dumps(completion, indent=2))

print("\\nSmoke test complete!")
print(f"Validation: {output_file}")
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


def create_kernel_metadata(ablation_id: str, config: dict) -> dict:
    """Create kernel-metadata.json for Kaggle deployment."""

    # Convert ablation_id to slug (1_no_mosaic -> mini-ablation-1-no-mosaic)
    slug = f"mini-ablation-{ablation_id.replace('_', '-')}-smoke-test"

    return {
        "id": f"bkdkaggle/{slug}",
        "title": f"{config['name']} - Smoke Test (CPU)",
        "code_file": f"{ablation_id}_smoke_test.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "bkdkaggle/wound-seg-code-v1",
            "bkdkaggle/baseline-outputs",
            "bkdkaggle/tiny-wound-seg-smoke-test"
        ],
        "kernel_sources": []
    }


def main():
    """Generate all mini-ablation smoke test notebooks."""

    # Priority order (from comparison document)
    priority_order = [
        "1_no_mosaic",
        "7_no_randaugment",
        "2_no_color",
        "4_minimal",
        "5_no_translate",
        "6_clinical_conservative",
        "3_no_scale",
    ]

    base_dir = Path(__file__).parent.parent / "notebooks" / "03_ablations" / "mini_ablations"
    base_dir.mkdir(parents=True, exist_ok=True)

    print("Building mini-ablation smoke test notebooks...")
    print(f"Output directory: {base_dir}")
    print()

    for ablation_id in priority_order:
        if ablation_id not in MINI_ABLATIONS:
            print(f"WARNING: {ablation_id} not found in configurations")
            continue

        config = MINI_ABLATIONS[ablation_id]

        # Create subdirectory
        ablation_dir = base_dir / ablation_id
        ablation_dir.mkdir(exist_ok=True)

        # Build notebook
        print(f"Building {config['name']}...")
        nb = build_smoke_test_notebook(ablation_id, config)

        # Write notebook
        notebook_path = ablation_dir / f"{ablation_id}_smoke_test.ipynb"
        with open(notebook_path, 'w') as f:
            nbf.write(nb, f)

        # Create kernel metadata
        metadata = create_kernel_metadata(ablation_id, config)
        metadata_path = ablation_dir / "kernel-metadata.json"

        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  PASS: Notebook: {notebook_path}")
        print(f"  PASS: Metadata: {metadata_path}")
        print()

    print(f"Generated {len(priority_order)} smoke test notebooks")
    print()
    print("Next steps:")
    print("1. Deploy to Kaggle: cd notebooks/03_ablations/mini_ablations/<ablation_id> && kaggle kernels push -p .")
    print("2. Monitor results: kaggle kernels status bkdkaggle/<kernel-slug>")
    print("3. Download outputs: kaggle kernels output bkdkaggle/<kernel-slug> -p ./results")


if __name__ == "__main__":
    main()
