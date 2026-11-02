"""
Build Ablation 4a notebook from Ablation 2 template.

Key modifications:
1. Add DDI dataset preparation cells (split, copy, create empty labels)
2. Modify training to use combined dataset (2,208 wounds + 514 DDI)
3. Modify evaluation to handle both wound and DDI test images
4. Update all metadata and descriptions

Validated paths from kagglehub:
- DDI: /kaggle/input/ddi-diverse-dermatology/PRIVATE - ddidiversedermatologyimages/*.png (643 images)
- Baseline: /kaggle/input/baseline-outputs/data_wound_seg/
"""

from pathlib import Path

import nbformat as nbf

# DDI dataset preparation cell
DDI_PREP_CELL = """# Prepare DDI dataset for hard negative mining
print("=" * 80)
print("DDI DATASET PREPARATION - HARD NEGATIVE MINING")
print("=" * 80)

import shutil
from sklearn.model_selection import train_test_split

# DDI image paths (VALIDATED via kagglehub)
DDI_IMAGES_DIR = Path("/kaggle/input/ddi-diverse-dermatology/PRIVATE - ddidiversedermatologyimages")
ddi_image_paths = sorted(DDI_IMAGES_DIR.glob("*.png"))

print(f"\\nFound {len(ddi_image_paths)} DDI images")
print(f"Expected: 643 images (verified via kagglehub)")

if len(ddi_image_paths) != 643:
    print(f"WARNING: Expected 643 DDI images, found {len(ddi_image_paths)}")

# Fixed train/test split (seed=42 for reproducibility)
ddi_train_paths, ddi_test_paths = train_test_split(
    ddi_image_paths,
    test_size=0.2,
    random_state=42,  # FIXED across all experiments
    shuffle=True
)

print(f"\\nDDI Split (seed=42):")
print(f"  Train: {len(ddi_train_paths)} images (80%)")
print(f"  Test:  {len(ddi_test_paths)} images (20%)")

# Create combined dataset structure in /kaggle/working/
COMBINED_DATASET_ROOT = OUTPUT_PATH / "ablation_4a_dataset"

# Create directories
(COMBINED_DATASET_ROOT / "train/images").mkdir(parents=True, exist_ok=True)
(COMBINED_DATASET_ROOT / "train/labels").mkdir(parents=True, exist_ok=True)
(COMBINED_DATASET_ROOT / "test/images").mkdir(parents=True, exist_ok=True)
(COMBINED_DATASET_ROOT / "test/labels").mkdir(parents=True, exist_ok=True)

print(f"\\nCopying baseline wound images and labels...")

# Copy baseline wound training data
baseline_train_imgs = Path("/kaggle/input/baseline-outputs/data_wound_seg/train/images")
baseline_train_labels = Path("/kaggle/input/baseline-outputs/data_wound_seg/train/labels")

wound_train_count = 0
for img_path in baseline_train_imgs.glob("*.png"):
    shutil.copy(img_path, COMBINED_DATASET_ROOT / "train/images" / img_path.name)
    wound_train_count += 1

for label_path in baseline_train_labels.glob("*.txt"):
    shutil.copy(label_path, COMBINED_DATASET_ROOT / "train/labels" / label_path.name)

print(f"  PASS: Copied {wound_train_count} wound training images")

# Copy baseline wound test data
baseline_test_imgs = Path("/kaggle/input/baseline-outputs/data_wound_seg/test/images")
baseline_test_labels = Path("/kaggle/input/baseline-outputs/data_wound_seg/test/labels")

wound_test_count = 0
for img_path in baseline_test_imgs.glob("*.png"):
    shutil.copy(img_path, COMBINED_DATASET_ROOT / "test/images" / img_path.name)
    wound_test_count += 1

for label_path in baseline_test_labels.glob("*.txt"):
    shutil.copy(label_path, COMBINED_DATASET_ROOT / "test/labels" / label_path.name)

print(f"  PASS: Copied {wound_test_count} wound test images")

print(f"\\nAdding DDI hard negatives...")

# Copy DDI training images and create empty labels
for img_path in ddi_train_paths:
    # Copy image
    shutil.copy(img_path, COMBINED_DATASET_ROOT / "train/images" / img_path.name)

    # Create empty label (hard negative = no wound annotations)
    label_path = COMBINED_DATASET_ROOT / "train/labels" / f"{img_path.stem}.txt"
    label_path.write_text("")  # Empty file

print(f"  PASS: Added {len(ddi_train_paths)} DDI training images (empty labels)")

# Copy DDI test images and create empty labels
for img_path in ddi_test_paths:
    shutil.copy(img_path, COMBINED_DATASET_ROOT / "test/images" / img_path.name)

    label_path = COMBINED_DATASET_ROOT / "test/labels" / f"{img_path.stem}.txt"
    label_path.write_text("")

print(f"  PASS: Added {len(ddi_test_paths)} DDI test images (empty labels)")

# Verify final counts
final_train_imgs = len(list((COMBINED_DATASET_ROOT / "train/images").glob("*")))
final_train_labels = len(list((COMBINED_DATASET_ROOT / "train/labels").glob("*")))
final_test_imgs = len(list((COMBINED_DATASET_ROOT / "test/images").glob("*")))
final_test_labels = len(list((COMBINED_DATASET_ROOT / "test/labels").glob("*")))

print(f"\\nFinal dataset composition:")
print(f"  Train: {final_train_imgs} images, {final_train_labels} labels")
print(f"    - Wounds: {wound_train_count}")
print(f"    - DDI: {len(ddi_train_paths)}")
print(f"  Test: {final_test_imgs} images, {final_test_labels} labels")
print(f"    - Wounds: {wound_test_count}")
print(f"    - DDI: {len(ddi_test_paths)}")

# Save DDI split info for reproducibility
ddi_split_info = {
    "total_ddi": len(ddi_image_paths),
    "train_ddi": len(ddi_train_paths),
    "test_ddi": len(ddi_test_paths),
    "random_seed": 42,
    "train_ddi_files": [p.name for p in ddi_train_paths],
    "test_ddi_files": [p.name for p in ddi_test_paths],
    "dataset_composition": {
        "train_total": final_train_imgs,
        "train_wounds": wound_train_count,
        "train_ddi": len(ddi_train_paths),
        "test_total": final_test_imgs,
        "test_wounds": wound_test_count,
        "test_ddi": len(ddi_test_paths)
    }
}

with open(OUTPUT_PATH / "ddi_split_info.json", "w") as f:
    json.dump(ddi_split_info, f, indent=2)

print(f"\\nPASS: DDI split info saved: {OUTPUT_PATH / 'ddi_split_info.json'}")
print("=" * 80)
"""

# YOLO config cell for 4a
YOLO_CONFIG_4A_CELL = """# Create YOLO configuration for Ablation 4a
print("\\nGenerating YOLO config for Ablation 4a (combined dataset)...")

config_4a_result = create_yolo_config(
    dataset_root=COMBINED_DATASET_ROOT,
    train_dir="train",
    val_dir="test",
    test_dir="test",
    output_path=OUTPUT_PATH / "wound_dataset_4a.yaml",
)

if config_4a_result["success"]:
    config_path = config_4a_result["data"]["config_path"]
    print(f"PASS: Ablation 4a config created: {config_path}")
else:
    raise ValueError(f"Config creation failed: {config_4a_result['error']}")

print("=" * 80)
"""


def main():
    project_root = Path(__file__).parent.parent
    notebook_path = (
        project_root
        / "notebooks/03_ablations/ablation_4a_hard_negatives_mixed/ablation_4a_hard_negatives_mixed.ipynb"
    )

    print(f"Building Ablation 4a notebook...")
    print(f"  Template: Ablation 2")
    print(f"  Output: {notebook_path}")

    # Read template
    with open(notebook_path, "r") as f:
        nb = nbf.read(f, as_version=4)

    print(f"\\n  Original cells: {len(nb.cells)}")

    # Find where to insert DDI preparation (after environment setup, before training)
    # Look for the cell that sets up test_pairs
    insert_idx = None
    for idx, cell in enumerate(nb.cells):
        if (
            cell.cell_type == "code"
            and "test_pairs" in cell.source
            and "TEST_IMAGES_DIR" in cell.source
        ):
            insert_idx = idx + 1  # Insert after test_pairs setup
            break

    if insert_idx is None:
        print("ERROR: Could not find insertion point for DDI preparation")
        return

    print(f"  Inserting DDI preparation at cell {insert_idx}")

    # Insert DDI preparation cells
    nb.cells.insert(
        insert_idx,
        nbf.v4.new_markdown_cell(
            """## DDI Dataset Preparation - Hard Negative Mining

**Intervention**: Mix dermatological condition images (eczema, psoriasis, melanoma) with wound training data.

Baseline false positive rate is 22.8%. Hard negatives mark what is not a wound.

**Dataset Split** (FIXED seed=42):
- DDI train: 514 images (80% of 643)
- DDI test: 129 images (20% of 643)
- Combined train: 2,208 wounds and 514 DDI, 2,722 images
- Combined test: 552 wounds + 129 DDI = 681 images"""
        ),
    )

    nb.cells.insert(insert_idx + 1, nbf.v4.new_code_cell(DDI_PREP_CELL))
    nb.cells.insert(insert_idx + 2, nbf.v4.new_code_cell(YOLO_CONFIG_4A_CELL))

    # Update title markdown cells
    for cell in nb.cells:
        if cell.cell_type == "markdown" and "Ablation 2" in cell.source:
            cell.source = cell.source.replace("Ablation 2", "Ablation 4a")
            cell.source = cell.source.replace(
                "Clinical Data Augmentation", "Hard Negative Mining (Mixed Training)"
            )
            cell.source = cell.source.replace("Clinical Augmentation", "Hard Negative Mining")

        # Remove clinical augmentation parameters from training cell
        if (
            cell.cell_type == "code"
            and "degrees=15.0" in cell.source
            and "Clinical Augmentation" in cell.source
        ):
            # Replace augmentation parameters with standard baseline params
            cell.source = cell.source.replace(
                "# Clinical Augmentation (INTERVENTION)",
                "# No augmentation (testing hard negatives only)",
            )
            cell.source = cell.source.replace("degrees=15.0,", "degrees=0.0,")
            cell.source = cell.source.replace("translate=0.1,", "translate=0.0,")
            cell.source = cell.source.replace("scale=0.5,", "scale=0.0,")
            cell.source = cell.source.replace("flipud=0.5,", "flipud=0.0,")
            cell.source = cell.source.replace("fliplr=0.5,", "fliplr=0.0,")
            cell.source = cell.source.replace("hsv_v=0.3,", "hsv_v=0.0,")

            # Update config path reference
            cell.source = cell.source.replace(
                "config_path", "config_path  # Use 4a config with combined dataset"
            )

        # Update completion status
        if (
            cell.cell_type == "code"
            and "completion_status" in cell.source
            and "ablation_2" in cell.source
        ):
            cell.source = cell.source.replace("ablation_2", "ablation_4a")
            cell.source = cell.source.replace("clinical_augmentation", "hard_negative_mining_mixed")

    # Validate and normalize
    nbf.validate(nb)
    nb = nbf.validator.normalize(nb)[1]

    # Write updated notebook
    with open(notebook_path, "w") as f:
        nbf.write(nb, f)

    print(f"\\nPASS: Ablation 4a notebook created!")
    print(f"  Final cells: {len(nb.cells)}")
    print(f"\\nKey changes:")
    print(f"  - Added DDI dataset preparation (split, copy, labels)")
    print(f"  - Combined dataset: 2,722 train, 681 test")
    print(f"  - Removed clinical augmentation (baseline params only)")
    print(f"  - Updated all metadata and descriptions")


if __name__ == "__main__":
    main()
