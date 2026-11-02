"""
DDI dataset preparation for hard negatives mining.

Stratified train/test split plus empty YOLO labels for the training images.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.preprocessing import create_ddi_train_test_split, create_empty_yolo_label


def main():
    """Run split and label generation."""
    print("=" * 80)
    print("DDI Dataset Preparation for Hard Negatives Mining")
    print("=" * 80)

    # Paths
    metadata_path = PROJECT_ROOT / "data" / "ddi_cleaned" / "ddi_metadata_cleaned.csv"
    train_output_path = PROJECT_ROOT / "data" / "ddi_cleaned" / "ddi_train.csv"
    test_output_path = PROJECT_ROOT / "data" / "ddi_cleaned" / "ddi_test.csv"
    yolo_labels_dir = PROJECT_ROOT / "data" / "ddi_yolo_labels" / "train"

    # Step 1: Create train/test split
    print("\nStep 1: Creating stratified train/test split...")
    print(f"Reading metadata: {metadata_path}")

    result = create_ddi_train_test_split(
        metadata_path=metadata_path,
        train_output_path=train_output_path,
        test_output_path=test_output_path,
        test_size=0.2,
        random_state=42,
    )

    print(f"\nSplit completed successfully!")
    print(f"  Train images: {result['train_count']}")
    print(f"  Test images: {result['test_count']}")
    print(f"  Train CSV: {result['train_path']}")
    print(f"  Test CSV: {result['test_path']}")

    print("\nDisease distribution (train):")
    for disease, count in sorted(result["disease_distribution"]["train"].items()):
        print(f"  {disease}: {count}")

    print("\nDisease distribution (test):")
    for disease, count in sorted(result["disease_distribution"]["test"].items()):
        print(f"  {disease}: {count}")

    # Step 2: Generate empty YOLO labels for training images
    print(f"\nStep 2: Generating empty YOLO labels for {result['train_count']} training images...")
    print(f"Output directory: {yolo_labels_dir}")

    # Read train CSV to get image filenames
    import pandas as pd
    train_df = pd.read_csv(train_output_path)

    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    created_count = 0
    for _, row in train_df.iterrows():
        image_filename = row["DDI_file"]
        label_filename = image_filename.replace(".png", ".txt")
        label_path = yolo_labels_dir / label_filename

        create_empty_yolo_label(label_path)
        created_count += 1

    print(f"\nGenerated {created_count} empty YOLO label files")

    # Verify a few labels
    print("\nVerifying labels (should be empty files)...")
    sample_labels = list(yolo_labels_dir.glob("*.txt"))[:3]
    for label_path in sample_labels:
        file_size = label_path.stat().st_size
        print(f"  {label_path.name}: {file_size} bytes (empty)")

    print("\n" + "=" * 80)
    print("DDI Dataset Preparation Complete!")
    print("=" * 80)
    print(f"\nReady for Ablation 4 experiments:")
    print(f"  - Training set: {result['train_count']} images with empty labels")
    print(f"  - Test set: {result['test_count']} images for FPR evaluation")
    print(f"  - Labels directory: {yolo_labels_dir}")


if __name__ == "__main__":
    main()
