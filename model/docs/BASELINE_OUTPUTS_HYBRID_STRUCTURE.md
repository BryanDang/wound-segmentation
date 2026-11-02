# Baseline-Outputs Hybrid Dataset Structure

Created 2025-01-16. Version 1.0.

## Purpose

Kaggle dataset: https://www.kaggle.com/datasets/bkdkaggle/baseline-outputs

It combines:
- Wound segmentation training and test data (2,760 images).
- Pre-converted YOLO labels (saves 5 minutes per experiment).
- Baseline results and trained weights (3 seeds).
- Configuration and metadata (correspondence tables, statistics).

It is the single source for all ablation experiments. No separate wound-segmentation-images dataset and no label conversion step.

## Contents

Total size 1.4GB.

```
baseline-outputs/
├── data_wound_seg/                    # image dataset
│   ├── train_images/                  # 2,208 PNG, 512x512
│   ├── train_masks/                   # 2,208 PNG, binary (0=background, 255=wound)
│   ├── test_images/                   # 552 PNG, 512x512
│   └── test_masks/                    # 552 PNG, binary
│
├── yolo_training/                     # pre-converted YOLO labels
│   ├── train/                         # 2,208 .txt (class x y w h)
│   └── test/                          # 552 .txt, normalized [0-1]
│
├── baseline_results/
│   ├── zero_shot/
│   │   ├── metrics_summary.json       # zero-shot: 4.9% Dice
│   │   ├── confusion_matrix.png
│   │   └── example_predictions.png
│   └── fine_tuned/
│       ├── seed_42/
│       │   ├── metrics_summary.json   # fine-tuned: 80% Dice
│       │   └── predictions/
│       ├── seed_123/
│       └── seed_456/
│
├── trained_models/                    # YOLOv8n weights
│   ├── seed_42/
│   │   ├── best.pt                    # 6.45MB
│   │   └── last.pt                    # 6.45MB
│   ├── seed_123/
│   └── seed_456/
│
├── metadata/
│   ├── correspondence_table.xlsx      # image-mask mapping
│   └── dataset_statistics.json
│
└── wound_dataset.yaml                 # YOLO config
```

### Statistics

- Images and masks: train 2,208 plus 2,208; test 552 plus 552; PNG 512x512; 2,760 matched pairs.
- YOLO labels: 2,208 train and 552 test .txt files, format class x_center y_center width height (normalized). Saves about 5 minutes per experiment.
- Models: seeds 42, 123, 456; best.pt and last.pt each; 6 files x 6.45MB = 38.7MB. Config batch=32, lr0=0.015, 100 epochs.
- Zero-shot: 4.9% Dice, 94.8% Precision, 2.5% Recall.
- Fine-tuned (3 seed average): 80.0% Dice, 85.2% Precision, 76.8% Recall.
- 5 runs per seed, Wilcoxon p < 0.001.

## Usage

Kernel metadata:
```python
{
    "dataset_sources": [
        "bkdkaggle/baseline-outputs",
        "bkdkaggle/wound-seg-code-v1"
    ]
}
```

Paths:
```python
BASE_INPUT = "/kaggle/input/baseline-outputs"

TRAIN_IMAGES = f"{BASE_INPUT}/data_wound_seg/train_images"
TRAIN_MASKS = f"{BASE_INPUT}/data_wound_seg/train_masks"
TEST_IMAGES = f"{BASE_INPUT}/data_wound_seg/test_images"
TEST_MASKS = f"{BASE_INPUT}/data_wound_seg/test_masks"

YOLO_TRAIN = f"{BASE_INPUT}/yolo_training/train"
YOLO_TEST = f"{BASE_INPUT}/yolo_training/test"

YOLO_CONFIG = f"{BASE_INPUT}/wound_dataset.yaml"

BASELINE_ZERO = f"{BASE_INPUT}/baseline_results/zero_shot"
BASELINE_FINE = f"{BASE_INPUT}/baseline_results/fine_tuned"
```

Training with no conversion step:
```python
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
results = model.train(
    data='/kaggle/input/baseline-outputs/wound_dataset.yaml',
    epochs=100,
    batch=32,
    lr0=0.015,
    imgsz=512,
    patience=10,
    workers=8,
    amp=False
)
```

Baseline comparison:
```python
import json

path = '/kaggle/input/baseline-outputs/baseline_results/fine_tuned/seed_42/metrics_summary.json'
with open(path) as f:
    baseline = json.load(f)

print(f"Baseline Dice: {baseline['dice']:.3f}")
print(f"Your Dice: {your_dice:.3f}")
print(f"Improvement: {your_dice - baseline['dice']:.3f}")
```

## Versus wound-segmentation-images

| Feature | baseline-outputs | wound-segmentation-images |
|---|---|---|
| Images/Masks | 2,760 (same) | 2,760 (same) |
| YOLO labels | Pre-converted | Needs conversion (about 5 min) |
| Baseline results | Included | Not available |
| Trained weights | 6 checkpoints | Not available |
| Metadata | Complete | Basic |
| Size | 1.4GB | 1.5GB |
| Startup | Fast | +5 min conversion |

Use `baseline-outputs` for all ablations. Use `wound-segmentation-images` only as the original source reference.

## Validation

After download:
```bash
kaggle datasets download -d bkdkaggle/baseline-outputs
unzip baseline-outputs.zip -d baseline-outputs/

cd baseline-outputs/
ls -lh data_wound_seg/train_images/ | wc -l  # 2,208
ls -lh data_wound_seg/test_images/ | wc -l   # 552
ls -lh yolo_training/train/ | wc -l          # 2,208
ls -lh yolo_training/test/ | wc -l           # 552
```

In a Kaggle notebook:
```python
from pathlib import Path

def validate_baseline_outputs():
    base = Path("/kaggle/input/baseline-outputs")

    assert (base / "data_wound_seg/train_images").exists()
    assert (base / "data_wound_seg/test_images").exists()
    assert (base / "yolo_training/train").exists()
    assert (base / "yolo_training/test").exists()

    train_images = len(list((base / "data_wound_seg/train_images").glob("*.png")))
    test_images = len(list((base / "data_wound_seg/test_images").glob("*.png")))
    train_labels = len(list((base / "yolo_training/train").glob("*.txt")))
    test_labels = len(list((base / "yolo_training/test").glob("*.txt")))

    assert train_images == 2208, f"Expected 2208 train images, got {train_images}"
    assert test_images == 552, f"Expected 552 test images, got {test_images}"
    assert train_labels == 2208, f"Expected 2208 train labels, got {train_labels}"
    assert test_labels == 552, f"Expected 552 test labels, got {test_labels}"

    print("Baseline-outputs validation passed")
    return True

validate_baseline_outputs()
```

## Troubleshooting

YOLO config not found. Error: `FileNotFoundError: Dataset '/kaggle/working/wound_dataset.yaml' images not found`. Reference the config from the input path; do not copy it to /kaggle/working:
```python
data_yaml = "/kaggle/input/baseline-outputs/wound_dataset.yaml"
model.train(data=data_yaml, ...)
```

Label-image mismatch. YOLO reports missing labels. Compare stems and check `metadata/correspondence_table.xlsx`:
```python
import pandas as pd
from pathlib import Path

path = "/kaggle/input/baseline-outputs/metadata/correspondence_table.xlsx"
corr = pd.read_excel(path)

train_images = set(Path("/kaggle/input/baseline-outputs/data_wound_seg/train_images").glob("*.png"))
train_labels = set(Path("/kaggle/input/baseline-outputs/yolo_training/train").glob("*.txt"))

image_stems = {p.stem for p in train_images}
label_stems = {p.stem for p in train_labels}

missing_labels = image_stems - label_stems
missing_images = label_stems - image_stems

assert len(missing_labels) == 0, f"Missing labels: {missing_labels}"
assert len(missing_images) == 0, f"Missing images: {missing_images}"
```

Baseline metrics JSON will not load. List the seeds and load one:
```python
import json
from pathlib import Path

baseline_dir = Path("/kaggle/input/baseline-outputs/baseline_results/fine_tuned")

seeds = [d.name for d in baseline_dir.iterdir() if d.is_dir()]
print(f"Available seeds: {seeds}")

with open(baseline_dir / "seed_42/metrics_summary.json") as f:
    metrics = json.load(f)

print(f"Baseline Dice: {metrics['dice']:.3f}")
```
