# Simulated Kaggle environment

## Structure

```
simulated_kaggle_data/
├── input/wound-segmentation-images/    # dataset (3 png images total)
│   ├── train_images/ (2 files)
│   ├── train_masks/ (2 files)
│   ├── test_images/ (1 file)
│   └── test_masks/ (1 file)
└── working/                            # all outputs (clear for fresh runs)
```

## Purpose

- Local testing: run notebooks before Kaggle deployment.
- Real processing: genuine computer vision and analysis.
- Path simulation: functions behave as if in the Kaggle environment.
- Output verification: shows what Kaggle would generate.

## Usage

Automatic detection: environment-detection functions are meant to pick up the simulated environment and return the "Simulated Kaggle" paths. The module that held them (`wound_seg_functions.py`) is not in the repository.

Clear outputs for a fresh run (keeps the dataset):
```bash
rm -rf simulated_kaggle_data/working/*
```

Expected outputs after a notebook run:
- `working/1a_environment_and_exploration/per_image_analysis.json`
- `working/yolo_labels/train/` (2 label files)
- `working/yolo_labels/test/` (1 label file)
- `working/wound_dataset.yaml` (YOLO configuration)
- `working/yolo_conversion_report.json`

## Known issues

- Functions detect and report an environment mismatch. This is expected behavior.
- Remaining text damaged in source.

## Environment info

`simulated_kaggle_data/environment_info.json` holds setup metadata:
- Creation timestamp
- Directory structure documentation
- Path mappings (relative)
- Dataset copy status

## Development workflow

1. Edit the notebook and test locally with the simulated environment.
2. Clear outputs so test runs are fresh.
3. Check the generated files in `working/`.
4. Deploy the tested notebook to Kaggle.

## Benefits

- No Kaggle upload and rerun cycle.
- Access to all intermediate files for debugging.
- Works offline.
- Behavior verified before deployment.
