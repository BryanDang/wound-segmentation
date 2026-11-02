# Mini-ablation smoke tests

Purpose: isolate individual augmentation parameters to find which contribute to the baseline's 80% Dice.

Strategy: test each augmentation change independently with a CPU smoke test before committing to GPU production runs.

## Notebooks (7 total)

### Priority 1: critical spatial augmentation
- [1_no_mosaic/](1_no_mosaic/): remove only mosaic.
  - Hypothesis: mosaic provides 5x more spatial diversity than translate.
  - Expected Dice: 70-75% (significant drop if mosaic is critical).
  - Critical test: confirms whether mosaic is essential.

### Priority 2: color fidelity
- [7_no_randaugment/](7_no_randaugment/): remove only RandAugment.
  - Hypothesis: unpredictable transforms (Solarize, Posterize) hurt clinical color fidelity.
  - Expected Dice: 78-82% (similar or better if RandAugment is harmful).
  - Critical test: isolates RandAugment from the other color augmentations.

### Priority 3: comprehensive color test
- [2_no_color/](2_no_color/): remove all color augmentations.
  - Hypothesis: color augmentations alter diagnostic features.
  - Expected Dice: 78-82% (similar or better if color is harmful).
  - Covers hsv_h, hsv_s, hsv_v, erasing and RandAugment together.

### Priority 4: baseline lower bound
- [4_minimal/](4_minimal/): mosaic and horizontal flip only.
  - Hypothesis: only the essential augmentations are needed.
  - Expected Dice: 75-80% (establishes minimum performance).
  - Configuration: absolute minimum (2 augmentations).

### Priority 5: spatial redundancy tests
- [5_no_translate/](5_no_translate/): remove only translate.
  - Hypothesis: translate (+/-51 px) is redundant with mosaic (256 px).
  - Expected Dice: 78-82% (similar if mosaic is sufficient).
- [3_no_scale/](3_no_scale/): remove only scale.
  - Hypothesis: mosaic provides sufficient scale diversity.
  - Expected Dice: 78-82% (similar if scale is redundant).

### Priority 6: recommended clinical configuration
- [6_clinical_conservative/](6_clinical_conservative/): clinically justified augmentations.
  - Hypothesis: reduced, safe augmentations match the baseline.
  - Expected Dice: 78-82% (validates the recommended approach).
  - Configuration: reduced scale (0.3), reduced hsv_v (0.2), no color, erasing or RandAugment.

## Smoke test specifications

All 7 smoke tests use:
- Dataset: TINY (2 test images, 4 train images)
- Training: 1 epoch, batch=4
- GPU: disabled (CPU only, no GPU quota used)
- Runtime: about 3 minutes per test
- Datasets attached:
  - `bkdkaggle/wound-seg-code-v1` (code modules)
  - `bkdkaggle/baseline-outputs` (pre-converted YOLO labels)
  - `bkdkaggle/tiny-wound-seg-smoke-test` (TINY dataset)

## Deployment workflow

Step 1: deploy all smoke tests (parallel).
```bash
# From this directory
for ablation in 1_no_mosaic 7_no_randaugment 2_no_color 4_minimal 5_no_translate 6_clinical_conservative 3_no_scale; do
    cd $ablation && kaggle kernels push -p . && cd ..
done
```

Step 2: monitor status.
```bash
# Check all 7 kernels
for ablation in 1_no_mosaic 7_no_randaugment 2_no_color 4_minimal 5_no_translate 6_clinical_conservative 3_no_scale; do
    slug="mini-ablation-${ablation//_/-}-smoke-test"
    kaggle kernels status bkdkaggle/$slug
done
```

Step 3: download results.
```bash
# Download all outputs
mkdir -p ../../../mini_ablation_smoke_results
for ablation in 1_no_mosaic 7_no_randaugment 2_no_color 4_minimal 5_no_translate 6_clinical_conservative 3_no_scale; do
    slug="mini-ablation-${ablation//_/-}-smoke-test"
    kaggle kernels output bkdkaggle/$slug -p ../../../mini_ablation_smoke_results/$ablation
done
```

Step 4: validate configurations.
- `completion_status.json` should read "complete".
- `*_smoke_test_validation.json` should hold the correct augmentation params.
- `yolo_training/smoke_test/weights/last.pt` should exist.

Step 5: deploy GPU production runs (if smoke tests pass).
- Use the validated configurations for full training (3 seeds, 100 epochs).
- Deploy sequentially (1 GPU session at a time).

## Directory structure

```
mini_ablations/
├── README.md
├── 1_no_mosaic/
│   ├── 1_no_mosaic_smoke_test.ipynb
│   └── kernel-metadata.json
├── 7_no_randaugment/
│   ├── 7_no_randaugment_smoke_test.ipynb
│   └── kernel-metadata.json
├── 2_no_color/
│   ├── 2_no_color_smoke_test.ipynb
│   └── kernel-metadata.json
├── 4_minimal/
│   ├── 4_minimal_smoke_test.ipynb
│   └── kernel-metadata.json
├── 5_no_translate/
│   ├── 5_no_translate_smoke_test.ipynb
│   └── kernel-metadata.json
├── 6_clinical_conservative/
│   ├── 6_clinical_conservative_smoke_test.ipynb
│   └── kernel-metadata.json
└── 3_no_scale/
    ├── 3_no_scale_smoke_test.ipynb
    └── kernel-metadata.json
```

## Expected outcomes

Best case:
- Mini-ablation 6 (clinical conservative) reaches >= 80% Dice.
- Mini-ablation 2 (no color) reaches >= 80% Dice.
- Conclusion: the baseline can be matched with clinically justified augmentations.

Worst case:
- All ablations drop below 78% Dice.
- Conclusion: the baseline's aggressive augmentations are necessary.

Most likely:
- Mini-ablation 1 (no mosaic) drops to 70-75%, confirming mosaic is critical.
- Mini-ablation 2 (no color) reaches 78-82%, so color augmentations are unnecessary.
- Mini-ablation 6 (clinical conservative) reaches 79-81%, validating the approach.
- Conclusion: mosaic essential, color optional, clinical approach viable.

## Documentation references

- [`/docs/YOLO_AUGMENTATION_ANALYSIS.md`](../../../docs/YOLO_AUGMENTATION_ANALYSIS.md): YOLO defaults analysis
- [`/docs/RANDAUGMENT_EXPLAINED.md`](../../../docs/RANDAUGMENT_EXPLAINED.md): RandAugment explained

## Builder script

All 7 notebooks are generated by:
```bash
python scripts/build_augmentation_mini_ablation_smoke_tests.py
```

The script:
- Uses `nbformat` for correct notebook structure.
- Validates all notebooks with `nbf.validate()`.
- Generates matching `kernel-metadata.json` files.
- Locks the NumPy version (prevents the 2.x upgrade).
- Includes hypothesis and expected outcomes in markdown cells.

## Timeline

Total: about 33 minutes from start to validated configurations.
- Deployment: 5 minutes (push all 7 in parallel)
- Execution: 21 minutes (3 minutes x 7, running simultaneously)
- Download: 2 minutes
- Validation: 5 minutes (manual inspection)

GPU savings: CPU smoke tests prevent wasted GPU hours on broken configurations.
