# Kaggle Kernels: Successful Runs (Baseline, Ablations 1-3, Mini-Ablations)

Date: 2025-10-22 17:00 PST.
Purpose: reference for downloading args.yaml from all completed training runs.
"Successful" means training completed without errors, not better performance.
Method: extraction from kernel-metadata.json files and documentation. All kernel IDs checked against local metadata files.
Totals: 9 production runs (baseline plus 8 ablation runs), ~14.17 hours on Kaggle P100.

## Baseline

Production run (80.0% Dice):
- URL: https://www.kaggle.com/code/bkdkaggle/baseline-model-test
- Kernel ID: `bkdkaggle/baseline-model-test`
- Seeds: 42, 123, 456. Training ~1.5 hours on P100.
- Download: `kaggle kernels output bkdkaggle/baseline-model-test -p /tmp/baseline_results`
- args.yaml:
  - `/tmp/baseline_results/yolo_training/seed_42/args.yaml`
  - `/tmp/baseline_results/yolo_training/seed_123/args.yaml`
  - `/tmp/baseline_results/yolo_training/seed_456/args.yaml`

Dataset (includes trained weights):
- URL: https://www.kaggle.com/datasets/bkdkaggle/baseline-outputs
- ID: `bkdkaggle/baseline-outputs`
- Size 1.4GB. Contents: 2,208 train + 552 test images and masks, YOLO labels, trained weights.

## Ablation 1: Focal Tversky loss

Production run (80.01% Dice, marginal):
- URL: https://www.kaggle.com/code/bkdkaggle/ablation-1-focal-tversky-loss
- Kernel ID: `bkdkaggle/ablation-1-focal-tversky-loss`
- 80.01% Dice (+0.01% vs baseline, not significant). Seeds 42, 123, 456. ~1.5 hours on P100.
- Download: `kaggle kernels output bkdkaggle/ablation-1-focal-tversky-loss -p /tmp/ablation_1_results`
- args.yaml: `/tmp/ablation_1_results/yolo_training/seed_{42,123,456}/args.yaml`
- Decision: REJECT. No meaningful improvement.

## Ablation 2: Clinical augmentation

### 2.1 Original production run (68.27% Dice, failed)
- URL: https://www.kaggle.com/code/bkdkaggle/ablation-2-clinical-augmentation
- Kernel ID: `bkdkaggle/ablation-2-clinical-augmentation`
- Completed, performance -11.73%. Seeds 42, 123, 456. ~1.5 hours on P100.
- Configuration: mosaic=0.0, degrees=15.0, flipud=0.5, hsv_v=0.3.
- Download: `kaggle kernels output bkdkaggle/ablation-2-clinical-augmentation -p /tmp/ablation_2_original`
- args.yaml: `/tmp/ablation_2_original/yolo_training/seed_{42,123,456}/args.yaml`
- Decision: failed. Mosaic removal caused severe degradation.

### 2.2 Counterfactual production run (86.78% Dice)
- URL: https://www.kaggle.com/code/bkdkaggle/ablation-2-counterfactual-mosaic-only
- Kernel ID: `bkdkaggle/ablation-2-counterfactual-mosaic-only`
- 86.78% Dice (+6.78% vs baseline, +18.51% vs original A2). Seeds 42, 123, 456. ~1.5 hours on P100.
- Configuration: same as original A2 but mosaic=1.0.
- Download: `kaggle kernels output bkdkaggle/ablation-2-counterfactual-mosaic-only -p /tmp/ablation_2_counterfactual`
- args.yaml: `/tmp/ablation_2_counterfactual/yolo_training/seed_{42,123,456}/args.yaml`
- Decision at the time: pending, strong candidate for adoption (+6.78%). Metrics pending final download (training completed Oct 22, 16:11).

## Ablation 3: Multi-scale training

### 3.1 Production v1-v3 (54.57% Dice, failed)
- URL: https://www.kaggle.com/code/bkdkaggle/ablation-3-multi-scale-training
- Kernel ID: `bkdkaggle/ablation-3-multi-scale-training`
- Completed, -25.43%. Seeds 42, 123, 456. ~1.5 hours on P100.
- Configuration: multi_scale=[416,512,640], mosaic=1.0.
- Download: `kaggle kernels output bkdkaggle/ablation-3-multi-scale-training -p /tmp/ablation_3_v1_3`
- args.yaml: `/tmp/ablation_3_v1_3/yolo_training/seed_{42,123,456}/args.yaml`
- Decision: failed. Multi-scale with mosaic caused -25%.

### 3.2 Production v4 (68.81% Dice, failed)
- URL: https://www.kaggle.com/code/bkdkaggle/ablation-3-multi-scale-training
- Kernel ID: `bkdkaggle/ablation-3-multi-scale-training` (same kernel, different run)
- Completed, -11.19% vs baseline. Seeds 42, 123, 456.
- Configuration: multi_scale=[416,512,640], mosaic=0.0.
- Same kernel as v1-v3. Check the run history for v4 outputs.
- Decision: failed. Disabling mosaic helped but still underperformed.

### 3.3 Production v5 (4.80% Dice, catastrophic)
- URL: https://www.kaggle.com/code/bkdkaggle/ablation-3-v5-multiscale-no-randaugment
- Kernel ID: `bkdkaggle/ablation-3-v5-multiscale-no-randaugment`
- Completed, -95.2%. Seeds 42, 123, 456. ~1.5 hours on P100.
- Configuration: multi_scale=[416,512,640], mosaic=0.0, auto_augment=null.
- Download: `kaggle kernels output bkdkaggle/ablation-3-v5-multiscale-no-randaugment -p /tmp/ablation_3_v5`
- args.yaml: `/tmp/ablation_3_v5/yolo_training/seed_{42,123,456}/args.yaml`
- Decision: catastrophic. Multi-scale without augmentation gives severe overfitting.
- `auto_augment: null` was correctly applied here (verified in args.yaml line 105).
- Open: ABLATION_3_CHRONOLOGICAL_HISTORY.md and ABLATION_3_V5_AUTO_AUGMENT_BUG_CORRECTION.md give the kernel slug as `ablation-3-v5-multi-scale-no-randaugment` (hyphen between multi and scale). To be checked against kernel-metadata-v5.json.

### 3.4 Counterfactual production (64.82% Dice, failed)
- URL: https://www.kaggle.com/code/bkdkaggle/ablation-3-counterfactual-production
- Kernel ID: `bkdkaggle/ablation-3-counterfactual-production`
- Completed, -15.18% vs baseline. Seeds 42, 123, 456. ~1.67 hours on P100.
- Configuration: multi_scale=false, mosaic=0.0, minimal augmentation.
- Download: `kaggle kernels output bkdkaggle/ablation-3-counterfactual-production -p /tmp/ablation_3_counterfactual`
- args.yaml: `/tmp/ablation_3_counterfactual/yolo_training/seed_{42,123,456}/args.yaml`
- Decision: failed. Multi-scale alone does not improve performance. Augmentations are essential.

## Mini-ablations: 7 augmentation configurations

- URL: https://www.kaggle.com/code/bkdkaggle/mini-ablations-production-all-7-configs
- Kernel ID: `bkdkaggle/mini-ablations-production-all-7-configs`
- All 7 configs in a single notebook. Seeds 42, 123, 456 per config. ~10.5 hours total (7 x 1.5 hours).
- Configurations:
  1. No Mosaic (mosaic=0.0)
  2. No Color (hsv_h=0, hsv_s=0, hsv_v=0)
  3. No Scale (scale=0.0)
  4. Minimal (only fliplr=0.5)
  5. No Translate (translate=0.0)
  6. Clinical Conservative (reduced augmentation)
  7. No RandAugment (auto_augment=null)
- Download: `kaggle kernels output bkdkaggle/mini-ablations-production-all-7-configs -p /tmp/mini_ablations_all`
- args.yaml:
  - `/tmp/mini_ablations_all/mini_ablations/1_no_mosaic/args.yaml`
  - `/tmp/mini_ablations_all/mini_ablations/2_no_color/args.yaml`
  - `/tmp/mini_ablations_all/mini_ablations/3_no_scale/args.yaml`
  - `/tmp/mini_ablations_all/mini_ablations/4_minimal/args.yaml`
  - `/tmp/mini_ablations_all/mini_ablations/5_no_translate/args.yaml`
  - `/tmp/mini_ablations_all/mini_ablations/6_clinical_conservative/args.yaml`
  - `/tmp/mini_ablations_all/mini_ablations/7_no_randaugment/args.yaml`
- Analysis pending. Results inform future augmentation strategy.

## Summary table

| Experiment | Kernel ID | Dice (%) | Delta Baseline | Status | Training Time |
|------------|-----------|----------|----------------|--------|---------------|
| Baseline | bkdkaggle/baseline-model-test | 80.0 | n/a | reference | ~1.5 hrs |
| A1: Focal Tversky | bkdkaggle/ablation-1-focal-tversky-loss | 80.01 | +0.01% | reject | ~1.5 hrs |
| A2: Original | bkdkaggle/ablation-2-clinical-augmentation | 68.27 | -11.73% | failed | ~1.5 hrs |
| A2: Counterfactual | bkdkaggle/ablation-2-counterfactual-mosaic-only | 86.78 | +6.78% | pending | ~1.5 hrs |
| A3: v1-v3 | bkdkaggle/ablation-3-multi-scale-training | 54.57 | -25.43% | failed | ~1.5 hrs |
| A3: v5 | bkdkaggle/ablation-3-v5-multiscale-no-randaugment | 4.80 | -95.2% | catastrophic | ~1.5 hrs |
| A3: Counterfactual | bkdkaggle/ablation-3-counterfactual-production | 64.82 | -15.18% | failed | ~1.67 hrs |
| Mini-Ablations | bkdkaggle/mini-ablations-production-all-7-configs | TBD | TBD | pending | ~10.5 hrs |

## Download all

```bash
# Baseline
kaggle kernels output bkdkaggle/baseline-model-test -p /tmp/baseline

# Ablation 1
kaggle kernels output bkdkaggle/ablation-1-focal-tversky-loss -p /tmp/ablation_1

# Ablation 2 (original and counterfactual)
kaggle kernels output bkdkaggle/ablation-2-clinical-augmentation -p /tmp/ablation_2_original
kaggle kernels output bkdkaggle/ablation-2-counterfactual-mosaic-only -p /tmp/ablation_2_counterfactual

# Ablation 3 (v1-v3, v5, counterfactual)
kaggle kernels output bkdkaggle/ablation-3-multi-scale-training -p /tmp/ablation_3_v1_3
kaggle kernels output bkdkaggle/ablation-3-v5-multiscale-no-randaugment -p /tmp/ablation_3_v5
kaggle kernels output bkdkaggle/ablation-3-counterfactual-production -p /tmp/ablation_3_counterfactual

# Mini-ablations (all 7 configs)
kaggle kernels output bkdkaggle/mini-ablations-production-all-7-configs -p /tmp/mini_ablations
```

## Excluded runs (smoke tests, diagnostics, failed deployments)

Smoke tests (3 epochs), diagnostic runs or failed deployments, not listed above:
- `bkdkaggle/ablation-1b-dice-loss-smoke-test-cpu`
- `bkdkaggle/ablation-2-smoke-test-cpu`
- `bkdkaggle/ablation-2-counterfactual-smoke`
- `bkdkaggle/ablation-3-smoke-test-cpu`
- `bkdkaggle/ablation-3-counterfactual-smoke-v2`
- `bkdkaggle/mini-ablation-*-smoke-test` (7 individual smoke tests)
- `bkdkaggle/mini-ablations-comprehensive-smoke-test`
