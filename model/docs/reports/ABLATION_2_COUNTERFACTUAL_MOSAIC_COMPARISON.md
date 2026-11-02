# Ablation 2: Mosaic Counterfactual Comparison

Date: 2025-10-22.
Type: counterfactual (controlled variable).
Question: what is the isolated effect of mosaic on YOLOv8 wound segmentation?
Hypothesis: the failure of original A2 (Dice 68.27%, -11.73%) came from removing mosaic, not from adding rotation and flipud.

This is the state at kernel launch. Final results are in ABLATION_2_COUNTERFACTUAL_COMPLETE_ANALYSIS.md.

Three configurations:
1. Original A2 (failed): mosaic=0.0, rotation, flipud. Dice 68.27%.
2. OLD counterfactual: mosaic=1.0, translate, scale. mAP50 85.99%.
3. NEW counterfactual: mosaic=1.0 only. mAP50 TBD at the time of writing.

## Configurations

### 1. Original A2 (clinical augmentation), failed

```yaml
mosaic: 0.0         # disabled, the critical change
degrees: 15.0       # added
flipud: 0.5         # added
translate: 0.0      # baseline default (disabled)
scale: 0.0          # baseline default (disabled)
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.3
```

Results: Dice 68.27% [65.8-70.6%], -11.73% vs baseline. Rejected. Removing mosaic destroyed performance despite the added clinical augmentations.

### 2. OLD counterfactual (mosaic, translate, scale)

```yaml
mosaic: 1.0         # enabled
degrees: 15.0
flipud: 0.5
translate: 0.1      # added
scale: 0.5          # added
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.3
```

Results, 3 seeds:
- Seed 42: mAP50(M) 86.78%, Precision(M) 88.24%, Recall(M) 81.89%.
- Seed 123: mAP50(M) 86.66%, Precision(M) 89.37%, Recall(M) 79.67%.
- Seed 456: mAP50(M) 84.55%, Precision(M) 87.46%, Recall(M) 78.17%.

Aggregate:
- mAP50 85.99% +- 1.25%.
- Precision 88.36% +- 0.97%.
- Recall 79.91% +- 1.87%.
- Improvement vs baseline (80.0% mAP50): +5.99%.

Adding translate and scale on top of mosaic kept segmentation strong.

### 3. NEW counterfactual (mosaic only)

As planned at launch:

```yaml
mosaic: 1.0         # isolated variable
degrees: 0.0
flipud: 0.0
translate: 0.0
scale: 0.0
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.3
```

Open: the launch plan has degrees=0.0 and flipud=0.0. ABLATION_2_COUNTERFACTUAL_COMPLETE_ANALYSIS.md records the completed run with degrees=15.0 and flipud=0.5. To be checked against args.yaml.

Kaggle:
- Kernel: `bkdkaggle/ablation-2-counterfactual-mosaic-only`.
- Start: 2025-10-22, exact time TBD.
- Expected duration: ~2.5-3 hours (P100, 100 epochs, 3 seeds).
- URL: https://www.kaggle.com/code/bkdkaggle/ablation-2-counterfactual-mosaic-only

Results at the time of writing: pending.

## Comparison table

| Configuration | Mosaic | Translate | Scale | Degrees | FlipUD | mAP50 (%) | Precision (%) | Recall (%) | Delta vs Baseline |
|---------------|--------|-----------|-------|---------|--------|-----------|---------------|------------|-------------------|
| Baseline | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 80.00 | 74.7 | 90.6 | n/a |
| Original A2 | 0.0 | 0.0 | 0.0 | 15.0 | 0.5 | ~54.6* | n/a | n/a | -25.4% |
| OLD Counterfactual | 1.0 | 0.1 | 0.5 | 15.0 | 0.5 | 85.99 | 88.36 | 79.91 | +5.99% |
| NEW Counterfactual | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | TBD | TBD | TBD | TBD |

*Original A2 reported Dice 68.27%. mAP50 estimated with the 0.8 Dice-to-mAP50 ratio.

## Questions the NEW run answers

1. Isolated mosaic effect: NEW counterfactual vs baseline. Expected: significant improvement if mosaic is critical.
2. Translate and scale effect: OLD counterfactual vs NEW counterfactual. Expected: small to moderate extra gain.
3. Why original A2 failed: original A2 (no mosaic) vs baseline (mosaic enabled). Hypothesis: removing mosaic removed augmentation diversity.
4. Adopt mosaic for wound segmentation. Criteria:
   - p < 0.00625 (Bonferroni).
   - Cohen's d > 0.8.
   - Meaningful precision and recall gain.

## Statistical plan

- H0: mosaic has no effect on segmentation performance.
- H1: mosaic significantly improves segmentation performance.
- Test: Welch's t-test (unequal variances).
- Threshold: alpha = 0.00625 (Bonferroni, 8 ablations).
- Effect size: Cohen's d. Adoption threshold d > 0.8.

Planned comparisons:
1. Mosaic effect: NEW counterfactual vs baseline.
2. Translate and scale effect: OLD counterfactual vs NEW counterfactual.
3. Mosaic removal effect: original A2 vs baseline.

Decisions to make from results:
1. Adopt mosaic if NEW counterfactual shows significant improvement.
2. Adopt translate and scale if OLD significantly outperforms NEW.
3. Select the augmentation combination for real-world performance.

## Implementation

Notebook: `ablation_2_counterfactual_mosaic_only.ipynb`
- Location: `/notebooks/03_ablations/ablation_2_counterfactual_mosaic_only/`.
- Kernel ID: `bkdkaggle/ablation-2-counterfactual-mosaic-only`.
- GPU: P100.
- Training: 100 epochs, patience=10, seeds 42, 123, 456.

Data:
- Training: 2,208 images from `bkdkaggle/baseline-outputs`.
- Testing: 552 images from `bkdkaggle/baseline-outputs`.
- Code: `bkdkaggle/wound-seg-code-v1`.

Output after completion:
- Local: `/kaggle_outputs/ablation_2_counterfactual_mosaic_only/`.
- Kaggle: `/kaggle/working/`, dataset export.

## Next

1. Check the Kaggle run every 30-60 minutes.
2. Download results to local kaggle_outputs.
3. Parse results.csv for all 3 seeds.
4. Compute p-values and effect sizes.
5. Complete the comparison table.
6. Decide ADOPT, MARGINAL or REJECT for mosaic.
7. Update docs/ABLATION_SPECIFICATIONS.md.

## References

- Original A2: `/tmp/abl2_final/completion_status.json`.
- OLD counterfactual: `/tmp/abl2_final/yolo_training/seed_*/results.csv`.
- Baseline: docs/ABLATION_SPECIFICATIONS.md (mAP50 = 80.0%).
- Deployment notebook: `/notebooks/03_ablations/ablation_2_counterfactual_mosaic_only/`.
