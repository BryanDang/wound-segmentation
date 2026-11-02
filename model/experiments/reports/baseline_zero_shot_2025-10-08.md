# Baseline model performance report: zero-shot YOLOv8n-seg evaluation

Experiment type: baseline zero-shot evaluation.
Date: 2025-10-08.
Model: YOLOv8n-seg (COCO pretrained weights).
Dataset: wound dataset (552 test images, specialist-annotated).
Evaluation type: zero-shot (no fine-tuning).
Status: historical record.

## Summary

The zero-shot COCO-pretrained YOLOv8n-seg model reached Dice 4.9% (95% CI: 4.2-5.6%). The model is unusable for wound segmentation without domain-specific fine-tuning.

Key finding: recall is high (76.7%) and precision is very low (2.6%). Rest of the finding (what the model detects, over-segmentation, deployment need): text damaged in source.

## Performance metrics

| Metric | Value | 95% CI low | 95% CI high |
|--------|-------|------------|-------------|
| Dice score | 4.9% | 4.2% | 5.6% |
| IoU (Jaccard) | 2.7% | 2.3% | 3.3% |
| Precision | 2.6% | 2.2% | 2.9% |
| Recall | 76.7% | 73.2% | 80.1% |
| Boundary F1 | 0.7% | 0.4% | 1.2% |

Confidence intervals: 10,000 bootstrap iterations.

Interpretation:
- Dice 4.9%: overlap between prediction and ground truth is minimal (rest damaged in source).
- IoU 2.7%: text damaged in source.
- Recall 76.7%: most wound pixels are found (rest damaged in source).
- Precision 2.6%: only 2.6% of predicted wound pixels are wound (97.4% false positives).
- Boundary F1 0.7%: near-zero boundary delineation (rest damaged in source).

## Confusion matrix

Pixel-level classification, aggregated over 552 images.

| | Predicted non-wound | Predicted wound |
|--|---------------------|-----------------|
| Actual non-wound | 118,834,419 (TN) | 3,765,479 (FP) |
| Actual wound | 355,912 (FN) | 1,169,870 (TP) |

Normalized:

| | Predicted non-wound | Predicted wound |
|--|---------------------|-----------------|
| Actual non-wound | 96.9% | 3.1% |
| Actual wound | 23.3% | 76.7% |

Analysis:
- TN (96.9%): text damaged in source (model identifies non-wound tissue).
- FP (3.1% of non-wounds): a small rate, but 3.76M mislabeled pixels, which produces low-precision masks.
- TP (76.7%): the model finds wound pixels.
- FN (23.3%): the model misses 23.3% of wound area.

Implication: the gap between precision (2.6%) and recall (76.7%) shows the model has no wound-boundary or texture understanding. Rest of the analysis (including the "damaged tissue" remark): text damaged in source.

## Example predictions: best, middle, worst

Visualization: 3x3 grid of predictions, saved as `example_predictions.png`.

Best 3 (Dice 15-20%):
- Moderate wound detection.
- Boundary cases.
- Remaining text damaged in source; the masks still lack precision.

Middle 3 (Dice 4-6%):
- Over-segmentation.
- Leakage around the wound periphery (partly damaged in source).

Worst 3 (Dice <1%):
- Cases with limb or body regions.
- Skin tone, shadows and other features (text partly damaged in source).
- Wound tissue not detected (text partly damaged in source).

Pattern observed: performance degrades with complex backgrounds, skin diversity and wound ambiguity. Domain-specific training and fine-tuning are needed.

## Literature comparison

Wang et al. (2020), wound dataset:
- Specialized CNN model: Dice 92.8% (fine-tuned).
- Zero-shot pretrained model: not reported.

Oota et al. (2023), WSNet architecture:
- Specialized model: Dice 89.3% (fine-tuned).
- Zero-shot baseline: not reported.

This baseline (zero-shot YOLOv8n-seg):
- Dice 4.9%.
- Literature fine-tuned range needed: 85-88% (line partly damaged in source).

Implication: the gap between a general pretrained model and domain-specialized architectures is as expected. Ablation 1 fine-tuning follows (rest damaged in source).

## Next: Ablation 1 fine-tuning

Hypothesis: fine-tuning YOLOv8n-seg with Focal Tversky Loss (alpha=0.7, beta=0.3, gamma=2.0) and augmentation (rotation +/-15 degrees, horizontal flip, rest damaged in source) reaches Dice > 85%, matching specialized models.

Planned interventions:
1. Loss function: Focal Tversky Loss (rationale damaged in source).
2. Data augmentation: dataset size 8x (rest damaged in source).
3. Multi-scale training: image sizes [416, 512, 640] (rest damaged in source).
4. Optimizer: AdamW with scheduling.

Rigor:
- 5 independent runs (random seeds).
- Wilcoxon signed-rank test (p < 0.001 required).
- Bonferroni correction (alpha = 0.0125).
- Bootstrap confidence intervals on metrics.

Success criteria: mean Dice > 85% with the 95% CI (rest damaged in source), and p < 0.001 against the zero-shot baseline.

## Technical environment

- Platform: Kaggle notebook (GPU P100)
- NumPy: 1.26.4 (force-reinstalled because of 2.x incompatibility)
- Ultralytics: 8.3.x (latest)
- OpenCV fix: Kaggle's OpenCV 4.12.0 is compiled against NumPy 1.x, so NumPy is force-reinstalled.

## Reproducibility

- Random seed: not applicable (zero-shot inference)
- Model weights: YOLOv8n-seg, COCO pretrained (default)
- Input resolution: 640x640 (letterbox)
- Confidence threshold: 0.25 (default)
- Bootstrap iterations: 10,000 (95% CI)

## Data notes

- Test set: 552 images, 100% specialist-annotated.
- Known issue: `fusc_0413` image and mask problem, excluded from analysis (detail damaged in source).
- Validation: all 552 images processed (rest damaged in source).

## Generated files

1. `baseline_summary_statistics.json`: machine-readable summary with 95% CI
2. `baseline_per_image_results.json`: per-image Dice, IoU, Precision, Recall, Boundary F1
3. `confusion_matrix.png`: pixel-level visualization
4. `example_predictions.png`: best, middle and worst prediction examples
5. `baseline_performance_report.md`: text report

## Conclusion

The zero-shot COCO-pretrained YOLOv8n-seg fails at wound segmentation: Dice 4.9%, precision 2.6%, recall 76.7%. Domain fine-tuning is needed. The model lacks wound-boundary discrimination (Boundary F1 near zero).

The baseline is established. Next: Ablation 1 fine-tuning with Focal Tversky Loss and augmentation (target Dice > 85%).
