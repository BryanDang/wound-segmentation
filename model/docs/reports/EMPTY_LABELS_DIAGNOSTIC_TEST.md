# YOLOv8 Empty Labels Validation Diagnostic Test

Deployed: 2025-10-20.
Kaggle notebook: https://www.kaggle.com/code/bkdkaggle/yolov8-empty-labels-validation-diagnostic
Purpose: determine whether empty label files affect YOLOv8 validation metrics.
Outcome: see ABLATION_4_EMPTY_LABELS_CONCLUSIVE_ANSWER.md. Empty labels are skipped.

## Background

Conflicting evidence from the ablation 4 analysis.

For "affects metrics":
- Code in `ultralytics/utils/metrics.py` suggests empty labels count as false positive sources.
- Official Ultralytics statement: "YOLO models do process these images during validation".

Against:
- Logs show images classified as "backgrounds" were skipped.
- Standard benchmarks do not include background-only images in validation.
- Strategies 4b/4c with 100% empty labels produced zero metrics.

## Test design: three scenarios

- Scenario A, labeled only (baseline): validation set of 10 wound images with annotations. Expected: normal mAP50.
- Scenario B, mixed (labeled + empty): 10 wound images + 5 DDI images (empty labels). The critical test. Lower mAP50 than A if empty labels affect metrics, same as A if not.
- Scenario C, empty only: 10 DDI images (all empty labels). Expected: zero metrics or validation failure. Confirms behavior with 100% backgrounds, as in 4b/4c.

## Datasets

Wound images (labeled):
- Source: `/kaggle/input/baseline-outputs/data_wound_seg/test_images/`.
- Labels: dummy segmentation annotations (for testing).
- Role: positive validation examples.

DDI images (empty labels):
- Source: `/kaggle/input/baseline-outputs/data_ddi_cleaned/`.
- Labels: 0-byte empty .txt files.
- Role: hard negatives.

Training set (minimal): 5 wound images, just enough to load the model.

## Predicted outcomes

If empty labels affect metrics:

| Scenario | mAP50 | Reason |
|----------|-------|--------|
| A (10 labeled) | ~0.50 | baseline |
| B (10 labeled + 5 empty) | ~0.33 | FP on 5 empty images lowers precision |
| C (10 empty) | 0.00 | no ground truth |

Interpretation: 4a's 87.51% mAP50 includes a penalty from 129 DDI images.

If empty labels do not affect metrics:

| Scenario | mAP50 | Reason |
|----------|-------|--------|
| A (10 labeled) | ~0.50 | baseline |
| B (10 labeled + 5 empty) | ~0.50 | empty images skipped |
| C (10 empty) | 0.00 | no ground truth, validation aborts |

Interpretation: 4a's 87.51% mAP50 computed on 552 wound images only, DDI skipped.

## Reading the results

Key comparison: scenario B vs scenario A.
- mAP50(B) < mAP50(A) - 0.05: empty labels are counted. FP on background images affect precision. 4a's improvement is larger than it looks, since it overcame the penalty.
- |mAP50(B) - mAP50(A)| < 0.01: empty labels are skipped. Validation uses only labeled images. 4a's improvement is purely from better training data.
- mAP50(B) > mAP50(A): unexpected, needs investigation.

## Implications for ablation 4

If empty labels affect metrics:
- 4a: trained on 2,208 wounds + 513 DDI, validated on 552 wounds + 129 DDI with FP penalty. 87.51% is despite the penalty.
- 4b/4c: validated on 129 DDI only, metrics from false positives alone. 0% because no ground truth to match.

If they do not:
- 4a: validated on 552 wounds, DDI skipped. 87.51% reflects wound performance only.
- 4b/4c: validated on 129 DDI only, no ground truth, YOLOv8 aborted validation. 0% because metrics cannot be computed.

## Steps after the run (~5-10 minutes on Kaggle)

1. Open the notebook URL and read the "CONCLUSIVE ANALYSIS" section. Compare mAP50 for scenarios A, B, C.
2. Download results:
```bash
kaggle kernels output bkdkaggle/yolov8-empty-labels-validation-diagnostic
```
3. Update ABLATION_4_ROOT_CAUSE_ANALYSIS.md and docs/ABLATION_SPECIFICATIONS.md with the answer.
4. Confirm 4a adoption with the correct interpretation, close the open question, proceed to ablation 5 (TTA).

## Why it matters

1. Establishes how YOLOv8 behaves, not speculation from docs or code.
2. Fixes how the 87.51% is interpreted.
3. Sets practice for validation set composition in hard negative mining.
4. Gives empirical evidence to cite in a thesis or publication.

Runtime 5-10 minutes (CPU only, small datasets). Cost $0 (free Kaggle CPU).

## Files

- Notebook: empty_labels_diagnostic.ipynb, deployed as the Kaggle kernel above.
- This document: `EMPTY_LABELS_DIAGNOSTIC_TEST.md`.
