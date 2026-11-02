# YOLOv8 Validation Metrics with Empty Label Files (Hard Negatives)

Research date: 2025-10-20. Ultralytics main branch (latest at the time). Scope: how empty label files affect validation mAP, precision, recall.

## Answer

Images with empty label files (hard negatives, background images) affect YOLOv8 validation metrics.

1. Empty labels are processed during validation, not skipped.
2. False positives on empty-label images count toward precision.
3. They do not contribute to recall (no ground truth, no missed detections).
4. mAP drops through the precision term in AP.
5. Intentional, built for hard negative mining.

Confidence: high (code analysis plus official responses).

## Code: `ConfusionMatrix.process_batch()`

File `ultralytics/utils/metrics.py`, lines about 600-650.

```python
def process_batch(
    self,
    detections: dict[str, torch.Tensor],
    batch: dict[str, Any],
    conf: float = 0.25,
    iou_thres: float = 0.45,
) -> None:
    """Update confusion matrix for object detection task."""
    gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]

    # Empty label handling
    if gt_cls.shape[0] == 0:  # labels empty
        if not no_pred:  # model made predictions
            detections = {k: detections[k][detections["conf"] > conf] for k in detections}
            detection_classes = detections["cls"].int().tolist()
            for i, dc in enumerate(detection_classes):
                self.matrix[dc, self.nc] += 1  # count as FALSE POSITIVE
                self._append_matches("FP", detections, i)
        return  # exit early, no TP/FN counting

    # normal processing for images with labels...
```

Effect: ground truth empty, any prediction counted as false positive, early return so no TP or FN. Precision decreases, recall unchanged, mAP decreases.

## Ultralytics GitHub issues

Issue #7981, "Is it OK to include images with no labels in the training data set?" (https://github.com/ultralytics/ultralytics/issues/7981). Official reply:

> "For time-lapse datasets with predominantly background images, you can include unlabeled background images in your validation set to measure false positive rates. YOLO models do process these images during validation and will count any predictions as false positives since there are no ground truth labels. This approach helps assess real-world precision."

- Training: "During training, null images help the model learn to reduce false positives by treating these as background (negative samples) and adjusting accordingly through backpropagation."
- Validation: "In the validation set, null images are used to evaluate the model's performance in rejecting false positives, ensuring it generalizes well."
- How: "Create an empty text file (with .txt extension) for each image with the same filename as the image. These empty annotation files signal to the dataset loader that these images should be included during training as valid samples with no objects."

Issue #3228, "Should I include background images in validation/test set?" (https://github.com/ultralytics/ultralytics/issues/3228). Official reply:

> "Including background images (unlabeled images) in the validation or test set may not provide any benefit for the overall performance evaluation... However, during the validation and testing stages, YOLOv8 does not make predictions for unlabelled background images."

This reply is outdated or misleading. Code analysis and the later #7981 reply show YOLOv8 does process empty-label images during validation.

## Metric effects

- Precision = TP / (TP + FP). A prediction on an empty-label image raises FP; precision drops.
- Recall = TP / (TP + FN). No ground truth, no FN; recall unchanged.
- mAP: lower precision lowers AP for affected classes; mAP drops.
- Confusion matrix: predictions on empty labels go to `matrix[predicted_class, nc]`; `nc` is the background column; FP count rises.

## Training versus validation

Training:
1. YOLOv5 could skip images without label files.
2. YOLOv8 needs empty `.txt` files for null images.
3. Purpose: teach the model to reject false positives.
4. Mechanism: backpropagation penalizes predictions on empty-label images.

Validation:
1. Empty-label images are processed.
2. Model predicts normally.
3. Predictions count as false positives.
4. Purpose: measure false positive rate under realistic conditions.

## For the wound project (DDI hard negatives)

Effects:
- Realistic precision measurement; tests false positive rate.
- Better clinical validation; simulates deployment.
- Prevents overfitting to annotated data.
- Precision decreases if the model predicts on background images; mAP falls in proportion; recall unchanged.

Setup:
1. Training set: 5-10% hard negatives with empty `.txt` files.
2. Validation set: similar proportion.
3. Test set: include hard negatives to measure deployment readiness.

Interpretation:
- Lower precision with hard negatives is more realistic, not a worse model.
- Use the same validation set for all ablations.
- State whether hard negatives are included in each result.

Ultralytics guidance: 0-10% of the dataset (higher if needed); empty `.txt` per background image; include in training and validation; purpose is fewer false positives in deployment.

Dataset layout:
```
dataset/
├── images/
│   ├── train/
│   │   ├── wound_001.jpg
│   │   ├── background_001.jpg  # hard negative
│   └── val/
│       ├── wound_050.jpg
│       ├── background_010.jpg  # hard negative
└── labels/
    ├── train/
    │   ├── wound_001.txt       # annotations
    │   ├── background_001.txt  # empty file (0 bytes)
    └── val/
        ├── wound_050.txt       # annotations
        ├── background_010.txt  # empty file (0 bytes)
```

## Worked example

10 validation images with wounds, all detected correctly; 1 hard negative where the model predicts 1 wound.

- Without the hard negative: TP 10, FP 0, FN 0; precision 10/10 = 100%; recall 100%.
- With it: TP 10, FP 1, FN 0; precision 10/11 = 90.9%; recall 100%.

`ultralytics/models/yolo/detect/val.py`, `DetectionValidator._process_batch(detections, batch)` takes `batch["cls"]` and calls `ConfusionMatrix.process_batch()`, which checks `gt_classes.shape[0] == 0`, counts predictions as FP if empty, and updates precision.

## Summary table

| Aspect | Empty labels in validation |
|---|---|
| Processed | Yes |
| Affect precision | Yes (decreases if predictions made) |
| Affect recall | No |
| Affect mAP | Yes (through precision) |
| Count as false positives | Yes (any prediction) |
| Recommended | Yes, for realistic evaluation |
| Typical proportion | 0-10% |

## For Ablation 4 (DDI)

- Expect lower precision and mAP than a baseline evaluated without hard negatives.
- Lower metrics do not mean a worse model.
- Document that hard negatives are included.
- Same validation set across ablations.

Related: a model that predicts nothing on every image gets its validation skipped and reports 0.0% (see ABLATION_4_INSIGHTS_FOR_FINAL_REPORT.md).

## References

- Source: `ultralytics/utils/metrics.py` (ConfusionMatrix), `ultralytics/models/yolo/detect/val.py` (DetectionValidator), `ultralytics/engine/validator.py` (BaseValidator).
- GitHub issues #7981 and #3228.
- Ultralytics docs: YOLO Validation Mode, dataset preparation best practices.
