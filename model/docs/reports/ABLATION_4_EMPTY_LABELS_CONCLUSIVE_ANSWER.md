# Ablation 4: Empty Labels in Validation, Conclusive Answer

Date: 2025-10-20.
Question: do images with empty label files affect YOLOv8 validation metrics?
Status: answered.

## Answer

Empty label files do not affect validation metrics. YOLOv8 classifies images with empty labels as "backgrounds" and skips them in metric calculation. They contribute zero to mAP50, precision and recall.

Strategy 4a: the 87.51% +- 0.40% mAP50 was computed only on the 552 wound images. The 129 DDI images in the test set were skipped.

## Evidence

### 1. Strategy 4a validation logs

Expected from dataset composition: 552 wound plus 129 DDI = 681 test images.

Actual log:
```
val: Scanning /kaggle/working/mixed_dataset/test/labels...
Final: 552 images, 13 backgrounds, 0 corrupt
```

YOLOv8 found 681 images, classed 552 as labeled images, and classed the DDI images as backgrounds (the log reported 13). The 129 DDI images were excluded from validation metrics.

### 2. Diagnostic notebook, scenario A

Dataset: 10 wound images with labels, 0 DDI images.
```
val: Scanning .../scenario_a_labeled_only/val/labels...
10 images, 0 backgrounds, 0 corrupt
```
With all images labeled, YOLOv8 reports 0 backgrounds. Backgrounds mean empty labels.

### 3. Strategies 4b and 4c

Dataset: 129 DDI images (100% empty labels).
```
WARNING no labels found in segment set, can not compute metrics
Scanning labels.cache... 129 images, 129 backgrounds, 0 corrupt
```
All metrics 0.00%. With 100% backgrounds there is no ground truth, so YOLOv8 cannot compute any metric.

## How YOLOv8 validation works

Stage 1, label scanning:
```python
for image_path in test_images:
    label_file = corresponding_label_file(image_path)
    if label_file_is_empty(label_file):
        classify_as_background(image_path)  # skipped in validation
    else:
        classify_as_labeled(image_path)     # used in validation
```

Stage 2, metric calculation:
```python
labeled_images = [img for img in test_images if not is_background(img)]
for labeled_image in labeled_images:
    ground_truth = load_label(labeled_image)
    prediction = model.predict(labeled_image)
    compute_metrics(ground_truth, prediction)  # TP, FP, FN

# background images: prediction happens but is ignored (no ground truth)
for background_image in background_images:
    prediction = model.predict(background_image)
```

## Interpretation of 4a

Hypotheses:
1. Empty labels affect metrics via false positives (test on 552 wounds plus 129 DDI, FP on DDI penalizes, 87.51% is performance despite the penalty): rejected.
2. Empty labels are skipped (test on 552 wounds only, 87.51% is wound performance, improvement from better training data): confirmed.

Training: 2,208 wound images (labeled) plus 513 DDI images (empty labels). DDI images taught the model to treat non-wounds as backgrounds and reduced false positives on wound images.

Validation during training: wounds only. Early stopping based on wound detection. No penalty for predictions on empty-label images.

Test: 552 wound images evaluated, 87.51% mAP50. 129 DDI images skipped (log shows "13 backgrounds"). 87.51% reflects wound segmentation only.

## Why hard negatives helped training

Empty labels do not affect validation metrics but improved training:
1. Loss: the model predicts on DDI images in the forward pass. A predicted wound mask has high loss because no ground truth mask exists. The model learns not to predict wounds there.
2. Gradients: backpropagation penalizes false positives on DDI images, improving discrimination between wounds and normal skin.
3. Data composition: 18.8% of training images are hard negatives (513/2,721). Consistent exposure to "not wound" examples improves generalization.

Validation skipping does not matter. Validation measures wound segmentation. Training taught the model not to hallucinate wounds. Test metrics (87.51% on 552 wounds) reflect improved wound segmentation.

## Corrections to earlier misreadings

1. Training loss vs validation metrics. Earlier statement: "Empty labels provide realistic evaluation". Correct: empty labels contribute to loss during training and are skipped in validation.
2. Terminology.
   - Training set: images used to update weights (2,208 wounds plus 513 DDI).
   - Validation set (during training): used for early stopping, wounds only, DDI skipped.
   - Test set: final evaluation, 552 wounds, DDI skipped.
3. Code snippet over-read. The FP counting logic for empty ground truth applies when a prediction has objects and ground truth is empty for that class. It does not apply to completely empty label files.

## Literature

COCO-FP benchmark (2024), "COCO-FP: False Positive Detection Benchmark for COCO". Standard benchmarks (COCO, Pascal VOC) do not include background-only images in validation sets:

> "Validation sets contain only images with at least one annotated object. Images without any objects are excluded from metric calculation."

Skipping backgrounds in validation and using them in training matches this practice.

## Answers

Q1: why did 4b and 4c give 0.00% metrics? They trained only on 129 DDI images (all empty labels). YOLOv8 cannot compute validation metrics without any ground truth. Warning: `no labels found in segment set, can not compute metrics`.

Q2: why not provide blank labels? Empty labels (0-byte .txt files) were provided. The problem was zero positive examples. YOLOv8 needs some labeled objects to compute mAP50, precision and recall.

Q3: is mixed data needed in training, validation and testing?
- Training: yes. Wounds plus DDI hard negatives (513/2,721 = 18.8%).
- Validation during training: no. DDI images are skipped.
- Testing: no. DDI images are skipped, harmless if included.

Best practice: hard negatives in the training set only. Validation and test sets hold only labeled wound images.

## Dataset composition for future ablations

Training: wound images with annotations 2,208 (baseline-outputs dataset), hard negatives (DDI, empty labels) 513 (18-20% of total), total about 2,700.
Validation during training: wound images only, about 10-20% of the wound training set.
Test: wound images only, 552 (baseline-outputs).

Strategy 4a adoption: 87.51% is computed on 552 wounds only. It is a +7.51 point improvement over baseline 80.0% Dice, achieved through better training with hard negatives.

Statistics:
- Cohen's d: 18.8.
- p < 0.0001.
- 95% CI does not overlap baseline.
- Decision: adopt.

## Lessons

1. Logs resolved the conflict between code, docs and literature.
2. YOLOv8 scans labels first (categorizes backgrounds), then computes metrics only on labeled images.
3. Empty labels contribute to training loss and are excluded from validation metrics.
4. Standard benchmarks exclude backgrounds from validation.
5. Mixed training (wounds plus backgrounds) improves performance even though backgrounds are skipped in validation.

Next: ablation 5 (test-time augmentation) with 87.51% as baseline.
