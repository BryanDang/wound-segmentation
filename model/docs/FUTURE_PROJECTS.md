# Future Projects: Wound Segmentation Model

All assume the baseline wound segmentation model is complete. Timelines assume full-time work. Citation dates are the most recent relevant publications. Publishability ratings reflect novelty and clinical relevance for 2025 conferences.

## 1. Post-hoc calibration for YOLOv8 segmentation

Objective: improve confidence calibration of YOLO predictions for clinical decision support.

Approach:
- Modify the YOLOv8 source to expose raw logits (pre-softmax). YOLO exposes only post-softmax probabilities.
- Temperature scaling on the logits.
- Validate with Expected Calibration Error (ECE).

Reason: overconfident predictions are unsafe clinically; clinicians need accurate uncertainty.

Expected: ECE from about 0.035 to about 0.012 (65% improvement), same Dice.

Citations:
- Kumar et al., "Post hoc calibration of medical segmentation models" (2025).
- Mask-TS Net (2024): temperature scaling for polyp segmentation.

Timeline 2-3 weeks:
- Week 1: modify the YOLO forward pass to return logits.
- Week 2: temperature scaling and validation.
- Week 3: experiments and statistical testing.

## 2. Knowledge distillation: U-Net to YOLOv8n-seg

Objective: U-Net accuracy at YOLO speed for mobile deployment.

Approach:
- Train a U-Net teacher (expected Dice 86-88%).
- Distill into a YOLOv8n-seg student.
- Loss: 0.3 x GT + 0.4 x Soft + 0.3 x Feature.
- Temperature T=4.0.

Reason: mobile wound assessment needs speed. U-Net is accurate (150 ms) but slow; YOLO is fast (30 ms) but less accurate. U-Net to YOLO distillation is unexplored for medical segmentation.

Expected:
- Distilled YOLO Dice about 84% (baseline 80%, teacher 87%).
- Inference 30 ms (teacher 150 ms).
- Model size 6MB (teacher 44MB).

Citations:
- MSKD (2024): +18.56% Dice via structured knowledge distillation.
- Multi-teacher distillation (Nature 2025): +32.6%.
- SAM distillation (Jan 2025): keeps accuracy, reduces complexity.

Timeline 3-4 weeks:
- Week 1: U-Net teacher (ResNet34 encoder).
- Week 2: distillation framework.
- Weeks 3-4: multi-seed experiments and statistical validation.

Publishability 9/10: novel architecture pairing, clinical relevance, strong 2024 citations.

## 3. Boundary refinement module (end-to-end trainable)

Objective: differentiable boundary refinement inside the YOLO architecture.

Approach:
- Lightweight boundary refinement head on YOLO.
- Train end-to-end with a boundary-focused loss.
- Boundary-aware attention.

End-to-end beats post-processing, keeps inference real-time.

Citations:
- Chen et al., MICCAI 2023: boundary refinement module.
- DeepLabv3+ (2018): atrous convolution in place of CRF.

Timeline 2 weeks.

## 4. Multi-model ensemble (architecture diversity)

Objective: robust predictions from different architectures.

Approach:
- Train YOLOv8n-seg, U-Net, DeepLabv3+.
- Weighted average of predictions.
- Ensemble weights tuned on the validation set.

Expected: Dice +2-3% over the best single model.

Timeline 4 weeks (3 models to train).

## 5. Self-distillation for boundary refinement

Objective: use the model's own predictions to refine boundaries.

Approach:
- Multi-stage training on its own outputs.
- Online distillation; student and teacher share weights initially.
- Progressive refinement.

Citations:
- Zhang et al., ICCV 2024: self-distillation meets self-supervision.
- Born-again networks (2018).

Timeline 3 weeks.

## 6. Active learning for hard negative mining

Objective: find confusable non-wound images automatically.

Approach:
- Train an initial model on the wound dataset.
- Run inference on a large dermatology dataset.
- Select high-confidence false positives as hard negatives.
- Retrain with them.

Advantage over the fixed DDI set: data-driven selection, hardness quantified by model confidence, iterative.

Timeline 2-3 weeks.

## Priority (impact x feasibility)

1. Knowledge distillation: high impact, medium effort, publishable.
2. Post-hoc calibration: medium impact, low effort, clinical value.
3. Boundary refinement module: medium impact, medium effort, modern.
4. Active learning hard negatives: medium impact, medium effort, principled.
5. Multi-model ensemble: low impact, high effort, standard technique.
6. Self-distillation: medium impact, high effort, exploratory.
