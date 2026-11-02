# Ablation 4: Hard Negative Mining with DDI, Methodology

Ablation 4a/4b/4c. Dataset: DDI (Diverse Dermatology Images), 642 validated dermatology images. Written 2025-10-16, before the runs. Results are in ABLATION_4_INSIGHTS_FOR_FINAL_REPORT.md.

## Research question

Can hard negative mining with dermatology conditions reduce the false positive rate without degrading wound detection?

Pre-registered hypothesis:
- Baseline FPR on DDI: 15-25% (visual similarity to wounds).
- Target FPR after intervention: below 15% (conservative), more than 40% reduction (optimistic).
- Wound Dice impact: under 1% degradation for the best strategy.

## DDI dataset

- 642 validated images, 78 conditions.
- Split: 513 train (80%), 129 test (20%), stratified by condition.
- Fitzpatrick skin types 1-6.
- Malignancy: 162 malignant (25.2%), 480 benign (74.8%).

Conditions resembling wounds:
- Melanocytic nevi (119 images): pigmented lesions.
- Seborrheic keratosis (58): raised pigmented lesions.
- Basal cell carcinoma (38): ulcerated lesions.
- Actinic keratosis, eczema, psoriasis: inflammatory, erythema.
- Many show redness, texture change and ulceration like chronic wounds.

Reasons for DDI: visual similarity to wounds, clinical relevance (deployment must separate wounds from other skin pathology), expert-validated diagnostic labels, skin tone diversity.

## Strategies

### 4a: mixed training from scratch

2,208 wound images plus 513 DDI images, trained together. Wound and non-wound features are learned together, with negatives throughout training.
- Expected: FPR reduction 15-25%, wound Dice -1 to -2%.
- Pro: simple, standard hard negative mining.
- Con: competing objectives may degrade wound detection.

### 4b: two-stage fine-tune, best checkpoint

Load `baseline_best.pt` (converged, about epoch 50), fine-tune on the 513 DDI images only. 25 epochs, lr 0.0015 (10x lower) to limit forgetting.
- Expected: FPR reduction 10-20%, wound Dice impact under 0.5%.
- Pro: little forgetting risk, efficient.
- Con: converged model may be rigid.

### 4c: two-stage fine-tune, early checkpoint

Load `epoch40.pt`, same fine-tuning as 4b (25 epochs, lr 0.0015). Early checkpoint is more plastic.
- Expected: FPR reduction 15-25%, wound Dice impact under 1%.
- Pro: adapts better to the new domain.
- Con: unstable start is possible.
- Support: foundational model literature (2024) reports earlier checkpoints adapt better than converged ones.

## Evaluation

Wound test set (552 images): specialist-annotated (Wang et al. 2020), unchanged from baseline. Metrics: Dice, Precision, Recall, Boundary F1.

DDI holdout (129 images): 20% of DDI, stratified by condition, used only for FPR (share of DDI images wrongly predicted as wounds).

Per-strategy acceptance:
- FPR reduction of at least 10%.
- Wound Dice degradation under 2%.
- p < 0.0056 (Bonferroni).

Best strategy: highest FPR reduction with lowest Dice loss. Clinically relevant: FPR under 15% with Dice at least 79%.

## Pre-deployment diagnostic

Run before the ablation to check the hypothesis.
1. Load `baseline_best.pt`.
2. Run inference on all 642 DDI images.
3. Compute baseline FPR (expected 15-25%).
4. Per-condition FPR breakdown.

Decision:
- FPR below 10%: consider skipping Ablation 4.
- FPR 10-20%: run 4b and 4c only.
- FPR above 20%: run 4a, 4b, 4c.

## Training configuration

4a:
```python
dataset = "wound_ddi_mixed.yaml"  # 2,208 wound + 513 DDI
epochs = 100
batch = 32
lr0 = 0.015
seeds = [42, 123, 456]
```

4b:
```python
checkpoint = "baseline_best.pt"  # converged, about epoch 50
dataset = "ddi_only.yaml"        # 513 DDI train split
epochs = 25
batch = 32
lr0 = 0.0015                     # 10x lower
seeds = [42, 123, 456]
```

4c:
```python
checkpoint = "epoch40.pt"
dataset = "ddi_only.yaml"
epochs = 25
batch = 32
lr0 = 0.0015
seeds = [42, 123, 456]
```

Time: 4a 1.5-2 h (3 seeds x 100 epochs); 4b 30-45 min; 4c 30-45 min. Total 3-4 h.

Note: 4b and 4c as specified train on empty labels only and collapsed to 0.0% metrics. See the insights doc.

## Literature

- Schuhmacher and Hann (2023): hard negatives cut FPR 15-25% in medical detection, under 1% primary task loss.
- BMC Medical Imaging (2022): hard negative mining matters for deployment; false positives are underreported in research.
- Chen and Feng (2024): limited fine-tuning at 0.1-0.2x base LR and under 30 epochs prevents catastrophic forgetting.
- Foundational models (2024): early checkpoints adapt better.

## Risks

1. Wound detection degradation. Multi-seed training; safety threshold Dice degradation under 2%; reject strategies that break it even if FPR improves.
2. Catastrophic forgetting (4b, 4c). LR 0.0015, 25 epochs vs 100, monitor wound test performance.
3. Instability (4c). Smoke test the early checkpoint first; multi-seed std flags it; fall back to 4b.

## Success indicators

- Primary: FPR reduction on the 129 DDI holdout images.
- Secondary: wound Dice on the 552-image test set.
- Statistical: p < 0.0056 (Bonferroni, 9 comparisons).
- Qualitative: per-condition FPR, visual inspection of remaining false positives, confidence of false positives (low vs high).

## Clinical impact

Estimated baseline FPR 15-25% on dermatology conditions. In a dermatology clinic about 1 in 5 non-wound conditions would trigger a wound detection: unnecessary referrals, patient anxiety, wasted clinical time, lower trust. Target FPR below 15% (conservative) to below 10% (optimistic).
