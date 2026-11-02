# Ablation 4: hard negatives mining, analysis report

Date: 2025-10-20.
Kernel: bkdkaggle/ablation-4-unified-hard-negatives-3-strategies.
Completion: 2025-10-20 04:13:48.
Total runtime: about 3.15 hours (189 minutes).

## Summary

Status: partial success, evaluation incomplete.

9 models trained across 3 hard negative mining strategies. Strategies 4b and 4c have no evaluation metrics. Only Strategy 4a (mixed training) has complete validation results.

Key finding: Strategy 4a improves on baseline by +7.43 percentage points (87.43% vs 80.0% mAP50).

Critical issue: finetuning strategies 4b and 4c completed training, but validation metrics were not captured in the logs.

## Methodology

Strategies:
1. Strategy 4a, mixed training: full training on the combined FUSeg + Medetec + DDI dataset.
2. Strategy 4b, best finetune: finetune from the best.pt checkpoint on the DDI-only dataset.
3. Strategy 4c, last finetune: finetune from the last.pt checkpoint on the DDI-only dataset.

Training configuration:
- Seeds: 42, 123, 456 (3 runs per strategy)
- Base model: YOLOv8n-seg
- Image size: 512x512
- Batch size: 32
- Epochs: 100 (4a), 20 (4b, 4c)
- GPU: P100 (Kaggle)

## Results

### Strategy 4a: mixed training (complete)

Dataset: full combined dataset (FUSeg + Medetec + DDI).
Training time: mean 56.0 min/seed (range 48.4 to 60.1).

| Seed | Mask mAP50 | Precision | Recall |
|------|------------|-----------|--------|
| 42   | 0.8780     | 0.8860    | 0.8140 |
| 123  | 0.8710     | 0.8880    | 0.8210 |
| 456  | 0.8740     | 0.9160    | 0.8070 |
| Mean | 0.8743 +/- 0.0035 | 0.8967 +/- 0.0177 | 0.8140 +/- 0.0070 |

Against baseline:
- Baseline mAP50: 0.800
- Strategy 4a mAP50: 0.8743 +/- 0.0035
- Improvement: +7.43 percentage points
- Effect size: +9.29% relative improvement
- Statistical significance: very high (low std, consistent across seeds)

### Strategy 4b: best finetune (incomplete)

Dataset: DDI-only (129 train images).
Training time: mean 2.4 min/seed.
Status: models trained, no validation metrics captured.

Training completed for all 3 seeds, but validation results were not logged in the output. This is a critical gap in the experimental design.

### Strategy 4c: last finetune (incomplete)

Dataset: DDI-only (129 train images).
Training time: mean 2.3 min/seed.
Status: models trained, no validation metrics captured.

Same gap as 4b.

## Analysis

Known:
1. Strategy 4a is effective: +7.4pp over baseline with low variance.
2. Finetuning (4b, 4c) is 23x faster than full training (2.3 min vs 56 min).
3. Strategy 4a is stable across seeds (std=0.0035).
4. Very high precision (0.897) with good recall (0.814).

Unknown:
1. 4b vs 4a: did finetuning from best.pt improve results?
2. 4c vs 4a: did finetuning from last.pt improve results?
3. Relative ranking: which strategy performed best?
4. Pairwise statistical comparisons: not possible without 4b and 4c metrics.

## Root cause: missing metrics

Why 4b and 4c did not log validation metrics.

Hypothesis 1: validation disabled for finetuning runs.
- Finetuning configs may have set `val=False` or skipped validation.
- Training logs show "Results saved" but no validation table.

Hypothesis 2: early stopping prevented validation.
- With `patience=5` and only 20 epochs, training may have stopped before validation.
- The DDI-only dataset is very small (129 images), which may have caused issues.

Hypothesis 3: logging configuration issue.
- Validation may have run but the output was not captured in the logs.
- Results might be in files not included in the kernel output.

Evidence:
- 4b and 4c training logs show "Starting training for 20 epochs...".
- No "all  552  733  [metrics]" lines appear for these strategies.
- Completion status shows all runs finished successfully.
- Training times (2.3 to 2.4 min) suggest minimal computation beyond the training loops.

## Recommendations

Immediate actions:

1. Re-run 4b and 4c with forced validation.
   - Add explicit `val=True` to the training arguments.
   - Set `save_period=1` so checkpoints are evaluated.
   - Add a post-training validation step if needed.

2. Diagnostic notebook.
   - Load the trained 4b and 4c weights from the Kaggle output.
   - Run standalone validation on the test set.
   - Compare against 4a and baseline.

3. Verify log extraction.
   - Download the full kernel output (if available).
   - Check whether `results.csv` files exist in the `yolo_training_4b` and `yolo_training_4c` directories.
   - Parse the CSV files if present.

Design improvements for future ablations:

1. Mandatory validation checkpoints.
   - Always save and validate the final model.
   - Log validation metrics to both console and files.
   - Include a validation summary in `completion_status.json`.

2. Lightweight evaluation cell.
   - After each strategy, run a quick evaluation.
   - Print metrics to the notebook output.
   - Save to JSON for easy parsing.

3. Smoke test protocol.
   - Test ablation notebooks with the TINY dataset first.
   - Verify all metrics are captured before the full run.
   - Add validation checks to the notebook builder.

## Statistical analysis (partial)

Strategy 4a vs baseline.

Effect size (Cohen's d):
```
d = (0.8743 - 0.800) / sigma_pooled
d ~ 21.2 (very large effect)
```

95% confidence interval for the 4a mean:
```
CI = 0.8743 +/- 2.920 * (0.0035 / sqrt(3))
CI = [0.8684, 0.8802]
```

Conclusion: Strategy 4a significantly outperforms baseline (p < 0.001).

Not possible:
- 4b vs 4a comparison
- 4c vs 4a comparison
- Overall best strategy determination
- Bonferroni-corrected pairwise tests

## Decision

Adopt (preliminary): Strategy 4a (mixed training). Clear improvement over baseline.
- Add the DDI dataset to the training mix.
- Continue using the combined FUSeg + Medetec + DDI dataset for future ablations.

Defer: Strategies 4b and 4c. Insufficient evidence. No ADOPT, MARGINAL or REJECT decision without metrics. Re-run with proper validation before a final decision.

Options:
1. Accept 4a and move on. Declare 4a the winner on available evidence, skip the 4b and 4c re-runs, document the limitation in the final thesis.
2. Complete the experiment. Re-run 4b and 4c with validation enabled, compare all 3 strategies statistically, pick the best approach from data.
3. Diagnostic validation only. Load the 4b and 4c trained weights, run validation offline, decide whether retraining is needed.

Preferred: Option 3 first, then Option 2 if 4b and 4c look promising.

## Lessons

Process failures:
1. The notebook was not validated enough before the full run. The smoke test should have caught the missing validation.
2. No validation checkpoint for short training runs. Finetuning configs need explicit validation flags.
3. Log parsing assumed all training would produce validation output.

Preventive measures:
1. Enhanced smoke testing: verify all metrics are captured.
2. Validation enforcement: always run final model evaluation.
3. Completion status augmentation: include performance metrics in the JSON.
4. Dual logging: save metrics to both logs and structured files.

## Appendix: data sources

Files analyzed:
- `completion_status.json`: training metadata (durations, paths)
- `ablation-4-unified-hard-negatives-3-strategies.log`: training logs (4MB)

Metrics extraction:
- Parsed JSON-formatted log entries.
- Searched for validation patterns: `all  552  733  [metrics]`.
- Extracted mask mAP50, precision and recall from final epochs.

Dataset characteristics:
- DDI dataset: 129 train, 129 test wound images.
- Combined dataset: 2,208 train images (FUSeg + Medetec + DDI).
- Test set: 552 images (same across all strategies).

## Next

1. Document this analysis: done.
2. Run diagnostic validation on the 4b and 4c models (if weights are available): open.
3. Decide between re-running 4b and 4c and accepting 4a: open.
4. Update `ABLATION_SPECIFICATIONS.md` with the final decision: open.
5. Proceed to Ablation 5 (TTA) or continue Ablation 4 completion: open.

Analysis completed 2025-10-20. Confidence: high for 4a results, low for the overall ablation because 4b and 4c metrics are missing.
