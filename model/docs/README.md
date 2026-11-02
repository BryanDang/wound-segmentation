# Docs Index

## model/docs/

Study design and results
- [EXPERIMENT-PROTOCOL.md](EXPERIMENT-PROTOCOL.md): data, fixed settings, measures, procedure, experiment register, results, open items, how to test.
- [ABLATION_SPECIFICATIONS.md](ABLATION_SPECIFICATIONS.md): per-ablation specification, status and results.
- [ABLATION_OUTPUTS_ORGANIZATION.md](ABLATION_OUTPUTS_ORGANIZATION.md): Kaggle output directory layout for ablations.
- [BASELINE_OUTPUTS_HYBRID_STRUCTURE.md](BASELINE_OUTPUTS_HYBRID_STRUCTURE.md): layout and use of the `baseline-outputs` Kaggle dataset.
- [COMPONENTS.md](COMPONENTS.md): reusable code, scripts and notebook families.

Ablation 4 (hard negatives)
- [ABLATION_4_METHODOLOGY_SUMMARY.md](ABLATION_4_METHODOLOGY_SUMMARY.md): pre-run design of strategies 4a, 4b, 4c.
- [ABLATION_4_INSIGHTS_FOR_FINAL_REPORT.md](ABLATION_4_INSIGHTS_FOR_FINAL_REPORT.md): results, failure causes, open FPR measurement.
- [YOLOV8_EMPTY_LABELS_VALIDATION_RESEARCH.md](YOLOV8_EMPTY_LABELS_VALIDATION_RESEARCH.md): how YOLOv8 validation treats empty label files.

Implementation plans
- [ABLATIONS_1B_5_6_IMPLEMENTATION_PLAN.md](ABLATIONS_1B_5_6_IMPLEMENTATION_PLAN.md): first combined draft for 1b, 5, 6.
- [ABLATION_1B_DICE_PLUS_PLUS_IMPLEMENTATION_PLAN.md](ABLATION_1B_DICE_PLUS_PLUS_IMPLEMENTATION_PLAN.md): Dice++ loss plan.
- [ABLATION_05_TTA_IMPLEMENTATION_PLAN.md](ABLATION_05_TTA_IMPLEMENTATION_PLAN.md): test-time augmentation plan.
- [ABLATION_06_LABEL_SMOOTHING_IMPLEMENTATION_PLAN.md](ABLATION_06_LABEL_SMOOTHING_IMPLEMENTATION_PLAN.md): label smoothing plan.

Research and reference
- [LOSS_FUNCTION_RESEARCH_2025.md](LOSS_FUNCTION_RESEARCH_2025.md): Boundary DoU, Weighted Tversky and ABeDice candidates.
- [YOLO_AUGMENTATION_ANALYSIS.md](YOLO_AUGMENTATION_ANALYSIS.md): YOLOv8 augmentation defaults and four proposed strategies.
- [RANDAUGMENT_EXPLAINED.md](RANDAUGMENT_EXPLAINED.md): the 14 RandAugment operations and their fit for wound images.
- [MANDATORY_KAGGLE_VALIDATION.md](MANDATORY_KAGGLE_VALIDATION.md): kernel metadata validation before every push.
- [FUTURE_PROJECTS.md](FUTURE_PROJECTS.md): six follow-up projects with timelines.

## model/docs/reports/

- [ABLATION_2_COMPLETE_PARAMETER_METRIC_TABLE.md](reports/ABLATION_2_COMPLETE_PARAMETER_METRIC_TABLE.md): all parameters and metrics, baseline vs A2 vs counterfactual.
- [ABLATION_2_COUNTERFACTUAL_COMPLETE_ANALYSIS.md](reports/ABLATION_2_COUNTERFACTUAL_COMPLETE_ANALYSIS.md): isolated effect of mosaic.
- [ABLATION_2_COUNTERFACTUAL_MOSAIC_COMPARISON.md](reports/ABLATION_2_COUNTERFACTUAL_MOSAIC_COMPARISON.md): mosaic counterfactual at kernel launch.
- [ABLATION_3_CHRONOLOGICAL_HISTORY.md](reports/ABLATION_3_CHRONOLOGICAL_HISTORY.md): every multi-scale iteration.
- [ABLATION_3_COMPLETE_PARAMETER_METRIC_TABLE.md](reports/ABLATION_3_COMPLETE_PARAMETER_METRIC_TABLE.md): parameters and metrics across Ablation 3 versions.
- [ABLATION_3_COUNTERFACTUAL_COMPLETE_ANALYSIS.md](reports/ABLATION_3_COUNTERFACTUAL_COMPLETE_ANALYSIS.md): multi-scale without augmentations.
- [ABLATION_3_V5_AUTO_AUGMENT_BUG_CORRECTION.md](reports/ABLATION_3_V5_AUTO_AUGMENT_BUG_CORRECTION.md): correction of the v5 auto_augment claim.
- [ABLATION_4D_V10_FAILURE_ANALYSIS.md](reports/ABLATION_4D_V10_FAILURE_ANALYSIS.md): Ablation 4d v10 failure analysis.
- [ABLATION_4D_V10_FAILURE_ANALYSIS.md](reports/ABLATION_4D_V10_FAILURE_ANALYSIS.md): v10 ModuleNotFoundError failure.
- [ABLATION_4_ALL_VARIANTS_DOCUMENTATION.md](reports/ABLATION_4_ALL_VARIANTS_DOCUMENTATION.md): all Ablation 4 notebooks and directories.
- [ABLATION_4_EMPTY_LABELS_CONCLUSIVE_ANSWER.md](reports/ABLATION_4_EMPTY_LABELS_CONCLUSIVE_ANSWER.md): empty labels in YOLOv8 validation.
- [ABLATION_4_KEY_FINDINGS.md](reports/ABLATION_4_KEY_FINDINGS.md): early Ablation 4 findings.
- [ABLATION_4_REDESIGN_PROPOSAL.md](reports/ABLATION_4_REDESIGN_PROPOSAL.md): corrected 4b/4c design.
- [ABLATION_4_RESULTS_SUMMARY.txt](reports/ABLATION_4_RESULTS_SUMMARY.txt): partial Ablation 4 results.
- [ABLATION_4_ROOT_CAUSE_ANALYSIS.md](reports/ABLATION_4_ROOT_CAUSE_ANALYSIS.md): why 4a worked and 4b/4c failed.
- [ABLATION_4_STRATEGY_ANALYSIS_FINAL.md](reports/ABLATION_4_STRATEGY_ANALYSIS_FINAL.md): strategy analysis after the diagnostic.
- [DDI_DATASET_RESOLUTION.md](reports/DDI_DATASET_RESOLUTION.md): DDI dataset path fix.
- [EMPTY_LABELS_DIAGNOSTIC_TEST.md](reports/EMPTY_LABELS_DIAGNOSTIC_TEST.md): Kaggle diagnostic for empty labels.
- [KAGGLE_KERNELS_ALL_SUCCESSFUL_RUNS.md](reports/KAGGLE_KERNELS_ALL_SUCCESSFUL_RUNS.md): successful kernel runs for baseline and Ablations 1-3.
