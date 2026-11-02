"""
Evaluation cell for Ablation 2 & 3 notebooks
Loads pre-trained weights if available, otherwise trains from scratch
Then evaluates on test set and populates all_training_results
"""

EVALUATION_CELL = """# Multi-seed training/evaluation with smart weight loading
RANDOM_SEEDS = [42, 123, 456]

# Storage for all training results
all_training_results = {}

print("=" * 80)
print("STARTING ABLATION EVALUATION")
print("=" * 80)

for seed_idx, seed in enumerate(RANDOM_SEEDS, 1):
    print("\\n" + "=" * 80)
    print(f"PROCESSING SEED {seed} ({seed_idx}/{len(RANDOM_SEEDS)})")
    print("=" * 80)

    # Check if pre-trained weights exist (from previous run uploaded as dataset)
    weights_path = OUTPUT_PATH / "yolo_training" / f"seed_{seed}" / "weights" / "best.pt"

    # ALSO check if weights exist in attached outputs dataset
    outputs_dataset_path = BASE_INPUT_PATH / "OUTPUTS_DATASET_NAME" / "yolo_training" / f"seed_{seed}" / "weights" / "best.pt"

    if outputs_dataset_path.exists():
        print(f"PASS: Loading pre-trained weights from outputs dataset: {outputs_dataset_path}")
        model = YOLO(str(outputs_dataset_path))
        print(f"  Training already completed - skipping to evaluation")
    elif weights_path.exists():
        print(f"PASS: Loading pre-trained weights from working directory: {weights_path}")
        model = YOLO(str(weights_path))
        print(f"  Training already completed - skipping to evaluation")
    else:
        print(f"WARNING: No pre-trained weights found - training from scratch")
        print(f"  This will take ~30-40 minutes on P100...")

        # Initialize model from pretrained COCO weights
        model = YOLO("yolov8n-seg.pt")

        # [TRAINING CODE GOES HERE - specific to each ablation]
        # Ablation 2: Add augmentation parameters
        # Ablation 3: Add imgsz=[416,512,640]

        results = model.train(
            data=str(config_path),
            batch=BATCH_SIZE,
            lr0=LEARNING_RATE,
            workers=WORKERS,
            amp=AMP,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            patience=PATIENCE,
            seed=seed,
            save=True,
            project=str(OUTPUT_PATH / "yolo_training"),
            name=f"seed_{seed}",
            exist_ok=True,
            verbose=False,
            # ABLATION-SPECIFIC PARAMETERS HERE
        )

        print(f"\\nPASS: Training completed")

    # EVALUATION (always runs, whether weights were loaded or trained)
    print(f"\\nEvaluating model on test set ({len(test_pairs)} images)...")

    seed_results = []
    for idx, (image_id, gt_mask_path) in enumerate(test_pairs):
        # Load image
        image_path = TEST_IMAGES_DIR / image_id
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load ground truth mask
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # Run inference
        yolo_results = model.predict(img_rgb, verbose=False)

        # Extract predicted mask
        mask_result = extract_mask_from_yolo_result(yolo_results[0])

        if mask_result["success"] and mask_result["data"]["has_prediction"]:
            pred_mask = mask_result["data"]["mask"]
        else:
            pred_mask = np.zeros_like(gt_mask)

        # Calculate metrics
        metrics = calculate_segmentation_metrics(pred_mask, gt_mask, image_id)
        seed_results.append(metrics)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_pairs)} images...")

    # Store results for this seed
    all_training_results[f"seed_{seed}"] = seed_results

    print(f"\\nPASS: Seed {seed} complete: {len(seed_results)} images evaluated")

print("\\n" + "=" * 80)
print("ALL TRAINING/EVALUATION RUNS COMPLETE")
print("=" * 80)
print(f"Total seeds: {len(all_training_results)}")
print(f"Images per seed: {len(next(iter(all_training_results.values())))}")
"""
