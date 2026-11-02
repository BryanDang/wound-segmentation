"""
Cell specifications for DDI Diagnostic Validation Notebook
Tests baseline model on dermatological images to ensure no false positives
"""

CELL_0_NUMPY_LOCK = """# Lock NumPy version BEFORE any imports
# Prevents ultralytics from upgrading NumPy to 2.x (breaks OpenCV compatibility)
import subprocess
import numpy as np

default_numpy = np.__version__
print(f"Kaggle default NumPy: {default_numpy}")

# Install packages
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)

# Force-reinstall NumPy to prevent 2.x upgrade
subprocess.run(
    ["pip", "install", "-q", "--force-reinstall", f"numpy=={default_numpy}"],
    check=True
)

import numpy as np
print(f"Final NumPy version: {np.__version__}")"""

CELL_1_IMPORTS = """# Standard library imports
import sys
from pathlib import Path
import json
import shutil

# Third-party imports
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add code dataset to path
sys.path.append('/kaggle/input/wound-seg-code-v1')

# Import project modules
from evaluation.metrics import calculate_segmentation_metrics
from evaluation.yolo_utils import extract_mask_from_yolo_result

print("All imports successful")"""

CELL_2_SETUP = """# Configuration
IN_KAGGLE = True
BASE_INPUT_PATH = Path('/kaggle/input')
OUTPUT_PATH = Path('/kaggle/working')

# Dataset paths
DDI_DATASET_PATH = BASE_INPUT_PATH / 'ddi-diverse-dermatology' / 'PRIVATE - ddidiversedermatologyimages'
DDI_METADATA_PATH = BASE_INPUT_PATH / 'ddi-diverse-dermatology' / 'ddi_metadata.csv'

# Baseline model (best checkpoint from fine-tuning)
BASELINE_MODEL_PATH = BASE_INPUT_PATH / 'baseline-outputs' / 'runs' / 'segment' / 'finetune_seed_42' / 'weights' / 'best.pt'

# Output directory
DIAGNOSTIC_OUTPUT_DIR = OUTPUT_PATH / '03_ddi_diagnostic'
DIAGNOSTIC_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"DDI images: {DDI_DATASET_PATH}")
print(f"DDI metadata: {DDI_METADATA_PATH}")
print(f"Baseline model: {BASELINE_MODEL_PATH}")
print(f"Output directory: {DIAGNOSTIC_OUTPUT_DIR}")

# Verify paths exist
assert DDI_DATASET_PATH.exists(), f"DDI dataset not found: {DDI_DATASET_PATH}"
assert DDI_METADATA_PATH.exists(), f"DDI metadata not found: {DDI_METADATA_PATH}"
assert BASELINE_MODEL_PATH.exists(), f"Baseline model not found: {BASELINE_MODEL_PATH}"

print("All paths validated")"""

CELL_3_LOAD_METADATA = """# Load DDI metadata
df_metadata = pd.read_csv(DDI_METADATA_PATH)

print(f"Original metadata records: {len(df_metadata)}")

# Get actual files on disk
actual_files = sorted([f.name for f in DDI_DATASET_PATH.glob("*.png")])
print(f"Actual files on disk: {len(actual_files)}")

# Filter to only files that exist (cleaned dataset)
df_metadata = df_metadata[df_metadata['DDI_file'].isin(actual_files)].reset_index(drop=True)

print(f"Cleaned metadata records: {len(df_metadata)}")
print(f"\\nDisease distribution (top 10):")
print(df_metadata['disease'].value_counts().head(10))

# Display sample
print(f"\\nFirst 5 records:")
df_metadata.head()"""

CELL_4_LOAD_MODEL = """# Load baseline model
print("Loading baseline model...")
model = YOLO(BASELINE_MODEL_PATH)

print(f"Model loaded: {BASELINE_MODEL_PATH.name}")
print(f"Model task: {model.task}")
print(f"Model type: {model.model.__class__.__name}")"""

CELL_5_PROCESS_IMAGES = """# Process all DDI images through baseline model
results = []

print(f"Processing {len(df_metadata)} dermatological images...")

for idx, row in df_metadata.iterrows():
    image_file = row['DDI_file']
    image_path = DDI_DATASET_PATH / image_file

    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    yolo_results = model.predict(img_rgb, verbose=False)

    # Extract mask using project utility
    mask_result = extract_mask_from_yolo_result(yolo_results[0])

    # Check if wound was detected
    wound_detected = mask_result["success"] and mask_result["data"]["has_prediction"]

    # Calculate wound area if detected
    wound_area = 0
    confidence = 0.0
    if wound_detected:
        pred_mask = mask_result["data"]["mask"]
        wound_area = int(np.sum(pred_mask > 0))
        confidence = float(mask_result["data"]["confidence"])

    # Store result
    results.append({
        "image_id": image_file,
        "DDI_ID": row['DDI_ID'],
        "disease": row['disease'],
        "malignant": row['malignant'],
        "skin_tone": row['skin_tone'],
        "wound_detected": wound_detected,
        "wound_area_pixels": wound_area,
        "confidence": confidence
    })

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1}/{len(df_metadata)} images...")

print(f"Processing complete: {len(results)} images")

# Convert to DataFrame
df_results = pd.DataFrame(results)
df_results.head(10)"""

CELL_6_ANALYZE_RESULTS = """# Analysis of false positive detections

# Overall statistics
total_images = len(df_results)
false_positives = df_results['wound_detected'].sum()
fp_rate = (false_positives / total_images) * 100

print("="*60)
print("DDI DIAGNOSTIC VALIDATION RESULTS")
print("="*60)
print(f"Total dermatological images: {total_images}")
print(f"False positive detections: {false_positives}")
print(f"False positive rate: {fp_rate:.2f}%")
print(f"True negative rate: {100 - fp_rate:.2f}%")
print("="*60)

# Breakdown by disease
print(f"\\nFalse positives by disease:")
fp_by_disease = df_results[df_results['wound_detected']].groupby('disease').size().sort_values(ascending=False)
print(fp_by_disease.head(20))

# Breakdown by malignancy
print(f"\\nFalse positives by malignancy:")
fp_by_malignancy = df_results[df_results['wound_detected']].groupby('malignant').size()
print(fp_by_malignancy)

# Breakdown by skin tone
print(f"\\nFalse positives by skin tone:")
fp_by_tone = df_results[df_results['wound_detected']].groupby('skin_tone').size()
print(fp_by_tone)

# Confidence statistics for false positives
if false_positives > 0:
    print(f"\\nFalse positive confidence statistics:")
    fp_confidences = df_results[df_results['wound_detected']]['confidence']
    print(f"Mean: {fp_confidences.mean():.3f}")
    print(f"Median: {fp_confidences.median():.3f}")
    print(f"Min: {fp_confidences.min():.3f}")
    print(f"Max: {fp_confidences.max():.3f}")"""

CELL_7_VISUALIZE_FP_DISTRIBUTION = """# Visualize false positive distribution

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. False positive rate by disease (top 15)
ax1 = axes[0, 0]
fp_diseases = df_results[df_results['wound_detected']].groupby('disease').size().sort_values(ascending=False).head(15)
fp_diseases.plot(kind='barh', ax=ax1, color='coral')
ax1.set_title('False Positives by Disease (Top 15)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Count', fontsize=12)
ax1.set_ylabel('Disease', fontsize=12)
ax1.grid(axis='x', alpha=0.3)

# 2. Overall detection rate
ax2 = axes[0, 1]
detection_counts = df_results['wound_detected'].value_counts()
labels = ['True Negatives\\n(No Detection)', 'False Positives\\n(Wound Detected)']
colors = ['lightgreen', 'coral']
ax2.pie(detection_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax2.set_title(f'Overall Detection Results\\n(n={total_images})', fontsize=14, fontweight='bold')

# 3. Confidence distribution for false positives
ax3 = axes[1, 0]
if false_positives > 0:
    fp_confidences = df_results[df_results['wound_detected']]['confidence']
    ax3.hist(fp_confidences, bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(fp_confidences.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {fp_confidences.mean():.3f}')
    ax3.set_title('False Positive Confidence Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Confidence Score', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'No false positives detected', ha='center', va='center', fontsize=14)
    ax3.set_title('False Positive Confidence Distribution', fontsize=14, fontweight='bold')

# 4. False positives by skin tone
ax4 = axes[1, 1]
fp_tone = df_results[df_results['wound_detected']].groupby('skin_tone').size()
fp_tone.plot(kind='bar', ax=ax4, color='coral', edgecolor='black')
ax4.set_title('False Positives by Skin Tone', fontsize=14, fontweight='bold')
ax4.set_xlabel('Fitzpatrick Scale', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(DIAGNOSTIC_OUTPUT_DIR / 'ddi_diagnostic_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Visualization saved: ddi_diagnostic_visualization.png")"""

CELL_8_VISUALIZE_FP_EXAMPLES = """# Visualize worst false positive examples

if false_positives > 0:
    # Get top 9 false positives by confidence
    df_fp = df_results[df_results['wound_detected']].sort_values('confidence', ascending=False).head(9)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(df_fp.iterrows()):
        ax = axes[idx]

        # Load image
        image_path = DDI_DATASET_PATH / row['image_id']
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference to get mask
        yolo_results = model.predict(img_rgb, verbose=False)
        mask_result = extract_mask_from_yolo_result(yolo_results[0])

        # Create overlay
        if mask_result["success"] and mask_result["data"]["has_prediction"]:
            pred_mask = mask_result["data"]["mask"]

            # Create red overlay for predicted wound
            overlay = img_rgb.copy()
            overlay[pred_mask > 0] = [255, 0, 0]

            # Blend
            display_img = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
        else:
            display_img = img_rgb

        ax.imshow(display_img)
        ax.set_title(f"{row['disease']}\\nConf: {row['confidence']:.3f} | Area: {row['wound_area_pixels']} px",
                     fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Worst False Positive Examples (Top 9 by Confidence)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(DIAGNOSTIC_OUTPUT_DIR / 'ddi_diagnostic_fp_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"False positive examples saved: ddi_diagnostic_fp_examples.png")
else:
    print("No false positives detected - no examples to visualize")"""

CELL_9_SAVE_RESULTS = """# Save detailed results

# Save per-image results
results_csv_path = DIAGNOSTIC_OUTPUT_DIR / 'ddi_diagnostic_per_image_results.csv'
df_results.to_csv(results_csv_path, index=False)
print(f"Per-image results saved: {results_csv_path.name}")

# Save summary statistics
summary = {
    "total_images": int(total_images),
    "false_positives": int(false_positives),
    "true_negatives": int(total_images - false_positives),
    "false_positive_rate": float(fp_rate),
    "true_negative_rate": float(100 - fp_rate),
    "false_positives_by_disease": fp_by_disease.to_dict() if false_positives > 0 else {},
    "false_positives_by_malignancy": fp_by_malignancy.to_dict() if false_positives > 0 else {},
    "false_positives_by_skin_tone": fp_by_tone.to_dict() if false_positives > 0 else {},
    "confidence_stats": {
        "mean": float(fp_confidences.mean()) if false_positives > 0 else 0.0,
        "median": float(fp_confidences.median()) if false_positives > 0 else 0.0,
        "min": float(fp_confidences.min()) if false_positives > 0 else 0.0,
        "max": float(fp_confidences.max()) if false_positives > 0 else 0.0
    } if false_positives > 0 else {}
}

summary_json_path = DIAGNOSTIC_OUTPUT_DIR / 'ddi_diagnostic_summary.json'
with open(summary_json_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary statistics saved: {summary_json_path.name}")

# Create completion marker
completion_status = {
    "notebook": "03_ddi_diagnostic",
    "status": "COMPLETE",
    "total_images_processed": int(total_images),
    "false_positive_rate": float(fp_rate),
    "outputs": [
        "ddi_diagnostic_per_image_results.csv",
        "ddi_diagnostic_summary.json",
        "ddi_diagnostic_visualization.png",
        "ddi_diagnostic_fp_examples.png" if false_positives > 0 else None
    ]
}

completion_path = DIAGNOSTIC_OUTPUT_DIR / 'completion_status.json'
with open(completion_path, 'w') as f:
    json.dump(completion_status, f, indent=2)

print(f"Completion status saved: {completion_path.name}")"""

CELL_10_INTERPRETATION = """# Interpretation and Next Steps

print("="*60)
print("DIAGNOSTIC INTERPRETATION")
print("="*60)
print(f"\\nBaseline Model Performance on Dermatological Images:")
print(f"  - False Positive Rate: {fp_rate:.2f}%")
print(f"  - True Negative Rate: {100 - fp_rate:.2f}%")

print(f"\\nInterpretation:")
if fp_rate < 10:
    print(" EXCELLENT: Model shows strong discrimination")
    print("     Hard negative mining may provide minimal benefit")
elif fp_rate < 20:
    print("  MODERATE: Model shows some confusion with dermatological conditions")
    print("     Hard negative mining recommended for refinement")
else:
    print(" POOR: Model struggles with dermatological conditions")
    print("     Hard negative mining CRITICAL for reducing false positives")

print(f"\\nRecommendation for Ablation 4:")
if fp_rate >= 15:
    print("  - Proceed with ALL variants (4a, 4b, 4c)")
    print("  - Expected FP reduction: {fp_rate:.1f}% to <15%")
    print("  - Focus on diseases with highest FP rates")
elif fp_rate >= 10:
    print("  - Proceed with two-stage variants (4b, 4c)")
    print("  - Expected FP reduction: {fp_rate:.1f}% to <10%")
    print("  - Mixed approach (4a) may be overkill")
else:
    print("  - Consider SKIPPING Ablation 4")
    print("  - Model already performs excellently on hard negatives")
    print("  - Focus resources on other ablations")

print("="*60)"""

# Cell specifications list
CELLS = [
    {"type": "code", "source": CELL_0_NUMPY_LOCK},
    {"type": "code", "source": CELL_1_IMPORTS},
    {"type": "code", "source": CELL_2_SETUP},
    {"type": "code", "source": CELL_3_LOAD_METADATA},
    {"type": "code", "source": CELL_4_LOAD_MODEL},
    {"type": "code", "source": CELL_5_PROCESS_IMAGES},
    {"type": "code", "source": CELL_6_ANALYZE_RESULTS},
    {"type": "code", "source": CELL_7_VISUALIZE_FP_DISTRIBUTION},
    {"type": "code", "source": CELL_8_VISUALIZE_FP_EXAMPLES},
    {"type": "code", "source": CELL_9_SAVE_RESULTS},
    {"type": "code", "source": CELL_10_INTERPRETATION},
]
