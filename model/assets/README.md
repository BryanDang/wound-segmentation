# Assets

Clinical examples, media files and supporting assets for the wound segmentation project.

Currently present: `confusion_matrix.png`. The subdirectories below are the planned layout.

## Planned layout

```
assets/
├── clinical_examples/
│   ├── validation_samples/    # clinical validation images
│   ├── model_predictions/     # example model outputs
│   ├── comparison_plots/      # performance comparison charts
│   └── statistical_analysis/  # statistical test results
├── documentation_media/       # images for documentation
├── presentation_materials/    # figures for presentations
└── reference_images/          # reference and calibration images
```

`clinical_examples/` holds validation examples and reference images for model performance demonstrations, clinical evaluation benchmarks, documentation and presentation material, and training and validation samples.

Contents of `clinical_examples/`:
- De-identified clinical images (when available)
- Ground truth segmentation masks
- Model prediction examples
- Before/after comparison images
- Statistical analysis visualizations

## Rules for adding assets

1. Privacy: all clinical images de-identified.
2. Naming: descriptive, consistent names.
3. Metadata: README for complex asset collections.
4. Size: compress images appropriately.
5. Sources: cite and attribute.

## Clinical data guidelines

Privacy and ethics:
- Clinical images are de-identified per HIPAA standards.
- Identifying information is removed or obscured.
- IRB approval is needed where required.
- Institutional data use policies apply.

Quality:
- High resolution preferred (minimum 512x512).
- Consistent lighting and image quality.
- Proper wound exposure and positioning.
- Clean backgrounds without distracting elements.

Annotation:
- Consistent annotation tools and formats.
- Follow clinical wound assessment protocols.
- Include wound boundary annotations.
- Document annotation methodology and inter-rater reliability.

## Usage examples

Loading a clinical example:
```python
import cv2
from pathlib import Path

# Load clinical validation image
image_path = Path("assets/clinical_examples/validation_samples/case_001.png")
image = cv2.imread(str(image_path))

# Load corresponding ground truth mask
mask_path = Path("assets/clinical_examples/validation_samples/case_001_mask.png")
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
```

Creating a comparison visualization:
```python
import matplotlib.pyplot as plt

# Create before/after comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title('Original Image')
axes[1].imshow(ground_truth_mask, cmap='gray')
axes[1].set_title('Ground Truth')
axes[2].imshow(model_prediction, cmap='gray')
axes[2].set_title('Model Prediction')

plt.savefig('assets/clinical_examples/comparison_plots/case_001_comparison.png')
```

## Dataset citations

1. Medetec Wound Database:
   ```
   Thomas, S. Stock pictures of wounds. Medetec Wound Database (2020).
   http://www.medetec.co.uk/files/medetec-image-databases.html
   ```

2. Wang et al. dataset:
   ```
   Wang, C., et al. "Fully automatic wound segmentation with deep convolutional neural networks."
   Sci Rep 10, 21897 (2020). https://doi.org/10.1038/s41598-020-78799-w
   ```

3. Oota et al. dataset:
   ```
   Oota, S.R., et al. "WSNet: Towards An Effective Method for Wound Image Segmentation."
   IEEE/CVF WACV 2023. doi: 10.1109/WACV56688.2023.00325
   ```

## Maintenance

Regular tasks:
- Review and update clinical examples quarterly.
- Verify compliance with privacy regulations.
- Update documentation and citations.
- Optimize file sizes and organization.
- Validate asset integrity and accessibility.

Quality control:
- Periodic review of clinical accuracy.
- Validation of annotation quality.
- Assessment of dataset representativeness.
- Documentation completeness checks.
