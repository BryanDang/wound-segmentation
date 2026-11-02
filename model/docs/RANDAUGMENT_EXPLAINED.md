# RandAugment in YOLOv8

With `auto_augment="randaugment"` (the YOLOv8 default), each training image gets 2 random transformations drawn from 14 operations. Clinical appropriateness is unknown or poor for most of them.

## The 14 operations

YOLOv8 uses `torchvision.transforms.RandAugment`.

| Transformation | Effect | Suitability for wounds |
|---|---|---|
| Identity | No change | Safe |
| AutoContrast | Maximize contrast (darkest to black, lightest to white) | Questionable: may alter wound appearance |
| Equalize | Histogram equalization | Questionable: may alter tissue color assessment |
| Rotate | Rotation, angle by magnitude | Harmful: wounds have gravitational orientation |
| Solarize | Invert pixels above threshold | Harmful, critical: unrealistic for medical images |
| Color | Adjust saturation | Harmful: tissue viability assessment needs accurate color |
| Posterize | Reduce bits per channel (2-8) | Harmful: loses diagnostic detail |
| Contrast | Adjust contrast | Questionable: may help with lighting variance |
| Brightness | Adjust brightness | Questionable: similar to hsv_v |
| Sharpness | Adjust sharpness | Questionable: may blur wound edges |
| ShearX | Horizontal shear | Harmful: no physical justification |
| ShearY | Vertical shear | Harmful: no physical justification |
| TranslateX | Horizontal translation | Safe: similar to translate |
| TranslateY | Vertical translation | Safe: similar to translate |

Safe 3/14 (21%). Questionable 5/14 (36%). Harmful 6/14 (43%).

## Mechanics

```python
auto_augment="randaugment"  # YOLOv8 default
```

torchvision parameters:
- `num_ops=2`: 2 random transformations per image.
- `magnitude=9`: intensity on a 0-30 scale.
- `num_magnitude_bins=31`.

Per training image: pick 2 of 14 operations at random, apply in sequence at magnitude 9. Different for every image.

Examples: AutoContrast then Rotate(15 degrees); Solarize then ShearX; Identity then Brightness(+20%); Posterize(4 bits) then Color(-30% saturation).

## Problem operations

Solarize (critical). Inverts pixels above a threshold. Red (R=200, G=100, B=100) becomes cyan (R=55, G=155, B=155). Red erythema turns cyan/blue (mimics cyanosis), pink granulation turns green/purple.

Posterize (critical). Reduces color depth.
```
8-bit (256 colors): R=200, G=150, B=100
4-bit (16 colors):  R=192, G=144, B=96
2-bit (4 colors):   R=192, G=128, B=64
```
Loses erythema gradations, cannot separate granulation from slough.

Color, Contrast, Brightness (moderate). Like hsv_s and hsv_v; random application gives unpredictable color shifts and may train the model to ignore diagnostic color.

Shear, Rotate (questionable). Wounds have physical orientation (gravity, drainage). Shear has no medical justification. Rotation above 5 degrees looks unrealistic.

## Effect on baseline training

Baseline configuration:
```python
auto_augment="randaugment"  # every training image
# + hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
# + erasing=0.4  (40% random patch removal)
```

Per-image pipeline:
1. Standard YOLO augmentation (mosaic, flip, translate, scale, hsv).
2. Random erasing (40%).
3. RandAugment: 2 of 14 operations.

Training images were heavily distorted in color and geometry, and the baseline still reached 80% Dice. Reading: the model learned to ignore color (unreliable under augmentation) and relied on texture and shape. Robust to color variation, may miss diagnostic color cues.

## Clinical implications

Robust to lighting conditions, color variation, geometric variation.

Possible loss of clinically relevant features: erythema intensity (inflammation), tissue viability (saturation, perfusion), necrotic tissue (black/brown), granulation tissue (pink/red). A wound could be segmented accurately while signs of infection, poor healing or necrosis are missed.

## Configurations to test

1. Baseline replica: `auto_augment="randaugment"`.
2. Clinical conservative: `auto_augment=None`.
3. Comparison: if performance drops without RandAugment, aggressive augmentation helps; if similar or better, RandAugment was unnecessary or harmful.

Deployment choice:
- Keep RandAugment: robust model that ignores color.
- Remove RandAugment: color-aware model, less robust.
- Chosen direction for clinical use: remove RandAugment, rely on mosaic and minimal safe augmentation, preserve color fidelity.

## Summary

| Issue | Severity | Impact |
|---|---|---|
| Solarize | Critical | Inverts colors (red to cyan) |
| Posterize | Critical | Removes diagnostic detail |
| Random color shifts | High | Unpredictable saturation and hue changes |
| Shear/Rotate | Moderate | No medical justification |
| Non-deterministic | Moderate | Different augmentation per image, hard to debug |

RandAugment applies 6 potentially harmful operations (43%) with no clinical oversight. Use `auto_augment=None` in all clinical augmentation strategies.

## Open

1. Add `auto_augment=None` to all proposed mini-ablations.
2. Test baseline with and without RandAugment.
3. Prohibit solarize and posterize in the PRD.
4. Color-aware evaluation: does the model use erythema for segmentation?
