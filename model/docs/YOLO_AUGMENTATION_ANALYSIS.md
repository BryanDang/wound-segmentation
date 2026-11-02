# YOLOv8 Augmentation Analysis for Wound Segmentation

## Default parameters

From `ultralytics/cfg/default.yaml`.

Geometric:
| Parameter | Default | Description |
|---|---|---|
| `degrees` | 0.0 | Rotation (+/- degrees) |
| `translate` | 0.1 | Translation (+/- fraction of image) |
| `scale` | 0.5 | Scaling (+/- gain) |
| `shear` | 0.0 | Shear (degrees) |
| `perspective` | 0.0 | Perspective (fraction) |
| `flipud` | 0.0 | Vertical flip probability |
| `fliplr` | 0.5 | Horizontal flip probability |

Color and intensity:
| Parameter | Default | Description |
|---|---|---|
| `hsv_h` | 0.015 | Hue (fraction) |
| `hsv_s` | 0.7 | Saturation (fraction) |
| `hsv_v` | 0.4 | Brightness (fraction) |
| `bgr` | 0.0 | RGB/BGR swap probability |

Advanced:
| Parameter | Default | Description |
|---|---|---|
| `mosaic` | 1.0 | Mosaic probability |
| `mixup` | 0.0 | MixUp probability |
| `cutmix` | 0.0 | CutMix probability |
| `copy_paste` | 0.0 | Copy-paste probability (segmentation) |
| `copy_paste_mode` | "flip" | Copy-paste strategy |
| `auto_augment` | "randaugment" | Auto augmentation policy |
| `erasing` | 0.4 | Random erasing probability |
| `close_mosaic` | 10 | Disable mosaic for the final N epochs |

## close_mosaic and early stopping

`close_mosaic=10` disables mosaic at epoch (total_epochs - 10). With `epochs=100`, that is epoch 90. If `patience=10` stops training at epoch 50, mosaic never disables (GitHub Issue #18013).

Baseline: expected mosaic on for epochs 1-90 and clean fine-tuning for 91-100. Actual: training stopped around epoch 40-70 with mosaic on throughout. The 80% Dice was reached without the clean fine-tuning phase, so mosaic does not need to be disabled for medical imaging.

## Baseline augmentation

The baseline notebook set no augmentation parameters, so YOLOv8 defaults applied:

```python
degrees=0.0        # no rotation
translate=0.1      # +/- 51 pixels (10% of 512)
scale=0.5          # +/- 50%
shear=0.0
perspective=0.0
flipud=0.0
fliplr=0.5
mosaic=1.0         # always on (never disabled, early stopping)
hsv_h=0.015
hsv_s=0.7
hsv_v=0.4
erasing=0.4
auto_augment="randaugment"
```

Baseline had 11 active augmentation parameters and still reached 80% Dice.

## Ablation 2

Set explicitly:
```python
degrees=15.0
fliplr=0.5
flipud=0.5
mosaic=0.0
```

Inherited defaults: translate 0.1, scale 0.5, hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, erasing 0.4, auto_augment "randaugment".

Result: 68% Dice (-12% vs baseline).

Cause:
- Lost: `mosaic=0.0` removed 4x positional diversity (256-pixel displacement vs 51-pixel translate).
- Gained: `degrees=15.0` (wounds have gravitational orientation) and `flipud=0.5` (wounds not vertically symmetric), both questionable.
- Net: worse, because mosaic's spatial diversity is not replaced.

## Proposed strategies

### 1. Baseline replica (control)

Replicate the baseline exactly, for fair comparison.

```python
model.train(
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4,
    auto_augment="randaugment", close_mosaic=10,
    # ... other training params
)
```

Expected: about 80% Dice. Purpose: confirm baseline reproduces with explicit parameters.

### 2. Clinical conservative

Keep only augmentations that preserve anatomical realism.

Kept:
1. `mosaic=1.0`: scale and position diversity.
2. `fliplr=0.5`: wounds equally likely on left and right limbs.
3. `translate=0.1`: camera positioning variance.
4. `scale=0.3`: reduced from 0.5 (+/-30% more realistic).
5. `hsv_v=0.2`: reduced from 0.4; lighting only, preserve color.

Removed:
- `degrees`: gravitational orientation (leg ulcers do not appear rotated).
- `flipud`: top and bottom differ (gravity, drainage).
- `hsv_h=0.0`: hue is critical (erythema, necrosis).
- `hsv_s=0.0`: saturation is critical (tissue viability).
- `erasing=0.0`: may remove diagnostic regions.
- `auto_augment=None`: RandAugment may apply unsafe transforms.
- `shear=0.0`: no physical basis.
- `perspective=0.0`: camera is typically perpendicular to the wound.

```python
model.train(
    mosaic=1.0, fliplr=0.5, translate=0.1,
    scale=0.3,           # reduced from 0.5
    hsv_v=0.2,           # reduced from 0.4
    degrees=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0,
    shear=0.0, perspective=0.0, erasing=0.0,
    auto_augment=None, mixup=0.0, copy_paste=0.0,
    close_mosaic=10,
    # ... other training params
)
```

Expected: 78-82% Dice. Every augmentation has an explicit medical or physical basis.

### 3. Minimal augmentation

Test whether augmentation is needed at all.

```python
model.train(
    mosaic=1.0, fliplr=0.5,
    degrees=0.0, translate=0.0, scale=0.0, flipud=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    shear=0.0, perspective=0.0, erasing=0.0,
    auto_augment=None, mixup=0.0, copy_paste=0.0,
    close_mosaic=10,
    # ... other training params
)
```

Expected: 75-80% Dice, possibly below baseline. Purpose: lower bound; do color and scale augmentation contribute?

### 4. Clinical enhanced

Maximize diversity within anatomical constraints.

```python
model.train(
    mosaic=1.0, fliplr=0.5,
    translate=0.15,      # up from 0.1 (+/- 77 pixels)
    scale=0.4,           # up from 0.3
    hsv_v=0.25,          # up from 0.2
    degrees=5.0,         # camera angle variance only
    flipud=0.0, hsv_h=0.0, hsv_s=0.0,
    shear=0.0, perspective=0.0, erasing=0.0,
    auto_augment=None, mixup=0.0, copy_paste=0.0,
    close_mosaic=10,
    # ... other training params
)
```

Expected: 79-83% Dice, possibly best. Purpose: does more spatial and lighting diversity help without losing clinical realism?

## Revised Ablation 2 design

Control: Strategy 1.
Treatments:
- 2a: clinical conservative.
- 2b: minimal augmentation (mosaic and flip only).
- 2c: clinical enhanced.

Hypothesis: conservative or enhanced matches or beats baseline and is clinically justifiable.

Metrics: Dice (primary), Boundary F1, per-class performance (wound vs background).

Success: Dice at least 80% with clinically justified augmentation only.

## Findings

1. Mosaic is critical: 5x more effective than translate for positional diversity (256 px vs 51 px displacement).
2. close_mosaic does not work with early stopping; baseline trained with mosaic throughout and still got 80%.
3. `hsv_h` and `hsv_s` alter wound appearance (erythema, necrosis, viability).
4. Rotation is questionable (drainage, edema).
5. Baseline used 11 active augmentations, not a minimal set.
6. Ablation 2 disabled mosaic while adding rotation and vertical flip, losing spatial diversity for little gain.

## Open

1. Implement and test Strategy 2 (clinical conservative) as the new baseline.
2. Run the ablation comparing all 4 strategies.
3. Update the PRD with a clinically justified augmentation protocol.
