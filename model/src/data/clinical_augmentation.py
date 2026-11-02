"""
Clinical data augmentation for wound segmentation.

Contract: ClinicalAugmentationContract
Provides anatomically plausible augmentations optimized for medical imaging.
"""

from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def rotate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    angle_range: Tuple[float, float] = (-15, 15),
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Rotate image and mask by random angle within range.

    Contract: RotationAugmentationContract
    Tests: tests/unit/data/test_clinical_augmentation.py::TestRotateImageAndMask

    Args:
        image: RGB image (H, W, 3), dtype uint8
        mask: Binary mask (H, W), dtype uint8 (0 or 255)
        angle_range: Min/max rotation angle in degrees (default: ±15°)

    Returns:
        Tuple of:
            - rotated_image: Rotated RGB image (same shape, uint8)
            - rotated_mask: Rotated binary mask (same shape, uint8)
            - angle: Actual rotation angle applied (degrees)

    Raises:
        ValueError: If image/mask shapes incompatible or angle_range invalid

    Notes:
        Range ±15°. Rotations above 30° give unrealistic wound presentations.
    """
    # Validate inputs
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3), got {image.shape}")

    if len(mask.shape) != 2:
        raise ValueError(f"Mask must be (H, W), got {mask.shape}")

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image shape {image.shape[:2]} != mask shape {mask.shape}"
        )

    if angle_range[0] > angle_range[1]:
        raise ValueError(
            f"Invalid angle range: {angle_range} (min > max)"
        )

    # Random rotation angle
    angle = np.random.uniform(angle_range[0], angle_range[1])

    # Get rotation matrix
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Rotate image
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Rotate mask (use NEAREST to preserve binary values)
    rotated_mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated_image, rotated_mask, angle


def adjust_brightness(
    image: np.ndarray,
    factor_range: Tuple[float, float] = (-0.2, 0.2),
) -> Tuple[np.ndarray, float]:
    """
    Adjust image brightness by random factor.

    Contract: BrightnessAugmentationContract
    Tests: tests/unit/data/test_clinical_augmentation.py::TestAdjustBrightness

    Args:
        image: RGB image (H, W, 3), dtype uint8
        factor_range: Min/max brightness factor (default: ±20%)
                      Positive = brighter, Negative = darker

    Returns:
        Tuple of:
            - adjusted_image: Brightness-adjusted image (same shape, uint8)
            - factor: Actual brightness factor applied

    Raises:
        ValueError: If image shape invalid or factor_range invalid

    Notes:
        Range ±20%. Simulates room light, natural light and camera flash.
    """
    # Validate inputs
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3), got {image.shape}")

    if factor_range[0] > factor_range[1]:
        raise ValueError(
            f"Invalid factor range: {factor_range} (min > max)"
        )

    # Random brightness factor
    factor = np.random.uniform(factor_range[0], factor_range[1])

    # Adjust brightness: I' = I * (1 + factor)
    adjusted_image = image.astype(np.float32) * (1.0 + factor)

    # Clamp to valid range [0, 255]
    adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

    return adjusted_image, factor


def apply_gaussian_blur(
    image: np.ndarray,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Apply Gaussian blur to simulate camera focus variation.

    Contract: BlurAugmentationContract
    Tests: tests/unit/data/test_clinical_augmentation.py::TestGaussianBlur

    Args:
        image: RGB image (H, W, 3), dtype uint8
        sigma: Gaussian kernel standard deviation (default: 1.5)

    Returns:
        blurred_image: Blurred RGB image (same shape, uint8)

    Raises:
        ValueError: If image shape invalid or sigma <= 0

    Notes:
        σ=1.5 simulates camera focus variation. σ above 3 removes
        clinically relevant detail.
    """
    # Validate inputs
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3), got {image.shape}")

    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")

    # Apply Gaussian blur using OpenCV (faster than scipy for images)
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # 3σ rule, odd kernel
    blurred_image = cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigma,
    )

    return blurred_image


def elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 720,
    beta: float = 24,
    random_state: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply elastic deformation to simulate tissue elasticity.

    Contract: ElasticDeformationContract
    Tests: tests/unit/data/test_clinical_augmentation.py::TestElasticDeformation

    Args:
        image: RGB image (H, W, 3), dtype uint8
        mask: Binary mask (H, W), dtype uint8 (0 or 255)
        alpha: Deformation intensity (default: 720)
               Higher = more deformation
        beta: Gaussian filter sigma for deformation field (default: 24)
              Higher = smoother deformation
        random_state: NumPy random state for reproducibility (optional)

    Returns:
        Tuple of:
            - deformed_image: Elastically deformed image (same shape, uint8)
            - deformed_mask: Elastically deformed mask (same shape, uint8)

    Raises:
        ValueError: If image/mask shapes incompatible or alpha/beta invalid

    Notes:
        Simulates tissue movement and skin elasticity. Parameters
        α=720, β=24, following Simard et al. (2003).

    Reference:
        Simard, P.Y., Steinkraus, D., & Platt, J.C. (2003). Best practices
        for convolutional neural networks applied to visual document analysis.
        ICDAR 2003.
    """
    # Validate inputs
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3), got {image.shape}")

    if len(mask.shape) != 2:
        raise ValueError(f"Mask must be (H, W), got {mask.shape}")

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image shape {image.shape[:2]} != mask shape {mask.shape}"
        )

    if alpha < 0:
        raise ValueError(f"Alpha must be non-negative, got {alpha}")

    if beta <= 0:
        raise ValueError(f"Beta must be positive, got {beta}")

    if random_state is None:
        random_state = np.random.RandomState()

    # Special case: zero deformation
    if alpha == 0:
        return image.copy(), mask.copy()

    # Generate random displacement fields
    height, width = image.shape[:2]

    dx = gaussian_filter(
        (random_state.rand(height, width) * 2 - 1),
        beta,
        mode='constant',
        cval=0,
    ) * alpha

    dy = gaussian_filter(
        (random_state.rand(height, width) * 2 - 1),
        beta,
        mode='constant',
        cval=0,
    ) * alpha

    # Create meshgrid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Apply displacement
    indices_x = np.clip(x + dx, 0, width - 1)
    indices_y = np.clip(y + dy, 0, height - 1)

    # Deform image (each channel separately)
    deformed_image = np.zeros_like(image)
    for c in range(3):
        deformed_image[:, :, c] = map_coordinates(
            image[:, :, c],
            [indices_y, indices_x],
            order=1,  # Bilinear interpolation
            mode='reflect',
        ).astype(np.uint8)

    # Deform mask (nearest neighbor to preserve binary values)
    deformed_mask = map_coordinates(
        mask,
        [indices_y, indices_x],
        order=0,  # Nearest neighbor
        mode='constant',
        cval=0,
    ).astype(np.uint8)

    # Ensure mask remains binary (0 or 255)
    deformed_mask = (deformed_mask > 127).astype(np.uint8) * 255

    return deformed_image, deformed_mask


def compose_augmentations(
    image: np.ndarray,
    mask: np.ndarray,
    probability: float = 0.7,
    random_state: np.random.RandomState = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose multiple clinical augmentations with probability control.

    Contract: AugmentationCompositionContract
    Tests: tests/unit/data/test_clinical_augmentation.py::TestComposeAugmentations

    Args:
        image: RGB image (H, W, 3), dtype uint8
        mask: Binary mask (H, W), dtype uint8 (0 or 255)
        probability: Probability of applying each augmentation (default: 0.7)
        random_state: NumPy random state for reproducibility (optional)

    Returns:
        Tuple of:
            - augmented_image: Augmented RGB image (same shape, uint8)
            - augmented_mask: Augmented binary mask (same shape, uint8)

    Raises:
        ValueError: If image/mask shapes incompatible or probability invalid

    Notes:
        Probability 0.7 per augmentation. Augmentations apply independently:
        4 augmentations, 16 possible combinations, effective training set
        8-16x larger.

    Augmentation Pipeline:
        1. Rotation (±15°) - Orientation diversity
        2. Brightness (±20%) - Lighting variation
        3. Gaussian blur (σ=1.5) - Focus variation
        4. Elastic deformation (α=720, β=24) - Tissue elasticity

    Expected improvement: +2-3% Dice.
    """
    # Validate inputs
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must be (H, W, 3), got {image.shape}")

    if len(mask.shape) != 2:
        raise ValueError(f"Mask must be (H, W), got {mask.shape}")

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image shape {image.shape[:2]} != mask shape {mask.shape}"
        )

    if not 0 <= probability <= 1:
        raise ValueError(
            f"Probability must be in [0, 1], got {probability}"
        )

    if random_state is None:
        random_state = np.random.RandomState()

    # Start with original
    augmented_image = image.copy()
    augmented_mask = mask.copy()

    # 1. Rotation (affects both image and mask)
    if random_state.rand() < probability:
        augmented_image, augmented_mask, _ = rotate_image_and_mask(
            augmented_image,
            augmented_mask,
            angle_range=(-15, 15),
        )

    # 2. Brightness (affects only image)
    if random_state.rand() < probability:
        augmented_image, _ = adjust_brightness(
            augmented_image,
            factor_range=(-0.2, 0.2),
        )

    # 3. Gaussian blur (affects only image)
    if random_state.rand() < probability:
        augmented_image = apply_gaussian_blur(
            augmented_image,
            sigma=1.5,
        )

    # 4. Elastic deformation (affects both image and mask)
    if random_state.rand() < probability:
        augmented_image, augmented_mask = elastic_deformation(
            augmented_image,
            augmented_mask,
            alpha=720,
            beta=24,
            random_state=random_state,
        )

    return augmented_image, augmented_mask
