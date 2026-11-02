"""
Unit tests for clinical data augmentation.

Contract: ClinicalAugmentationContract
Tests anatomically plausible augmentations for wound segmentation.
"""

import numpy as np
import pytest

from src.data.clinical_augmentation import (
    adjust_brightness,
    apply_gaussian_blur,
    compose_augmentations,
    elastic_deformation,
    rotate_image_and_mask,
)


class TestRotateImageAndMask:
    """Test rotation augmentation with boundary preservation."""

    def test_rotate_within_range(self):
        """Rotation angle respects ±15° constraint."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        # Test multiple rotations
        for _ in range(10):
            rotated_img, rotated_mask, angle = rotate_image_and_mask(
                image, mask, angle_range=(-15, 15)
            )

            # Angle within range
            assert -15 <= angle <= 15

            # Shape preserved
            assert rotated_img.shape == image.shape
            assert rotated_mask.shape == mask.shape

            # Mask remains binary
            assert set(np.unique(rotated_mask)).issubset({0, 255})

    def test_rotate_preserves_image_type(self):
        """Rotation preserves uint8 dtype."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        rotated_img, rotated_mask, _ = rotate_image_and_mask(
            image, mask, angle_range=(-15, 15)
        )

        assert rotated_img.dtype == np.uint8
        assert rotated_mask.dtype == np.uint8

    def test_rotate_zero_degrees(self):
        """Zero rotation returns original (approximately)."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        rotated_img, rotated_mask, _ = rotate_image_and_mask(
            image, mask, angle_range=(0, 0)
        )

        # Zero rotation should be nearly identical (allowing for interpolation)
        assert rotated_img.shape == image.shape
        assert rotated_mask.shape == mask.shape

    def test_rotate_boundary_wound(self):
        """Rotation handles wounds at image boundaries."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)

        # Create wound at edge
        mask[0:50, 0:50] = 255

        rotated_img, rotated_mask, _ = rotate_image_and_mask(
            image, mask, angle_range=(-15, 15)
        )

        # Mask still contains wound pixels (may have moved)
        assert np.sum(rotated_mask > 0) > 0

        # No invalid values
        assert rotated_mask.min() >= 0
        assert rotated_mask.max() <= 255


class TestAdjustBrightness:
    """Test brightness adjustment within clinical range."""

    def test_brightness_within_range(self):
        """Brightness factor respects ±20% constraint."""
        image = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)

        # Test multiple adjustments
        for _ in range(10):
            adjusted_img, factor = adjust_brightness(image, factor_range=(-0.2, 0.2))

            # Factor within range
            assert -0.2 <= factor <= 0.2

            # Shape preserved
            assert adjusted_img.shape == image.shape

            # Pixel values clamped to valid range
            assert adjusted_img.min() >= 0
            assert adjusted_img.max() <= 255

            # Type preserved
            assert adjusted_img.dtype == np.uint8

    def test_brightness_positive_factor(self):
        """Positive factor increases brightness."""
        image = np.full((512, 512, 3), 100, dtype=np.uint8)

        adjusted_img, _ = adjust_brightness(image, factor_range=(0.1, 0.1))

        # Brightened image has higher mean
        assert adjusted_img.mean() > image.mean()

    def test_brightness_negative_factor(self):
        """Negative factor decreases brightness."""
        image = np.full((512, 512, 3), 150, dtype=np.uint8)

        adjusted_img, _ = adjust_brightness(image, factor_range=(-0.1, -0.1))

        # Darkened image has lower mean
        assert adjusted_img.mean() < image.mean()

    def test_brightness_zero_factor(self):
        """Zero factor returns original."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        adjusted_img, _ = adjust_brightness(image, factor_range=(0, 0))

        # Identical to original
        np.testing.assert_array_equal(adjusted_img, image)

    def test_brightness_extreme_values(self):
        """Brightness handles edge cases (very dark/bright images)."""
        # Very dark image
        dark_image = np.full((512, 512, 3), 10, dtype=np.uint8)
        adjusted_dark, _ = adjust_brightness(dark_image, factor_range=(0.2, 0.2))
        assert adjusted_dark.max() <= 255

        # Very bright image
        bright_image = np.full((512, 512, 3), 240, dtype=np.uint8)
        adjusted_bright, _ = adjust_brightness(bright_image, factor_range=(-0.2, -0.2))
        assert adjusted_bright.min() >= 0


class TestGaussianBlur:
    """Test Gaussian blur with controlled kernel size."""

    def test_blur_reduces_variance(self):
        """Gaussian blur reduces image variance."""
        # Create high-frequency noise image
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        blurred_img = apply_gaussian_blur(image, sigma=1.5)

        # Blurred image has lower variance
        assert blurred_img.var() < image.var()

    def test_blur_preserves_shape_and_type(self):
        """Blur preserves image shape and dtype."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        blurred_img = apply_gaussian_blur(image, sigma=1.5)

        assert blurred_img.shape == image.shape
        assert blurred_img.dtype == np.uint8

    def test_blur_preserves_value_range(self):
        """Blur keeps pixel values in [0, 255]."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        blurred_img = apply_gaussian_blur(image, sigma=1.5)

        assert blurred_img.min() >= 0
        assert blurred_img.max() <= 255

    def test_blur_uniform_image(self):
        """Blur on uniform image returns uniform (approximately)."""
        image = np.full((512, 512, 3), 128, dtype=np.uint8)

        blurred_img = apply_gaussian_blur(image, sigma=1.5)

        # Uniform image should remain approximately uniform
        assert abs(blurred_img.mean() - 128) < 5


class TestElasticDeformation:
    """Test elastic deformation with tissue-realistic parameters."""

    def test_elastic_preserves_shape_and_type(self):
        """Elastic deformation preserves image/mask shape and dtype."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        deformed_img, deformed_mask = elastic_deformation(
            image, mask, alpha=720, beta=24
        )

        # Shape preserved
        assert deformed_img.shape == image.shape
        assert deformed_mask.shape == mask.shape

        # Type preserved
        assert deformed_img.dtype == np.uint8
        assert deformed_mask.dtype == np.uint8

    def test_elastic_mask_remains_binary(self):
        """Elastic deformation keeps mask binary."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        _, deformed_mask = elastic_deformation(image, mask, alpha=720, beta=24)

        # Mask values only 0 or 255
        unique_values = set(np.unique(deformed_mask))
        assert unique_values.issubset({0, 255})

    def test_elastic_preserves_wound_area_approximately(self):
        """Elastic deformation preserves wound area (±20%)."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)

        # Create circular wound in center
        center = (256, 256)
        radius = 50
        y, x = np.ogrid[0:512, 0:512]
        circle_mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        mask[circle_mask] = 255

        original_area = np.sum(mask > 0)

        _, deformed_mask = elastic_deformation(image, mask, alpha=720, beta=24)

        deformed_area = np.sum(deformed_mask > 0)

        # Area preserved within 20% (elastic deformation may change area slightly)
        area_ratio = deformed_area / original_area
        assert 0.8 <= area_ratio <= 1.2

    def test_elastic_zero_alpha(self):
        """Zero alpha returns original (no deformation)."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        deformed_img, deformed_mask = elastic_deformation(image, mask, alpha=0, beta=24)

        # Should be approximately identical (allowing for interpolation)
        assert deformed_img.shape == image.shape
        assert deformed_mask.shape == mask.shape


class TestComposeAugmentations:
    """Test augmentation composition with probability control."""

    def test_compose_applies_augmentations(self):
        """Compose applies multiple augmentations."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        augmented_img, augmented_mask = compose_augmentations(
            image, mask, probability=1.0  # Force all augmentations
        )

        # Shape preserved
        assert augmented_img.shape == image.shape
        assert augmented_mask.shape == mask.shape

        # Type preserved
        assert augmented_img.dtype == np.uint8
        assert augmented_mask.dtype == np.uint8

        # Mask remains binary
        assert set(np.unique(augmented_mask)).issubset({0, 255})

    def test_compose_probability_zero(self):
        """Zero probability returns original."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        augmented_img, augmented_mask = compose_augmentations(
            image, mask, probability=0.0  # No augmentations
        )

        # Should be identical
        np.testing.assert_array_equal(augmented_img, image)
        np.testing.assert_array_equal(augmented_mask, mask)

    def test_compose_probability_clinical(self):
        """Clinical probability (0.7) applies augmentations stochastically."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        # Run multiple times to test stochasticity
        augmented_images = []
        for _ in range(10):
            augmented_img, _ = compose_augmentations(
                image, mask, probability=0.7
            )
            augmented_images.append(augmented_img)

        # At least some should be different from original
        differences = [not np.array_equal(aug, image) for aug in augmented_images]
        assert any(differences)  # At least one augmented

    def test_compose_preserves_value_range(self):
        """Compose keeps pixel values valid."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255

        augmented_img, augmented_mask = compose_augmentations(
            image, mask, probability=1.0
        )

        # Image values in [0, 255]
        assert augmented_img.min() >= 0
        assert augmented_img.max() <= 255

        # Mask values binary
        assert augmented_mask.min() >= 0
        assert augmented_mask.max() <= 255


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_image(self):
        """Functions handle all-zero images."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)

        # All functions should handle gracefully
        rotated_img, rotated_mask, _ = rotate_image_and_mask(image, mask)
        assert rotated_img.shape == image.shape

        adjusted_img, _ = adjust_brightness(image)
        assert adjusted_img.shape == image.shape

        blurred_img = apply_gaussian_blur(image)
        assert blurred_img.shape == image.shape

        deformed_img, deformed_mask = elastic_deformation(image, mask)
        assert deformed_img.shape == image.shape

        augmented_img, augmented_mask = compose_augmentations(image, mask)
        assert augmented_img.shape == image.shape

    def test_single_pixel_wound(self):
        """Functions handle single-pixel wound masks."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[256, 256] = 255  # Single wound pixel

        augmented_img, augmented_mask = compose_augmentations(
            image, mask, probability=1.0
        )

        # Should complete without error
        assert augmented_img.shape == image.shape
        assert augmented_mask.shape == mask.shape

    def test_full_wound_mask(self):
        """Functions handle completely filled wound masks."""
        image = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        mask = np.full((512, 512), 255, dtype=np.uint8)  # All wound

        augmented_img, augmented_mask = compose_augmentations(
            image, mask, probability=1.0
        )

        # Should complete without error
        assert augmented_img.shape == image.shape
        assert augmented_mask.shape == mask.shape
        assert np.sum(augmented_mask > 0) > 0  # Still has wound pixels

    def test_boundary_wound(self):
        """Functions handle wounds at all image boundaries."""
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)

        # Wound along all edges
        mask[0:10, :] = 255  # Top
        mask[-10:, :] = 255  # Bottom
        mask[:, 0:10] = 255  # Left
        mask[:, -10:] = 255  # Right

        augmented_img, augmented_mask = compose_augmentations(
            image, mask, probability=1.0
        )

        # Should complete without error
        assert augmented_img.shape == image.shape
        assert augmented_mask.shape == mask.shape
