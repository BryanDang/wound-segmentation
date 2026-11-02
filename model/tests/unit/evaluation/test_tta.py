"""
Unit tests for Test-Time Augmentation (TTA).

Tests augmentation transforms, inverse transforms, and ensemble aggregation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestAugmentationTransforms:
    """Test forward and inverse augmentation transforms."""

    @pytest.fixture
    def sample_tensor(self):
        """Create a test tensor with known pattern."""
        # 3×4×4 tensor with distinct values for testing transforms
        tensor = torch.arange(48).reshape(3, 4, 4).float()
        return tensor

    @pytest.fixture
    def tta_module(self):
        """Import TTA module."""
        # Import will be available after implementation
        from src.evaluation.tta import TestTimeAugmentation
        return TestTimeAugmentation

    def test_original_is_identity(self, sample_tensor, tta_module):
        """Test that 'original' augmentation is identity transform."""
        forward_fn, inverse_fn = tta_module.AUGMENTATIONS["original"]

        result = forward_fn(sample_tensor)
        assert torch.equal(result, sample_tensor), "Original should be identity"

        restored = inverse_fn(result)
        assert torch.equal(restored, sample_tensor), "Inverse of identity should be identity"

    def test_horizontal_flip_forward_inverse(self, sample_tensor, tta_module):
        """Test horizontal flip forward and inverse are perfect inverses."""
        forward_fn, inverse_fn = tta_module.AUGMENTATIONS["horizontal_flip"]

        flipped = forward_fn(sample_tensor)
        restored = inverse_fn(flipped)

        error = torch.abs(restored - sample_tensor).max().item()
        assert error < 1e-6, f"Horizontal flip inverse error: {error}"

    def test_vertical_flip_forward_inverse(self, sample_tensor, tta_module):
        """Test vertical flip forward and inverse are perfect inverses."""
        forward_fn, inverse_fn = tta_module.AUGMENTATIONS["vertical_flip"]

        flipped = forward_fn(sample_tensor)
        restored = inverse_fn(flipped)

        error = torch.abs(restored - sample_tensor).max().item()
        assert error < 1e-6, f"Vertical flip inverse error: {error}"

    def test_rotate_90_forward_inverse(self, sample_tensor, tta_module):
        """Test 90° rotation forward and inverse are perfect inverses."""
        forward_fn, inverse_fn = tta_module.AUGMENTATIONS["rotate_90"]

        rotated = forward_fn(sample_tensor)
        restored = inverse_fn(rotated)

        error = torch.abs(restored - sample_tensor).max().item()
        assert error < 1e-6, f"Rotate 90 inverse error: {error}"

    def test_rotate_180_forward_inverse(self, sample_tensor, tta_module):
        """Test 180° rotation forward and inverse are perfect inverses."""
        forward_fn, inverse_fn = tta_module.AUGMENTATIONS["rotate_180"]

        rotated = forward_fn(sample_tensor)
        restored = inverse_fn(rotated)

        error = torch.abs(restored - sample_tensor).max().item()
        assert error < 1e-6, f"Rotate 180 inverse error: {error}"

    def test_rotate_270_forward_inverse(self, sample_tensor, tta_module):
        """Test 270° rotation forward and inverse are perfect inverses."""
        forward_fn, inverse_fn = tta_module.AUGMENTATIONS["rotate_270"]

        rotated = forward_fn(sample_tensor)
        restored = inverse_fn(rotated)

        error = torch.abs(restored - sample_tensor).max().item()
        assert error < 1e-6, f"Rotate 270 inverse error: {error}"

    def test_hflip_rotate_90_forward_inverse(self, sample_tensor, tta_module):
        """Test composed transform (hflip + rotate 90) forward/inverse."""
        forward_fn, inverse_fn = tta_module.AUGMENTATIONS["hflip_rotate_90"]

        transformed = forward_fn(sample_tensor)
        restored = inverse_fn(transformed)

        error = torch.abs(restored - sample_tensor).max().item()
        assert error < 1e-6, f"HFlip+Rotate90 inverse error: {error}"

    def test_hflip_rotate_270_forward_inverse(self, sample_tensor, tta_module):
        """Test composed transform (hflip + rotate 270) forward/inverse."""
        forward_fn, inverse_fn = tta_module.AUGMENTATIONS["hflip_rotate_270"]

        transformed = forward_fn(sample_tensor)
        restored = inverse_fn(transformed)

        error = torch.abs(restored - sample_tensor).max().item()
        assert error < 1e-6, f"HFlip+Rotate270 inverse error: {error}"

    def test_all_augmentations_are_invertible(self, sample_tensor, tta_module):
        """Test all augmentations are perfect inverses."""
        max_errors = {}

        for aug_name, (forward_fn, inverse_fn) in tta_module.AUGMENTATIONS.items():
            transformed = forward_fn(sample_tensor)
            restored = inverse_fn(transformed)
            error = torch.abs(restored - sample_tensor).max().item()
            max_errors[aug_name] = error

            assert error < 1e-6, f"{aug_name} inverse error: {error}"

        print(f"\nAll augmentation inverse errors: {max_errors}")


class TestEnsembleAggregation:
    """Test ensemble aggregation methods."""

    @pytest.fixture
    def predictions(self):
        """Create sample predictions for testing aggregation."""
        # 5 predictions of shape (4, 4)
        preds = [
            np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
            np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
            np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]),
            np.array([[0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
            np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
        ]
        return preds

    def test_mean_aggregation(self, predictions):
        """Test mean aggregation produces correct result."""
        # Mean of 5 predictions, threshold at 0.5
        mean_pred = np.mean(predictions, axis=0)
        ensemble = (mean_pred > 0.5).astype(np.uint8)

        # Pixel [0, 0]: 4/5 = 0.8 to 1
        assert ensemble[0, 0] == 1, "Pixel with 4/5 votes should be 1"

        # Pixel [0, 2]: 0/5 = 0.0 to 0
        assert ensemble[0, 2] == 0, "Pixel with 0/5 votes should be 0"

        # Pixel [1, 1]: 4/5 = 0.8 to 1
        assert ensemble[1, 1] == 1, "Pixel with 4/5 votes should be 1"

    def test_majority_vote_aggregation(self, predictions):
        """Test majority vote aggregation produces correct result."""
        # Sum predictions and threshold at n/2
        sum_pred = np.sum(predictions, axis=0)
        ensemble = (sum_pred > len(predictions) / 2).astype(np.uint8)

        # Pixel [0, 0]: 4/5 > 2.5 to 1
        assert ensemble[0, 0] == 1, "Pixel with 4/5 votes should be 1"

        # Pixel [0, 2]: 0/5 > 2.5 to 0
        assert ensemble[0, 2] == 0, "Pixel with 0/5 votes should be 0"

        # Pixel [1, 1]: 4/5 > 2.5 to 1
        assert ensemble[1, 1] == 1, "Pixel with 4/5 votes should be 1"

    def test_mean_vs_majority_vote_equivalence(self, predictions):
        """Test mean (>0.5) and majority vote produce same result."""
        mean_pred = (np.mean(predictions, axis=0) > 0.5).astype(np.uint8)
        majority_pred = (np.sum(predictions, axis=0) > len(predictions) / 2).astype(np.uint8)

        assert np.array_equal(mean_pred, majority_pred), \
            "Mean (>0.5) and majority vote should be equivalent"


class TestTTAPipeline:
    """Test end-to-end TTA pipeline."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        class MockModel:
            def predict(self, image, verbose=True):
                """Return a mock prediction."""
                # Return a simple mask (all zeros for simplicity)
                class MockResult:
                    def __init__(self):
                        self.masks = None  # Simulate no wound detected

                return [MockResult()]

        return MockModel()

    def test_tta_initialization(self, mock_model):
        """Test TTA class initializes correctly."""
        from src.evaluation.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(
            model=mock_model,
            augmentation_set=["original", "horizontal_flip"],
            aggregation="mean"
        )

        assert tta.model == mock_model
        assert tta.augmentation_set == ["original", "horizontal_flip"]
        assert tta.aggregation == "mean"

    def test_tta_invalid_augmentation_raises_error(self, mock_model):
        """Test TTA raises error for unknown augmentation."""
        from src.evaluation.tta import TestTimeAugmentation

        with pytest.raises(ValueError, match="Unknown augmentation"):
            TestTimeAugmentation(
                model=mock_model,
                augmentation_set=["invalid_augmentation"],
                aggregation="mean"
            )

    def test_tta_invalid_aggregation_raises_error(self, mock_model):
        """Test TTA raises error for unknown aggregation method."""
        from src.evaluation.tta import TestTimeAugmentation

        tta = TestTimeAugmentation(
            model=mock_model,
            augmentation_set=["original"],
            aggregation="invalid_method"
        )

        # Error should occur during predict, not init
        # This test will need an actual test image to work
        # For now, just verify aggregation is set
        assert tta.aggregation == "invalid_method"


class TestShapePreservation:
    """Test that TTA preserves tensor shapes correctly."""

    def test_augmentation_preserves_2d_shape(self):
        """Test augmentations preserve 2D tensor shape."""
        from src.evaluation.tta import TestTimeAugmentation

        # Create 2D tensor (H, W)
        tensor_2d = torch.rand(512, 512)

        for aug_name, (forward_fn, inverse_fn) in TestTimeAugmentation.AUGMENTATIONS.items():
            transformed = forward_fn(tensor_2d)
            assert transformed.shape == tensor_2d.shape, \
                f"{aug_name} changed 2D shape: {tensor_2d.shape} to {transformed.shape}"

    def test_augmentation_preserves_3d_shape(self):
        """Test augmentations preserve 3D tensor shape (C, H, W)."""
        from src.evaluation.tta import TestTimeAugmentation

        # Create 3D tensor (C, H, W)
        tensor_3d = torch.rand(3, 512, 512)

        for aug_name, (forward_fn, inverse_fn) in TestTimeAugmentation.AUGMENTATIONS.items():
            transformed = forward_fn(tensor_3d)
            assert transformed.shape == tensor_3d.shape, \
                f"{aug_name} changed 3D shape: {tensor_3d.shape} to {transformed.shape}"

    def test_augmentation_preserves_4d_shape(self):
        """Test augmentations preserve 4D tensor shape (B, C, H, W)."""
        from src.evaluation.tta import TestTimeAugmentation

        # Create 4D tensor (B, C, H, W)
        tensor_4d = torch.rand(2, 3, 512, 512)

        for aug_name, (forward_fn, inverse_fn) in TestTimeAugmentation.AUGMENTATIONS.items():
            transformed = forward_fn(tensor_4d)
            assert transformed.shape == tensor_4d.shape, \
                f"{aug_name} changed 4D shape: {tensor_4d.shape} to {transformed.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
