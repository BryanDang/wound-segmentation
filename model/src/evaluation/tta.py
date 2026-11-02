"""
Test-Time Augmentation (TTA) for YOLOv8 Segmentation.

Applies multiple augmentations at inference time and ensembles predictions.
"""

import torch
import numpy as np
from typing import List, Tuple, Callable, Union
from pathlib import Path
import cv2


class TestTimeAugmentation:
    """
    Test-Time Augmentation for YOLOv8 segmentation.

    Applies multiple augmentations at inference time and ensembles
    predictions.

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("best.pt")
        >>> tta = TestTimeAugmentation(model, aggregation="mean")
        >>> mask = tta.predict("image.png")
        >>> # Returns ensembled prediction from 8 augmentations

    Args:
        model: YOLOv8 model instance
        augmentation_set: List of augmentation names (default: all 8)
        aggregation: 'majority_vote' or 'mean' (default: 'mean')

    Returns:
        Binary segmentation mask (H, W) as numpy array
    """

    # Define augmentation transforms (forward, inverse) pairs
    AUGMENTATIONS = {
        "original": (
            lambda x: x,
            lambda x: x
        ),
        "horizontal_flip": (
            lambda x: torch.flip(x, dims=[-1]),
            lambda x: torch.flip(x, dims=[-1])
        ),
        "vertical_flip": (
            lambda x: torch.flip(x, dims=[-2]),
            lambda x: torch.flip(x, dims=[-2])
        ),
        "rotate_90": (
            lambda x: torch.rot90(x, k=1, dims=[-2, -1]),
            lambda x: torch.rot90(x, k=-1, dims=[-2, -1])
        ),
        "rotate_180": (
            lambda x: torch.rot90(x, k=2, dims=[-2, -1]),
            lambda x: torch.rot90(x, k=-2, dims=[-2, -1])
        ),
        "rotate_270": (
            lambda x: torch.rot90(x, k=3, dims=[-2, -1]),
            lambda x: torch.rot90(x, k=-3, dims=[-2, -1])
        ),
        "hflip_rotate_90": (
            lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=1, dims=[-2, -1]),
            lambda x: torch.flip(torch.rot90(x, k=-1, dims=[-2, -1]), dims=[-1])
        ),
        "hflip_rotate_270": (
            lambda x: torch.rot90(torch.flip(x, dims=[-1]), k=3, dims=[-2, -1]),
            lambda x: torch.flip(torch.rot90(x, k=-3, dims=[-2, -1]), dims=[-1])
        ),
    }

    def __init__(
        self,
        model,
        augmentation_set: List[str] = None,
        aggregation: str = "mean"
    ):
        """
        Initialize TTA.

        Args:
            model: YOLOv8 model instance
            augmentation_set: List of augmentation names (None = use all)
            aggregation: 'mean' or 'majority_vote'

        Raises:
            ValueError: If unknown augmentation in augmentation_set
        """
        self.model = model
        self.augmentation_set = augmentation_set or list(self.AUGMENTATIONS.keys())
        self.aggregation = aggregation

        # Validate augmentation set
        for aug_name in self.augmentation_set:
            if aug_name not in self.AUGMENTATIONS:
                raise ValueError(
                    f"Unknown augmentation: {aug_name}. "
                    f"Available: {list(self.AUGMENTATIONS.keys())}"
                )

    def predict(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Predict with TTA ensemble.

        Args:
            image_path: Path to input image

        Returns:
            mask: (H, W) binary segmentation mask (0=background, 1=wound)

        Process:
            1. Load image
            2. For each augmentation:
                a. Apply forward transform
                b. Run YOLO prediction
                c. Extract mask (native YOLO resolution)
                d. Apply inverse transform
                e. Resize to original shape
            3. Aggregate predictions (mean or majority vote)
            4. Return ensembled mask
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # Convert to RGB (YOLO expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store original shape for final resize
        original_shape = image.shape[:2]  # (H, W)

        # Collect predictions from all augmentations
        predictions = []

        for aug_name in self.augmentation_set:
            forward_fn, inverse_fn = self.AUGMENTATIONS[aug_name]

            # Apply augmentation to IMAGE (not tensor yet)
            aug_image = self._apply_augmentation_to_image(image, forward_fn)

            # Predict with YOLO
            result = self.model.predict(aug_image, verbose=False)[0]

            # Extract mask from YOLO result (NO RESIZING - keep native resolution)
            pred_mask = self._extract_mask_native(result)

            # Convert to tensor for inverse transform
            pred_tensor = torch.from_numpy(pred_mask).float()

            # Reverse augmentation (this restores correct spatial orientation)
            pred_reversed = inverse_fn(pred_tensor)

            # Convert back to numpy
            pred_numpy = pred_reversed.numpy()

            # NOW resize to original shape (after inverse transform)
            if pred_numpy.shape != original_shape:
                pred_resized = cv2.resize(
                    pred_numpy,
                    (original_shape[1], original_shape[0]),  # (W, H) for cv2
                    interpolation=cv2.INTER_LINEAR
                )
                # Re-threshold after resize
                pred_resized = (pred_resized > 0.5).astype(np.uint8)
            else:
                pred_resized = pred_numpy.astype(np.uint8)

            predictions.append(pred_resized)

        # Aggregate predictions
        ensemble_mask = self._aggregate_predictions(predictions)

        return ensemble_mask

    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[np.ndarray]:
        """
        Predict on multiple images with TTA.

        Args:
            image_paths: List of paths to input images

        Returns:
            masks: List of (H, W) binary segmentation masks
        """
        return [self.predict(path) for path in image_paths]

    def _apply_augmentation_to_image(
        self,
        image: np.ndarray,
        transform_fn: Callable
    ) -> np.ndarray:
        """
        Apply augmentation transform to image.

        Args:
            image: (H, W, 3) numpy array
            transform_fn: Torch augmentation function

        Returns:
            aug_image: (H, W, 3) numpy array (augmented)
        """
        # Convert to torch tensor (H, W, 3) to (3, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        # Apply transform
        aug_tensor = transform_fn(image_tensor)

        # Convert back to numpy (3, H, W) to (H, W, 3)
        aug_image = aug_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

        return aug_image

    def _extract_mask_native(
        self,
        yolo_result
    ) -> np.ndarray:
        """
        Extract binary mask from YOLO result at native resolution.

        Args:
            yolo_result: YOLO prediction result

        Returns:
            mask: (H, W) binary mask at YOLO's native resolution (0 or 1)

        Note:
            Returns mask WITHOUT resizing. Caller should resize after
            applying inverse augmentation transform.
        """
        # Check if any masks detected
        if yolo_result.masks is None or len(yolo_result.masks.data) == 0:
            # No wound detected - return zeros at a default size
            # YOLO typically returns 160x160 masks
            return np.zeros((160, 160), dtype=np.uint8)

        # Extract first mask (highest confidence)
        mask = yolo_result.masks.data[0].cpu().numpy()

        # Threshold to binary
        binary_mask = (mask > 0.5).astype(np.uint8)

        return binary_mask

    def _extract_mask(
        self,
        yolo_result,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract binary mask from YOLO result.

        Args:
            yolo_result: YOLO prediction result
            target_shape: (H, W) target shape

        Returns:
            mask: (H, W) binary mask (0 or 1)

        Note:
            This method is kept for backward compatibility.
            TTA pipeline now uses _extract_mask_native() instead.
        """
        # Check if any masks detected
        if yolo_result.masks is None or len(yolo_result.masks.data) == 0:
            # No wound detected - return all zeros
            return np.zeros(target_shape, dtype=np.uint8)

        # Extract first mask (highest confidence)
        mask = yolo_result.masks.data[0].cpu().numpy()

        # Resize to target shape if needed
        if mask.shape != target_shape:
            mask = cv2.resize(
                mask,
                (target_shape[1], target_shape[0]),  # (W, H) for cv2
                interpolation=cv2.INTER_LINEAR
            )

        # Threshold to binary
        binary_mask = (mask > 0.5).astype(np.uint8)

        return binary_mask

    def _aggregate_predictions(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate predictions using specified method.

        Args:
            predictions: List of (H, W) binary masks

        Returns:
            ensemble_mask: (H, W) binary mask

        Raises:
            ValueError: If unknown aggregation method
        """
        # Stack predictions
        pred_stack = np.stack(predictions, axis=0)  # (N, H, W)

        if self.aggregation == "mean":
            # Mean probability, threshold at 0.5
            ensemble_mask = (np.mean(pred_stack, axis=0) > 0.5).astype(np.uint8)

        elif self.aggregation == "majority_vote":
            # Majority vote (>50% of predictions)
            sum_pred = np.sum(pred_stack, axis=0)
            ensemble_mask = (sum_pred > len(predictions) / 2).astype(np.uint8)

        else:
            raise ValueError(
                f"Unknown aggregation: {self.aggregation}. "
                f"Available: ['mean', 'majority_vote']"
            )

        return ensemble_mask

    def get_config(self) -> dict:
        """
        Return TTA configuration for logging.

        Returns:
            config: Dictionary with TTA settings
        """
        return {
            "augmentation_set": self.augmentation_set,
            "n_augmentations": len(self.augmentation_set),
            "aggregation": self.aggregation,
        }


def validate_augmentation_inverses(verbose: bool = True) -> dict:
    """
    Validate that all augmentation inverse transforms are correct.

    Args:
        verbose: Print validation results

    Returns:
        results: Dict with max errors for each augmentation

    Example:
        >>> errors = validate_augmentation_inverses(verbose=True)
        >>> assert all(err < 1e-6 for err in errors.values())
    """
    # Create test tensor with known pattern
    test_tensor = torch.arange(48).reshape(3, 4, 4).float()

    max_errors = {}

    for aug_name, (forward_fn, inverse_fn) in TestTimeAugmentation.AUGMENTATIONS.items():
        # Apply forward then inverse
        transformed = forward_fn(test_tensor)
        restored = inverse_fn(transformed)

        # Compute max error
        error = torch.abs(restored - test_tensor).max().item()
        max_errors[aug_name] = error

        if verbose:
            status = "PASS" if error < 1e-6 else "FAIL"
            print(f"{status} {aug_name:20s}: error = {error:.2e}")

    return max_errors


if __name__ == "__main__":
    # Validate augmentation inverses
    print("Validating TTA augmentation transforms...")
    errors = validate_augmentation_inverses(verbose=True)

    all_valid = all(err < 1e-6 for err in errors.values())
    print(f"\n{'PASS' if all_valid else 'FAIL'} All augmentations valid: {all_valid}")
