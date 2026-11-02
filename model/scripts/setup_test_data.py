import os

import cv2
import numpy as np

# Simulated data root
base_img_path = "simulated_kaggle_data/input/wound-segmentation-images/train_images/"
base_mask_path = "simulated_kaggle_data/input/wound-segmentation-images/train_masks/"

# Directories
os.makedirs(base_img_path, exist_ok=True)
os.makedirs(base_mask_path, exist_ok=True)

# --- Case 1: Image with a MISSING mask ---
print("Creating image with no mask...")
img_no_mask = np.zeros((50, 50, 3), dtype=np.uint8)  # 50x50 black image
cv2.imwrite(os.path.join(base_img_path, "fusc_0004_no_mask.png"), img_no_mask)
print(f" -> Created fusc_0004_no_mask.png in {base_img_path}")


# --- Case 2: Image with an EMPTY mask ---
print("Creating image with an empty (all black) mask...")
# 60x60 noise image
img_with_empty_mask = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
# Matching empty mask
empty_mask = np.zeros((60, 60), dtype=np.uint8)

cv2.imwrite(os.path.join(base_img_path, "fusc_0005_empty_mask.png"), img_with_empty_mask)
cv2.imwrite(os.path.join(base_mask_path, "fusc_0005_empty_mask.png"), empty_mask)
print(
    f" -> Created fusc_0005_empty_mask.png in {base_img_path} and an empty mask in {base_mask_path}"
)

print("\nTest data setup complete.")
