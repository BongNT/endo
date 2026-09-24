import cv2
import numpy as np


def overlay_mask(
    img_path,
    mask_path,
    out_path="overlay.png",
    color=(0, 255, 255),  # Red in BGR
    alpha=0.4,
):
    # Read image (BGR)
    img = cv2.imread(img_path)
    print("img shape:", img.shape)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Read mask (grayscale)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print("mask shape:", mask.shape, np.unique(mask))

    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    # Ensure same size
    if img.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Binarize mask (handle 0/1 or 0/255)
    mask_bin = (mask > 0).astype(np.uint8)

    # Create colored mask
    color_mask = np.zeros_like(img)
    color_mask[mask_bin == 1] = color

    # Overlay
    overlay = img.copy()
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)

    # Save
    cv2.imwrite(out_path, overlay)
    print(f"Saved overlay to {out_path}")

    return overlay


if __name__ == "__main__":
    img_path = "/home/bongmedai/Endo/datasets/nnUNet_raw/Dataset001_Endo2D/imagesTr/case0001_0000.png"
    mask_path = "/home/bongmedai/Endo/datasets/nnUNet_raw/Dataset001_Endo2D/labelsTr/case0001.png"
    out_path = "/home/bongmedai/Endo/datasets/test/overlay.png"

    overlay_mask(img_path, mask_path, out_path)
