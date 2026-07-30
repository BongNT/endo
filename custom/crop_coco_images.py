import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

# ================= CONFIG =================
SRC_ROOT = "/home/bongmedai/Endo/datasets/endo_coco_seg"
DST_ROOT = "/home/bongmedai/Endo/datasets/endo_coco_crop"

SPLITS = ["train", "val", "test"]
IMG_EXTS = (".jpg", ".jpeg", ".png")
# =========================================


def crop_endo_safe(img, black_v_thresh=15, black_ratio_thresh=0.3, border_check_ratio=0.08):

    H, W = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]

    black_mask = v < black_v_thresh
    endo_mask = ~black_mask

    # ### NEW: Remove text and small noise using Morphological Opening
    # This kernel size (5x5) is big enough to eat the thin text
    # but small enough to preserve the main image.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # We use this 'clean' mask for calculations, but apply coordinates to original img
    clean_mask = cv2.morphologyEx(endo_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=4)
    # cv2.imwrite("clean_mask.png", clean_mask*255)
    # Update logic to use clean_mask for projections
    # Note: We invert clean_mask back to get the equivalent 'black_mask' logic if needed,
    # or just use clean_mask directly for sums.

    # Row / column black ratios (Use original mask for safety or clean_mask?
    # Better to use clean_mask to ignore text in border checks too)
    row_black_ratio = 1.0 - np.mean(clean_mask, axis=1)  # Inverted logic
    1.0 - np.mean(clean_mask, axis=0)

    # Projections (Use clean_mask)
    row_sum = np.sum(clean_mask, axis=1).astype(np.float32)
    col_sum = np.sum(clean_mask, axis=0).astype(np.float32)

    # --- The rest of your logic remains exactly the same ---
    row_sum /= max(row_sum.max(), 1)
    col_sum /= max(col_sum.max(), 1)

    row_grad = np.gradient(row_sum)
    col_grad = np.gradient(col_sum)

    # Candidate boundaries
    top_cand = np.argmax(row_grad)
    bottom_cand = np.argmin(row_grad)
    left_cand = np.argmax(col_grad)
    right_cand = np.argmin(col_grad)

    top_cand, bottom_cand = sorted([top_cand, bottom_cand])
    left_cand, right_cand = sorted([left_cand, right_cand])

    # Border validation
    border_rows = int(border_check_ratio * H)
    int(border_check_ratio * W)

    # Top
    if np.mean(row_black_ratio[:border_rows]) < black_ratio_thresh:
        top = 0
    else:
        top = top_cand

    # Bottom
    if np.mean(row_black_ratio[-border_rows:]) < black_ratio_thresh:
        bottom = H
    else:
        bottom = bottom_cand

    # # Left
    # if np.mean(col_black_ratio[:border_cols]) < black_ratio_thresh:
    #     left = 0
    # else:
    #     left = left_cand

    # # Right
    # if np.mean(col_black_ratio[-border_cols:]) < black_ratio_thresh:
    #     right = W
    # else:
    #     right = right_cand

    # Final sanity check
    if bottom - top < 0.5 * H:
        top, bottom = 0, H
    # if right - left < 0.5 * W:
    #     left, right = 0, W

    return top, bottom, left_cand, right_cand


def process_label(label_path, bbox, orig_shape, new_shape):
    """Update YOLO segmentation labels after cropping."""
    top, _bottom, left, _right = bbox
    H, W = orig_shape
    new_H, new_W = new_shape

    new_lines = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = parts[0]
            coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)

            # normalized → absolute
            coords[:, 0] *= W
            coords[:, 1] *= H

            # apply crop
            coords[:, 0] -= left
            coords[:, 1] -= top

            # keep only points inside crop
            valid = (coords[:, 0] >= 0) & (coords[:, 0] <= new_W) & (coords[:, 1] >= 0) & (coords[:, 1] <= new_H)

            if np.sum(valid) < 3:
                continue  # invalid polygon

            coords = coords[valid]

            # absolute → normalized
            coords[:, 0] /= new_W
            coords[:, 1] /= new_H

            flat = " ".join(f"{v:.6f}" for v in coords.flatten())
            new_lines.append(f"{cls} {flat}")

    return new_lines


if __name__ == "__main__":
    # ================ MAIN =================
    for split in SPLITS:
        img_in_dir = os.path.join(SRC_ROOT, "images", split)
        lbl_in_dir = os.path.join(SRC_ROOT, "labels", split)

        img_out_dir = os.path.join(DST_ROOT, "images", split)
        lbl_out_dir = os.path.join(DST_ROOT, "labels", split)

        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        imgs = [f for f in os.listdir(img_in_dir) if f.lower().endswith(IMG_EXTS)]

        for fname in tqdm(imgs, desc=f"Processing {split}"):
            img_path = os.path.join(img_in_dir, fname)
            lbl_path = os.path.join(lbl_in_dir, fname.replace(".png", ".txt"))

            img = cv2.imread(img_path)
            if img is None:
                continue

            bbox = crop_endo_safe(img, black_ratio_thresh=0.3, border_check_ratio=0.08)
            if bbox is None:
                continue

            top, bottom, left, right = bbox
            cropped = img[top:bottom, left:right]

            cv2.imwrite(os.path.join(img_out_dir, fname), cropped)

            if not os.path.exists(lbl_path):
                open(os.path.join(lbl_out_dir, fname.replace(".png", ".txt")), "w").close()
                continue

            new_labels = process_label(
                lbl_path,
                bbox,
                orig_shape=img.shape[:2],
                new_shape=cropped.shape[:2],
            )

            with open(os.path.join(lbl_out_dir, fname.replace(".png", ".txt")), "w") as f:
                f.write("\n".join(new_labels))

    # copy dataset.yaml
    try:
        shutil.copy(
            os.path.join(SRC_ROOT, "dataset.yaml"),
            os.path.join(DST_ROOT, "dataset.yaml"),
        )
    except Exception as e:
        print(f"Failed to copy dataset.yaml: {e}")

    print("✅ Cropping + label remapping completed!")
