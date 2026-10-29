import argparse
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def txt2mask(dataset_root, output_dir, split="train"):
    images_dir = os.path.join(dataset_root, "images", split)
    labels_dir = os.path.join(dataset_root, "labels", split)

    # Check if directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Directory not found: {images_dir} or {labels_dir}")
        return

    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(images_dir, ext)))

    pattern_files = sorted(image_files)

    print(f"Processing {len(pattern_files)} images for split: {split}")

    for img_path in tqdm(pattern_files):
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        label_path = os.path.join(labels_dir, f"{name}.txt")
        output_path = os.path.join(output_split_dir, f"{name}.png")

        if not os.path.exists(label_path):
            # If no label file, create empty mask? Or skip?
            # Usually empty mask is safer if we want to preserve 1:1 mapping
            # Let's read image to get size
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(output_path, mask)
            continue

        # Read image dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        h, w = img.shape[:2]

        # Create empty mask
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:  # class + at least 1 point (degenerate)
                continue

            # parts[0] is class_id (ignored for binary mask, or could be used for value)
            # parts[1:] are normalized coordinates x1 y1 x2 y2 ...
            coords = np.array([float(x) for x in parts[1:]])

            # Reshape to (-1, 2)
            if len(coords) % 2 != 0:
                print(f"Warning: Odd number of coordinates in {label_path}")
                continue

            points_norm = coords.reshape(-1, 2)

            # Denormalize
            points_abs = points_norm * np.array([w, h])
            points_abs = points_abs.astype(np.int32)

            # Draw polygon
            # Use 255 for white mask
            cv2.fillPoly(mask, [points_abs], color=255)

        # Save mask
        cv2.imwrite(output_path, mask)


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO txt polygons to binary masks")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/bongmedai/Endo/datasets/medai_endo_data",
        help="Root path of the dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="masks_converted",
        help="Output directory name (created inside dataset root or absolute)",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to process")

    args = parser.parse_args()

    # Handle output dir
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(args.dataset_root, args.output_dir)

    print(f"Dataset root: {args.dataset_root}")
    print(f"Output directory: {args.output_dir}")

    for split in args.splits:
        txt2mask(args.dataset_root, args.output_dir, split)


if __name__ == "__main__":
    main()
