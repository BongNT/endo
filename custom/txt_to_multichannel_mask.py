import os

import cv2
import numpy as np
from PIL import Image

# ================= CONFIG =================
DATASET_ROOT = "/home/bongmedai/Endo/datasets/endo_coco_seg3"
SPLIT = "val"  # train / val
OUT_ROOT = "masks"  # output folder
NUM_CLASSES = 2  # adenoma, carcinoma
# =========================================

IMG_DIR = os.path.join(DATASET_ROOT, "images", SPLIT)
LBL_DIR = os.path.join(DATASET_ROOT, "labels", SPLIT)
OUT_DIR = os.path.join(DATASET_ROOT, OUT_ROOT, SPLIT)

# LBL_DIR = "/home/bongmedai/Endo/ultralytics/runs/segment/predict/labels"
# OUT_DIR = "/home/bongmedai/Endo/ultralytics/runs/segment/predict/masks"
os.makedirs(OUT_DIR, exist_ok=True)


def convert_one(img_path, label_path, out_path):
    with Image.open(img_path) as im:
        w, h = im.size

    mask = np.zeros((h, w, NUM_CLASSES), dtype=np.uint8)

    if not os.path.exists(label_path):
        np.save(out_path, mask)
        return

    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 7:
                continue

            cls = int(parts[0])  # 0=adenoma, 1=carcinoma
            pts = parts[1:]

            poly = np.array([[int(pts[i] * w), int(pts[i + 1] * h)] for i in range(0, len(pts), 2)], np.int32)

            # 🔥 FIX: draw on a contiguous array
            tmp = mask[:, :, cls].copy()
            cv2.fillPoly(tmp, [poly], 1)
            mask[:, :, cls] = tmp

    np.save(out_path, mask)


images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])

print(f"🔄 Converting {len(images)} images to multi-channel masks...")

for img_name in images:
    img_path = os.path.join(IMG_DIR, img_name)
    label_path = os.path.join(LBL_DIR, img_name.replace(".png", ".txt"))
    out_path = os.path.join(OUT_DIR, img_name.replace(".png", ".npy"))

    convert_one(img_path, label_path, out_path)

print(f"✅ Multi-channel masks saved to {OUT_DIR}")
