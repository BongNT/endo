import csv
import json
import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_ROOT = "/home/bongmedai/Endo/datasets/endo_coco_seg3"
DST_ROOT = "/home/bongmedai/Endo/datasets/nnUNet_raw/Dataset001_Endo2D"

IMG_TR = os.path.join(DST_ROOT, "imagesTr")
LBL_TR = os.path.join(DST_ROOT, "labelsTr")

os.makedirs(IMG_TR, exist_ok=True)
os.makedirs(LBL_TR, exist_ok=True)

case_idx = 1
mapping_rows = []
train_cases = []
val_cases = []

for split in ["train", "val"]:
    img_dir = os.path.join(SRC_ROOT, "images", split)
    msk_dir = os.path.join(SRC_ROOT, "masks", split)

    for fname in tqdm(sorted(os.listdir(img_dir))):
        if not fname.endswith(".png"):
            continue

        # fname is already like: case0001.png
        stem = os.path.splitext(fname)[0]  # case0001

        img_path = os.path.join(img_dir, fname)
        msk_path = os.path.join(msk_dir, f"{stem}.npy")

        if not os.path.exists(msk_path):
            raise FileNotFoundError(msk_path)

        # ---- load HWC mask ----
        mask = np.load(msk_path)  # (H, W, C)

        bg = mask.sum(axis=-1) == 0
        label = np.argmax(mask, axis=-1) + 1
        label[bg] = 0
        label = label.astype(np.uint8)

        # ---- nnU-Net case id ----
        case_id = f"case{case_idx:04d}"

        # ---- copy image with nnU-Net naming ----
        shutil.copy(img_path, os.path.join(IMG_TR, f"{case_id}_0000.png"))

        # ---- save label ----
        Image.fromarray(label).save(os.path.join(LBL_TR, f"{case_id}.png"))

        # ---- mapping ----
        mapping_rows.append([case_id, split, fname, os.path.basename(msk_path)])

        if split == "train":
            train_cases.append(case_id)
        else:
            val_cases.append(case_id)

        case_idx += 1

# ---- save mapping ----
with open(os.path.join(DST_ROOT, "mapping.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["nnunet_id", "original_split", "original_image", "original_mask"])
    writer.writerows(mapping_rows)

# ---- save split file ----
splits = [{"train": train_cases, "val": val_cases}]
with open(os.path.join(DST_ROOT, "splits_final.json"), "w") as f:
    json.dump(splits, f, indent=2)

print("✅ COCO → nnU-Net v2 conversion finished")
