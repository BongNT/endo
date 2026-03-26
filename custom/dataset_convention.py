import csv  # ✅ NEW
import json
import os
import random
import shutil

from PIL import Image
from tqdm import tqdm

# ================= CONFIG =================
DATAPATH = "/home/bongmedai/Endo/datasets/"
SRC_ROOT = DATAPATH + "Endoscopy_raw"
OUT_ROOT = DATAPATH + "endo_coco_seg3"
SPLIT_RATIO = 0.8  # train / val

CATEGORIES = {"adenoma": 0, "carcinoma": 1}
# ==========================================


def normalize_polygon(xs, ys, w, h):
    points = []
    for x, y in zip(xs, ys):
        points.append(round(x / w, 6))
        points.append(round(y / h, 6))
    return points


def main():
    samples = []

    # -------- collect all samples --------
    for cls_name, cls_id in CATEGORIES.items():
        cls_dir = os.path.join(SRC_ROOT, cls_name)
        label_path = os.path.join(cls_dir, "label.json")

        with open(label_path) as f:
            data = json.load(f)

        for _, v in data.items():
            img_path = os.path.join(cls_dir, v["filename"])
            regions = list(v["regions"].values())
            samples.append((cls_name, cls_id, img_path, regions))

    random.shuffle(samples)
    split_idx = int(len(samples) * SPLIT_RATIO)

    splits = {"train": samples[:split_idx], "val": samples[split_idx:]}

    # -------- create folders --------
    for split in ["train", "val"]:
        os.makedirs(f"{OUT_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_ROOT}/labels/{split}", exist_ok=True)

    # -------- NEW: mapping holder --------
    mapping = []

    # -------- process --------
    case_id = 1
    for split, items in splits.items():
        for cls_name, cls_id, img_path, regions in tqdm(items):
            case_name = f"case{case_id:04d}"
            img_name = f"{case_name}.png"

            # copy image
            shutil.copy(img_path, f"{OUT_ROOT}/images/{split}/{img_name}")

            with Image.open(img_path) as im:
                w, h = im.size

            label_lines = []

            for r in regions:
                xs = r["shape_attributes"]["all_points_x"]
                ys = r["shape_attributes"]["all_points_y"]

                poly = normalize_polygon(xs, ys, w, h)
                line = str(cls_id) + " " + " ".join(map(str, poly))
                label_lines.append(line)

            with open(f"{OUT_ROOT}/labels/{split}/{case_name}.txt", "w") as f:
                f.write("\n".join(label_lines))

            # -------- NEW: save mapping --------
            mapping.append([case_name, split, os.path.basename(img_path), cls_name])

            case_id += 1

    # -------- NEW: write mapping file --------
    mapping_path = os.path.join(OUT_ROOT, "mapping.csv")
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "split", "original_image", "original_class"])
        writer.writerows(mapping)

    print("✅ YOLOv8 segmentation dataset created successfully")
    print(f"📄 Mapping saved to {mapping_path}")


if __name__ == "__main__":
    main()
