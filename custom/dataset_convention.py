"""
Docstring for ultralytics.custom.dataset_convention
coco_seg/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── instances_train.json
    └── instances_val.json

"""
import os
import json
import shutil
import random
from PIL import Image

# ================= CONFIG =================
DATAPATH = "/home/bongmedai/Endo/datasets/"
SRC_ROOT = DATAPATH + "Endoscopy_raw"
OUT_ROOT = DATAPATH + "endo_coco_seg"
SPLIT_RATIO = 0.8   # train / val

CATEGORIES = {
    "adenoma": 1,
    "carcinoma": 2
}
# ==========================================


def polygon_to_bbox(xs, ys):
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min),
        float(y_max - y_min)
    ]


def polygon_area(xs, ys):
    area = 0.0
    n = len(xs)
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


def convert(split="train", items=None):
    images = []
    annotations = []
    img_id = 1
    ann_id = 1

    out_img_dir = os.path.join(OUT_ROOT, "images", split)
    os.makedirs(out_img_dir, exist_ok=True)

    for cls, img_path, regions in items:
        filename = os.path.basename(img_path)

        # copy image
        shutil.copy(img_path, os.path.join(out_img_dir, filename))

        with Image.open(img_path) as im:
            w, h = im.size

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h
        })

        for r in regions:
            xs = r["shape_attributes"]["all_points_x"]
            ys = r["shape_attributes"]["all_points_y"]

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CATEGORIES[cls],
                "segmentation": [sum(zip(xs, ys), ())],
                "bbox": polygon_to_bbox(xs, ys),
                "area": polygon_area(xs, ys),
                "iscrowd": 0
            })
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": v, "name": k, "supercategory": "lesion"}
            for k, v in CATEGORIES.items()
        ]
    }

    out_ann = os.path.join(OUT_ROOT, "annotations", f"instances_{split}.json")
    os.makedirs(os.path.dirname(out_ann), exist_ok=True)

    with open(out_ann, "w") as f:
        json.dump(coco, f, indent=2)


def main():
    samples = []

    for cls in CATEGORIES.keys():
        cls_dir = os.path.join(SRC_ROOT, cls)
        label_file = os.path.join(cls_dir, "label.json")

        with open(label_file) as f:
            data = json.load(f)

        for k, v in data.items():
            img_path = os.path.join(cls_dir, v["filename"])
            regions = list(v["regions"].values())
            samples.append((cls, img_path, regions))

    random.shuffle(samples)
    split_idx = int(len(samples) * SPLIT_RATIO)

    convert("train", samples[:split_idx])
    convert("val", samples[split_idx:])

    print("✅ COCO segmentation dataset created successfully")


if __name__ == "__main__":
    main()
