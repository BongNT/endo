import argparse
import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def analyze_dataset(root, out_dir="analysis"):
    out_dir = os.path.join(root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats = {
        "images": {},
        "labels": {},
        "instances": {},
        "instances_per_split": {},
        "empty_labels": {},
        "no_label_images": {},
        "image_sizes": {},
        "labels_per_image": {},
        "labels_per_image_detail": {},
    }

    for split in ["train", "val"]:
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        stats["instances_per_split"][split] = {}
        stats["images"][split] = len(images)
        stats["labels"][split] = 0
        stats["empty_labels"][split] = 0
        stats["no_label_images"][split] = 0
        stats["image_sizes"][split] = {}
        stats["labels_per_image"][split] = {}
        stats["labels_per_image_detail"][split] = {}

        for img in images:
            stem = img.replace(".png", "")
            img_path = os.path.join(img_dir, img)
            label_path = os.path.join(lbl_dir, stem + ".txt")

            # -------- image size --------
            with Image.open(img_path) as im:
                w, h = im.size
            size_key = f"{w}x{h}"
            stats["image_sizes"][split][size_key] = stats["image_sizes"][split].get(size_key, 0) + 1

            # -------- labels per image --------
            if not os.path.exists(label_path):
                n_labels = 0
                stats["no_label_images"][split] += 1
            else:
                with open(label_path) as f:
                    lines = [l.strip() for l in f if l.strip()]

                if not lines:
                    n_labels = 0
                    stats["empty_labels"][split] += 1
                else:
                    n_labels = len(lines)
                    stats["labels"][split] += 1

                    for l in lines:
                        cls = int(l.split()[0])

                        # global instances
                        stats["instances"][cls] = stats["instances"].get(cls, 0) + 1

                        # per split instances
                        split_dict = stats["instances_per_split"][split]
                        split_dict[cls] = split_dict.get(cls, 0) + 1

            # -------- aggregate counts --------
            key = str(n_labels)
            stats["labels_per_image"][split][key] = stats["labels_per_image"][split].get(key, 0) + 1

            # -------- store detail --------
            stats["labels_per_image_detail"][split].setdefault(key, []).append(img)

    # -------- save JSON --------
    json_path = os.path.join(out_dir, "dataset_analysis.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"📊 Analysis saved to:\n- {json_path}")

    return stats


def plot_samples(root, split="train", num=5):
    out_dir = os.path.join(root, "plots", split)
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)
    print(out_dir)
    os.makedirs(os.path.join(out_dir), exist_ok=True)

    images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    samples = random.sample(images, min(num, len(images)))

    for img_name in samples:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".png", ".txt"))

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for l in f:
                    parts = list(map(float, l.split()))
                    cls = int(parts[0])
                    pts = parts[1:]

                    if len(pts) < 6:
                        continue  # invalid polygon

                    poly = []
                    for i in range(0, len(pts), 2):
                        x = int(pts[i] * w)
                        y = int(pts[i + 1] * h)
                        poly.append([x, y])

                    poly = np.array(poly, np.int32)
                    cv2.polylines(img, [poly], True, (0, 255, 0), 2)
                    cv2.putText(img, f"class {cls}", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, img)

        print(f"✅ Saved: {out_path}")


def plot_samples_show_window(root, split="train", num=5):
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)

    images = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    samples = random.sample(images, min(num, len(images)))

    for img_name in samples:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace(".png", ".txt"))

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                lines = f.readlines()

            for l in lines:
                parts = list(map(float, l.split()))
                cls = int(parts[0])
                pts = parts[1:]

                poly = []
                for i in range(0, len(pts), 2):
                    x = int(pts[i] * w)
                    y = int(pts[i + 1] * h)
                    poly.append([x, y])

                poly = np.array(poly, np.int32)
                cv2.polylines(img, [poly], True, (0, 255, 0), 2)
                cv2.putText(img, f"class {cls}", tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(img_name)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to coco8-seg dataset")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--num", type=int, default=5, help="Number of images to plot")
    args = parser.parse_args()

    analyze_dataset(args.dataset)
    plot_samples(args.dataset, args.split, args.num)


if __name__ == "__main__":
    # python ultralytics/custom/analyze_and_plot_cocoseg.py --dataset datasets/endo_coco_seg3 --split val --num 10000
    main()
