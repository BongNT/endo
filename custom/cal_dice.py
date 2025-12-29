import os
import argparse
import numpy as np
import csv


def load_npy_mask(path, threshold=0.5):
    mask = np.load(path)
    # squeeze channel if needed
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    # binarize
    if mask.dtype != bool:
        mask = mask > threshold

    return mask


def dice_score(mask1, mask2, convert_to_fg_bg=True):
    mask1 = mask1.astype(bool) #HWC: (480, 640, 2)
    mask2 = mask2.astype(bool) #HWC
    if convert_to_fg_bg:
        # convert multi-class to binary (foreground vs background)
        mask1 = np.any(mask1, axis=-1) 
        mask2 = np.any(mask2, axis=-1)
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()

    if union == 0:
        return 1.0  # both empty → perfect match

    return 2.0 * inter / union


def main(gt_dir, pred_dir, out_dir, threshold):
    os.makedirs(out_dir, exist_ok=True)

    scores = []
    results = []

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".npy")])

    for fname in gt_files:
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.exists(pred_path):
            print(f"⚠ Missing prediction: {fname}")
            continue

        gt = load_npy_mask(gt_path, threshold)
        pred = load_npy_mask(pred_path, threshold)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {fname}: {gt.shape} vs {pred.shape}")

        d = dice_score(gt, pred)
        scores.append(d)
        results.append((fname, float(d)))

        print(f"{fname}: Dice = {d:.4f}")

    mean_dice = float(np.mean(scores)) if scores else 0.0

    # -------- save CSV --------
    csv_path = os.path.join(out_dir, "dice_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "dice"])
        writer.writerows(results)

    # -------- save TXT --------
    txt_path = os.path.join(out_dir, "dice_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"Mean Dice Score: {mean_dice:.4f}\n")
        f.write(f"Number of samples: {len(scores)}\n")
        f.write(f"Threshold: {threshold}\n")

    print("\n✅ Dice evaluation completed")
    print(f"📊 Mean Dice: {mean_dice:.4f}")
    print(f"📁 Results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground truth npy folder")
    parser.add_argument("--pred", required=True, help="Prediction npy folder")
    parser.add_argument("--out", default="dice_results", help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binarizing masks")
    args = parser.parse_args()

    main(args.gt, args.pred, args.out, args.threshold)
