import os
import argparse
import numpy as np
import csv


# -----------------------------
# Load npy mask
# -----------------------------
def load_npy_mask(path, threshold=0.5):
    mask = np.load(path)

    # squeeze last channel if (H, W, 1)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    # ensure boolean
    if mask.dtype != bool:
        mask = mask > threshold

    return mask


# -----------------------------
# Binary Dice
# -----------------------------
def dice_binary(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum()

    if union == 0:
        return 1.0  # both empty

    return 2.0 * inter / union


# -----------------------------
# Dice per class + all class
# -----------------------------
def dice_per_class_and_all(mask1, mask2):
    """
    mask shape: (H, W, C)
    returns:
        per_class_dice: list[C]
        all_class_dice: float (foreground vs background)
        mean_class_dice: float
    """
    assert mask1.shape == mask2.shape
    assert mask1.ndim == 3  # HWC

    num_classes = mask1.shape[-1]

    # per-class dice
    per_class_dice = []
    for c in range(num_classes):
        d = dice_binary(mask1[..., c], mask2[..., c])
        per_class_dice.append(d)

    mean_class_dice = float(np.mean(per_class_dice))

    # all-class (foreground vs background)
    fg1 = np.any(mask1, axis=-1)
    fg2 = np.any(mask2, axis=-1)
    all_class_dice = dice_binary(fg1, fg2)

    return per_class_dice, all_class_dice, mean_class_dice


# -----------------------------
# Main
# -----------------------------
def main(gt_dir, pred_dir, out_dir, threshold):
    os.makedirs(out_dir, exist_ok=True)

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".npy")])

    results = []
    all_mean_dices = []
    all_fg_dices = []
    per_class_collect = None

    for fname in gt_files:
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        if not os.path.exists(pred_path):
            print(f"⚠ Missing prediction: {fname}")
            continue

        gt = load_npy_mask(gt_path, threshold)
        pred = load_npy_mask(pred_path, threshold)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: {fname} {gt.shape} vs {pred.shape}")

        # ensure HWC
        if gt.ndim == 2:
            gt = gt[..., None]
            pred = pred[..., None]

        per_class_dice, fg_dice, mean_dice = dice_per_class_and_all(gt, pred)

        # init collector
        if per_class_collect is None:
            per_class_collect = [[] for _ in range(len(per_class_dice))]

        for i, d in enumerate(per_class_dice):
            per_class_collect[i].append(d)

        all_mean_dices.append(mean_dice)
        all_fg_dices.append(fg_dice)

        results.append(
            [fname, mean_dice, fg_dice] + per_class_dice
        )

        print(
            f"{fname} | Mean Dice: {mean_dice:.4f} | FG Dice: {fg_dice:.4f} | "
            f"Per-class: {[round(d,4) for d in per_class_dice]}"
        )

    # -----------------------------
    # Save CSV
    # -----------------------------
    csv_path = os.path.join(out_dir, "dice_scores.csv")
    num_classes = len(per_class_collect)

    header = (
        ["filename", "mean_dice", "fg_dice"]
        + [f"dice_class_{i}" for i in range(num_classes)]
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    # -----------------------------
    # Save TXT summary
    # -----------------------------
    txt_path = os.path.join(out_dir, "dice_summary.txt")

    with open(txt_path, "w") as f:
        f.write(f"Number of samples: {len(all_mean_dices)}\n")
        f.write(f"Threshold: {threshold}\n\n")

        f.write(f"Mean Dice (per-image mean over classes): {np.mean(all_mean_dices):.4f}\n")
        f.write(f"Mean Dice (foreground all-class): {np.mean(all_fg_dices):.4f}\n\n")

        for i, vals in enumerate(per_class_collect):
            f.write(f"Class {i} Mean Dice: {np.mean(vals):.4f}\n")

    print("\n✅ Dice evaluation completed")
    print(f"📊 Mean Dice (class-mean): {np.mean(all_mean_dices):.4f}")
    print(f"📊 Mean Dice (foreground): {np.mean(all_fg_dices):.4f}")
    print(f"📁 Results saved to: {out_dir}")


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Ground truth npy folder")
    parser.add_argument("--pred", required=True, help="Prediction npy folder")
    parser.add_argument("--out", default="dice_results", help="Output folder")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args.gt, args.pred, args.out, args.threshold)
