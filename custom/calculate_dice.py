
import os
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from glob import glob


def calculate_metrics(gt_mask, pred_mask):
    """
    Calculate Dice, Precision, and Recall between two binary masks.
    Masks are expected to be 0 or 255 (or boolean).
    
    Returns:
        dict with 'dice', 'precision', 'recall'
    """
    # Ensure boolean
    gt = gt_mask > 128
    pred = pred_mask > 128
    
    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()
    
    # Dice = 2*TP / (2*TP + FP + FN)
    dice_denom = 2 * tp + fp + fn
    if dice_denom == 0:
        dice = 1.0  # Both empty
    else:
        dice = 2.0 * tp / dice_denom
    
    # Precision = TP / (TP + FP)
    if tp + fp == 0:
        precision = 1.0 if tp + fn == 0 else 0.0  # No predictions
    else:
        precision = tp / (tp + fp)
    
    # Recall = TP / (TP + FN)
    if tp + fn == 0:
        recall = 1.0  # No ground truth
    else:
        recall = tp / (tp + fn)
    
    return {"dice": dice, "precision": precision, "recall": recall}


def evaluate_folder(gt_dir, pred_dir, output_csv):
    """
    Evaluate Dice, Precision, Recall between GT and Pred folders.
    Matches files by name.
    """
    if not os.path.exists(gt_dir):
        print(f"GT directory not found: {gt_dir}")
        return
    if not os.path.exists(pred_dir):
        print(f"Pred directory not found: {pred_dir}")
        return

    pred_files = sorted(glob(os.path.join(pred_dir, "*.png")) + glob(os.path.join(pred_dir, "*.jpg")))
    
    results = []
    
    print(f"Evaluating {len(pred_files)} prediction files in {pred_dir}...")
    
    for pred_path in tqdm(pred_files):
        filename = os.path.basename(pred_path)
        stem, ext = os.path.splitext(filename)
        
        # Try to find matching GT file (assuming png or jpg)
        gt_path = os.path.join(gt_dir, filename)
        if not os.path.exists(gt_path):
            # Try swapping extension
            if ext == ".png":
                gt_path = os.path.join(gt_dir, stem + ".jpg")
            else:
                gt_path = os.path.join(gt_dir, stem + ".png")
                
        if not os.path.exists(gt_path):
            print(f"Warning: No GT found for {filename}")
            results.append({"case_id": stem, "dice": np.nan, "precision": np.nan, "recall": np.nan, "status": "missing_gt"})
            continue
            
        # Load masks
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if pred_mask is None or gt_mask is None:
            print(f"Error reading masks for {filename}")
            results.append({"case_id": stem, "dice": np.nan, "precision": np.nan, "recall": np.nan, "status": "read_error"})
            continue
            
        # Resize pred to match GT if needed (though they should match)
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        metrics = calculate_metrics(gt_mask, pred_mask)
        results.append({
            "case_id": stem,
            "dice": metrics["dice"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "status": "ok"
        })
        
    df = pd.DataFrame(results)
    if not df.empty:
        ok_mask = df["status"] == "ok"
        mean_dice = df.loc[ok_mask, "dice"].mean()
        mean_precision = df.loc[ok_mask, "precision"].mean()
        mean_recall = df.loc[ok_mask, "recall"].mean()
        
        print(f"Mean Dice:      {mean_dice:.4f}")
        print(f"Mean Precision: {mean_precision:.4f}")
        print(f"Mean Recall:    {mean_recall:.4f}")
        
        # Append mean row
        new_row = pd.DataFrame([{
            "case_id": "MEAN",
            "dice": mean_dice,
            "precision": mean_precision,
            "recall": mean_recall,
            "status": "summary"
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Calculate Dice Score")
    parser.add_argument("--gt_root", type=str, default="/home/bongmedai/Endo/datasets/medai_endo_data/masks_converted", help="Root of GT masks")
    parser.add_argument("--pred_root", type=str, default="/home/bongmedai/Endo/sam3_results", help="Root of Predicted masks")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for CSVs")
    parser.add_argument("--splits", nargs='+', default=["val", "test"], help="Splits to evaluate")
    
    args = parser.parse_args()
    
    for split in args.splits:
        print(f"\n--- Evaluating Split: {split} ---")
        gt_dir = os.path.join(args.gt_root, split)
        # Predictions might be in {pred_root}/{split}/masks or just {pred_root}/{split}
        # Based on previous script: output_mask_dir = os.path.join(args.output_dir, args.split, "masks")
        pred_dir = os.path.join(args.pred_root, split, "masks") 
        
        output_csv = os.path.join(args.output_dir, f"dice_results_{split}.csv")
        
        evaluate_folder(gt_dir, pred_dir, output_csv)

if __name__ == "__main__":
    main()
