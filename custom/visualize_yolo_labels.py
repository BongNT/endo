import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import argparse

def visualize_yolo_labels(labels_dir, images_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    print(f"Found {len(label_files)} label files in {labels_dir}")

    for label_file in tqdm(label_files, desc="Visualizing labels"):
        basename = os.path.basename(label_file)
        file_id = os.path.splitext(basename)[0]
        
        # Construct image path (assuming .png extension based on prior exploration)
        image_path = os.path.join(images_dir, file_id + ".png")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {basename} at {image_path}")
            continue
            
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Read labels
        with open(label_file, "r") as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            coords = parts[1:]
            
            # YOLO format: normalized coordinates
            # Reshape strings to (N, 2)
            points = np.array(coords).reshape(-1, 2)
            
            # Denormalize
            points[:, 0] *= w
            points[:, 1] *= h
            
            points = points.astype(np.int32)
            
            if points.size == 0:
                continue

            # Define a list of colors (BGR)
            colors = [
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 0, 0),    # Blue
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (0, 165, 255),  # Orange
                (128, 0, 128),  # Purple
                (128, 128, 0),  # Teal
                (0, 128, 128)   # Olive
            ]
            
            # Select color based on instance index (i) since class_id is always 0
            color = colors[i % len(colors)]

            # Draw polygon
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
            
            # Optional: Draw filled polygon with transparency
            if points.shape[0] > 0:
                overlay = img.copy()
                cv2.fillPoly(overlay, [points], color=color)
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Save result
        output_path = os.path.join(output_dir, file_id + "_vis.png")
        cv2.imwrite(output_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images")
    parser.add_argument("--labels_dir", type=str, default="/home/bongmedai/Endo/ultralytics/runs/segment/val8/labels", help="Path to labels directory")
    parser.add_argument("--images_dir", type=str, default="/home/bongmedai/Endo/datasets/medai_endo_data/images/val", help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="/home/bongmedai/Endo/ultralytics/runs/segment/val8/vis", help="Path to output directory")
    
    args = parser.parse_args()
    
    visualize_yolo_labels(args.labels_dir, args.images_dir, args.output_dir)
