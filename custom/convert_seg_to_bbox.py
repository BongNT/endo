import os
import glob
import tqdm
import numpy as np

def convert_polygon_to_bbox(polygon):
    """
    Converts a polygon (normalized) to a bounding box (normalized center_x, center_y, width, height).
    Polygon format: [x1, y1, x2, y2, ...]
    """
    x_coords = polygon[0::2]
    y_coords = polygon[1::2]
    
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width / 2
    center_y = min_y + height / 2
    
    return center_x, center_y, width, height

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    print(f"Processing {len(txt_files)} files in {input_dir}...")
    
    for txt_file in tqdm.tqdm(txt_files):
        basename = os.path.basename(txt_file)
        output_path = os.path.join(output_dir, basename)
        
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 3: # Need at least class + 1 point (though usually more)
                continue
                
            class_id = int(parts[0])
            polygon = parts[1:]
            
            # Simple check if there are enough points (3 points = 6 coords)
            if len(polygon) < 6:
                print(f"Warning: Skipping invalid polygon in {basename}")
                continue
            
            cx, cy, w, h = convert_polygon_to_bbox(polygon)
            
            # Clamp values to [0, 1] just in case
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            new_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
        with open(output_path, "w") as f:
            f.writelines(new_lines)

def main():
    base_input_dir = "/home/bongmedai/Endo/datasets/medai_endo_data/labels"
    base_output_dir = "/home/bongmedai/Endo/datasets/medai_endo_data/labels_bbox"
    
    for split in ["train", "val", "test"]:
        input_split_dir = os.path.join(base_input_dir, split)
        output_split_dir = os.path.join(base_output_dir, split)
        
        if os.path.exists(input_split_dir):
            process_directory(input_split_dir, output_split_dir)
        else:
            print(f"Directory not found: {input_split_dir}")

if __name__ == "__main__":
    main()
