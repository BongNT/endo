"""
YOLO segmentation 데이터셋에 대한 오프라인 augmentation 스크립트.

[INFO] 지원하는 augmentation:
    1. Horizontal flip - x 좌표 반전
    2. Vertical flip - y 좌표 반전
    3. Gaussian blur - 이미지 블러 (라벨 변경 없음)
    4. Rotation - 지정 각도로 회전 (좌표 변환 포함)

[INFO] 입력/출력:
    - 원본 데이터셋 구조를 유지하면서 augmented 이미지/라벨 추가
    - 파일명에 augmentation 타입 suffix 추가 (예: case0001_hflip.jpg)
"""

import os
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm


# ================= CONFIG =================
DATAPATH = "C:/Users/user/endo/endo/datasets/"
SRC_ROOT = DATAPATH + "hyperkvasir_polyps"
OUT_ROOT = DATAPATH + "hyperkvasir_polyps_augmented"

# Augmentation 설정
AUGMENTATIONS = {
    "hflip": True,      # Horizontal flip
    "vflip": True,      # Vertical flip
    "blur": True,       # Gaussian blur
    "rotate": True,     # Rotation
}

# Rotation angles (degrees)
ROTATION_ANGLES = [90, 180, 270]  # 90도 단위 회전 (좌표 변환이 깔끔함)

# Gaussian blur kernel sizes
BLUR_KERNEL_SIZES = [5, 7, 9]

# Train만 augment할지 여부 (일반적으로 validation은 augment 안함)
AUGMENT_TRAIN_ONLY = True

# [INFO] 테스트 모드 - None이면 전체 처리, 숫자면 해당 개수만 처리
TEST_LIMIT = 5
# ==========================================


def parse_yolo_label(label_path: str) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    YOLO segmentation 라벨 파일을 파싱한다.
    
    Returns:
        [(class_id, [(x1, y1), (x2, y2), ...]), ...]
    """
    objects = []
    
    if not os.path.exists(label_path):
        return objects
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + 최소 3개 점 (6개 좌표)
                continue
            
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # (x, y) 쌍으로 변환
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            objects.append((class_id, points))
    
    return objects


def save_yolo_label(label_path: str, objects: List[Tuple[int, List[Tuple[float, float]]]]):
    """
    YOLO segmentation 라벨을 파일로 저장한다.
    """
    lines = []
    for class_id, points in objects:
        # 좌표를 0~1 범위로 클램핑
        clamped_points = [(max(0, min(1, x)), max(0, min(1, y))) for x, y in points]
        coords_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in clamped_points)
        lines.append(f"{class_id} {coords_str}")
    
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def horizontal_flip_image(img: np.ndarray) -> np.ndarray:
    """이미지를 수평 반전한다."""
    return cv2.flip(img, 1)


def horizontal_flip_label(objects: List[Tuple[int, List[Tuple[float, float]]]]) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """라벨의 x 좌표를 반전한다 (x -> 1-x)."""
    flipped = []
    for class_id, points in objects:
        new_points = [(1.0 - x, y) for x, y in points]
        # 점 순서를 뒤집어야 polygon 방향이 유지됨
        new_points = new_points[::-1]
        flipped.append((class_id, new_points))
    return flipped


def vertical_flip_image(img: np.ndarray) -> np.ndarray:
    """이미지를 수직 반전한다."""
    return cv2.flip(img, 0)


def vertical_flip_label(objects: List[Tuple[int, List[Tuple[float, float]]]]) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """라벨의 y 좌표를 반전한다 (y -> 1-y)."""
    flipped = []
    for class_id, points in objects:
        new_points = [(x, 1.0 - y) for x, y in points]
        # 점 순서를 뒤집어야 polygon 방향이 유지됨
        new_points = new_points[::-1]
        flipped.append((class_id, new_points))
    return flipped


def gaussian_blur_image(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """이미지에 Gaussian blur를 적용한다."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def rotate_image_90(img: np.ndarray, angle: int) -> np.ndarray:
    """
    이미지를 90도 단위로 회전한다.
    
    Args:
        angle: 90, 180, 270 중 하나
    """
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def rotate_label_90(
    objects: List[Tuple[int, List[Tuple[float, float]]]], 
    angle: int
) -> List[Tuple[int, List[Tuple[float, float]]]]:
    """
    라벨 좌표를 90도 단위로 회전한다.
    
    90도 시계방향: (x, y) -> (1-y, x)
    180도: (x, y) -> (1-x, 1-y)
    270도 시계방향: (x, y) -> (y, 1-x)
    """
    rotated = []
    for class_id, points in objects:
        if angle == 90:
            new_points = [(1.0 - y, x) for x, y in points]
        elif angle == 180:
            new_points = [(1.0 - x, 1.0 - y) for x, y in points]
        elif angle == 270:
            new_points = [(y, 1.0 - x) for x, y in points]
        else:
            new_points = points
        rotated.append((class_id, new_points))
    return rotated


def process_single_image(
    img_path: str,
    label_path: str,
    out_img_dir: str,
    out_label_dir: str,
    base_name: str
) -> int:
    """
    단일 이미지에 대해 모든 augmentation을 적용하고 저장한다.
    
    Returns:
        생성된 augmented 이미지 수
    """
    # 원본 이미지/라벨 로드
    img = cv2.imread(img_path)
    if img is None:
        return 0
    
    objects = parse_yolo_label(label_path)
    
    count = 0
    
    # 1. Horizontal flip
    if AUGMENTATIONS.get("hflip"):
        aug_img = horizontal_flip_image(img)
        aug_objects = horizontal_flip_label(objects)
        
        out_img_path = os.path.join(out_img_dir, f"{base_name}_hflip.jpg")
        out_label_path = os.path.join(out_label_dir, f"{base_name}_hflip.txt")
        
        cv2.imwrite(out_img_path, aug_img)
        save_yolo_label(out_label_path, aug_objects)
        count += 1
    
    # 2. Vertical flip
    if AUGMENTATIONS.get("vflip"):
        aug_img = vertical_flip_image(img)
        aug_objects = vertical_flip_label(objects)
        
        out_img_path = os.path.join(out_img_dir, f"{base_name}_vflip.jpg")
        out_label_path = os.path.join(out_label_dir, f"{base_name}_vflip.txt")
        
        cv2.imwrite(out_img_path, aug_img)
        save_yolo_label(out_label_path, aug_objects)
        count += 1
    
    # 3. Gaussian blur (라벨은 원본 그대로)
    if AUGMENTATIONS.get("blur"):
        kernel_size = random.choice(BLUR_KERNEL_SIZES)
        aug_img = gaussian_blur_image(img, kernel_size)
        
        out_img_path = os.path.join(out_img_dir, f"{base_name}_blur.jpg")
        out_label_path = os.path.join(out_label_dir, f"{base_name}_blur.txt")
        
        cv2.imwrite(out_img_path, aug_img)
        save_yolo_label(out_label_path, objects)  # 원본 라벨 사용
        count += 1
    
    # 4. Rotation (90, 180, 270도)
    if AUGMENTATIONS.get("rotate"):
        for angle in ROTATION_ANGLES:
            aug_img = rotate_image_90(img, angle)
            aug_objects = rotate_label_90(objects, angle)
            
            out_img_path = os.path.join(out_img_dir, f"{base_name}_rot{angle}.jpg")
            out_label_path = os.path.join(out_label_dir, f"{base_name}_rot{angle}.txt")
            
            cv2.imwrite(out_img_path, aug_img)
            save_yolo_label(out_label_path, aug_objects)
            count += 1
    
    return count


def main():
    """메인 augmentation 로직."""
    print("=" * 60)
    print("[INFO] YOLO Segmentation Dataset Augmentation")
    print("=" * 60)
    
    splits_to_process = ["train"] if AUGMENT_TRAIN_ONLY else ["train", "val"]
    
    total_original = 0
    total_augmented = 0
    
    for split in splits_to_process:
        img_dir = os.path.join(SRC_ROOT, "images", split)
        label_dir = os.path.join(SRC_ROOT, "labels", split)
        out_img_dir = os.path.join(OUT_ROOT, "images", split)
        out_label_dir = os.path.join(OUT_ROOT, "labels", split)
        
        # 출력 폴더 생성
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        if not os.path.exists(out_label_dir):
            os.makedirs(out_label_dir)
        
        if not os.path.exists(img_dir):
            print(f"[WARN] {img_dir} not found, skipping...")
            continue
        
        # 이미지 파일 목록
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if TEST_LIMIT:
            img_files = img_files[:TEST_LIMIT]
        
        print(f"\n[INFO] Processing {split}: {len(img_files)} images")
        total_original += len(img_files)
        
        split_augmented = 0
        
        for img_file in tqdm(img_files, desc=f"{split} augmentation"):
            base_name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            
            count = process_single_image(
                img_path=img_path,
                label_path=label_path,
                out_img_dir=out_img_dir,
                out_label_dir=out_label_dir,
                base_name=base_name
            )
            split_augmented += count
        
        print(f"[OK] {split}: {split_augmented} augmented images created")
        total_augmented += split_augmented
    
    print("\n" + "=" * 60)
    print(f"[OK] Augmentation complete!")
    print(f"[INFO] Original images: {total_original}")
    print(f"[INFO] Augmented images: {total_augmented}")
    print(f"[INFO] Total images: {total_original + total_augmented}")
    print("=" * 60)
    
    # Augmentation 요약
    aug_count = sum(1 for v in AUGMENTATIONS.values() if v)
    if AUGMENTATIONS.get("rotate"):
        aug_count += len(ROTATION_ANGLES) - 1  # rotate는 여러 각도
    print(f"\n[INFO] Applied augmentations:")
    if AUGMENTATIONS.get("hflip"):
        print("  - Horizontal flip")
    if AUGMENTATIONS.get("vflip"):
        print("  - Vertical flip")
    if AUGMENTATIONS.get("blur"):
        print(f"  - Gaussian blur (kernel: {BLUR_KERNEL_SIZES})")
    if AUGMENTATIONS.get("rotate"):
        print(f"  - Rotation ({ROTATION_ANGLES} degrees)")


if __name__ == "__main__":
    main()

