"""
segmented-images 데이터셋을 YOLOv8 segmentation 형식으로 변환하는 스크립트.

[INFO] 입력 데이터셋 구조:
    segmented-images/
    ├── images/          # 원본 이미지
    ├── masks/           # 바이너리 마스크 이미지
    └── bounding-boxes.json  # (사용하지 않음 - polygon 추출)

[INFO] 출력 데이터셋 구조 (YOLOv8 segmentation):
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── mapping.csv
"""

import csv
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# ================= CONFIG =================
# Windows 경로 (사용자 환경에 맞게 수정)
DATAPATH = "C:/Users/user/endo/endo/datasets/"
SRC_ROOT = DATAPATH + "segmented-images"
OUT_ROOT = DATAPATH + "segmented_yolo_seg"

SPLIT_RATIO = 0.8  # train / val 비율
MIN_CONTOUR_AREA = 100  # 최소 contour 면적 (노이즈 제거용)
SIMPLIFY_EPSILON = 2.0  # polygon 단순화 정도 (0=단순화 안함)

# [INFO] 테스트 모드 - None이면 전체 처리, 숫자면 해당 개수만 처리
TEST_LIMIT = None

# [INFO] segmented-images는 단일 클래스(polyp)
CATEGORIES = {"polyp": 0}
# ==========================================


def extract_polygons_from_mask(
    mask_path: str, 
    min_area: float = MIN_CONTOUR_AREA,
    epsilon: float = SIMPLIFY_EPSILON
) -> List[np.ndarray]:
    """
    마스크 이미지에서 polygon contour를 추출한다.
    
    Args:
        mask_path: 마스크 이미지 경로
        min_area: 최소 contour 면적 (이보다 작으면 무시)
        epsilon: polygon 단순화 정도
        
    Returns:
        contour 리스트 (각 contour는 Nx1x2 numpy array)
    """
    # 마스크 이미지 로드 (그레이스케일)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"마스크 이미지를 찾을 수 없음: {mask_path}")
    
    # 이진화 (임계값 127 기준)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # contour 추출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 필터링 및 단순화
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        # polygon 단순화 (점 개수 줄이기)
        if epsilon > 0:
            cnt = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 최소 3개 점 필요
        if len(cnt) >= 3:
            valid_contours.append(cnt)
    
    return valid_contours


def normalize_contour(contour: np.ndarray, w: int, h: int) -> List[float]:
    """
    contour를 YOLO 형식의 정규화된 좌표로 변환한다.
    
    Args:
        contour: Nx1x2 형태의 contour array
        w: 이미지 너비
        h: 이미지 높이
        
    Returns:
        [x1, y1, x2, y2, ...] 형태의 정규화된 좌표 리스트
    """
    points = []
    for point in contour:
        x, y = point[0]  # contour는 (N, 1, 2) 형태
        points.append(round(x / w, 6))
        points.append(round(y / h, 6))
    return points


def main():
    """메인 변환 로직."""
    print("=" * 50)
    print("[INFO] segmented-images -> YOLOv8 segmentation 변환")
    print("=" * 50)
    
    images_dir = os.path.join(SRC_ROOT, "images")
    masks_dir = os.path.join(SRC_ROOT, "masks")
    
    # 이미지 파일 목록 수집
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"[INFO] 총 이미지 수: {len(image_files)}")
    
    # 테스트 모드: 제한된 개수만 처리
    if TEST_LIMIT is not None:
        image_files = image_files[:TEST_LIMIT]
        print(f"[TEST] 테스트 모드: {TEST_LIMIT}개 파일만 처리")
    
    # 샘플 수집
    samples = []
    skipped = 0
    
    for img_file in tqdm(image_files, desc="마스크에서 polygon 추출"):
        img_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        
        # 마스크 파일 찾기 (확장자가 다를 수 있음)
        mask_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = os.path.join(masks_dir, img_id + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path is None:
            skipped += 1
            continue
        
        try:
            contours = extract_polygons_from_mask(mask_path)
            if len(contours) == 0:
                skipped += 1
                continue
                
            samples.append({
                "img_id": img_id,
                "img_path": img_path,
                "mask_path": mask_path,
                "contours": contours,
                "cls_name": "polyp",
                "cls_id": CATEGORIES["polyp"]
            })
        except Exception as e:
            print(f"[WARN] {img_id} 처리 실패: {e}")
            skipped += 1
    
    print(f"[OK] 유효한 샘플: {len(samples)}, 스킵: {skipped}")
    
    # 셔플 및 분할
    random.shuffle(samples)
    split_idx = int(len(samples) * SPLIT_RATIO)
    splits = {
        "train": samples[:split_idx],
        "val": samples[split_idx:]
    }
    
    print(f"[INFO] Train: {len(splits['train'])}, Val: {len(splits['val'])}")
    
    # 출력 폴더 생성
    for split in ["train", "val"]:
        os.makedirs(f"{OUT_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_ROOT}/labels/{split}", exist_ok=True)
    
    # 매핑 정보 저장용
    mapping = []
    
    # 데이터 처리
    case_id = 1
    for split, items in splits.items():
        for sample in tqdm(items, desc=f"{split} 처리"):
            case_name = f"case{case_id:04d}"
            img_ext = os.path.splitext(sample["img_path"])[1]
            img_name = f"{case_name}{img_ext}"
            
            # 이미지 복사
            shutil.copy(sample["img_path"], f"{OUT_ROOT}/images/{split}/{img_name}")
            
            # 이미지 크기 확인
            with Image.open(sample["img_path"]) as im:
                w, h = im.size
            
            # 라벨 파일 생성
            label_lines = []
            for contour in sample["contours"]:
                poly = normalize_contour(contour, w, h)
                line = str(sample["cls_id"]) + " " + " ".join(map(str, poly))
                label_lines.append(line)
            
            # 라벨 저장
            with open(f"{OUT_ROOT}/labels/{split}/{case_name}.txt", "w") as f:
                f.write("\n".join(label_lines))
            
            # 매핑 저장
            mapping.append([case_name, split, sample["img_id"], sample["cls_name"]])
            case_id += 1
    
    # 매핑 CSV 저장
    mapping_path = os.path.join(OUT_ROOT, "mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "split", "original_id", "class"])
        writer.writerows(mapping)
    
    # YOLO 데이터셋 설정 파일 생성
    yaml_content = f"""# YOLOv8 segmentation dataset configuration
# Auto-generated from segmented-images

path: {OUT_ROOT}
train: images/train
val: images/val

# Classes
names:
  0: polyp
"""
    yaml_path = os.path.join(OUT_ROOT, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print("\n" + "=" * 50)
    print("[OK] YOLOv8 segmentation dataset 생성 완료!")
    print(f"[PATH] 출력 경로: {OUT_ROOT}")
    print(f"[FILE] 매핑 파일: {mapping_path}")
    print(f"[FILE] YAML 설정: {yaml_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

