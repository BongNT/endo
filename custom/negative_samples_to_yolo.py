"""
anatomical-landmarks 폴더의 이미지들을 YOLO negative sample로 변환하는 스크립트.

[INFO] Negative sample이란?
    - 객체가 없는 배경 이미지
    - YOLO 학습 시 False Positive를 줄이는 데 도움
    - 라벨 파일(.txt)이 비어있거나 아예 없음

[INFO] 입력 데이터셋 구조:
    labeled-images/upper-gi-tract/anatomical-landmarks/
    ├── z-line/           # 이미지만 있음 (polyp 없음)
    ├── retroflex-stomach/
    └── pylorus/

[INFO] 출력: 기존 YOLO 데이터셋에 negative sample 추가
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/   ← 빈 .txt 파일 (negative sample)
    │   └── val/
    └── mapping.csv
"""

import csv
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


# ================= CONFIG =================
# Windows 경로 (사용자 환경에 맞게 수정)
DATAPATH = "C:/Users/user/endo/endo/datasets/"
SRC_ROOT = DATAPATH + "labeled-images/upper-gi-tract/anatomical-landmarks"
OUT_ROOT = DATAPATH + "negative_yolo"

SPLIT_RATIO = 0.8  # train / val 비율

# [INFO] 테스트 모드 - None이면 전체 처리, 숫자면 해당 개수만 처리
TEST_LIMIT = None

# [INFO] 하위 폴더 목록 (모두 negative sample)
SUBFOLDERS = ["z-line", "retroflex-stomach", "pylorus"]
# ==========================================


def collect_images(src_root: str, subfolders: List[str]) -> List[Tuple[str, str, str]]:
    """
    하위 폴더들에서 이미지 파일을 수집한다.
    
    Returns:
        [(이미지 경로, 원본 파일명, 원본 폴더명), ...]
    """
    images = []
    
    for subfolder in subfolders:
        folder_path = os.path.join(src_root, subfolder)
        if not os.path.exists(folder_path):
            print(f"[WARN] 폴더가 존재하지 않음: {folder_path}")
            continue
            
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                images.append((img_path, filename, subfolder))
    
    return images


def main():
    """메인 변환 로직."""
    print("=" * 50)
    print("[INFO] anatomical-landmarks -> YOLO negative samples")
    print("=" * 50)
    
    # 이미지 수집
    images = collect_images(SRC_ROOT, SUBFOLDERS)
    print(f"[INFO] 총 이미지 수: {len(images)}")
    
    # 테스트 모드
    if TEST_LIMIT is not None:
        images = images[:TEST_LIMIT]
        print(f"[TEST] 테스트 모드: {TEST_LIMIT}개 파일만 처리")
    
    if len(images) == 0:
        print("[ERROR] 처리할 이미지가 없습니다!")
        return
    
    # 셔플 및 분할
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    splits = {
        "train": images[:split_idx],
        "val": images[split_idx:]
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
        for img_path, original_filename, source_folder in tqdm(items, desc=f"{split} 처리"):
            case_name = f"neg{case_id:04d}"
            img_ext = os.path.splitext(img_path)[1]
            img_name = f"{case_name}{img_ext}"
            
            # 이미지 복사
            shutil.copy(img_path, f"{OUT_ROOT}/images/{split}/{img_name}")
            
            # [INFO] 빈 라벨 파일 생성 (negative sample)
            with open(f"{OUT_ROOT}/labels/{split}/{case_name}.txt", "w") as f:
                pass  # 빈 파일
            
            # 매핑 저장
            mapping.append([case_name, split, original_filename, source_folder, "negative"])
            case_id += 1
    
    # 매핑 CSV 저장
    mapping_path = os.path.join(OUT_ROOT, "mapping.csv")
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "split", "original_filename", "source_folder", "label_type"])
        writer.writerows(mapping)
    
    # YOLO 데이터셋 설정 파일 생성
    yaml_content = f"""# YOLOv8 negative samples dataset configuration
# Auto-generated from anatomical-landmarks

path: {OUT_ROOT}
train: images/train
val: images/val

# Classes (same as positive dataset)
names:
  0: polyp
"""
    yaml_path = os.path.join(OUT_ROOT, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print("\n" + "=" * 50)
    print("[OK] YOLO negative samples dataset 생성 완료!")
    print(f"[PATH] 출력 경로: {OUT_ROOT}")
    print(f"[FILE] 매핑 파일: {mapping_path}")
    print(f"[FILE] YAML 설정: {yaml_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

