"""
여러 YOLO 데이터셋을 하나로 병합하는 스크립트.

각 데이터셋에 고유한 prefix를 붙여 파일명 충돌을 방지하고,
데이터셋별로 샘플 수를 지정할 수 있습니다.

Usage:
    python custom/merge_datasets.py
"""

import random
import shutil
from pathlib import Path
from dataclasses import dataclass


# =============================================================================
# [INFO] 설정 영역 - 여기서 각 데이터셋의 샘플 수를 조정하세요
# =============================================================================

RANDOM_SEED = 42  # 재현성을 위한 시드 (None이면 랜덤)

@dataclass
class DatasetConfig:
    """데이터셋 설정"""
    name: str           # 데이터셋 이름
    path: str           # 데이터셋 경로
    prefix: str         # 파일명 prefix
    train_count: int    # train에서 가져올 샘플 수 (-1이면 전체)
    val_count: int      # val에서 가져올 샘플 수 (-1이면 전체, 0이면 없음)
    class_mapping: dict | None = None  # 클래스 인덱스 매핑


# [INFO] 여기서 각 데이터셋의 샘플 수를 설정하세요
DATASETS = [
    DatasetConfig(
        name="hyperkvasir_polyps",
        path="datasets/hyperkvasir_polyps",
        prefix="hk_",
        train_count=5,   # 800개 중 500개만 사용
        val_count=1,     # 200개 중 100개만 사용
        class_mapping=None,
    ),
    DatasetConfig(
        name="hyperkvasir_polyps_augmented",
        path="datasets/hyperkvasir_polyps_augmented",
        prefix="hka_",
        train_count=10,  # augmented 데이터에서 1000개
        val_count=0,       # val 없음
        class_mapping=None,
    ),
    DatasetConfig(
        name="negative_yolo",
        path="datasets/negative_yolo",
        prefix="",         # 이미 neg_ prefix 있음
        train_count=5,   # 2156개 중 500개만 사용
        val_count=1,     # 539개 중 100개만 사용
        class_mapping=None,
    ),
]

TARGET_DIR = Path("datasets/data_with_augmentation")

# =============================================================================


def get_file_pairs(
    img_dir: Path,
    lbl_dir: Path,
    img_extensions: tuple[str, ...] = (".jpg", ".png", ".jpeg"),
) -> list[tuple[Path, Path]]:
    """
    이미지와 라벨 파일 쌍을 찾아 반환합니다.
    
    Args:
        img_dir: 이미지 디렉토리
        lbl_dir: 라벨 디렉토리
        img_extensions: 이미지 확장자
        
    Returns:
        (이미지 경로, 라벨 경로) 튜플의 리스트
    """
    pairs = []
    
    if not img_dir.exists():
        return pairs
    
    for img_path in img_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in img_extensions:
            # 대응되는 라벨 파일 찾기
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
            else:
                # 라벨이 없어도 이미지만 추가 (negative 샘플 등)
                pairs.append((img_path, None))
    
    return pairs


def copy_pairs(
    pairs: list[tuple[Path, Path | None]],
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    prefix: str,
    class_mapping: dict | None = None,
) -> tuple[int, int]:
    """
    이미지-라벨 쌍을 대상 디렉토리로 복사합니다.
    
    Args:
        pairs: (이미지, 라벨) 경로 쌍 리스트
        dst_img_dir: 대상 이미지 디렉토리
        dst_lbl_dir: 대상 라벨 디렉토리
        prefix: 파일명 prefix
        class_mapping: 클래스 인덱스 매핑
        
    Returns:
        (복사된 이미지 수, 복사된 라벨 수)
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    img_count = 0
    lbl_count = 0
    
    for img_path, lbl_path in pairs:
        # 이미지 복사
        new_img_name = f"{prefix}{img_path.name}"
        dst_img_path = dst_img_dir / new_img_name
        
        if dst_img_path.exists():
            print(f"  [WARN] 건너뜀 (이미 존재): {dst_img_path.name}")
            continue
        
        shutil.copy2(img_path, dst_img_path)
        img_count += 1
        
        # 라벨 복사
        if lbl_path and lbl_path.exists():
            new_lbl_name = f"{prefix}{lbl_path.name}"
            dst_lbl_path = dst_lbl_dir / new_lbl_name
            
            if class_mapping:
                # 클래스 인덱스 변환
                with open(lbl_path, "r") as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        old_class = int(parts[0])
                        new_class = class_mapping.get(old_class, old_class)
                        parts[0] = str(new_class)
                        new_lines.append(" ".join(parts) + "\n")
                
                with open(dst_lbl_path, "w") as f:
                    f.writelines(new_lines)
            else:
                shutil.copy2(lbl_path, dst_lbl_path)
            
            lbl_count += 1
    
    return img_count, lbl_count


def sample_pairs(
    pairs: list[tuple[Path, Path | None]],
    count: int,
) -> list[tuple[Path, Path | None]]:
    """
    지정된 수만큼 샘플링합니다.
    
    Args:
        pairs: 전체 쌍 리스트
        count: 샘플링할 개수 (-1이면 전체)
        
    Returns:
        샘플링된 쌍 리스트
    """
    if count < 0 or count >= len(pairs):
        return pairs
    
    return random.sample(pairs, count)


def merge_datasets():
    """메인 병합 함수"""
    
    # 시드 설정
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        print(f"[INFO] 랜덤 시드: {RANDOM_SEED}")
    
    print("=" * 70)
    print("데이터셋 병합 시작")
    print(f"대상 디렉토리: {TARGET_DIR}")
    print("=" * 70)
    
    # 설정 요약 출력
    print("\n[설정 요약]")
    print("-" * 70)
    print(f"{'데이터셋':<35} {'prefix':<10} {'train':<10} {'val':<10}")
    print("-" * 70)
    for ds in DATASETS:
        train_str = "전체" if ds.train_count < 0 else str(ds.train_count)
        val_str = "전체" if ds.val_count < 0 else ("없음" if ds.val_count == 0 else str(ds.val_count))
        print(f"{ds.name:<35} {ds.prefix or '(없음)':<10} {train_str:<10} {val_str:<10}")
    print("-" * 70)
    
    total_train_img = 0
    total_train_lbl = 0
    total_val_img = 0
    total_val_lbl = 0
    
    for ds in DATASETS:
        print(f"\n[INFO] {ds.name} 처리 중...")
        src_path = Path(ds.path)
        
        # Train 처리
        if ds.train_count != 0:
            train_pairs = get_file_pairs(
                src_path / "images" / "train",
                src_path / "labels" / "train",
            )
            print(f"  - train 전체: {len(train_pairs)}개")
            
            sampled_train = sample_pairs(train_pairs, ds.train_count)
            print(f"  - train 샘플링: {len(sampled_train)}개")
            
            img_cnt, lbl_cnt = copy_pairs(
                sampled_train,
                TARGET_DIR / "images" / "train",
                TARGET_DIR / "labels" / "train",
                ds.prefix,
                ds.class_mapping,
            )
            print(f"  - train 복사 완료: 이미지 {img_cnt}개, 라벨 {lbl_cnt}개")
            total_train_img += img_cnt
            total_train_lbl += lbl_cnt
        
        # Val 처리
        if ds.val_count != 0:
            val_pairs = get_file_pairs(
                src_path / "images" / "val",
                src_path / "labels" / "val",
            )
            
            if val_pairs:
                print(f"  - val 전체: {len(val_pairs)}개")
                
                sampled_val = sample_pairs(val_pairs, ds.val_count)
                print(f"  - val 샘플링: {len(sampled_val)}개")
                
                img_cnt, lbl_cnt = copy_pairs(
                    sampled_val,
                    TARGET_DIR / "images" / "val",
                    TARGET_DIR / "labels" / "val",
                    ds.prefix,
                    ds.class_mapping,
                )
                print(f"  - val 복사 완료: 이미지 {img_cnt}개, 라벨 {lbl_cnt}개")
                total_val_img += img_cnt
                total_val_lbl += lbl_cnt
    
    print("\n" + "=" * 70)
    print("병합 완료!")
    print(f"  Train: 이미지 {total_train_img}개, 라벨 {total_train_lbl}개")
    print(f"  Val:   이미지 {total_val_img}개, 라벨 {total_val_lbl}개")
    print(f"  총합:  이미지 {total_train_img + total_val_img}개")
    print("=" * 70)
    
    # 캐시 파일 삭제 안내
    cache_files = list(TARGET_DIR.glob("**/*.cache"))
    if cache_files:
        print("\n[WARN] 기존 캐시 파일을 삭제하세요:")
        for cf in cache_files:
            print(f"  rm {cf}")


if __name__ == "__main__":
    merge_datasets()
