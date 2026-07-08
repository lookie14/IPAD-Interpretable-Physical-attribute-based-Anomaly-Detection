"""MVTec-AD style folder access (reference selection + test listing).

Expected layout::

    data_root/
      <category>/
        train/good/*.png
        test/<anomaly_type>/*.png
        ground_truth/<anomaly_type>/*_mask.png   (unused in Stage 1)
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _sorted_images(pattern_dir: str) -> List[str]:
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(pattern_dir, f"*{ext}")))
    return sorted(files)


def select_reference_paths(
    data_root: str, category: str, n: int, seed: int = 42,
    indices: Optional[List[int]] = None,
) -> List[str]:
    """Pick normal reference images from ``train/good``.

    두 가지 모드:
      - ``indices`` 지정 시: 파일 숫자 stem(예: 12 -> ``012.png``)이 그 목록에 든
        이미지를 선택한다(seed/n 무시). 재현·직접 지정용.
      - ``indices`` 미지정 시: seed 로 고정된 무작위 n 장(기존 동작).
    반환은 항상 정렬해 순서 결정성 유지.

    Returns:
        List[str] (indices 모드면 그 개수, 아니면 최대 n).
    """
    good_dir = os.path.join(data_root, category, "train", "good")
    files = _sorted_images(good_dir)
    if len(files) == 0:
        raise FileNotFoundError(
            f"[dataset] No reference images found in '{good_dir}'."
        )

    if indices:                                    # 직접 인덱스 선택
        want = {int(i) for i in indices}
        picked = [
            f for f in files
            if os.path.splitext(os.path.basename(f))[0].isdigit()
            and int(os.path.splitext(os.path.basename(f))[0]) in want
        ]
        if not picked:
            raise FileNotFoundError(
                f"[dataset] reference_indices {sorted(want)} 에 해당하는 이미지가 "
                f"'{good_dir}' 에 없습니다."
            )
        return picked

    import random

    if len(files) <= n:
        print(f"[dataset] Requested {n} refs but only {len(files)} available; using all.")
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, n))


def list_test_images(
    data_root: str,
    category: str,
    anomaly_types: Optional[List[str]] = None,
    indices: Optional[List[int]] = None,
) -> List[Tuple[str, str]]:
    """List test images as (path, anomaly_type), with optional filtering.

    ``anomaly_type`` is the immediate subfolder under ``test`` (e.g. 'good',
    'broken', 'bent').

    Args:
        anomaly_types: if given, keep only these subfolders (e.g. ['bent', 'good']).
        indices: if given, keep only images whose numeric stem is in this set
                 (e.g. [0, 1] -> '000.png', '001.png').

    Returns:
        List[(image_path, anomaly_type)].
    """
    test_dir = os.path.join(data_root, category, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"[dataset] Test dir not found: '{test_dir}'.")

    want_types = set(anomaly_types) if anomaly_types else None
    want_idx = set(int(i) for i in indices) if indices else None

    items: List[Tuple[str, str]] = []
    for anomaly_type in sorted(os.listdir(test_dir)):
        if want_types is not None and anomaly_type not in want_types:
            continue
        sub = os.path.join(test_dir, anomaly_type)
        if not os.path.isdir(sub):
            continue
        for path in _sorted_images(sub):
            if want_idx is not None:
                stem = os.path.splitext(os.path.basename(path))[0]
                try:
                    if int(stem) not in want_idx:
                        continue
                except ValueError:
                    continue
            items.append((path, anomaly_type))
    if not items:
        raise FileNotFoundError(
            f"[dataset] No test images matched under '{test_dir}' "
            f"(anomaly_types={anomaly_types}, indices={indices})."
        )
    return items


def ground_truth_path(
    data_root: str, category: str, anomaly_type: str, image_path: str
) -> Optional[str]:
    """Resolve the ground-truth mask path for a test image.

    MVTec layout: ``ground_truth/<anomaly_type>/<stem>_mask.png``.
    Returns None when no mask exists (e.g. 'good' images, or missing files).
    """
    if anomaly_type == "good":
        return None
    stem = os.path.splitext(os.path.basename(image_path))[0]
    gt_dir = os.path.join(data_root, category, "ground_truth", anomaly_type)
    for cand in (f"{stem}_mask.png", f"{stem}.png"):
        p = os.path.join(gt_dir, cand)
        if os.path.isfile(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# Synthetic dataset — smoke-testing the pipeline without real MVTec data.
# --------------------------------------------------------------------------- #
def make_synthetic_dataset(
    data_root: str,
    category: str,
    image_size: int = 224,
    n_train: int = 6,
    n_test_good: int = 2,
    n_test_defect: int = 3,
    seed: int = 42,
) -> None:
    """Generate a tiny MVTec-like dataset (normal texture + defective blobs).

    Only used for the dummy smoke test. No-op-safe: overwrites/creates files.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)

    def _normal(size: int) -> np.ndarray:
        base = np.full((size, size, 3), 180, dtype=np.float32)
        base += rng.normal(0, 6, size=base.shape)  # mild texture
        return np.clip(base, 0, 255).astype(np.uint8)

    def _defect(size: int) -> np.ndarray:
        img = _normal(size).astype(np.float32)
        # paint a bright square "defect"
        h = rng.integers(size // 8, size // 4)
        y = rng.integers(0, size - h)
        x = rng.integers(0, size - h)
        img[y : y + h, x : x + h, :] = rng.integers(0, 60)  # dark blob
        return np.clip(img, 0, 255).astype(np.uint8)

    def _save(arr: np.ndarray, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(arr).save(path)

    root = os.path.join(data_root, category)
    for i in range(n_train):
        _save(_normal(image_size), os.path.join(root, "train", "good", f"{i:03d}.png"))
    for i in range(n_test_good):
        _save(_normal(image_size), os.path.join(root, "test", "good", f"{i:03d}.png"))
    for i in range(n_test_defect):
        _save(_defect(image_size), os.path.join(root, "test", "defect", f"{i:03d}.png"))

    print(
        f"[dataset] Synthetic dataset written to '{root}' "
        f"(train/good={n_train}, test/good={n_test_good}, test/defect={n_test_defect})."
    )
