"""stage1/2 파이프라인 연결 전, stage3 UI 개발/테스트용 더미 데이터 생성기.

실제 파이프라인이 준비되면 이 모듈 대신 candidate/scoring 결과를 PatchScore로
변환하는 어댑터로 교체하면 된다 (stage3_core.py의 인터페이스는 그대로 유지).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.scoring.patch_score import PatchScore

DEFAULT_ATTRS = ("scratch", "dent", "stain", "crack", "discoloration")


def make_dummy_image(size: tuple[int, int] = (384, 384), seed: int = 0) -> Image.Image:
    """단순 노이즈 배경 + 중앙 그라데이션의 더미 이미지."""
    rng = np.random.default_rng(seed)
    w, h = size
    base = rng.integers(90, 140, size=(h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist = (dist / dist.max() * 60).astype(np.uint8)
    base = np.clip(base.astype(np.int16) + dist[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(base, "RGB")


def make_dummy_patches(
    image_size: tuple[int, int] = (384, 384),
    grid: tuple[int, int] = (6, 6),
    attrs: tuple[str, ...] = DEFAULT_ATTRS,
    seed: int = 42,
    image_id: str = "dummy",
    anomaly_ratio: float = 1.0,
) -> list[PatchScore]:
    """이미지를 grid(cols, rows)로 균등 분할해 더미 PatchScore 리스트 생성.

    Args:
        anomaly_ratio: grid 셀 중 실제로 "이상 패치"로 뽑힐 비율 (0~1).
            1.0이면 모든 셀이 patches로 나옴(기존 동작). 1.0 미만이면
            stage2가 일부 패치만 이상으로 확정해 넘기는 상황을 흉내낸다
            (나머지 셀은 patches 리스트에 아예 포함되지 않음).
    """
    rng = np.random.default_rng(seed)
    w, h = image_size
    cols, rows = grid
    pw, ph = w // cols, h // rows

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    if anomaly_ratio < 1.0:
        n_keep = max(1, round(len(cells) * anomaly_ratio))
        keep_idx = rng.choice(len(cells), size=n_keep, replace=False)
        cells = [cells[i] for i in sorted(keep_idx)]

    patches: list[PatchScore] = []
    for patch_id, (r, c) in enumerate(cells):
        geo_score = float(rng.random())
        attr_scores = {a: float(rng.random()) for a in attrs}
        patches.append(
            PatchScore(
                patch_id=patch_id,
                x=c * pw,
                y=r * ph,
                w=pw,
                h=ph,
                geo_score=geo_score,
                attr_scores=attr_scores,
                image_id=image_id,
            )
        )
    return patches
