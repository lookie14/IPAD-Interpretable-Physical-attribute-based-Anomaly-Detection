"""PatchScore: stage2에서 확정된 이상 패치 1개에 대한 점수 데이터.

stage3(위치 시각화 / 속성 집계)의 유일한 입력 스키마.
실제 stage1/stage2 파이프라인이 준비되면 이 dataclass만 채워서 넘기면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PatchScore:
    """단일 패치의 이상 위치/속성 점수.

    Attributes:
        patch_id: 패치 인덱스 (이미지 내 고유).
        x, y: 패치 좌상단 좌표 (원본 이미지 픽셀 기준).
        w, h: 패치 폭/높이 (픽셀).
        geo_score: 위치 시각화용 이상 점수 (0~1). 값이 높을수록 이상 가능성 ↑.
        attr_scores: 속성명 -> 점수. 이 패치가 어떤 이상 속성(예: scratch, dent 등)에
            해당하는지에 대한 점수 딕셔너리.
        image_id: 어느 이미지에 속한 패치인지 구분용 (배치 처리 대비).
    """

    patch_id: int
    x: int
    y: int
    w: int
    h: int
    geo_score: float
    attr_scores: dict[str, float] = field(default_factory=dict)
    image_id: str = "default"

    @property
    def top1_attr(self) -> str:
        """이 패치에서 점수가 가장 높은 속성명."""
        if not self.attr_scores:
            raise ValueError(f"patch {self.patch_id}: attr_scores is empty")
        return max(self.attr_scores, key=self.attr_scores.get)

    @property
    def top1_score(self) -> float:
        """top1_attr의 점수."""
        return self.attr_scores[self.top1_attr]

    @property
    def coords(self) -> tuple[int, int, int, int]:
        """(x, y, w, h) 튜플."""
        return (self.x, self.y, self.w, self.h)

    @property
    def box(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) 튜플 — PIL/matplotlib 등에서 바로 사용."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)
