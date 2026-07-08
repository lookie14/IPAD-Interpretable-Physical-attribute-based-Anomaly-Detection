"""Stage3 핵심 로직: 위치 시각화 + 속성 집계.

입력은 오직 list[PatchScore] (+ 원본 이미지) 뿐이며, streamlit에 의존하지 않는다.
(-> 함수 단위 테스트/재사용이 쉽도록 UI 코드와 분리)

diagram:
    patches ─┬─> to_grid -> blend -> draw_patch_labels     -> overlay
             └─> aggregate_by_attr -> argmax -> bar_chart   -> attr_chart, predicted_attr
"""
from __future__ import annotations

from collections import defaultdict

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from src.scoring.patch_score import PatchScore

matplotlib.use("Agg")  # streamlit/서버 환경에서 GUI 백엔드 불필요


# --------------------------------------------------------------------------- #
# 디자인 토큰 (dataviz 스킬의 검증된 팔레트에서 그대로 가져옴 — 임의 색상 아님)
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_BLUE = "#2a78d6"      # 일반 속성 막대 (categorical slot 1)
STATUS_CRITICAL = "#d03b3b"  # 예측(top-1) 속성 강조 (status: critical)
STATUS_ORANGE = "#eb6834"    # heat colormap 중간 톤 (categorical slot 8)
STATUS_YELLOW = "#eda100"    # heat colormap 상단 톤 (categorical slot 3)

# "semantic heat" 시퀀셜 예외 — jet(무지개) 대신 어둠→red→orange→yellow 로
# 단조 증가하는 밝기의 heat 컬러맵. 전부 팔레트에 이미 있는 hex 값이라 별도
# CVD 검증 없이도 안전 (팔레트 자체가 이미 검증됨).
_HEAT_STOPS = [
    (0.00, "#241009"),
    (0.35, "#7a1c12"),
    (0.62, STATUS_CRITICAL),
    (0.82, STATUS_ORANGE),
    (1.00, STATUS_YELLOW),
]
_ANOMALY_HEAT_NAME = "anomaly_heat"
try:
    matplotlib.colormaps.register(
        LinearSegmentedColormap.from_list(_ANOMALY_HEAT_NAME, _HEAT_STOPS), name=_ANOMALY_HEAT_NAME,
    )
except ValueError:
    pass  # 이미 등록됨 (streamlit 재실행으로 모듈이 다시 import 될 때)


def _setup_korean_font() -> None:
    """시스템에 설치된 한글 폰트를 찾아 matplotlib 기본 폰트로 설정.

    못 찾으면 기본 폰트를 그대로 두되(글자가 깨질 수 있음) 예외는 던지지 않는다.
    """
    candidates = (
        "Malgun Gothic",  # Windows
        "AppleGothic",  # macOS
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    )
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_korean_font()


def _load_label_font(size: int) -> ImageFont.ImageFont:
    """패치 라벨용 트루타입 폰트 로드 (있으면), 없으면 PIL 기본 비트맵 폰트."""
    for name in ("DejaVuSans-Bold.ttf", "arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


# ── 위치 시각화 ──────────────────────────────────────────────


def to_grid(patches: list[PatchScore], image_size: tuple[int, int]) -> np.ndarray:
    """패치별 geo_score를 원본 이미지 해상도의 2D map으로 변환.

    Args:
        patches: PatchScore 리스트.
        image_size: (W, H) 원본 이미지 크기.

    Returns:
        (H, W) float32 array. 값은 각 패치의 geo_score (겹치는 영역이 없다고 가정).
    """
    w, h = image_size
    grid = np.zeros((h, w), dtype=np.float32)
    for p in patches:
        x0, y0 = max(p.x, 0), max(p.y, 0)
        x1, y1 = min(p.x + p.w, w), min(p.y + p.h, h)
        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] = p.geo_score
    return grid


def to_mask(patches: list[PatchScore], image_size: tuple[int, int]) -> np.ndarray:
    """patches가 실제로 덮는 픽셀 영역을 bool mask로 반환.

    이상 패치가 이미지 일부(예: grid 중 몇 개)만 있을 때, 나머지 영역과
    구분하기 위해 blend()에 넘겨준다.

    Args:
        patches: PatchScore 리스트 (stage2가 이상으로 확정한 패치만).
        image_size: (W, H) 원본 이미지 크기.

    Returns:
        (H, W) bool array. True인 픽셀만 히트맵이 칠해진다.
    """
    w, h = image_size
    mask = np.zeros((h, w), dtype=bool)
    for p in patches:
        x0, y0 = max(p.x, 0), max(p.y, 0)
        x1, y1 = min(p.x + p.w, w), min(p.y + p.h, h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def blend(
    image: Image.Image,
    geo_map: np.ndarray,
    mask: np.ndarray | None = None,
    alpha: float = 0.45,
    cmap_name: str = _ANOMALY_HEAT_NAME,
) -> Image.Image:
    """원본 이미지 위에 geo_map을 히트맵으로 얹어 반환.

    mask를 주면 mask==True인 픽셀(=실제 이상 패치 영역)만 칠해지고,
    나머지 영역은 원본 그대로 남는다 (이상 패치가 이미지 일부에만 있는
    경우를 위함). mask를 안 주면 이미지 전체를 칠한다 (기존 동작).

    기본 컬러맵은 무지개(jet) 대신 어둠→red→orange→yellow 로 밝기가 단조
    증가하는 "semantic heat" 컬러맵(anomaly_heat) — 값이 클수록(더 이상할수록)
    더 밝고 뜨거운 색으로 읽힌다.

    Args:
        image: 원본 PIL 이미지.
        geo_map: to_grid()의 출력 (H, W).
        mask: to_mask()의 출력 (H, W) bool. None이면 전체 이미지를 칠함.
        alpha: 히트맵 블렌딩 강도 (0=원본만, 1=히트맵만).
        cmap_name: matplotlib colormap 이름.
    """
    base = image.convert("RGBA")
    norm = geo_map.astype(np.float32)

    # 정규화 기준값은 실제 칠해질 영역(mask) 안에서만 계산해야 대비가 뚜렷해진다.
    ref = norm[mask] if mask is not None else norm
    vmin, vmax = (float(ref.min()), float(ref.max())) if ref.size else (0.0, 0.0)
    normed = (norm - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(norm)

    cmap = matplotlib.colormaps[cmap_name]
    heat_rgb = (cmap(normed)[..., :3] * 255).astype(np.uint8)  # (H, W, 3)

    if mask is not None:
        alpha_channel = np.where(mask, int(round(alpha * 255)), 0).astype(np.uint8)
    else:
        alpha_channel = np.full(normed.shape, int(round(alpha * 255)), dtype=np.uint8)

    heat_rgba = np.dstack([heat_rgb, alpha_channel])
    heat_img = Image.fromarray(heat_rgba, mode="RGBA")
    if heat_img.size != base.size:
        heat_img = heat_img.resize(base.size, resample=Image.NEAREST)

    blended = Image.alpha_composite(base, heat_img)
    return blended.convert("RGB")


def heat_scale_bar(width: int = 220, height: int = 14, cmap_name: str = _ANOMALY_HEAT_NAME) -> Image.Image:
    """anomaly_heat 컬러맵의 가로 스케일 바 (범례용, "낮음 → 높음" 텍스트는 호출부에서)."""
    cmap = matplotlib.colormaps[cmap_name]
    ramp = (cmap(np.linspace(0, 1, width))[:, :3] * 255).astype(np.uint8)  # (width, 3)
    arr = np.tile(ramp[None, :, :], (height, 1, 1))
    return Image.fromarray(arr, mode="RGB")


def draw_patch_labels(
    overlay: Image.Image,
    patches: list[PatchScore],
    box_color: str = "#ffffff",
    text_color: str = "#ffffff",
    backing_color: tuple[int, int, int, int] = (11, 11, 11, 175),
) -> Image.Image:
    """각 패치 위치(patch_coords)에 top-1 속성명을 직접 라벨링.

    라벨 텍스트 밑에 반투명 어두운 배경을 깔아, heat 컬러맵 위에서도(밝은
    노란 영역 포함) 흰 글씨가 항상 또렷하게 읽히도록 한다.

    Args:
        overlay: blend()의 출력 이미지 (수정하지 않고 복사본에 그림).
        patches: PatchScore 리스트.
    """
    img = overlay.convert("RGBA")
    label_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(label_layer)
    font = _load_label_font(size=12)

    for p in patches:
        x0, y0, x1, y1 = p.box
        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=1)

        label = f"{p.top1_attr} {p.top1_score:.2f}"
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:  # 구버전 Pillow fallback
            tw, th = draw.textsize(label, font=font)

        pad = 2
        bx0, by0 = x0, max(0, y0 - th - 2 * pad)
        draw.rectangle([bx0, by0, bx0 + tw + 2 * pad, by0 + th + 2 * pad], fill=backing_color)
        draw.text((bx0 + pad, by0 + pad), label, fill=text_color, font=font)

    return Image.alpha_composite(img, label_layer).convert("RGB")


# ── 속성 집계 ────────────────────────────────────────────────


def aggregate_by_attr(patches: list[PatchScore]) -> dict[str, float]:
    """패치별 top-1 속성명 기준으로 점수를 집계 (평균).

    같은 top-1 속성을 가진 패치들의 top1_score를 평균낸다.

    Returns:
        {attr_name: score} — 내부값. UI에는 attr_chart로만 노출.
    """
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for p in patches:
        attr = p.top1_attr
        sums[attr] += p.top1_score
        counts[attr] += 1
    return {attr: sums[attr] / counts[attr] for attr in sums}


def predict_attr(attr_scores: dict[str, float]) -> str:
    """attr_scores에서 argmax로 이미지 전체 대표 속성 1개를 결정."""
    if not attr_scores:
        raise ValueError("attr_scores is empty")
    return max(attr_scores, key=attr_scores.get)


def bar_chart(
    attr_scores: dict[str, float],
    highlight: str | None = None,
    title: str = "속성별 집계 점수",
) -> Figure:
    """속성별 집계 점수를 막대그래프로 렌더링 (predicted_attr 강조).

    마크: 얇은 막대(슬롯의 60%만 채움) + 대비되는 예측 속성 색(critical red) +
    recessive gridline/spine + 범례(색이 곧 정체성이 되지 않도록 텍스트 라벨 동반).
    """
    names = sorted(attr_scores, key=attr_scores.get, reverse=True)
    values = [attr_scores[n] for n in names]
    colors = [STATUS_CRITICAL if n == highlight else SERIES_BLUE for n in names]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1.0, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)
    ax.tick_params(axis="both", colors=TEXT_MUTED, labelsize=9)

    bars = ax.bar(names, values, width=0.6, color=colors, zorder=3)
    ax.set_ylabel("score", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.set_ylim(0, max(values + [1.0]) * 1.18)

    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{v:.2f}",
            ha="center", va="bottom", fontsize=9, color=TEXT_PRIMARY,
        )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color=TEXT_SECONDARY)

    if highlight is not None and highlight in names:
        legend_handles = [
            mpatches.Patch(facecolor=SERIES_BLUE, edgecolor="none", label="속성별 점수"),
            mpatches.Patch(facecolor=STATUS_CRITICAL, edgecolor="none", label="예측 속성 (top-1)"),
        ]
        ax.legend(
            handles=legend_handles, frameon=False, loc="upper right", fontsize=8.5,
            labelcolor=TEXT_SECONDARY,
        )

    fig.tight_layout()
    return fig
