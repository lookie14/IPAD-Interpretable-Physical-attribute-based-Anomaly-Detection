"""Padded-crop geometry helpers for Stage 2 (bbox expansion, clamping, local coords)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def expand_bbox(
    bbox: BBox,
    image_hw: Tuple[int, int],
    padding_ratio: float = 1.0,
    padding_px: Optional[int] = None,
    min_crop_size: int = 0,
    square: bool = True,
) -> BBox:
    """Expand ``bbox`` with padding, clamped to image bounds.

    ``padding_px`` (if given) is an absolute pixel margin added on each side;
    otherwise ``padding_ratio`` scales the margin relative to the bbox's own
    size (``margin = padding_ratio * max(bbox_w, bbox_h)``).

    ``min_crop_size`` enforces a lower bound on the padded crop's side length
    so very small candidate boxes still get enough context/resolution once
    resized for CLIP.

    ``square=True`` forces the padded box to be square *before* clamping, so
    that (for the typical case where the source image is itself square,
    e.g. Stage 1's resized input) CLIP's own resize does not need to
    center-crop the result -- avoiding any risk of cropping the emphasized
    region out of a non-square window.
    """
    h, w = image_hw
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    if padding_px is not None:
        margin_x = margin_y = float(padding_px)
    else:
        margin = float(padding_ratio) * max(bw, bh)
        margin_x = margin_y = margin

    half_w = bw / 2.0 + margin_x
    half_h = bh / 2.0 + margin_y
    if square:
        half_w = half_h = max(half_w, half_h)
    half_w = max(half_w, min_crop_size / 2.0)
    half_h = max(half_h, min_crop_size / 2.0)

    px1 = int(round(cx - half_w))
    py1 = int(round(cy - half_h))
    px2 = int(round(cx + half_w))
    py2 = int(round(cy + half_h))

    # Clamp to image bounds by sliding the window inward on overflow, rather
    # than truncating it, so the requested crop size is preserved whenever
    # the image is large enough to hold it.
    pw, ph = px2 - px1, py2 - py1
    if px1 < 0:
        px1, px2 = 0, min(w, pw)
    if px2 > w:
        px2 = w
        px1 = max(0, w - pw)
    if py1 < 0:
        py1, py2 = 0, min(h, ph)
    if py2 > h:
        py2 = h
        py1 = max(0, h - ph)

    return int(px1), int(py1), int(px2), int(py2)


def to_local_bbox(bbox: BBox, padded_bbox: BBox) -> BBox:
    """Translate ``bbox`` (original-image coords) into ``padded_bbox``-local coords."""
    px1, py1, px2, py2 = padded_bbox
    x1, y1, x2, y2 = bbox
    crop_w, crop_h = px2 - px1, py2 - py1
    lx1 = min(max(x1 - px1, 0), crop_w)
    ly1 = min(max(y1 - py1, 0), crop_h)
    lx2 = min(max(x2 - px1, 0), crop_w)
    ly2 = min(max(y2 - py1, 0), crop_h)
    return int(lx1), int(ly1), int(lx2), int(ly2)


def crop_array(img: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop ``img`` [H, W, 3] to ``bbox`` (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2].copy()
