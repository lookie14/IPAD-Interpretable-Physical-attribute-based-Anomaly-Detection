"""Visualization: heatmap, overlay, candidate boxes (matplotlib, no cv2)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F


def upsample_score_map(
    score_map: torch.Tensor, image_size: Tuple[int, int]
) -> np.ndarray:
    """Bilinearly upsample a [H_p, W_p] score map to (H, W).

    Returns:
        np.ndarray[H, W] float32.
    """
    h, w = image_size
    x = score_map.detach().float().cpu()[None, None]     # [1, 1, H_p, W_p]
    up = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def normalize01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; constant maps -> zeros."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def save_heatmap(score_map_up: np.ndarray, out_path: str) -> None:
    """Save a normalized colormapped heatmap PNG."""
    norm = normalize01(score_map_up)
    plt.figure(figsize=(4, 4))
    plt.imshow(norm, cmap="jet")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close()


def save_overlay(
    orig_img: np.ndarray,          # [H, W, 3] uint8
    score_map_up: np.ndarray,      # [H, W]
    out_path: str,
    alpha: float = 0.5,
) -> None:
    """Overlay the heatmap on the original image."""
    norm = normalize01(score_map_up)
    plt.figure(figsize=(4, 4))
    plt.imshow(orig_img)
    plt.imshow(norm, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close()


def save_comparison_panel(
    orig_img: np.ndarray,               # [H, W, 3] uint8
    gt_mask: Optional[np.ndarray],      # [H, W] uint8 {0,255} or None (e.g. 'good')
    score_map_up: np.ndarray,           # [H, W]
    out_path: str,
    alpha: float = 0.5,
    title: str = "",
) -> None:
    """Save a single figure with three panels: original | ground truth | overlay."""
    norm = normalize01(score_map_up)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    axes[0].imshow(orig_img)
    axes[0].set_title("original", fontsize=10)

    if gt_mask is None:
        # No ground truth (normal/'good' image): show a blank panel.
        axes[1].imshow(np.zeros_like(orig_img))
        axes[1].set_title("ground truth (none)", fontsize=10)
    else:
        axes[1].imshow(orig_img)
        axes[1].imshow(gt_mask, cmap="Reds", alpha=(gt_mask > 0) * 0.6)
        axes[1].set_title("ground truth", fontsize=10)

    axes[2].imshow(orig_img)
    axes[2].imshow(norm, cmap="jet", alpha=alpha)
    axes[2].set_title("overlay (anomaly score)", fontsize=10)

    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=120)
    plt.close(fig)


def draw_candidate_boxes(
    ax,
    orig_img: np.ndarray,          # [H, W, 3] uint8
    candidates: List[Dict],
    show_score: bool = True,
    fontsize: int = 6,
) -> None:
    """Draw Top-K candidate bboxes (with score labels) onto an existing axis."""
    ax.imshow(orig_img)
    for c in candidates:
        x1, y1, x2, y2 = c["bbox"]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        if show_score:
            ax.text(
                x1, max(0, y1 - 2), f"{c['score']:.2f}",
                color="lime", fontsize=fontsize, va="bottom",
            )
    ax.axis("off")


def save_candidate_panel(
    orig_img: np.ndarray,               # [H, W, 3] uint8
    candidates: List[Dict],
    score_map_up: np.ndarray,           # [H, W]
    out_path: str,
    alpha: float = 0.5,
    title: str = "",
) -> None:
    """Save a single figure with three panels: original | candidate boxes | overlay."""
    norm = normalize01(score_map_up)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    axes[0].imshow(orig_img)
    axes[0].set_title("original", fontsize=10)

    draw_candidate_boxes(axes[1], orig_img, candidates)
    axes[1].set_title(f"candidate boxes ({len(candidates)})", fontsize=10)

    axes[2].imshow(orig_img)
    axes[2].imshow(norm, cmap="jet", alpha=alpha)
    axes[2].set_title("overlay (anomaly score)", fontsize=10)

    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=120)
    plt.close(fig)


def save_candidate_boxes(
    orig_img: np.ndarray,          # [H, W, 3] uint8
    candidates: List[Dict],
    out_path: str,
) -> None:
    """Draw Top-K candidate bboxes (with score labels) on the original image."""
    plt.figure(figsize=(4, 4))
    ax = plt.gca()
    draw_candidate_boxes(ax, orig_img, candidates)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=120)
    plt.close()


# --------------------------------------------------------------------------- #
# Stage 2 — masked-crop pipeline visualizations
# --------------------------------------------------------------------------- #
def save_image(img: np.ndarray, out_path: str, title: str = "") -> None:
    """Save a single [H, W, 3] uint8 image as a PNG (Stage 2 intermediate dumps)."""
    plt.figure(figsize=(3.5, 3.5))
    plt.imshow(img)
    if title:
        plt.title(title, fontsize=9)
    plt.axis("off")
    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=120)
    plt.close()


def save_topk_bar(topk: List[Dict], out_path: str, title: str = "") -> None:
    """Horizontal bar chart of Top-K physical-property cosine similarities."""
    labels = [t["label"] for t in topk][::-1]
    scores = [t["cosine"] for t in topk][::-1]
    plt.figure(figsize=(5.5, 0.45 * len(labels) + 1))
    plt.barh(labels, scores, color="steelblue")
    plt.xlabel("cosine similarity")
    if title:
        plt.title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=120)
    plt.close()
