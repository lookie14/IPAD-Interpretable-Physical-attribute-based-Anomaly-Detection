"""Candidate patch extraction -> Stage 2 handoff (global-NN multi-layer 버전).

score 는 layer 평균 global-NN 거리. 각 candidate 에 layer 별 score 도 함께 담는다.
선택 방식: Top-K (개수/비율) 또는 threshold.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def _build_record(
    i: int, j: int,
    score_map: torch.Tensor,            # [H_p, W_p]
    layer_score_maps: torch.Tensor,     # [H_p, W_p, L]
    target_feature: Optional[torch.Tensor],  # [C_total] or None
    patch_h: float, patch_w: float,
    save_feature: bool,
) -> Dict:
    x1 = int(round(j * patch_w))
    y1 = int(round(i * patch_h))
    x2 = int(round((j + 1) * patch_w))
    y2 = int(round((i + 1) * patch_h))
    rec: Dict = {
        "grid_idx": [int(i), int(j)],
        "bbox": [x1, y1, x2, y2],
        "score": float(score_map[i, j].item()),
        "layer_scores": [float(v) for v in layer_score_maps[i, j].tolist()],
    }
    if save_feature and target_feature is not None:
        rec["feature"] = [float(v) for v in target_feature.tolist()]
    return rec


def select_topk(
    score_map: torch.Tensor,            # [H_p, W_p]
    layer_score_maps: torch.Tensor,     # [H_p, W_p, L]
    image_size: Tuple[int, int],        # (H, W)
    patch_grid_size: Tuple[int, int],   # (H_p, W_p)
    topk_ratio: float = 0.05,
    topk_k: Optional[int] = None,
    feature_map: Optional[torch.Tensor] = None,   # [H_p, W_p, C_total] (concat layers) or None
    save_feature: bool = True,
) -> List[Dict]:
    """점수 상위 K개 patch → candidate. topk_k 지정 시 topk_ratio 무시."""
    h_p, w_p = patch_grid_size
    img_h, img_w = image_size
    patch_h, patch_w = img_h / h_p, img_w / w_p
    total = h_p * w_p
    k = int(topk_k) if topk_k is not None else int(round(total * topk_ratio))
    k = max(1, min(k, total))

    idxs = torch.topk(score_map.reshape(-1), k=k, largest=True).indices.tolist()
    cands = []
    for fi in idxs:
        i, j = fi // w_p, fi % w_p
        feat = feature_map[i, j] if (feature_map is not None) else None
        cands.append(_build_record(i, j, score_map, layer_score_maps, feat,
                                   patch_h, patch_w, save_feature))
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def select_by_threshold(
    score_map: torch.Tensor,            # [H_p, W_p]
    layer_score_maps: torch.Tensor,     # [H_p, W_p, L]
    image_size: Tuple[int, int],
    patch_grid_size: Tuple[int, int],
    threshold: float,
    feature_map: Optional[torch.Tensor] = None,
    save_feature: bool = True,
) -> List[Dict]:
    """score >= threshold 인 patch 전부 → candidate (없으면 빈 리스트)."""
    h_p, w_p = patch_grid_size
    img_h, img_w = image_size
    patch_h, patch_w = img_h / h_p, img_w / w_p

    ys, xs = torch.where(score_map >= float(threshold))
    cands = []
    for i, j in zip(ys.tolist(), xs.tolist()):
        feat = feature_map[i, j] if (feature_map is not None) else None
        cands.append(_build_record(i, j, score_map, layer_score_maps, feat,
                                   patch_h, patch_w, save_feature))
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands
