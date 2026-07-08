"""Multi-layer position-wise scoring with optional neighborhood fusion.

Project3 keeps the main assumption from ``project``: a target patch at grid
position (i, j) is compared only with reference patches at the same position.
It adds the useful parts of ``project2``: multi-layer feature extraction,
per-layer score maps, and a shared runner/evaluator shape.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def build_local_descriptors(features: torch.Tensor, window: int) -> torch.Tensor:
    """[..., H, W, C] -> [..., H, W, (2w+1)^2*C].

    Neighbor features are concatenated in row-major order. They are not averaged.
    Border positions use replicate padding so every grid cell has a full window.
    """
    if window <= 0:
        return features
    lead = features.shape[:-3]
    h, w, c = features.shape[-3:]
    x = features.reshape(-1, h, w, c).permute(0, 3, 1, 2)
    x = F.pad(x, (window, window, window, window), mode="replicate")
    k = 2 * window + 1
    tiles = [x[:, :, dy:dy + h, dx:dx + w] for dy in range(k) for dx in range(k)]
    out = torch.cat(tiles, dim=1).permute(0, 2, 3, 1)
    return out.reshape(*lead, h, w, k * k * c)


def position_wise_score_map(
    target_features: torch.Tensor,
    ref_features: torch.Tensor,
    metric: str = "cosine",
    neighborhood: int = 0,
    k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score one layer by comparing only matching grid positions.

    Args:
        target_features: [H_p, W_p, C].
        ref_features: [N_ref, H_p, W_p, C].
        metric: "cosine" or "l2".
        neighborhood: 0=1x1, 1=3x3 concat, 2=5x5 concat.
        k: average the nearest k references at each position. k=1 is min distance.

    Returns:
        score_map: [H_p, W_p].
        ref_distance_map: [H_p, W_p, N_ref].
        nearest_ref_idx_map: [H_p, W_p].
    """
    if target_features.dim() != 3:
        raise ValueError(f"target_features must be [H,W,C], got {tuple(target_features.shape)}")
    if ref_features.dim() != 4:
        raise ValueError(f"ref_features must be [N,H,W,C], got {tuple(ref_features.shape)}")

    target_desc = build_local_descriptors(target_features, neighborhood)
    ref_desc = build_local_descriptors(ref_features, neighborhood)

    ref = ref_desc.permute(1, 2, 0, 3)      # [H, W, N, D]
    tgt = target_desc.unsqueeze(2)          # [H, W, 1, D]

    if metric == "cosine":
        t = F.normalize(tgt, p=2, dim=-1)
        r = F.normalize(ref, p=2, dim=-1)
        ref_distance_map = 1.0 - (r * t).sum(dim=-1)
    elif metric == "l2":
        ref_distance_map = torch.norm(ref - tgt, dim=-1)
    else:
        raise ValueError(f"[scoring] unknown metric '{metric}'.")

    kk = max(1, min(int(k), ref_distance_map.shape[-1]))
    if kk == 1:
        score_map, nearest_ref_idx_map = ref_distance_map.min(dim=-1)
    else:
        top = ref_distance_map.topk(kk, dim=-1, largest=False)
        score_map = top.values.mean(dim=-1)
        nearest_ref_idx_map = top.indices[..., 0]
    return score_map, ref_distance_map, nearest_ref_idx_map


def _fuse_maps(maps: List[torch.Tensor], weights, normalize: str) -> torch.Tensor:
    if not maps:
        raise ValueError("[scoring] no maps to fuse.")
    if len(maps) == 1:
        return maps[0]
    w = torch.tensor([float(x) for x in (weights or [1.0] * len(maps))],
                     dtype=maps[0].dtype, device=maps[0].device)
    w = w / (w.sum() + 1e-8)

    def norm(m: torch.Tensor) -> torch.Tensor:
        if normalize == "minmax":
            lo, hi = m.min(), m.max()
            return (m - lo) / (hi - lo + 1e-8)
        if normalize == "zscore":
            return (m - m.mean()) / (m.std() + 1e-8)
        if normalize == "none":
            return m
        raise ValueError(f"[scoring] unknown fuse normalize mode '{normalize}'.")

    out = torch.zeros_like(maps[0])
    for wi, m in zip(w, maps):
        out = out + wi * norm(m)
    return out


def resolve_fuse(scoring_cfg: dict):
    """Return (neighborhoods, weights, normalize) from config.scoring."""
    fc = scoring_cfg.get("fuse") or {}
    if fc.get("enabled", False):
        return [int(x) for x in fc.get("neighborhoods", [0, 1])], fc.get("weights"), fc.get("normalize", "none")
    return [int(scoring_cfg.get("neighborhood", 0))], None, "none"


def score(
    target_feats: Dict[int, torch.Tensor],
    ref_features: Dict[int, torch.Tensor],
    layers: List[int],
    neighborhoods: List[int] = (0,),
    weights=None,
    normalize: str = "none",
    metric: str = "cosine",
    k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score all layers, fuse neighborhoods inside each layer, then average layers.

    Returns:
        score_map: [H_p, W_p].
        layer_score_maps: [H_p, W_p, L].
        first_ref_distance_map: [H_p, W_p, N_ref] from first layer/neighborhood.
        first_nearest_ref_idx_map: [H_p, W_p] from first layer/neighborhood.
    """
    layer_maps = []
    first_ref_distance_map = None
    first_nearest_ref_idx_map = None

    for li in layers:
        nb_maps = []
        for nb in neighborhoods:
            sm, rd, ni = position_wise_score_map(
                target_feats[li], ref_features[li], metric=metric, neighborhood=nb, k=k
            )
            if first_ref_distance_map is None:
                first_ref_distance_map = rd
                first_nearest_ref_idx_map = ni
            nb_maps.append(sm)
        layer_maps.append(_fuse_maps(nb_maps, weights, normalize))

    stack = torch.stack(layer_maps, dim=-1)
    return stack.mean(dim=-1), stack, first_ref_distance_map, first_nearest_ref_idx_map
