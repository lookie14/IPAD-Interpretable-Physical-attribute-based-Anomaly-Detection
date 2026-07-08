"""Frozen DINOv3 MULTI-LAYER patch-token feature extractor.

project(단일 레이어)와 달리, 지정한 여러 block(예: [2, 5, 8, 11])의 patch-token
feature map 을 각각 반환한다. class token 은 제거하고 patch grid 로만 반환.
학습 없음(frozen).

TODO: support DINOv2 / CLIP / Swin backbone (wrapper 인터페이스 유지).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Dummy backbone: 가중치/데이터 없이 흐름 검증용.
# --------------------------------------------------------------------------- #
class _DummyDINO(nn.Module):
    """랜덤 초기화 ViT 스텁 — DINOv3 hub 인터페이스(get_intermediate_layers) 흉내."""

    def __init__(self, embed_dim: int = 768, patch_size: int = 16, depth: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(depth)])

    def get_intermediate_layers(self, x, n, reshape: bool = True, norm: bool = True):
        feat = self.patch_embed(x)                       # [B, C, H_p, W_p]
        # layer 마다 살짝 다른 표현이 나오도록 index 로 스케일 변주 (검증용).
        idxs = list(n) if isinstance(n, (list, tuple)) else list(range(self.depth - int(n), self.depth))
        outs = []
        for k in idxs:
            outs.append(feat * (1.0 + 0.05 * float(k)))
        return tuple(outs)


# --------------------------------------------------------------------------- #
# Backbone loading — project 와 동일 (meta_local / dummy).
# --------------------------------------------------------------------------- #
def load_backbone(cfg: Dict[str, Any]) -> nn.Module:
    """frozen backbone 로드. source: 'meta_local' | 'dummy'."""
    bb = cfg["backbone"]
    source = bb.get("source", "meta_local")

    if source == "dummy":
        name = bb.get("name", "dinov3_vitb16")
        embed_dim = 384 if "vits" in name else 1024 if "vitl" in name else 768
        return _DummyDINO(embed_dim=embed_dim, patch_size=16, depth=12)

    if source == "meta_local":
        repo_dir = bb.get("repo_dir")
        weights_path = bb.get("weights_path")
        name = bb.get("name", "dinov3_vitb16")
        if not repo_dir or not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                "[backbone] source='meta_local' 에는 로컬 DINOv3 repo clone 이 필요합니다.\n"
                f"  repo_dir='{repo_dir}' 없음. "
                "git clone https://github.com/facebookresearch/dinov3 <repo_dir>"
            )
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                "[backbone] source='meta_local' 에는 Meta 원본 .pth 가중치가 필요합니다.\n"
                f"  weights_path='{weights_path}' 없음."
            )
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        try:
            from dinov3.hub import backbones as _bk  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"[backbone] dinov3 backbones import 실패 ({repo_dir}): {type(e).__name__}: {e}"
            ) from e
        entry = getattr(_bk, name, None)
        if entry is None:
            raise ValueError(f"[backbone] entrypoint '{name}' 없음 (예: dinov3_vitb16).")
        try:
            return entry(pretrained=True, weights=weights_path)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"[backbone] '{name}' 빌드 실패 (weights={weights_path}): {type(e).__name__}: {e}"
            ) from e

    raise ValueError(f"[backbone] 알 수 없는 source '{source}'. 'meta_local' 또는 'dummy'.")


# --------------------------------------------------------------------------- #
# Multi-layer extractor
# --------------------------------------------------------------------------- #
class DINOv3MultiLayerExtractor(nn.Module):
    """Frozen backbone → 여러 layer 의 patch grid feature.

    forward -> Dict[layer_index, Tensor[B, H_p, W_p, C]]
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        bb = cfg["backbone"]
        self.backbone = load_backbone(cfg)
        self.normalize_feature: bool = bool(bb.get("normalize_feature", True))

        depth = self._infer_depth()
        raw = bb.get("layer_indices", [2, 5, 8, 11])
        self.layer_indices: List[int] = [max(0, min(int(i), depth - 1)) for i in raw]

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _infer_depth(self) -> int:
        for attr in ("blocks", "layers"):
            mod = getattr(self.backbone, attr, None)
            if mod is not None:
                try:
                    return len(mod)
                except TypeError:
                    pass
        return int(getattr(self.backbone, "depth", 12))

    def train(self, mode: bool = True):  # keep frozen
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> Dict[int, torch.Tensor]:
        """[B, 3, H, W] -> {layer_index: [B, H_p, W_p, C]} (patch token only).

        cosine 비교를 위해 normalize_feature 시 각 layer feature 를 채널축 L2 정규화.
        """
        feats = self.backbone.get_intermediate_layers(
            images, n=self.layer_indices, reshape=True, norm=True
        )  # tuple of [B, C, H_p, W_p], layer_indices 순서
        out: Dict[int, torch.Tensor] = {}
        for li, f in zip(self.layer_indices, feats):
            g = f.permute(0, 2, 3, 1).contiguous()       # [B, H_p, W_p, C]
            if self.normalize_feature:
                g = F.normalize(g, p=2, dim=-1)
            out[li] = g
        return out
