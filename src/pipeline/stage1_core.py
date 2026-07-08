"""Stage1 핵심 로직: DINOv3 특징 추출 + position-wise 스코어링 + candidate 선택.

streamlit에 의존하지 않는 순수 함수들 (src/viz/stage3_core.py와 동일한 패턴).
아카이브에서 이식한 src/models, src/scoring, src/candidate, src/utils 위에 얹은
오케스트레이션 레이어 — pages/1_Stage1.py 가 이 함수들을 호출한다.

diagram:
    ref_pils, test_pil
        │
        ▼
    (옵션) 회전 정렬 ── align_reference_set / derotate_pil
        │
        ▼
    build_extractor ── DINOv3MultiLayerExtractor.forward
        │
        ▼
    build_reference_features (ref) , score_test_image (test)
        │
        ▼
    select_candidates ── select_topk | select_by_threshold
        │
        ▼
    overlay_heatmap + draw_candidates  →  stage1 최종 output
    (candidates 리스트는 "Stage 2 handoff" 스키마 그대로)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.candidate.topk_selector import select_by_threshold, select_topk
from src.models.dinov3_feature_extractor import DINOv3MultiLayerExtractor
from src.scoring.position_wise_multi_layer_score import resolve_fuse, score as score_fn
from src.utils.image_utils import denormalize, image_hw, preprocess
from src.utils.rotation_adapter import align_reference_set, derotate_pil
from src.utils.visualize import normalize01, upsample_score_map

import matplotlib


# ── 더미 데이터 (실제 이미지 없이 UI 테스트용) ──────────────────────────


def make_dummy_ref_and_test(
    image_size: int = 224, n_ref: int = 4, seed: int = 0,
) -> Tuple[List[Image.Image], Image.Image]:
    """실제 데이터 없이 UI를 테스트하기 위한 간단한 합성 이미지 세트.

    reference: 옅은 노이즈 텍스처 n장. test: 같은 텍스처 + 이상 블롭 하나.
    """
    rng = np.random.default_rng(seed)

    def _normal() -> Image.Image:
        base = np.full((image_size, image_size, 3), 180, dtype=np.float32)
        base += rng.normal(0, 6, size=base.shape)
        return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")

    ref_pils = [_normal() for _ in range(n_ref)]

    test_arr = np.asarray(_normal(), dtype=np.float32)
    h = int(rng.integers(image_size // 8, image_size // 4))
    y = int(rng.integers(0, image_size - h))
    x = int(rng.integers(0, image_size - h))
    test_arr[y:y + h, x:x + h, :] = rng.integers(0, 60)
    test_pil = Image.fromarray(np.clip(test_arr, 0, 255).astype(np.uint8), "RGB")
    return ref_pils, test_pil


# ── extractor / feature 추출 ────────────────────────────────────────


def build_backbone_cfg(
    source: str = "dummy",
    name: str = "dinov3_vitb16",
    repo_dir: str = "",
    weights_path: str = "",
    layer_indices: Optional[List[int]] = None,
    normalize_feature: bool = True,
) -> dict:
    """DINOv3MultiLayerExtractor 가 기대하는 cfg dict 생성."""
    return {
        "backbone": {
            "source": source,
            "name": name,
            "repo_dir": repo_dir,
            "weights_path": weights_path,
            "layer_indices": list(layer_indices or [8]),
            "normalize_feature": normalize_feature,
        }
    }


def build_extractor(backbone_cfg: dict, device: torch.device) -> DINOv3MultiLayerExtractor:
    """extractor 생성 + device 이동 (무거운 연산 — 호출부에서 캐싱 권장)."""
    return DINOv3MultiLayerExtractor(backbone_cfg).to(device)


def extract_one(
    extractor: DINOv3MultiLayerExtractor, pil: Image.Image, image_size: int, device: torch.device,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """이미지 한 장 -> ({layer: [H_p,W_p,C]}, 전처리된 tensor[3,H,W])."""
    t = preprocess(pil, image_size).unsqueeze(0).to(device)
    feats = extractor(t)
    return {li: f[0] for li, f in feats.items()}, t[0]


def build_reference_features(
    extractor: DINOv3MultiLayerExtractor,
    ref_pils: List[Image.Image],
    ref_paths_for_alignment: Optional[List[str]],
    image_size: int,
    device: torch.device,
    rotation_cfg: Optional[dict] = None,
):
    """reference 이미지들을 (옵션) 정렬 후 layer별 feature 스택으로 변환.

    회전 정렬은 파일 경로 기반(align_reference_set)이라, 업로드된 이미지를
    쓰려면 호출부가 임시 파일로 저장해 ``ref_paths_for_alignment`` 로 넘겨야 한다.

    Returns:
        ref_features_dev: {layer: Tensor[N_ref, H_p, W_p, C]} (device 위).
        aligner: RotationAligner | None.
        aligned_ref_pils: 정렬된 reference PIL 리스트 (디버그 표시용).
    """
    rot_cfg = rotation_cfg or {}
    aligner = None
    if rot_cfg.get("enabled", False):
        if not ref_paths_for_alignment:
            raise ValueError("회전 정렬을 쓰려면 reference 이미지의 임시 파일 경로가 필요합니다.")
        aligner, aligned_ref_pils = align_reference_set(
            ref_paths_for_alignment,
            size=int(rot_cfg.get("size", 256)),
            method=rot_cfg.get("method", "direct"),
            refine=bool(rot_cfg.get("refine", True)),
            background=rot_cfg.get("background", "edge"),
            iters=int(rot_cfg.get("ref_align_iters", 2)),
            center=bool(rot_cfg.get("center", False)),
            tol=float(rot_cfg.get("center_tol", 0.12)),
            matte=bool(rot_cfg.get("matte", False)),
            matte_fill=rot_cfg.get("matte_fill", "auto"),
            matte_model=rot_cfg.get("matte_model", "u2net"),
        )
    else:
        aligned_ref_pils = list(ref_pils)

    layers = extractor.layer_indices
    per_layer: Dict[int, List[torch.Tensor]] = {li: [] for li in layers}
    for pil in aligned_ref_pils:
        feats, _ = extract_one(extractor, pil, image_size, device)
        for li in layers:
            per_layer[li].append(feats[li].cpu())
    ref_features_cpu = {li: torch.stack(v, 0) for li, v in per_layer.items()}
    ref_features_dev = {li: v.to(device) for li, v in ref_features_cpu.items()}
    return ref_features_dev, aligner, aligned_ref_pils


def score_test_image(
    extractor: DINOv3MultiLayerExtractor,
    test_pil: Image.Image,
    ref_features_dev: Dict[int, torch.Tensor],
    image_size: int,
    device: torch.device,
    scoring_cfg: dict,
    aligner=None,
    rotation_cfg: Optional[dict] = None,
) -> dict:
    """test 이미지를 (옵션) 정렬 후 스코어링.

    Returns dict: score_map, layer_score_maps, feature_map, orig_image(PIL),
    image_hw, align_info, aligned_pil.
    """
    rot_cfg = rotation_cfg or {}
    align_info = {"applied": False, "angle": 0.0, "score": None}
    pil = test_pil
    if rot_cfg.get("enabled", False) and aligner is not None:
        rotated, angle, rscore = derotate_pil(
            pil, aligner,
            size=int(rot_cfg.get("size", 256)),
            background=rot_cfg.get("background", "edge"),
            center=bool(rot_cfg.get("center", False)),
            tol=float(rot_cfg.get("center_tol", 0.12)),
            matte=bool(rot_cfg.get("matte", False)),
            matte_fill=rot_cfg.get("matte_fill", "auto"),
            matte_model=rot_cfg.get("matte_model", "u2net"),
        )
        if rscore >= float(rot_cfg.get("min_score", 0.0)):
            pil = rotated
            align_info = {"applied": True, "angle": float(angle), "score": float(rscore)}
        else:
            align_info = {"applied": False, "angle": float(angle), "score": float(rscore)}

    layers = extractor.layer_indices
    metric = scoring_cfg.get("distance_metric", "cosine")
    k_nn = int(scoring_cfg.get("k", 1))
    neighborhoods, weights, norm = resolve_fuse(scoring_cfg)

    feats, t = extract_one(extractor, pil, image_size, device)
    score_map, layer_score_maps, _rd, _ni = score_fn(
        feats, ref_features_dev, layers, neighborhoods,
        weights=weights, normalize=norm, metric=metric, k=k_nn,
    )
    feat_map = torch.cat([feats[li].cpu() for li in layers], dim=-1)

    orig = denormalize(t)
    img_hw = image_hw(t)
    return {
        "score_map": score_map.cpu(),
        "layer_score_maps": layer_score_maps.cpu(),
        "feature_map": feat_map,
        "orig_image": Image.fromarray(orig, "RGB"),
        "image_hw": img_hw,
        "align_info": align_info,
        "aligned_pil": pil,
    }


# ── candidate 선택 (Stage 2 handoff) ────────────────────────────────


def select_candidates(
    score_map: torch.Tensor,
    layer_score_maps: torch.Tensor,
    feature_map: torch.Tensor,
    image_hw: Tuple[int, int],
    patch_grid_size: Tuple[int, int],
    method: str = "threshold",
    threshold: float = 0.3,
    topk_ratio: float = 0.1,
    topk_k: Optional[int] = None,
    save_feature: bool = True,
) -> List[Dict]:
    """method 에 따라 select_by_threshold / select_topk 로 위임."""
    if method == "threshold":
        return select_by_threshold(
            score_map, layer_score_maps, image_hw, patch_grid_size,
            threshold=threshold, feature_map=feature_map, save_feature=save_feature,
        )
    return select_topk(
        score_map, layer_score_maps, image_hw, patch_grid_size,
        topk_ratio=topk_ratio, topk_k=topk_k, feature_map=feature_map, save_feature=save_feature,
    )


# ── 시각화 ───────────────────────────────────────────────────────────


def overlay_heatmap(
    orig_image: Image.Image, score_map: torch.Tensor, alpha: float = 0.5, cmap_name: str = "jet",
) -> Image.Image:
    """score_map(H_p,W_p)을 원본 해상도로 업샘플해 히트맵으로 전체 이미지에 얹는다."""
    w, h = orig_image.size
    score_up = upsample_score_map(score_map, (h, w))
    norm = normalize01(score_up)
    cmap = matplotlib.colormaps[cmap_name]
    heat_rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb, mode="RGB").convert("RGBA")
    heat_img.putalpha(int(round(max(0.0, min(1.0, alpha)) * 255)))
    base = orig_image.convert("RGBA")
    return Image.alpha_composite(base, heat_img).convert("RGB")


def draw_candidates(
    orig_image: Image.Image, candidates: List[Dict], box_color: str = "lime",
) -> Image.Image:
    """candidate bbox + score 라벨을 이미지 위에 그린다."""
    img = orig_image.copy()
    draw = ImageDraw.Draw(img)
    for c in candidates:
        x1, y1, x2, y2 = c["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), f"{c['score']:.2f}", fill=box_color)
    return img


def candidates_table(candidates: List[Dict]) -> List[Dict]:
    """streamlit st.dataframe 표시용으로 candidate 리스트를 평탄화."""
    return [
        {
            "grid_i": c["grid_idx"][0],
            "grid_j": c["grid_idx"][1],
            "bbox": tuple(c["bbox"]),
            "score": round(c["score"], 4),
            "layer_scores": [round(v, 3) for v in c.get("layer_scores", [])],
        }
        for c in candidates
    ]
