"""Stage2 핵심 로직: CoOp 학습된 physical-property 프롬프트로 이상 속성 분류.

design spec(1.png):
    ⓪ Prompt tuning (오프라인, 학습 완료됨): CoOp 방식 learnable token V(n_ctx개) +
       physical property 클래스명만으로 [SOS][V1..Vn][classname][EOS] 구성,
       CLIP 인코더는 frozen. coop_dtd_best_ctx_3.pt 가 이미 MVTec 데이터로 학습된
       체크포인트 (physical-property 클래스 20개 + 학습된 ctx).
    ① Input: Stage 1 candidate 영역(bbox, patch_coords=grid_idx, geo_score) + test 이미지
    ② Crop preprocessing: bbox padding("pad context") 후 crop. resize는 CLIP 자체 전처리가 담당.
    ③ CLIP image branch: crop -> frozen CLIP image encoder -> z_img
    ④ CLIP text branch: 학습된 ctx + physical property 클래스명 -> frozen CoOp CLIP
       text encoder -> z_text (체크포인트 로딩 시 1회 계산 — TextureAttributeClassifier)
    ⑤ Similarity + softmax: cos(z_img, z_text) * logit_scale -> softmax -> attr_probs
    ⑥ Typed output: TypedRegion{bbox, patch_coords, geo_score, pred_attr, attr_probs,
       sem_score} -> Stage 3 로 전달

Stage 1 candidate 각각을 독립된 "candidate region"으로 처리한다 — 클러스터링이나
시각적 강조(highlight/blur) 없이 스펙에 그려진 흐름만 그대로 구현한다.
sem_score 는 pred_attr 의 확률(top-1 prob)로 정의한다.

streamlit에 의존하지 않는 순수 함수들 (src/pipeline/stage1_core.py 와 동일한 패턴).
pages/2_Stage2.py 가 이 함수들을 호출한다. scripts/run_stage2.py(파일 기반 배치)도
동일 함수를 재사용한다.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.models.texture_attribute_classifier import TextureAttributeClassifier
from src.scoring.patch_score import PatchScore
from src.stage2.crop_utils import crop_array, expand_bbox


def build_classifier(checkpoint_path: str, device: torch.device) -> TextureAttributeClassifier:
    """CoOp 체크포인트(학습된 ctx + physical-property 클래스 bank) 로딩.

    무거운 연산(체크포인트 로드 + CLIP 백본 로드) — 호출부에서 st.cache_resource 권장.
    """
    return TextureAttributeClassifier(checkpoint_path, device)


def process_candidates(
    orig_image: Image.Image,
    candidates: List[Dict],
    image_hw: Tuple[int, int],
    classifier: TextureAttributeClassifier,
    *,
    padding_ratio: float = 1.0,
    padding_px: Optional[int] = None,
    min_crop_size: int = 64,
    square: bool = True,
    top_k: int = 5,
    sem_score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """Stage 1 candidate 각각 -> padded crop -> CoOp CLIP 분류 -> TypedRegion.

    sem_score_threshold: 이 값 미만인 region(= CoOp 예측 확신도가 낮은 candidate)은
    결과에서 제외한다. Stage 3는 이 함수의 반환값만 이어받으므로, 여기서 거르면
    Stage 3에도 자동으로 나타나지 않는다.

    Returns:
        list[TypedRegion] (geo_score 내림차순), 각 원소::

            {
              "bbox": [x1,y1,x2,y2], "padded_bbox": [...],
              "patch_coords": [i, j],       # Stage 1 grid_idx
              "geo_score": float,           # Stage 1 anomaly score
              "pred_attr": str,             # top-1 physical-property label
              "attr_probs": {label: prob},  # top-k 확률
              "sem_score": float,           # attr_probs[pred_attr]
              "topk": [...],                # classifier.classify() 원본 출력
              "padded_crop": np.ndarray,    # CLIP 에 들어간 crop (시각화용)
            }
    """
    if not candidates:
        return []

    orig = np.asarray(orig_image.convert("RGB"), dtype=np.uint8)

    regions: List[Dict[str, Any]] = []
    for cand in candidates:
        bbox = tuple(cand["bbox"])
        padded_bbox = expand_bbox(
            bbox, image_hw,
            padding_ratio=padding_ratio, padding_px=padding_px,
            min_crop_size=min_crop_size, square=square,
        )
        padded_crop = crop_array(orig, padded_bbox)
        pil_crop = Image.fromarray(padded_crop)

        topk_results, _preview = classifier.classify(pil_crop, top_k=top_k)
        attr_probs = {t["label"]: t["prob"] for t in topk_results}
        pred_attr = topk_results[0]["label"]
        sem_score = topk_results[0]["prob"]

        regions.append({
            "bbox": list(bbox),
            "padded_bbox": list(padded_bbox),
            "patch_coords": list(cand["grid_idx"]),
            "geo_score": float(cand["score"]),
            "pred_attr": pred_attr,
            "attr_probs": attr_probs,
            "sem_score": float(sem_score),
            "topk": topk_results,
            "padded_crop": padded_crop,
        })

    if sem_score_threshold > 0.0:
        regions = [r for r in regions if r["sem_score"] >= sem_score_threshold]

    regions.sort(key=lambda r: r["geo_score"], reverse=True)
    return regions


# ── Stage 3 handoff ──────────────────────────────────────────────────


def to_patch_scores(stage2_result: Dict[str, Any], image_id: str = "stage2") -> List[PatchScore]:
    """st.session_state["stage2_result"] (TypedRegion 리스트) -> Stage 3 입력 스키마."""
    patches: List[PatchScore] = []
    for patch_id, r in enumerate(stage2_result.get("regions", [])):
        x1, y1, x2, y2 = r["bbox"]
        patches.append(PatchScore(
            patch_id=patch_id,
            x=int(x1), y=int(y1),
            w=int(x2 - x1), h=int(y2 - y1),
            geo_score=float(r["geo_score"]),
            attr_scores=dict(r["attr_probs"]),
            image_id=image_id,
        ))
    return patches
