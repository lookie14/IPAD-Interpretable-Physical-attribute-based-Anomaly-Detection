"""Stage 1 — DINOv3 위치 기반(position-wise) 이상 탐지.

reference(정상) 이미지 여러 장 + test(검사 대상) 이미지 1장을 받아:
    1) DINOv3 멀티레이어 특징 추출
    2) 회전 정렬로 test 를 정상 pose 에 맞춤
    3) 같은 grid 위치끼리 비교하는 position-wise 이상 점수 계산
    4) threshold 로 이상 후보 patch 선정 (Stage 2 handoff)

backbone/scoring/image_size/seed/device 는 configs/stage1_default.yaml 값으로
고정된다 — UI에서 바꿀 수 없다. 회전 정렬(rotation)과 candidate threshold(Stage 2
handoff 기준)는 사이드바에서 조절 가능하다 (기본값은 config 값에서 가져옴).

결과는 st.session_state["stage1_result"] 에 저장되어 Stage 2가 그대로 이어받는다.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import torch
from PIL import Image

from src.pipeline.stage1_core import (
    build_backbone_cfg,
    build_extractor,
    build_reference_features,
    candidates_table,
    draw_candidates,
    overlay_heatmap,
    score_test_image,
    select_candidates,
)
from src.utils.io_utils import load_config, resolve_device
from src.utils.seed import set_seed

st.set_page_config(page_title="Stage 1 — 이상 위치 탐지", layout="wide")
st.title("Stage 1 — DINOv3 위치 기반 이상 탐지")

# ── configs/stage1_default.yaml 을 고정 설정으로 사용 (UI에서 변경 불가) ──
CFG = load_config(str(ROOT / "configs" / "stage1_default.yaml"))
IMAGE_SIZE = int(CFG["image_size"])
SEED = int(CFG.get("seed", 42))
DEVICE_NAME = CFG.get("device", "cuda")
ROTATION_CFG = dict(CFG.get("rotation") or {})
SCORING_CFG = dict(CFG["scoring"])
CAND_CFG = CFG["candidate"]
ALPHA = float(CFG["visualization"].get("overlay_alpha", 0.5))

BACKBONE_CFG = CFG["backbone"]
backbone_cfg = build_backbone_cfg(
    source=BACKBONE_CFG.get("source", "meta_local"),
    name=BACKBONE_CFG.get("name", "dinov3_vitb16"),
    repo_dir=str(ROOT / BACKBONE_CFG["repo_dir"]),
    weights_path=str(ROOT / BACKBONE_CFG["weights_path"]),
    layer_indices=BACKBONE_CFG.get("layer_indices", [8]),
)

st.caption(
    f"고정 설정 (configs/stage1_default.yaml) — "
    f"backbone={BACKBONE_CFG.get('name')} layer={BACKBONE_CFG.get('layer_indices')} | "
    f"scoring=cosine 1x1+3x3 융합 | image_size={IMAGE_SIZE}"
)


# ── 사이드바: 회전 정렬 + candidate threshold (조절 가능) ──────────────
with st.sidebar:
    st.header("회전 정렬")
    rot_enabled = st.checkbox(
        "회전 정렬 사용", value=bool(ROTATION_CFG.get("enabled", False)),
        help="reference/test 이미지 포즈가 서로 다를 때만 켜세요. 느려집니다.",
    )
    rotation_cfg = {"enabled": rot_enabled}
    if rot_enabled:
        method_options = ["direct", "fourier"]
        default_method = ROTATION_CFG.get("method", "direct")
        rotation_cfg.update({
            "method": st.selectbox(
                "정렬 방법", method_options,
                index=method_options.index(default_method) if default_method in method_options else 0,
            ),
            "size": st.number_input("정렬 해상도", value=int(ROTATION_CFG.get("size", 256)), step=32),
            "refine": st.checkbox("미세 조정(refine)", value=bool(ROTATION_CFG.get("refine", True))),
            "center": st.checkbox("회전 전 중앙 정렬", value=bool(ROTATION_CFG.get("center", True))),
            "background": ROTATION_CFG.get("background", "edge"),
            "ref_align_iters": int(ROTATION_CFG.get("ref_align_iters", 2)),
            "min_score": float(ROTATION_CFG.get("min_score", 0.0)),
            "center_tol": float(ROTATION_CFG.get("center_tol", 0.12)),
            "matte": st.checkbox(
                "배경 제거(rembg) 사용", value=bool(ROTATION_CFG.get("matte", False)),
                help="rembg 패키지 필요",
            ),
            "matte_fill": ROTATION_CFG.get("matte_fill", "auto"),
            "matte_model": ROTATION_CFG.get("matte_model", "u2net"),
        })

    st.header("Candidate 선택 (Stage 2 handoff)")
    method_options = ["threshold", "topk"]
    default_cand_method = CAND_CFG.get("method", "threshold")
    cand_method = st.selectbox(
        "방법", method_options,
        index=method_options.index(default_cand_method) if default_cand_method in method_options else 0,
    )
    if cand_method == "threshold":
        threshold = st.slider(
            "score threshold", 0.0, 1.0, float(CAND_CFG.get("score_threshold", 0.3)), 0.01,
        )
        topk_ratio, topk_k = float(CAND_CFG.get("topk_ratio", 0.1)), None
    else:
        topk_ratio = st.slider(
            "topk 비율", 0.01, 0.5, float(CAND_CFG.get("topk_ratio", 0.1)), 0.01,
        )
        threshold, topk_k = float(CAND_CFG.get("score_threshold", 0.3)), None


# ── 입력 이미지 업로드 ──────────────────────────────────────────────
ref_paths_tmp = None

st.subheader("1) Reference (정상) 이미지 업로드")
ref_files = st.file_uploader(
    "정상 이미지 여러 장", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
)
st.subheader("2) Test (검사 대상) 이미지 업로드")
test_file = st.file_uploader("검사할 이미지 1장", type=["png", "jpg", "jpeg"])

if not ref_files or test_file is None:
    st.info("Reference 이미지(1장 이상)와 Test 이미지(1장)를 모두 업로드하세요.")
    st.stop()

ref_pils = [Image.open(f).convert("RGB") for f in ref_files]
test_pil = Image.open(test_file).convert("RGB")

if rotation_cfg.get("enabled", False):
    tmp_dir = tempfile.mkdtemp(prefix="stage1_refs_")
    ref_paths_tmp = []
    for i, p in enumerate(ref_pils):
        path = os.path.join(tmp_dir, f"ref_{i:03d}.png")
        p.save(path)
        ref_paths_tmp.append(path)


run = st.button("분석 실행", type="primary")
if not run:
    st.stop()


# ── 파이프라인 실행 ──────────────────────────────────────────────────
set_seed(SEED)
device = resolve_device(DEVICE_NAME)


@st.cache_resource(show_spinner=False)
def _cached_extractor(backbone_cfg_json: str, device_str: str):
    cfg = json.loads(backbone_cfg_json)
    return build_extractor(cfg, torch.device(device_str))


try:
    with st.spinner("Backbone 로딩 중..."):
        extractor = _cached_extractor(json.dumps(backbone_cfg, sort_keys=True), str(device))
except (FileNotFoundError, RuntimeError, ValueError) as e:
    st.error(f"Backbone 로드 실패: {e}")
    st.stop()

with st.spinner("Reference 특징 추출 중..."):
    ref_features_dev, aligner, aligned_ref_pils = build_reference_features(
        extractor, ref_pils, ref_paths_tmp, IMAGE_SIZE, device, rotation_cfg,
    )

with st.spinner("Test 이미지 스코어링 중..."):
    result = score_test_image(
        extractor, test_pil, ref_features_dev, IMAGE_SIZE, device,
        SCORING_CFG, aligner=aligner, rotation_cfg=rotation_cfg,
    )

h_p, w_p = result["score_map"].shape
candidates = select_candidates(
    result["score_map"], result["layer_score_maps"], result["feature_map"],
    result["image_hw"], (h_p, w_p),
    method=cand_method, threshold=threshold, topk_ratio=topk_ratio, topk_k=topk_k,
)

overlay = overlay_heatmap(result["orig_image"], result["score_map"], alpha=ALPHA)
boxed = draw_candidates(overlay, candidates)


# ── stage1 최종 output ──────────────────────────────────────────────
col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("이상 위치 히트맵 + candidate box")
    st.image(boxed, width="stretch")
    if result["align_info"]["score"] is not None:
        ai = result["align_info"]
        st.caption(f"회전 정렬: applied={ai['applied']}  angle={ai['angle']:.1f}°  score={ai['score']:.2f}")
with col2:
    st.subheader(f"Candidates ({len(candidates)}) — Stage 2 handoff")
    st.metric(
        "score 범위 (grid, H_p x W_p = %dx%d)" % (h_p, w_p),
        f"{result['score_map'].min():.3f} ~ {result['score_map'].max():.3f}",
    )
    st.dataframe(candidates_table(candidates), width="stretch")

with st.expander("Reference 이미지 미리보기"):
    st.image(aligned_ref_pils, width=120)

st.session_state["stage1_result"] = {
    "image": result["orig_image"],
    "image_size": result["image_hw"],
    "patch_grid_size": (h_p, w_p),
    "candidates": candidates,
    "layer_indices": extractor.layer_indices,
    "align_info": result["align_info"],
}
st.success(
    "Stage 1 결과가 st.session_state['stage1_result'] 에 저장되었습니다. "
    "Stage 2 구현 시 여기서 이어받으면 됩니다."
)
