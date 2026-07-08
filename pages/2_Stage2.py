"""Stage 2 — CoOp 학습된 physical-property 프롬프트로 이상 속성 분류.

design spec(1.png) 그대로: 클러스터링도, crop 위 시각적 강조(highlight/blur)도 없다.
Stage 1에서 st.session_state["stage1_result"] 로 넘어온 candidate patch 각각을:
    1) padding 을 더해 crop (resize는 CLIP 자체 전처리가 담당)
    2) CoOp 로 학습된 CLIP(coop_dtd_best_ctx_3.pt: learned ctx + physical-property
       클래스명)으로 분류 — 자연어 zero-shot 이 아니라 학습된 soft prompt
    3) TypedRegion{bbox, patch_coords, geo_score, pred_attr, attr_probs, sem_score} 로 출력

파일 저장 없이 메모리에서 바로 이어받는다 (scripts/run_stage2.py 는 동일 로직의
파일 기반 배치 버전).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import torch

from src.pipeline.stage2_core import build_classifier, process_candidates
from src.utils.io_utils import resolve_device

st.set_page_config(page_title="Stage 2 — 이상 속성 분류", layout="wide")
st.title("Stage 2 — CoOp 기반 이상 속성 분류")

stage1_result = st.session_state.get("stage1_result")
if not stage1_result:
    st.warning("Stage 1 결과가 없습니다. 먼저 **Stage 1** 페이지에서 분석을 실행하세요.")
    st.stop()

candidates = stage1_result["candidates"]
orig_image = stage1_result["image"]           # PIL.Image (RGB)
image_hw = stage1_result["image_size"]         # (H, W)

st.caption(
    f"Stage 1 결과를 이어받았습니다: candidates {len(candidates)}개, "
    f"image {image_hw[0]}x{image_hw[1]}."
)

# ── 사이드바: 파라미터 ───────────────────────────────────────────────
with st.sidebar:
    st.header("CoOp 체크포인트")
    checkpoint_path = st.text_input(
        "checkpoint (.pt)",
        str(ROOT / "coop_dtd_best_ctx_3.pt"),
        help="MVTec 데이터로 학습된 physical-property CoOp 체크포인트 경로.",
    )
    device_choice = st.selectbox("device", ["auto", "cpu", "cuda", "mps"], index=0)

    st.header("Crop")
    padding_ratio = st.slider("padding_ratio", 0.0, 3.0, 1.0, 0.1,
                               help="여백 = padding_ratio * max(bbox 폭, 높이)")
    min_crop_size = st.number_input("min_crop_size (px)", value=64, min_value=1, step=8)
    square = st.checkbox("정사각형 crop", value=True)

    st.header("결과")
    top_k = st.slider("Top-K", 1, 20, 5)
    sem_score_threshold = st.slider(
        "sem_score threshold", 0.0, 1.0, 0.0, 0.01,
        help="CoOp 예측 확신도(sem_score)가 이 값 미만인 region은 결과에서 제외됩니다 "
             "(Stage 3에도 전달되지 않습니다). 0이면 필터링 없음.",
    )


run = st.button("Stage 2 실행", type="primary")
if not run:
    st.stop()

if not candidates:
    st.warning("Stage 1 candidate가 없어 Stage 2를 실행할 것이 없습니다.")
    st.stop()


# ── 파이프라인 실행 ──────────────────────────────────────────────────
device = resolve_device("cuda" if device_choice == "auto" else device_choice)


@st.cache_resource(show_spinner=False)
def _cached_classifier(checkpoint_path: str, device_str: str):
    return build_classifier(checkpoint_path, torch.device(device_str))


try:
    with st.spinner("CoOp 체크포인트 + CLIP 로딩 중..."):
        classifier = _cached_classifier(checkpoint_path, str(device))
except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
    st.error(f"CoOp 체크포인트 로드 실패: {e}")
    st.stop()

st.caption(f"physical-property 클래스 {len(classifier.labels)}개 로드됨: {', '.join(classifier.labels)}")

with st.spinner("crop + CoOp 분류 중..."):
    regions = process_candidates(
        orig_image, candidates, image_hw, classifier,
        padding_ratio=float(padding_ratio),
        min_crop_size=int(min_crop_size),
        square=square,
        top_k=int(top_k),
        sem_score_threshold=float(sem_score_threshold),
    )

st.success(
    f"{len(regions)}개 region 분류 완료."
    + (f" (sem_score < {sem_score_threshold:.2f} 인 candidate는 제외됨)" if sem_score_threshold > 0 else "")
)
if not regions:
    st.warning("threshold를 넘는 region이 없습니다. Stage 3에 넘길 것이 없습니다.")

# ── TypedRegion 결과 ──────────────────────────────────────────────────
st.subheader(f"TypedRegion 결과 ({len(regions)}개, geo_score 내림차순)")
for i, r in enumerate(regions):
    with st.expander(
        f"region {i:02d}  patch_coords={r['patch_coords']}  geo_score={r['geo_score']:.3f}  "
        f"→ pred_attr: {r['pred_attr']} (sem_score={r['sem_score']:.2f})",
        expanded=(i == 0),
    ):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.image(r["padded_crop"], caption="padded crop (CLIP 입력)", width="stretch")
        with c2:
            st.dataframe(
                [
                    {"label": t["label"], "cosine": round(t["cosine"], 3), "prob": round(t["prob"], 3)}
                    for t in r["topk"]
                ],
                width="stretch",
            )

# ── stage2 최종 output ──────────────────────────────────────────────
st.session_state["stage2_result"] = {
    "regions": [{k: v for k, v in r.items() if k != "padded_crop"} for r in regions],
}
st.info(
    "Stage 2 결과가 st.session_state['stage2_result'] 에 저장되었습니다. "
    "Stage 3 연동 시 여기서 이어받으면 됩니다."
)
