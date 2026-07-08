"""Stage 3 — 이상 위치 시각화 + 속성 집계.

입력: patches: list[PatchScore] (stage2에서 확정된 이상 패치, 이미지 전체)

    patches ─┬─ 위치 시각화: to_grid -> blend -> draw_patch_labels -> overlay
             └─ 속성 집계  : aggregate_by_attr -> predict_attr -> bar_chart -> attr_chart, predicted_attr

st.session_state["stage1_result"] / ["stage2_result"] 가 모두 있으면 실제 파이프라인
결과(stage2_core.to_patch_scores 로 변환)를 쓰고, 없으면 더미 PatchScore로 UI만
테스트한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image

from src.pipeline.stage2_core import to_patch_scores
from src.viz.dummy_data import DEFAULT_ATTRS, make_dummy_image, make_dummy_patches
from src.viz.stage3_core import (
    aggregate_by_attr,
    bar_chart,
    blend,
    draw_patch_labels,
    heat_scale_bar,
    predict_attr,
    to_grid,
    to_mask,
)

st.set_page_config(page_title="Stage 3 — 위치/속성 시각화", layout="wide")
st.title("Stage 3 — 이상 위치 시각화 & 속성 집계")

stage1_result = st.session_state.get("stage1_result")
stage2_result = st.session_state.get("stage2_result")
have_pipeline_result = stage1_result is not None and stage2_result is not None

with st.sidebar:
    st.header("입력 설정")
    if have_pipeline_result:
        use_dummy = st.checkbox(
            "더미 데이터 사용", value=False,
            help="체크하면 Stage 1/2 결과 대신 합성 데이터로 UI만 테스트합니다.",
        )
    else:
        st.info("Stage 1/2 결과가 없어 더미 데이터로 동작합니다. (Stage 1 -> Stage 2 순서로 먼저 실행하세요)")
        use_dummy = True

    if use_dummy:
        grid_cols = st.slider("grid 열", 2, 12, 6)
        grid_rows = st.slider("grid 행", 2, 12, 6)
        anomaly_ratio = st.slider(
            "이상 패치 비율",
            0.05,
            1.0,
            0.25,
            0.05,
            help="grid 셀 중 실제로 이상으로 확정될 비율. stage2가 일부 패치만 넘기는 상황을 흉내냅니다 (나머지 영역은 원본 그대로 유지).",
        )
        seed = st.number_input("더미 데이터 seed", value=42, step=1)

    alpha = st.slider("히트맵 투명도", 0.0, 1.0, 0.45, 0.05)
    show_labels = st.checkbox("패치 라벨 표시", value=True)

if use_dummy:
    if have_pipeline_result:
        image = make_dummy_image()
    else:
        uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
        if uploaded is None:
            st.warning("이미지를 업로드하거나 Stage 1 -> Stage 2를 먼저 실행하세요.")
            st.stop()
        image = Image.open(uploaded).convert("RGB")
        st.caption("※ Stage 1/2 미연동 상태이므로 patches는 더미 값으로 생성됩니다.")
    patches = make_dummy_patches(
        image_size=image.size,
        grid=(grid_cols, grid_rows),
        attrs=DEFAULT_ATTRS,
        seed=int(seed),
        anomaly_ratio=anomaly_ratio,
    )
    st.caption(f"[더미] 이상 패치로 확정된 개수: {len(patches)} / {grid_cols * grid_rows} (grid 셀 기준)")
else:
    image = stage1_result["image"]
    patches = to_patch_scores(stage2_result)
    st.caption(f"Stage 2 결과를 이어받았습니다: TypedRegion {len(patches)}개")

if not patches:
    st.warning("표시할 patch가 없습니다 (Stage 1/2에서 확정된 이상 후보가 없습니다).")
    st.stop()

# ── 위치 시각화 ──
geo_map = to_grid(patches, image.size)
mask = to_mask(patches, image.size)
overlay = blend(image, geo_map, mask=mask, alpha=alpha)
if show_labels:
    overlay = draw_patch_labels(overlay, patches)

# ── 속성 집계 ──
attr_scores = aggregate_by_attr(patches)  # 내부값 (top-1 속성별 평균 점수)
predicted_attr = predict_attr(attr_scores)
attr_chart = bar_chart(attr_scores, highlight=predicted_attr)

# ── stage3 최종 output ──
col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("위치 시각화 (overlay)")
    st.image(overlay, width="stretch")
    legend_col1, legend_col2, legend_col3 = st.columns([1, 4, 1])
    with legend_col2:
        st.image(heat_scale_bar(), width="stretch")
        lo, hi = st.columns(2)
        lo.caption("낮음 (정상에 가까움)")
        hi.markdown(
            "<div style='text-align:right; color:#898781; font-size:0.8rem;'>높음 (이상 가능성 ↑)</div>",
            unsafe_allow_html=True,
        )
with col2:
    st.subheader("속성별 막대그래프 (attr_chart)")
    st.pyplot(attr_chart, width="stretch")
    st.metric("예측 속성 (predicted_attr)", predicted_attr, f"{attr_scores[predicted_attr]:.3f}")

with st.expander("Raw PatchScore 데이터"):
    st.dataframe(
        [
            {
                "id": p.patch_id,
                "x": p.x,
                "y": p.y,
                "geo_score": round(p.geo_score, 3),
                "top1_attr": p.top1_attr,
                "top1_score": round(p.top1_score, 3),
            }
            for p in patches
        ],
        width="stretch",
    )
