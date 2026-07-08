"""이상 탐지 파이프라인 — Home.

stage1 -> stage2 -> stage3 순서로 pages/ 아래에 각 단계가 구현된다.
좌측 사이드바에서 단계를 선택해 이동.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Anomaly Detection Pipeline", layout="wide")

st.title("이상 탐지 파이프라인")
st.markdown(
    """
좌측 사이드바에서 단계를 선택하세요. Stage 1 → 2 → 3 순서로 실행하면
`st.session_state` 를 통해 결과가 자동으로 다음 단계로 이어집니다.

- **Stage 1** — DINOv3 위치 기반 이상 탐지 (구현 완료)
- **Stage 2** — CoOp 학습된 physical-property 프롬프트 기반 이상 속성 분류 (구현 완료, Stage 1 결과 필요)
- **Stage 3** — 이상 위치 시각화 + 속성 집계 (구현 완료, Stage 1/2 결과 없으면 더미 데이터로 동작)
"""
)
