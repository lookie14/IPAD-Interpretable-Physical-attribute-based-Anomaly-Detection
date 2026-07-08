# IPAD — Interpretable Physical-attribute-based Anomaly Detection

DINOv3 기반 위치 탐지(Stage 1) → CoOp 학습된 CLIP 기반 속성 분류(Stage 2) → 시각화/집계(Stage 3)
3단계로 이루어진 이상 탐지 파이프라인. Streamlit UI와 CLI 배치 스크립트를 모두 지원한다.

```
Stage 1 (위치)              Stage 2 (속성)                    Stage 3 (시각화)
─────────────────────      ────────────────────────────      ─────────────────────
정상 참조 N장 + test 1장  →  Stage 1 candidate 각각 crop   →  patch 위치 히트맵 오버레이
DINOv3 patch token 추출      CoOp 학습된 physical-property     + 속성별 집계 막대그래프
(layer 3,6,9,11 평균)        프롬프트로 CLIP 분류
position-wise 이상 점수      (coop_dtd_best_ctx_3.pt)
threshold → candidate        → TypedRegion{bbox, patch_coords,
                                geo_score, pred_attr,
                                attr_probs, sem_score}
```

## 폴더 구조

```
app.py                          Streamlit 홈 (사이드바로 Stage 1/2/3 이동)
pages/
  1_Stage1.py                   Stage 1 — 위치 탐지 UI (업로드 → candidate)
  2_Stage2.py                   Stage 2 — 속성 분류 UI (Stage 1 결과 이어받음)
  3_Stage3.py                   Stage 3 — 시각화/집계 UI (Stage 1+2 결과 이어받음, 없으면 더미)
configs/
  stage1_default.yaml           Stage 1 전체 설정 (data_root, backbone, rotation, scoring, candidate)
  stage2_default.yaml           Stage 2 전체 설정 (checkpoint, crop, topk)
scripts/
  run_stage1.py                 Stage 1 CLI (카테고리 전체/일부 배치 실행)
  run_stage2.py                 Stage 2 CLI (Stage 1 결과 폴더를 읽어 이어서 실행)
  run_experiments.sh            여러 Stage 1 실험 설정을 순차 실행하는 래퍼
src/
  pipeline/stage1_core.py       Stage 1 핵심 로직 (streamlit 비의존, UI/CLI 공용)
  pipeline/stage2_core.py       Stage 2 핵심 로직 + Stage 2→3 변환(to_patch_scores)
  models/dinov3_feature_extractor.py   DINOv3 멀티레이어 patch feature 추출 (meta_local/dummy)
  models/texture_attribute_classifier.py  CoOp 체크포인트 로딩 + CLIP 분류
  scoring/                      position-wise 멀티레이어 이상 점수 계산
  candidate/topk_selector.py    threshold/topk candidate 선정
  stage2/crop_utils.py          candidate bbox padding/crop 유틸
  datasets/mvtec_dataset.py     MVTec AD 폴더 접근 (train/good, test/*, ground_truth/*)
  utils/                        rotation 정렬, 이미지 전처리, 설정 I/O, 시각화
  viz/stage3_core.py            위치 히트맵 오버레이 + 속성 집계 막대그래프
  viz/dummy_data.py             Stage 1/2 결과 없을 때 Stage 3 UI 테스트용 더미 데이터
load_dtd_prompts.py             CoOp 체크포인트에서 학습된 context 복원 (texture_attribute_classifier.py 가 사용)
coop_dtd_best_ctx_3.pt          MVTec 데이터로 학습된 CoOp 체크포인트 (physical-property 20종)
prompt.json                     결함 속성 설명 문장 (참고용, 현재 Stage 2 파이프라인엔 미사용)
third_party/dinov3/             Meta DINOv3 원본 저장소 (별도 clone 필요, git 미포함)
data/                           MVTec AD 데이터셋 (별도 다운로드 필요, git 미포함)
weights/                        DINOv3 사전학습 가중치 (별도 다운로드 필요, git 미포함)
```

`third_party/dinov3/`, `data/`, `weights/`, `outputs/`는 용량이 크거나(수백MB~수GB) 라이선스 동의가
필요해서 `.gitignore`에 포함되어 있다 — 아래 설정 과정에서 각자 받아야 한다.

## 설정

### 1) 의존성 설치

```bash
pip install -r requirements.txt
```

- `torch`, `numpy`, `pyyaml`, `pillow`, `matplotlib`, `scikit-image`, `scipy`, `streamlit`
- `ftfy`, `regex`, `git+https://github.com/openai/CLIP.git` (Stage 2 CoOp 분류용 — git 필요)
- `rembg`는 Stage 1 회전 정렬의 배경 제거(matte) 옵션을 쓸 때만 필요 (기본은 꺼져 있음)

### 2) DINOv3 원본 저장소 clone

```bash
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
```

### 3) DINOv3 사전학습 가중치

Meta 라이선스 동의가 필요하다 (자동 다운로드 불가):
1. https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ 에서 신청 → 승인되면 이메일로 URL 목록 수신
2. **ViT-B/16 distilled (LVD-1689M)** 을 받는다 (`configs/*.yaml`의 기본 백본과 일치)
3. 받은 `.pth` 파일을 아래 경로에 배치:
   ```
   weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
   ```
   (파일명이 다르면 `configs/stage1_default.yaml`의 `backbone.weights_path`를 실제 파일명에 맞게 수정)

### 4) MVTec AD 데이터셋 (CLI 배치 실행 시에만 필요)

https://www.mvtec.com/company/research/datasets/mvtec-ad 에서 받아 압축을 풀고, 카테고리 폴더
(`bottle/`, `grid/`, `leather/` ...)가 아래처럼 바로 오도록 배치:

```
data/mvtec_anomaly_detection/
  grid/
    train/good/*.png
    test/{good,bent,broken,...}/*.png
    ground_truth/{bent,broken,...}/*_mask.png
  bottle/
    ...
```

Streamlit UI로 본인 이미지를 직접 업로드해서 쓸 거라면 이 단계는 필요 없다.

## 실행

### Streamlit (권장)

```bash
streamlit run app.py
```

좌측 사이드바에서 Stage 1 → 2 → 3 순서로 이동. 각 단계 결과는 `st.session_state`를 통해
자동으로 다음 단계에 이어진다.

1. **Stage 1**: reference(정상) 이미지 여러 장 + test(검사 대상) 이미지 1장 업로드 → 분석 실행.
   backbone/scoring은 `configs/stage1_default.yaml` 고정값을 쓰고, 회전 정렬과 candidate
   threshold/topk만 사이드바에서 조절 가능.
2. **Stage 2**: Stage 1 결과가 있어야 진행 가능. CoOp 체크포인트 경로와 crop 파라미터를
   확인하고 실행하면 candidate마다 physical-property 예측(TypedRegion)을 보여준다.
3. **Stage 3**: Stage 1+2 결과가 있으면 그걸로, 없으면 더미 데이터로 위치 히트맵과
   속성별 집계 막대그래프를 보여준다.

### CLI 배치 실행

MVTec 데이터셋이 준비되어 있어야 한다.

```bash
# Stage 1: configs/stage1_default.yaml 의 category(기본 grid)로 실행
python3 scripts/run_stage1.py --config configs/stage1_default.yaml

# 설정 override 예시
python3 scripts/run_stage1.py --config configs/stage1_default.yaml \
    --override category=leather device=cpu

# Stage 2: Stage 1 출력 폴더(outputs/stage1/<category>/<run_name>/candidates/)를 읽어 이어서 실행
python3 scripts/run_stage2.py --config configs/stage2_default.yaml

# 여러 Stage 1 실험을 한 번에
bash scripts/run_experiments.sh
```

가중치 없이 파이프라인 동작만 빠르게 확인하고 싶다면:

```bash
python3 scripts/run_stage1.py --config configs/stage1_default.yaml \
    --make-synthetic --override backbone.source=dummy device=cpu
```

## 참고

- `configs/stage1_default.yaml`의 `layer_indices: [3, 6, 9, 11]`은 DINOv3 여러 block의
  patch feature를 평균 융합하는 설정이다.
- `coop_dtd_best_ctx_3.pt`는 MVTec 데이터로 학습된 CoOp 체크포인트로, physical-property
  클래스(예: crack, dent_depression, bending_deformation 등) 20종을 예측한다 — 새 카테고리로
  재학습하려면 별도 학습 스크립트가 필요하다(이 저장소엔 포함되어 있지 않음).
- 설계 스펙은 `1.png` 참고.
