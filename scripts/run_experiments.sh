#!/usr/bin/env bash
# =============================================================================
# 여러 Stage 1 실험을 한 번에 순차 실행.
# 각 실험은 (이름 | override 문자열) 형식. output_dir 을 실험별로 분리해 결과가 안 섞이게 함.
#
# 사용:
#   bash scripts/run_experiments.sh                # 아래 EXPERIMENTS 전부 실행
#   DEVICE=cpu bash scripts/run_experiments.sh     # 디바이스 바꿔서
#   CATEGORY=leather bash scripts/run_experiments.sh
# =============================================================================
set -uo pipefail

# --- 프로젝트 루트로 이동 (한글 경로여도 동작) ---
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- 공통 설정 (환경변수로 덮어쓰기 가능) ---
CONFIG="${CONFIG:-configs/stage1_default.yaml}"
DEVICE="${DEVICE:-mps}"                 # cuda / mps / cpu
CATEGORY="${CATEGORY:-grid}"
OUT_ROOT="${OUT_ROOT:-./outputs/stage1}"
# 테스트 이미지 필터 (전체 돌리려면 아래 두 줄 비우기)
COMMON_OVERRIDES=( "category=${CATEGORY}" "device=${DEVICE}" "test_indices=[0,1,2,3,4]" )

# --- 실험 목록: "이름 | override1 override2 ..." ---
# 이름은 출력 폴더명이 됨 -> ${OUT_ROOT}/<이름>/<category>/...
# category 는 override 에 그냥 넣으면 됨. COMMON_OVERRIDES 의 기본 category 보다
# 나중에 전달되므로 실험별 값이 이김. (미지정 실험은 기본 CATEGORY 사용)
#   - 단일:   category=leather
#   - 전체:   category=all
#   - 여러개: "categories=[grid,leather]"
EXPERIMENTS=(
  "1x1       | category=grid    scoring.neighborhood=0"
  "3x3          | category=grid    scoring.neighborhood=1"
  "leather_single    | scoring.neighborhood=0"
  "leather_3x3       | scoring.neighborhood=1"
  "bottle_fuse_8_2   | category=bottle  scoring.fuse.enabled=true scoring.fuse.neighborhoods=[0,1] scoring.fuse.weights=[0.8,0.2]"
  "allcat_3x3        | category=all     scoring.neighborhood=1"
)

# --- 실행 루프 ---
total=${#EXPERIMENTS[@]}
i=0
fail=0
start_all=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
  i=$((i+1))
  name="$(echo "${entry%%|*}" | xargs)"       # '|' 앞 = 이름 (공백 제거)
  overrides="$(echo "${entry#*|}" | xargs)"   # '|' 뒤 = override 문자열
  out_dir="${OUT_ROOT}/${name}"

  echo ""
  echo "=================================================================="
  echo "[$i/$total] $name"
  echo "   overrides: $overrides"
  echo "   output   : $out_dir"
  echo "=================================================================="
  t0=$(date +%s)

  # shellcheck disable=SC2086  # overrides 는 의도적으로 단어 분리
  python3 scripts/run_stage1.py --config "$CONFIG" \
      --override "${COMMON_OVERRIDES[@]}" output_dir="$out_dir" $overrides
  rc=$?

  t1=$(date +%s)
  if [ $rc -ne 0 ]; then
    echo ">> [$name] FAILED (exit $rc)"
    fail=$((fail+1))
  else
    echo ">> [$name] done in $((t1-t0))s"
  fi
done

echo ""
echo "=================================================================="
echo "완료: $((total-fail))/$total 성공, $fail 실패, 총 $(( $(date +%s)-start_all ))s"
echo "결과: $OUT_ROOT/<실험이름>/${CATEGORY}/"
echo "=================================================================="
