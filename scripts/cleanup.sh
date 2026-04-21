#!/bin/bash
# YOLO_Grasp 프로젝트 cleanup 스크립트
# 사용법:
#   bash scripts/cleanup.sh              # dry-run (삭제 대상과 크기만 출력)
#   bash scripts/cleanup.sh --apply      # 실제 삭제
#   bash scripts/cleanup.sh --stage 1    # 특정 stage 만 (1~5)
#   bash scripts/cleanup.sh --stage 1 --apply
#
# Stage 1: YOLO 구버전 학습 run (v1, v2 + epoch*.pt)         ~3.7 GB
# Stage 2: Flow 구버전 학습 run (sweep, smoke, cmp, v2~v5)  ~2.9 GB
# Stage 3: 구 YOLO cache + dataset (v1/v2)                 ~100 MB
# Stage 4: 개발 중 viz/test 산출물 (scripts/_*)             ~73 MB
# Stage 5: __pycache__ + 기타 부스러기                      ~2 MB
# Stage 6: 구버전 코드 (파이프라인 미사용 21개)              ~230 KB
# =========================================================================

ROOT="/home/robotics/Competition/YOLO_Grasp"
APPLY=0
STAGE_FILTER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --stage) STAGE_FILTER="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

CLR_BLUE='\033[0;34m'; CLR_RED='\033[0;31m'; CLR_GRN='\033[0;32m'; CLR_YLW='\033[0;33m'; CLR_END='\033[0m'

run_stage() {
  local stage=$1; local title=$2; shift 2
  if [[ -n "$STAGE_FILTER" && "$STAGE_FILTER" != "$stage" ]]; then return; fi
  echo
  echo -e "${CLR_BLUE}=== Stage $stage: $title ===${CLR_END}"
  local total=0
  for p in "$@"; do
    if [[ -e "$p" ]]; then
      size=$(du -sb "$p" 2>/dev/null | cut -f1)
      human=$(du -sh "$p" 2>/dev/null | cut -f1)
      echo -e "  ${CLR_YLW}[del]${CLR_END} $p  (${human})"
      total=$((total + size))
      if [[ $APPLY -eq 1 ]]; then rm -rf "$p"; fi
    else
      echo -e "  ${CLR_GRN}[skip]${CLR_END} $p  (이미 없음)"
    fi
  done
  human_total=$(numfmt --to=iec-i --suffix=B $total 2>/dev/null || echo "$total bytes")
  echo -e "  ${CLR_RED}→ stage $stage 합계: $human_total${CLR_END}"
}

# ================ Stage 1 ================
# 구 YOLO 학습 run (v1, v2 는 v3_1280 로 대체), 중간 epoch 체크포인트
run_stage 1 "YOLO 구버전 + epoch*.pt 중간 체크포인트" \
  "$ROOT/runs/yolov8m_seg" \
  "$ROOT/runs/yolov8m_seg_v2" \
  "$ROOT/runs/yolov8m_seg_v3_1280/weights/epoch0.pt" \
  "$ROOT/runs/yolov8m_seg_v3_1280/weights/epoch10.pt" \
  "$ROOT/runs/yolov8m_seg_v3_1280/weights/epoch20.pt" \
  "$ROOT/runs/yolov8m_seg_v3_1280/weights/epoch30.pt" \
  "$ROOT/runs/yolov8m_seg_v3_1280/weights/epoch40.pt"

# ================ Stage 2 ================
# Flow Matching 학습 구 run (v6_150ep 만 최종)
run_stage 2 "Flow Matching 구 run (v6 만 유지)" \
  "$ROOT/runs/yolograsp_v2/sweep_v1" \
  "$ROOT/runs/yolograsp_v2/v2_posnorm" \
  "$ROOT/runs/yolograsp_v2/v3_tier2" \
  "$ROOT/runs/yolograsp_v2/v4_markerboost" \
  "$ROOT/runs/yolograsp_v2/v5_100ep_big" \
  "$ROOT/runs/yolograsp_v2/smoke" \
  "$ROOT/runs/yolograsp_v2/smoke2" \
  "$ROOT/runs/yolograsp_v2/smoke_marker" \
  "$ROOT/runs/yolograsp_v2/smoke_tier2" \
  "$ROOT/runs/yolograsp_v2/smoke_tier2_fix" \
  "$ROOT/runs/yolograsp_v2/postfix_smoke" \
  "$ROOT/runs/yolograsp_v2/cmp_adaln" \
  "$ROOT/runs/yolograsp_v2/cmp_adaln_fair" \
  "$ROOT/runs/yolograsp_v2/cmp_film"

# ================ Stage 3 ================
# 구 YOLO 캐시 + dataset (v3_1280 + dataset_v2 가 유일 유효)
run_stage 3 "구 YOLO cache + Roboflow dataset v1" \
  "$ROOT/img_dataset/yolo_cache" \
  "$ROOT/img_dataset/yolo_cache_v2" \
  "$ROOT/dataset"

# ================ Stage 4 ================
# 개발 중 viz/test 산출물. 모든 후속은 deploy/viz/ 로 통합됨
run_stage 4 "개발 중 viz/test 산출물 (scripts/_*)" \
  "$ROOT/scripts/_infer_viz" \
  "$ROOT/scripts/_grasp_3d_viz" \
  "$ROOT/scripts/_scene_pipeline_viz" \
  "$ROOT/scripts/_aug_viz" \
  "$ROOT/scripts/_icp_grasp_viz" \
  "$ROOT/scripts/_icp_test_out" \
  "$ROOT/scripts/_icp_test_out_v2" \
  "$ROOT/scripts/_icp_test_out_v3" \
  "$ROOT/scripts/_icp_compare_v1_v2" \
  "$ROOT/scripts/_icp_compare_v1_v3"

# ================ Stage 5 ================
# __pycache__ + wandb run logs + test/diagnose run 디렉토리
run_stage 5 "__pycache__ + wandb + 개별 run 아티팩트" \
  "$ROOT/scripts/__pycache__" \
  "$ROOT/src/__pycache__" \
  "$ROOT/wandb" \
  "$ROOT/runs/test_center" \
  "$ROOT/runs/test_results" \
  "$ROOT/runs/diagnose_preproc" \
  "$ROOT/runs/uv_preview" \
  "$ROOT/runs/detection_viz" \
  "$ROOT/runs/detection_viz_v3" \
  "$ROOT/runs/detection_viz_v3_fixed"

# ================ Stage 6 ================
# 구버전 코드 — 현재 파이프라인에서 사용되지 않음.
# 각 파일의 삭제 이유는 scripts/cleanup.sh 주석 참고 (git log 에도 기록).
run_stage 6 "구버전 코드 (파이프라인 미사용)" \
  "$ROOT/train.py" \
  "$ROOT/test.py" \
  "$ROOT/test_center.py" \
  "$ROOT/split_dataset.py" \
  "$ROOT/sweep_v1.yaml" \
  "$ROOT/scripts/batch_yolo_cache.py" \
  "$ROOT/scripts/batch_yolo_cache_v2.py" \
  "$ROOT/scripts/sweep_v1.py" \
  "$ROOT/scripts/diagnose_preprocessing.py" \
  "$ROOT/scripts/preview_uv_strategies.py" \
  "$ROOT/scripts/test_icp_small.py" \
  "$ROOT/scripts/test_icp_v2.py" \
  "$ROOT/scripts/test_icp_v3.py" \
  "$ROOT/scripts/test_yaw_aug.py" \
  "$ROOT/scripts/visualize_augmentation.py" \
  "$ROOT/scripts/visualize_augmentation_3d.py" \
  "$ROOT/scripts/visualize_detections.py" \
  "$ROOT/scripts/visualize_grasps_3d.py" \
  "$ROOT/scripts/visualize_icp_2d.py" \
  "$ROOT/scripts/visualize_icp_2d_v3.py" \
  "$ROOT/scripts/visualize_icp_overlay.py" \
  "$ROOT/scripts/visualize_icp_with_grasps.py" \
  "$ROOT/scripts/visualize_masked_pointcloud.py" \
  "$ROOT/scripts/visualize_scene_pipeline.py"

# ================ 요약 ================
echo
if [[ $APPLY -eq 0 ]]; then
  echo -e "${CLR_YLW}[dry-run]${CLR_END} 삭제되지 않았습니다. 실제 실행하려면 --apply 추가."
else
  echo -e "${CLR_GRN}[done]${CLR_END} 삭제 완료. 남은 크기 확인:"
  du -sh "$ROOT" 2>/dev/null
fi
