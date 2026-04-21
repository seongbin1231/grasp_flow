---
name: yolo-grasp-orchestrator
description: "YOLO_Grasp 프로젝트의 top-down 4-DoF grasp 예측 모델 구축을 위한 6인 에이전트 팀 오케스트레이터. 1280×720 카메라 프레임 전용 파이프라인. 데이터는 이미 확보됨(588장). 현재 YOLO v3 학습·재추론까지 완료, 다음은 ICP → grasp 합성 → 데이터셋 → 모델 → ONNX. YOLO_Grasp·grasp 모델·파지자세 예측·Simulink 데이터 파이프라인·ONNX 배포 작업 시 반드시 사용. 후속 작업(재실행, 부분 수정, threshold 조정, 재학습, 데이터 추가, 배포 재검증, 이전 결과 개선)도 이 스킬을 사용."
---

# YOLO_Grasp Orchestrator — 6인 팀 조율

RoboCup ARM 2026용 top-down 4-DoF grasp 예측 모델 구축을 담당하는 에이전트 팀 오케스트레이터. 카메라 프레임 전용 파이프라인, base 변환은 MATLAB 런타임.

## 실행 모드: 에이전트 팀

파이프라인 + 팬아웃 복합. 데이터 라벨 품질과 학습/배포가 강결합이라 팀 모드로 SendMessage 피드백 활용.

## 🎯 전역 환경 상수 (모든 에이전트가 준수)

| 항목 | 값 |
|---|---|
| 해상도 | 1280×720 |
| Camera K | fx=fy=1109, cx=640, cy=360 |
| Depth | uint16 PNG mm → Python에서 `/1000.0` meter |
| 좌표 프레임 | **카메라 프레임 전용** |
| YOLO numpy 입력 | **BGR** (ultralytics 규약) |
| YOLO 모델 | `runs/yolov8m_seg_v3_1280/weights/best.pt` |
| Python | `/home/robotics/anaconda3/bin/python` (conda base) |
| Cube 통합 | 하류에서 4종 → `cube` 1종 |

## 📍 현재 진행 상황 (2026-04-21)

| 단계 | 상태 | 산출물 |
|---|---|---|
| 1. 데이터 확보 | ✅ **완료** | [img_dataset/captured_images/](/home/robotics/Competition/YOLO_Grasp/img_dataset/captured_images/) 588장 + depth |
| 2. YOLO v3 학습 | ✅ **완료** | `runs/yolov8m_seg_v3_1280/weights/best.pt` (mAP50-95 0.982) |
| 3. YOLO v3 재추론 | ✅ **완료** | `img_dataset/yolo_cache_v3/detections.h5` (3242 passed) |
| 4. PC sanity check | ✅ **완료** | `runs/pc_sanity_v3/` |
| 5. ICP 라벨링 | ✅ **완료** | `img_dataset/icp_cache/poses.h5` (2238/2238 pass, fitness mean 0.46) |
| 6. Grasp 정책 확정 | ✅ **완료** | standing 16 / lying 8 or 6 / cube 2 |
| 7. Grasp 합성 (6-DoF SE(3)) | ✅ **완료** | `img_dataset/grasp_cache/grasps.h5` (pose_cam 7D quat) |
| 8. HDF5 v2 통합 | ✅ **완료** | `datasets/grasp_v2.h5` (train 13640 / val 2600 rows) |
| 9. 모델 학습 (Rectified Flow) | ✅ **완료** | **`runs/yolograsp_v2/v6_150ep/.../best.pt`** (ep 118, val_flow=0.3640) |
| 10. ONNX + MATLAB | ✅ **완료** | `deploy/onnx/{encoder,velocity}.onnx + meta.json + README.md` |
| 11. 배포 아티팩트 | ✅ **완료** | `deploy/yolo/yolov8m_seg_1280.onnx` (MATLAB 팀용) |
| 12. Viz 파이프 | ✅ **완료** | `deploy/viz/{index.html,live/,gt/}` (캐시 7 카테고리 + 라이브 ROS + GT) |

**현재 초점**: MATLAB 팀 통합 + marker 성능 추적. 재학습 없이 **YOLO-anchored fallback** 으로 marker 문제 임시 완화. 추후 marker augmentation / object-size-adaptive crop 재학습 트리거 판단 대기.

## 🎯 Grasp 정책 (최종 확정, 2026-04-19)

| 모드 | 클래스 | 구성 | grasp 수 |
|---|---|---|---|
| **standing** | bottle / can / marker / spam | **top-down 8 yaw + side-horizontal 8 방위 (정확히 90° 직각)** | **16** |
| **lying** | bottle / can | 긴축 N=4 × 180° 대칭, approach = cam +Z | 8 |
| **lying** | marker / spam | 긴축 N=3 × 180° 대칭, approach = cam +Z | 6 |
| **cube** | cube_* (4종) | R_icp edge column 정렬 2 yaw (대각선 ❌), approach = cam +Z | 2 |

- Standing side-cap은 **approach가 cam Z에 수직** (horizontal), 8 azimuth × TCP=cap 밴드 (tip에서 15mm 아래)
- Cube는 **면 edge 정렬** 필수, 대각선 yaw 금지
- 출력: **6-DoF SE(3) `pose_cam(n,7) = [x,y,z, qw,qx,qy,qz]`** (이전 4-DoF 폐기)

## 🧠 모델 아키텍처 (확정)

**Conditional Rectified Flow (Flow Matching)** — 다봉 분포 샘플링으로 여러 후보 grasp 생성 + 충돌 필터로 선택.
- 학습 공간: `g = (x, y, z, a_x, a_y, a_z, sin_yaw, cos_yaw)` 8D
- Loss: `‖v_θ(g_t, t, c) − (g_1 − g_0)‖²`
- 추론: Reflow 1-step, N=32~64 병렬 샘플
- ONNX: velocity MLP만 export. MATLAB이 노이즈 생성, Euler step, 충돌 필터, CameraTform 수행
- 참조: `/home/robotics/Competition/grasp_model/graspflow/` (SE(3) 구현체, 우리는 8D 파라미터화로 단순화)

## 에이전트 구성

| 팀원 | 타입 | 역할 | 스킬 | 주요 출력 |
|---|---|---|---|---|
| matlab-collector | custom | 캡처 데이터 유지/추가 수집 | `/matlab-data-collection` | `img_dataset/captured_images/` |
| yolo-extractor | custom | YOLO v3 배치 추론 (BGR!) | `/yolo-batch-inference` | `yolo_cache_v3/detections.h5` |
| icp-labeler | custom | open3d ICP, 카메라 프레임 pose | `/icp-quality-management` | `icp_cache/poses.h5` |
| grasp-synthesizer | custom | Top-down 4-DoF grasp 합성 | `/top-down-grasp-synthesis` | `grasp_cache/grasps.h5` |
| dataset-curator | custom | HDF5 통합 + cube 통합 + QA | `/hdf5-dataset-management` | `datasets/grasp_v1.h5` |
| model-engineer | custom | PyTorch 학습 + ONNX + MATLAB | `/depth-uv-grasp-training`, `/onnx-matlab-integration` | `runs/yolograsp_v1/` |

## 데이터 흐름

```
captured_images (588장, ✅완료) ──┐
                                  ├→ yolo-extractor (✅완료)  ──┐
                                  │      detections.h5           │
                                  │                               ├→ dataset-curator → model-engineer
                                  └→ icp-labeler ──→ grasp-synth ─┘
                                        poses.h5      grasps.h5
                                                                (카메라 프레임 전용)
```

## 워크플로우

### Phase 0: 컨텍스트 확인

반드시 시작 시 다음을 확인하고 부분 재실행 여부를 결정:

1. `_workspace/` 존재 여부
2. 기존 산출물 체크:
   - `img_dataset/yolo_cache_v3/detections.h5` ✅ 존재 (3242 passed)
   - `img_dataset/icp_cache/poses.h5` ✅ 존재 (2238 pass gate)
   - `img_dataset/grasp_cache/grasps.h5` — 🔜 다음 생성 대상
   - `datasets/grasp_v2.h5` — pending (6-DoF 스키마)
   - `runs/yolograsp_v2/checkpoints/best.pt` — pending
3. 사용자 요청 파싱:
   - "YOLO 재추론" → 2단계부터
   - "ICP threshold 조정 / 재실행" → 5단계부터
   - "Grasp 정책 변경" → 7단계부터 (ICP 재활용)
   - "모델 재학습" → 9단계부터
   - "ONNX 재export" → 10단계만
   - "처음부터" → `_workspace_{ts}/`로 백업 후 Phase 1
4. 변경 감지:
   - **YOLO v3 재추론 · ICP v3 배치 모두 완료**. 명시 요청 없으면 재호출 금지.
   - **Grasp 합성부터 시작이 현재 기본 경로** (2026-04-19 기준).

### Phase 1: 준비

1. 사용자 확정 파라미터 (기본값 제시 후 컨펌):
   - ICP fitness threshold (기본 0.02)
   - grasp 합성 긴축 샘플 N (lying bottle/can: 4, marker/spam: 3)
   - 학습 epoch (기본 50), batch (기본 8, 4070 VRAM)
2. `_workspace/` 생성 (없으면)
3. 00_config.yaml에 파라미터 고정

### Phase 2: 팀 구성

```
TeamCreate(
  team_name: "yolograsp-team",
  members: [
    # matlab-collector는 이미 데이터 확보 완료. 감사·유지 용도로만 포함
    { name: "matlab-collector", agent_type: "matlab-collector", model: "opus",
      prompt: "img_dataset/ 데이터 무결성 확인. 추가 수집 요청 없으면 대기. 다른 팀원 질의 응답." },
    { name: "yolo-extractor", agent_type: "yolo-extractor", model: "opus",
      prompt: "yolo_cache_v3/detections.h5가 이미 존재. 사용자가 재추론 요청 시에만 실행. BGR 규약 엄수." },
    { name: "icp-labeler", agent_type: "icp-labeler", model: "opus",
      prompt: "detections.h5 + depth + PLY로 open3d ICP 배치 실행 → poses.h5. 카메라 프레임 pose만 저장." },
    { name: "grasp-synthesizer", agent_type: "grasp-synthesizer", model: "opus",
      prompt: "poses.h5 + depth로 4-DoF grasp 합성 → grasps.h5. grasps_cam만 저장, base/quat 금지." },
    { name: "dataset-curator", agent_type: "dataset-curator", model: "opus",
      prompt: "캐시 3종 + depth → grasp_v1.h5. cube 4종→1종 통합, scene split, 경계면 QA." },
    { name: "model-engineer", agent_type: "model-engineer", model: "opus",
      prompt: "grasp_v1.h5로 학습 → ONNX export → predict_grasp.m. MATLAB에서 CameraTform으로 base 변환." },
  ]
)
TaskCreate(tasks with dependencies)
```

### Phase 3: 파이프라인 실행

팀원들이 의존성 해결된 작업을 claim, SendMessage로 산출물 공유. 리더는 주요 지표 모니터링:
- ICP 통과율 (현재 **100%** of aligned, 2238/2238) ✅
- **Grasp 합성**: mode별 count 검증 (standing 16 / lying 8|6 / cube 2), collision reject <30%
- **모델 val metrics** (Flow Matching): Position MAE <10mm, Rotation geodesic <15°, **Mode coverage ≥80%** (32 sample 중 GT mode 커버)
- ONNX velocity net PT ↔ ONNX diff <1e-4
- MATLAB predict_grasp.m: Euler 1-step + 충돌 필터 + CameraTform 통합, 최소 1개 생존자 보장

### Phase 4: 최종 보고

`_workspace/FINAL_REPORT.md`:
- 각 단계 지표
- 모델 경로 + ONNX + predict_grasp.m
- 이전 실행 대비 변화 (부분 재실행 시)

### Phase 5: 정리

TeamDelete, `_workspace/` 보존.

## 에러 핸들링

| 상황 | 전략 |
|---|---|
| YOLO conf가 비정상적으로 낮음 (특히 cube_red=0) | **BGR 버그 재발 의심**. 경로 전달 vs numpy 전달 결과 비교 테스트 |
| ICP 통과율 <20% | 1회 threshold 완화 → 여전히 낮으면 K/PLY 조사 (현재 100% 달성) |
| Grasp 합성 실패 (모두 충돌) | 그리퍼 파라미터/테이블 높이 재확인 |
| Standing grasp 개수 ≠ 16 | mode 분기 (`cos_z > 0.7`) + top-down/side 두 루프 모두 실행되는지 |
| Cube yaw가 대각선 | R_icp 수직축 탐지(`argmax(|R[2,:]|)`) + edge column 정렬 검증 |
| Flow Matching loss NaN | lr 절반 + grad clip 확인 |
| Mode diversity 부족 (sample 모두 같은 grasp) | cond dropout ↑, g_0 분산 ↑ |
| ONNX export 실패 | 모델 단순화 (커스텀 op 제거) |
| 좌표 프레임 오염 (grasps_base 발견) | dataset-curator에 즉시 중단 + 스키마 위반 보고 |

## 테스트 시나리오

### 정상 흐름 (현재 상태부터)
1. 사용자 "grasp_v1 파이프라인 이어서" 요청
2. Phase 0: yolo_cache_v3 존재 확인 → ICP부터 시작
3. Phase 1: config 확정
4. Phase 2: 6인 팀, icp-labeler 먼저 실행
5. Phase 3: ICP → grasp-synth → curator → model-engineer 순차
6. Phase 4: FINAL_REPORT

### 부분 재실행
1. "ICP fitness 0.03으로 재시도"
2. Phase 0: icp-labeler, grasp-synth, curator, model-engineer 재호출
3. yolo-extractor 재실행 안 함 (기존 cache 유지)

### BGR 버그 재발 의심 흐름
1. 사용자가 "검출 이상하다" 보고
2. 리더가 yolo-extractor에 경로 전달 vs numpy 전달 비교 지시
3. 결과가 다르면 스크립트의 `cv2.cvtColor(BGR→RGB)` 유무 검토 → 제거

## 후속 작업 키워드

- 다시 실행 / 재실행 / 재학습 / 재배포
- ICP threshold 조정 / grasp 합성 전략 변경
- 모델만 학습 / ONNX 재export / MATLAB 통합 재검증
- 이전 결과 개선 / 충돌률 줄이기
- 데이터 더 수집해서 학습
- "BGR 맞는지 확인"
