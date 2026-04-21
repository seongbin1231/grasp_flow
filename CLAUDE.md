# YOLO_Grasp — RoboCup ARM 2026

Top-view Depth + YOLO (u,v) 입력으로 **6-DoF SE(3) grasp** `[x,y,z, qw,qx,qy,qz]`를 **카메라 프레임**에서 생성하는 모델을 구축. **Conditional Rectified Flow (Flow Matching)** 기반으로 다봉 분포 샘플링 (N=32~64). ONNX velocity MLP 만 export → MATLAB `predict_grasp.m`이 Euler 1-step + 충돌 필터 + `CameraTform` 곱해 base frame pose 생성 → 기존 UR5e 파이프라인 연결.

## 하네스: YOLO_Grasp Pipeline

**목표:** MATLAB/Simulink 취득 1280×720 데이터 + ICP pseudo-GT + **mode별 6-DoF grasp 정책** 합성으로 학습 데이터셋 (`grasp_v2.h5`) 생성 → Flow Matching 학습 → ONNX + MATLAB 통합.

**트리거:** YOLO_Grasp 관련 작업(데이터 수집/ICP 라벨링/**grasp 합성 6-DoF**/HDF5 데이터셋/**Flow Matching 모델 학습**/ONNX 배포/MATLAB 통합) 요청 시 반드시 `yolo-grasp-orchestrator` 스킬을 사용한다. 단순 파일 열람 질문은 직접 응답 가능.

## 핵심 환경 상수 (모든 스킬이 따름)

| 항목 | 값 |
|---|---|
| RGB/Depth 해상도 | **1280×720** |
| Camera K | `fx=fy=1109, cx=640, cy=360` |
| Depth 포맷 | **uint16 PNG, 밀리미터 단위** → Python에서 `/1000.0`로 meter 변환 |
| YOLO numpy 입력 | **BGR** (ultralytics 공식 규약). `cv2.imread` 그대로 전달, RGB 변환 금지 |
| Python 실행 | `/home/robotics/anaconda3/bin/python` (conda base). `python3`(isaacsim kit)는 의존성 없음 |
| 최신 YOLO 모델 | `runs/yolov8m_seg_v3_1280/weights/best.pt` (1280 학습, 8 클래스, 2026-04-18) |
| 좌표 프레임 | **카메라 프레임 전용**. Base 변환은 MATLAB 런타임에서 `CameraTform`으로 처리 |
| Cube 클래스 | 하류(ICP·grasp·학습)에서 `cube_blue/green/p/red` 4종을 `cube` 1종으로 통합 |
| Grasp 표현 | **6-DoF SE(3) 7D `[x,y,z, qw,qx,qy,qz]`** (4-DoF `[x,y,z,yaw]` 폐기, 2026-04-19) |

## Grasp 정책 (2026-04-19 확정)

| 모드 | 클래스 | 구성 | grasp 수 |
|---|---|---|---|
| standing | bottle/can/marker/spam | top-down 8 yaw + side-horizontal 8 방위 (approach ⊥ cam Z, 정확히 90°) | **16** |
| lying | bottle/can | 긴축 N=4 × 180° 대칭 | 8 |
| lying | marker/spam | N=3 × 180° 대칭 | 6 |
| cube | cube_* (4종) | R_icp edge column 정렬 2 yaw (대각선 금지) | 2 |

## 모델 아키텍처

**Conditional Rectified Flow** — Lipman 2023. 단일 회귀의 mode collapse, Set prediction의 slot 제약을 회피. Velocity MLP만 ONNX export, MATLAB 런타임이 노이즈 N개 생성 → 1-step Euler → 충돌 필터 → 최적 grasp.

## 관련 외부 디렉터리

- [RoboCup_ARM/scripts/](/home/robotics/Competition/RoboCup_ARM/scripts/) — MATLAB ICP/IK 코드 (참조, 포팅 기준)
- [RoboCup_ARM/models/ply/](/home/robotics/Competition/RoboCup_ARM/models/ply/) — 8종 물체 PLY 모델
- [grasp_model/](/home/robotics/Competition/grasp_model/) — 참고용 GraspFlow 구현

## 변경 이력

| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-04-18 | 초기 구성 | 전체 | 하네스 최초 구축 (6인 팀 + 7 스킬) |
| 2026-04-18 | 해상도 480×640 → **1280×720** 전면 수정 | 전체 스킬 | 실제 Simulink 캡처 해상도 확인 (1280×720 PNG, K fx=fy=1109, cx=640, cy=360) |
| 2026-04-18 | **BGR 입력 규약** 크리티컬 추가 | yolo-batch-inference, yolo-extractor | 버그 진단: ultralytics는 numpy 입력을 BGR로 가정. `cv2.cvtColor(BGR→RGB)` 후 predict에 전달하면 R·B 채널 역전으로 conf 붕괴 (v3 1783→3242로 복구 확인) |
| 2026-04-18 | **카메라 프레임 전용**으로 단순화, CameraTform 수집 제거 | icp-labeler, grasp-synthesizer, dataset-curator, model-engineer | 사용자 확정: base 변환은 MATLAB 런타임에서 처리. 학습 데이터 수집·라벨·모델 모두 카메라 프레임에서만 동작 |
| 2026-04-18 | cube_blue/green/p/red 4종을 하류에서 `cube` 1종으로 통합 | yolo-extractor, grasp-synthesizer, dataset-curator | 사용자 확정: depth 상 형상 동일. 색은 학습에 기여 없음 |
| 2026-04-18 | YOLO 모델 v3_1280 finetune 완료 | — | Roboflow v2(1870장, 원본 1280×720) + yolov8m-seg finetune 50 epoch. Mask mAP50-95=0.982. Roboflow는 Fit within 대신 리사이즈 없음 옵션 사용 |
| 2026-04-18 | Python 환경 conda base 고정 | 전체 | 시스템 `python3`는 isaacsim kit으로 redirect, 의존성 없음 |
| 2026-04-18 | ICP 포팅은 open3d (MATLAB 외부 호출 대신 Python 내장) | icp-labeler, icp-quality-management | open3d 0.19.0 설치 완료. batch 처리 용이 |
| 2026-04-19 | **ICP v3 배치 완료** (PCA 4-flip + identity fallback, MATLAB 고정 distance 0.048/0.020/0.005, statistical outlier removal) | `img_dataset/icp_cache/poses.h5` | 2238/2238 pass gate (100%), fitness mean 0.46, rmse 2.1mm 평균. bbox margin<20px 31% 자동 외곽 제외 |
| 2026-04-19 | **Grasp 정책 4-DoF → 6-DoF SE(3) 변경** | grasp-synthesizer, dataset-curator, model-engineer + 스킬 4종 | Standing bottle/can에 "측면 horizontal approach" 추가 (reachability 개선). approach 방향이 cam +Z 외에 horizontal도 존재 → full SE(3) 불가피. 출력 `[x,y,z,yaw]` → `[x,y,z, qw,qx,qy,qz]` |
| 2026-04-19 | **Grasp 정책 최종 확정** | grasp-synthesizer + /top-down-grasp-synthesis | Standing(모든 클래스): top-down 8 + side-horizontal 8 = 16개, 두 그룹 90° 직각. Lying bottle/can 8, marker/spam 6. Cube: 면 edge 정렬 2 (대각선 금지) |
| 2026-04-19 | **모델 Flow Matching 채택** | model-engineer + /depth-uv-grasp-training, /onnx-matlab-integration | Conditional Rectified Flow. 단일 회귀 mode collapse, Set prediction slot 제약을 회피. 8D 파라미터화 (pos+approach+sincos yaw). ONNX velocity MLP만, MATLAB이 노이즈+Euler+필터 |
| 2026-04-19 | 데이터셋 스키마 v1(4-DoF) → **v2(6-DoF)** | /hdf5-dataset-management, dataset-curator | `grasps_cam(N,7)` quaternion, per-object 공통 uv(YOLO centroid), unrolled rows. schema_version="v2". 예상 N_rows ≈ 16,000 |
| 2026-04-19 | 훈련 개선 (sweep → posnorm → tier2 → marker_boost → big+long) | model-engineer | val_flow 궤적: sweep 0.26 → v2 posnorm + symmetric_min_loss + dim weights + marker_boost 4/spam_boost 2.5 → v5 interrupted → **v6_150ep** 재학습 |
| 2026-04-20 | **v6_150ep 학습 완료** (hidden 768, n_blocks 8, AdaLN-Zero, EMA 0.9998, warmup_frac 0.04, seed 42, wandb yolograsp-v2) | `runs/yolograsp_v2/v6_150ep/` | ep 118/150 best, val_flow=0.3640. v5 ep69 val=0.4055 대비 10.2% ↓. 현재 프로젝트 최종 모델 |
| 2026-04-20 | **ONNX export 완료** (encoder/velocity 분리 + meta.json) | `deploy/onnx/` | encoder.onnx(2.4MB, scene당 1회) + velocity.onnx(57MB, T×2 CFG) + meta.json(pos_mean/std, 상수) + README.md(357줄 MATLAB 통합 스펙). opset 17, onnxruntime round-trip max\|Δ\|=6.2e-06 |
| 2026-04-20 | **충돌/contact 필터 확정** | scripts/demo_inference.py, scripts/infer_live.py | body_margin=5mm (wrist+stem, palm 제외) + **tip→TCP sweep contact 15mm** (단순 tip 거리는 narrow bottle side grasp 오탈락. sweep이 그리퍼 closure path를 훑어 정답) |
| 2026-04-21 | **라이브 ROS 추론 파이프 구축** | scripts/ros_capture_once.py + scripts/infer_live.py | DOMAIN_ID=13, python3(isaacsim kit ROS bindings)로 capture → conda base python으로 infer. 1 frame 실측 완료 |
| 2026-04-21 | **best-pick 선정 로직 추가** | scripts/demo_inference.py, scripts/infer_live.py | top-down 우선(\|a_z\|>0.7) → pool 내 YOLO uv 3D 최단 거리. 골드 색상(#ffd700) 구분 시각화 |
| 2026-04-21 | **deploy/yolo/ 신설** (MATLAB 팀용 YOLO ONNX 1280) | `deploy/yolo/yolov8m_seg_1280.onnx` | ultralytics export, opset=12, simplify=True. 팀 기존 weights/best.onnx(imgsz=640) 과 **같은 best.pt** 지만 imgsz 1280 유지로 Flow 훈련 uv 일관성 보장 |
| 2026-04-21 | GT grasp 시각화 (val scene random6) | scripts/demo_gt.py → deploy/viz/gt/ | 정책 검증용. uv 매칭 버그(grasp_v2 object_ref 는 split 전역 번호) 수정 필요했음 — 추후 dataset 호출 시 uv≈target으로 매칭 |
| 2026-04-21 | **marker diagnosis 확정** | — (분석) | marker 훈련 실제 10.6% (1450/13640) + boost 4× = 42% 유효 비중. 수량 부족 아님. **원인: 물리 얇음(∅10mm) → 192×192 local crop 98%가 배경 → condition signal 약함**. 권장 MATLAB fallback: kept=0 시 YOLO uv 3D + top-down 강제 |
