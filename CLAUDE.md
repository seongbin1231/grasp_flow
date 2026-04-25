# YOLO_Grasp — Pixel-Conditioned Multi-Modal 6-DoF Grasp via Flow Matching

Depth 이미지 + 픽셀 (u,v) 입력으로 **6-DoF SE(3) grasp** `[x,y,z, qw,qx,qy,qz]`를 **카메라 프레임**에서 생성하는 모델. **Conditional Rectified Flow (Flow Matching)** 기반으로 멀티 모달 분포 샘플링 (N=32~64). ONNX velocity MLP 만 export → MATLAB `predict_grasp.m`이 Euler step + 충돌 필터 + `CameraTform` 곱해 base frame pose 생성 → UR5e 파이프라인 연결. **즉시 응용**: RoboCup ARM 2026.

## 연구 발전 방향 (IEEE 급 논문 목표)

**제목**: *Pixel-Conditioned Multi-Modal 6-DoF Grasp Pose Generation via Conditional Flow Matching*
**핵심 기여 (2개)**: ① 다수 객체 scene 의 픽셀 (u,v) 컨디셔닝 타겟 파지, ② Conditional Flow Matching 으로 멀티 모달 분포 직접 학습·샘플링.
**Future work**: VLM 어텐션 → (u,v) → grasp 의 자연어 명령 기반 end-to-end manipulation.
**GitHub**: https://github.com/seongbin1231/grasp_flow
**상세 계획**: 메모리 `research_paper_plan.md` 참조.

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

## Grasp 정책 v4 (2026-04-22 최종, 6dof-v4)

| 모드 | 클래스 | 구성 | grasp 수 |
|---|---|---|---|
| standing | bottle/can | **3-layer tilt**: top-down 8 yaw (θ=0°) + **side-45° 8 azimuth** + side-cap 8 azimuth (θ=90°) | **24** |
| lying | bottle/can | N=4 위치 × **tilt {−30°, 0°, +30°}** × 180° 대칭 | **24** |
| lying | marker/spam | N=3 위치 × **tilt {−15°, 0°, +15°}** × 180° 대칭 | **18** |
| cube | cube_* (4종) | R_icp edge column 정렬 2 yaw (대각선 금지) | 2 |

**정책 제외** (grasp 0개 생성):
- `marker_standing`: 7 obj (물리적으로 서있지 않음)
- `spam_standing`: 4 obj (동일)
- `spam_cap_not_up`: 149 obj (spam lying 중 라벨/뚜껑 면이 측면 → medium 축이 카메라 Z 와 정렬 안 됨. threshold 0.7)

**핵심 구현 디테일**:
- Lying tilt: long 축 주위 Rodrigues 회전, TCP = `p_center − app·short_r` (원통 곡면 추종)
- Standing side-45: TCP 선형 보간 `0.5·p_cap_top + 0.5·p_cap_side`, yaw 는 `Tool Y ⊥ cap 축` 이 되도록 역산
- grasp_group id: top-down=0, side-cap=1, lying=2, cube=3, **side-45=4** (v4 추가)

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
| 2026-04-22 | **ICP 필터 업그레이드** (Phase 5) | scripts/batch_icp_v3.py | `bbox_margin=20px` (31% drop) → **mask_px per-class p10** (10.1% drop). fitness gate 0.02 → **0.30**, rmse gate 5mm → **3mm**. stable 2,238 → **2,485 (+11%)**. PLY 파일 git 에서 복구 필요했음 |
| 2026-04-22 | **Grasp 정책 v4 확정** (Phase 6) | scripts/batch_grasp_synthesis.py + 2 viz 스크립트 | v1→v4: (1) marker/spam standing 제외, (2) spam_lying 은 medium 축(라벨 normal) 수직일 때만, (3) lying bottle/can tilt ±30°, marker/spam ±15° (Rodrigues + 곡면 추종), (4) standing 3-layer (0°/45°/90°), (5) side-45 는 Tool Y ⊥ cap 축 yaw 역산. 총 17,370 → **40,984 grasp (+136%)** |
| 2026-04-22 | **Dataset 재빌드 v4** | scripts/build_grasp_v2.py + datasets/grasp_v2.h5 | excluded object skip, group_names 에 `side-45` 추가. train 33,744 / val 7,240 rows. policy_version `6dof-v4` |
| 2026-04-22 | **v7_v4policy_big 학습** | runs/yolograsp_v2/v7_v4policy_big/ | hidden 1024, n_blocks 12, **35.28M params** (v6 대비 +36%). 250ep 중 **ep 226 best val_flow=0.3676** (v6 0.3640 와 동등). batch 16, lr 1e-3, marker_boost 1.5, spam_boost 2.5 |
| 2026-04-22 | **deploy2 신규 배포** | deploy2/onnx + yolo + viz + README | v7 ONNX: encoder(2.4MB) + velocity(**139MB**) + meta.json(v4 정책 메타 포함). round-trip max\|Δ\|=5.96e-06. 인터페이스 동일 (MATLAB 팀 코드 변경 불필요). deploy/(v6) 병존 유지 (rollback 용) |
| 2026-04-22 | **추론 검증 (random6 val scene)** | deploy2/viz/ | 7 카테고리 kept=28~32/32 (collision filter 정상). standing 에서 `kept split: topdown=X side=Y mid=Z` 확인 → **3-layer 모두 학습 성공**. lying_marker 3.9cm 오차는 알려진 약점 (물리 얇음) |
| 2026-04-23 | **논문 개요 작성** (교수 보고용) | 외부 산출물 | 8 섹션 (Contribution/Background/Objective/Originality/Main contents/Validation/Conclusion/Discussion). 핵심 기여 2개로 정제. Originality 는 직접 회귀 vs 샘플링-스코어링 vs Flow Matching 3분할 비교 |
| 2026-04-23 | **논문 제목 확정** | — | *Pixel-Conditioned Multi-Modal 6-DoF Grasp Pose Generation via Conditional Flow Matching* (4 키워드 모두 포함) |
| 2026-04-23 | **VLM 결합 future work 명시** | — | "자연어 명령 → VLM attention → (u,v) → grasp" end-to-end pipeline 확장 방향. open-vocabulary + 자연어 인터페이스 |
| 2026-04-24 | **GitHub 리포지토리 신설 + push** | github.com/seongbin1231/grasp_flow | main 브랜치 4 커밋 push (de0d9a8 → e69f453). origin tracking 설정. 텍스트만 (.gitignore 로 바이너리 제외) |
| 2026-04-25 | **연구 목표 IEEE 급으로 격상** | CLAUDE.md, MEMORY.md | 단순 대회 모델 → 일반화 멀티 모달 grasp 생성 framework 로 확장. 메모리에 `research_paper_plan.md` 신설하여 제목·기여·future work·인용 문헌 정리 |
