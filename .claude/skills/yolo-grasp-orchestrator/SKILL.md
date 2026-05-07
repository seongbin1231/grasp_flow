---
name: yolo-grasp-orchestrator
description: "YOLO_Grasp 프로젝트의 **6-DoF SE(3) Conditional Flow Matching grasp** 예측 모델 + **IEEE RA-L 8p 논문화 작업** (3주 일정, ieee-paper branch) 의 7인 에이전트 팀 오케스트레이터. 1280×720 카메라 프레임 전용. **zhou_9d_full_250ep (9D Zhou, val 0.2419)** 가 production. CFG=1.0 default (sweep 결과 GT 33% 근접). 사실 체크 12건 정정 완료 (9D stratified 재설계 / Stage 2 76× 과소 / RegionNormalizedGrasp NO LICENSE). YOLO_Grasp · grasp 모델 · 파지자세 예측 · Simulink 파이프라인 · ONNX 배포 · **8D vs 9D + depth vs PC ablation · 옵션 A~E 학습 트릭 (stratified noise · CFG interval · t-schedule · post-hoc EMA · Reflow) · GraspNet-1B Stage 2 fine-tune (1M subsample) · RegionNormalizedGrasp baseline · 논문 figure/table 생성** 작업 시 반드시 사용. 후속 작업 (재실행, 부분 수정, 재학습, 데이터 추가, paper figure 갱신, ONNX export, RA-L 진행, ieee-paper branch 작업) 도 이 스킬."
---

# YOLO_Grasp Orchestrator — 7인 팀 조율 (모델 + 논문화)

**Pixel-Conditioned Multi-Modal 6-DoF Grasp Pose Generation via Conditional Flow Matching** 모델 (IEEE 급 논문 목표) + RoboCup ARM 2026 즉시 응용 모델을 담당하는 에이전트 팀 오케스트레이터. 카메라 프레임 전용 파이프라인, base 변환은 MATLAB 런타임.

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

## 📍 현재 진행 상황 (2026-05-07)

| 단계 | 상태 | 산출물 |
|---|---|---|
| 1~16 데이터·YOLO·ICP·Grasp·Dataset·학습·ONNX·viz·Direct·Fig·8D/9D·PC ablation | ✅ | (이전 변경 이력 표 참조) |
| 17. **zhou_9d depth full 250ep** (production 후보) | ✅ | `runs/.../zhou_9d_full_250ep/.../best.pt` (ep250, **val_flow=0.2419**, 14.8M, 9D Zhou) |
| 18. **CFG sweep diagnostic + Fig 3 CFG=1.0 채택** | ✅ | `paper_figs/diag_cfg_sweep.{png,pdf}` (Fig 4 신설). CFG=0:78%/1:34%/2.5:22%/3.5:12% — **CFG=1.0 default** 결정 |
| 19. **Fig 1/3 마진+gripper+title 갱신** | ✅ | gripper 80%/130%, title 22pt bold, 마진 0, scene PC s=4.5 |
| 20. **venue RA-L 8p 결정 + ieee-paper branch + W0 skeleton** | ✅ | `ieee-paper` 843b2f6: 신규 11 파일 (sweep_t_schedule / posthoc_ema / reflow_data_gen / graspnet1b/{6} / eval_rngnet / zhou6d_to_graspgroup17) + demo_inference 수정 (stratified + cfg_at_t helper). 모두 NotImplementedError + TODO |
| 21. **W0 P1 9D SO(3) stratified noise 실제 구현** | ⏳ pending | `scripts/demo_inference.py:sample_g0_stratified_9d` skeleton 만 — SO(3) prior 구현 필요 |
| 22. **W0 Fig 1/3/Table 1 9D + CFG=1.0 + stratified 재생성** | ⏳ pending | |
| 23. **W1 P3/P4 t-schedule 50ep × 4 + winner 250ep** | ⏳ pending | logit-normal/cosine/uniform/lognorm_neg |
| 24. **W1 RegionNormalizedGrasp baseline** | ⏳ pending | CUDA Toolkit 설치 + monkey-patch (NO LICENSE → inference-only) |
| 25. **W2 GraspNet-1B 다운 + Stage 2 fine-tune (1M subsample)** | ⏳ pending | **디스크 44GB 제약**: 외장 SSD 또는 subsample 권장 |
| 26. **W2 P2 post-hoc EMA** | ⏳ pending | Karras 2024 |
| 27. **W3 GraspNet 공식 AP eval + Reflow 1라운드** | ⏳ pending | |
| 28. **W3 paper draft** | ⏳ pending | 8 page |

**현재 초점 (2026-05-07~)**: RA-L 8p 3주 일정. Branch `ieee-paper` 위에 skeleton 11 파일 ready. **다음**: W0 P1 9D stratified 구현 → Fig/Table 9D 갱신 → W1 시작. **상세**: 메모리 `ral_8p_plan.md` (사실 체크 12건 정정 + 컴퓨팅 47h GPU/112h dev).

## 사실 체크 결과 (2026-05-07, 3 병렬 Explore agent, 12건)

🔴 Critical:
1. 9D Zhou 의 stratified noise 는 8D 코드 그대로 못 씀 (g[5]=R[2,0] ≠ a_z) → SO(3) 직접 샘플
2. Stage 2 fine-tune 76× 과소 추정 (실제 612h) → **1M subsample (~13h)**
3. RegionNormalizedGrasp NO LICENSE + 640×360 internal + API 부재

🟠 High: CFG interval [0.3,0.7] paper 미증명 / SD3 logit-normal image-only / Reflow FID 4.85=+distill
🟡 Medium: snapshot 10~20 / t-sampling L357 만 / branch ieee-paper / PointNet2 nvcc / 함수명 정정 등

상세: 메모리 `ral_8p_plan.md` § 사실 체크

**핵심 ablation 결정 (2026-05-01~03)**:
- **회전 표현**: 8D approach_yaw → **9D Zhou 6D** (−34%, Zhou19 CVPR 인용)
- **입력 표현**: PC token → **depth raster + (u,v)** (−13.6%, small-data 친화)

## 🏆 핵심 정량 결과 (val 400 obj)

| Mode | Direct MLP COV | Ours (Flow) COV | Direct APD | Ours APD |
|---|---|---|---|---|
| standing (3 모드) | **33.3%** | **99.2%** | 0.24 cm | 0.50 cm |
| lying (1 모드) | 100% | 100% | 0.19 cm | 6.26 cm |
| cube (1 모드) | 100% | 100% | 0.31 cm | 0.35 cm |
| **all** | 93.3% | 99.9% | **0.22 cm** | **4.32 cm** (**20×**) |

threshold 5cm/30° (Sundermeyer ICRA 2021), COV (Achlioptas ICML 2018). 상세: 메모리 `paper_metrics_official.md`

## 🎯 Grasp 정책 v4 (2026-04-22 확정)

| 모드 | 클래스 | 구성 | grasp 수 |
|---|---|---|---|
| standing | bottle/can | **3-layer**: top-down 8 + side-45 8 + side-cap 8 (θ=0°/45°/90°) | **24** |
| lying | bottle/can | N=4 위치 × tilt {-30°, 0°, +30°} × 180° 대칭 | **24** |
| lying | marker/spam | N=3 위치 × tilt {-15°, 0°, +15°} × 180° 대칭 | **18** |
| cube | cube_* (4종) | R_icp edge 정렬 2 yaw (대각선 ❌) | 2 |

**Exclusion**: marker/spam standing (물리 실패), spam_lying 의 medium 축이 카메라 Z 미정렬일 때.
출력: **6-DoF SE(3)** `pose_cam(N,7) = [x,y,z, qw,qx,qy,qz]`.

## 🧠 모델 아키텍처 (확정)

**Conditional Rectified Flow (Flow Matching)** — 다봉 분포 샘플링으로 여러 후보 grasp 생성 + 충돌 필터로 선택.
- **학습 공간 (현 production 후보)**: `g = (x, y, z, R00, R10, R20, R01, R11, R21)` **9D Zhou 6D** (`--rot_repr zhou6d`). 추론 후 Gram-Schmidt 로 SO(3) 복원 → quat. Legacy 8D 도 `--rot_repr approach_yaw` 로 호환 (deploy2/v7 가 이거 사용 중)
- **입력 분기**: depth+uv (`FlowGraspNet`, 기본) vs PC (`FlowGraspNetPC`, `--use_pc_only`, 별도 클래스). depth 가 13.6% 우세 — production 은 depth
- Loss: `‖v_θ(g_t, t, c) − (g_1 − g_0)‖²` + symmetric_min_loss (lying 180°)
- 추론: Reflow 1-step, N=32 병렬 샘플
- ONNX: velocity MLP만 export. MATLAB이 노이즈 생성, Euler step, 충돌 필터, CameraTform 수행
- 참조: `/home/robotics/Competition/grasp_model/graspflow/` (SE(3) 구현체, 우리는 9D Zhou 로 단순화)

## 에이전트 구성 (7인)

| 팀원 | 타입 | 역할 | 스킬 | 주요 출력 |
|---|---|---|---|---|
| matlab-collector | custom | 캡처 데이터 유지/추가 수집 | `/matlab-data-collection` | `img_dataset/captured_images/` |
| yolo-extractor | custom | YOLO v3 배치 추론 (BGR!) | `/yolo-batch-inference` | `yolo_cache_v3/detections.h5` |
| icp-labeler | custom | open3d multi-scale ICP, 카메라 프레임 pose | `/icp-quality-management` | `icp_cache/poses.h5` (2,485 stable) |
| grasp-synthesizer | custom | **6-DoF SE(3) grasp 정책 v4** 합성 | `/top-down-grasp-synthesis` | `grasp_cache/grasps.h5` (40,984) |
| dataset-curator | custom | HDF5 v2 통합 + cube 통합 + QA | `/hdf5-dataset-management` | `datasets/grasp_v2.h5` |
| model-engineer | custom | Flow Matching 학습 + ONNX + MATLAB + **Direct MLP baseline + v8 cross-attn + v9 PC-only + 8D vs 9D Zhou + depth vs PC ablation** | `/depth-uv-grasp-training`, `/onnx-matlab-integration` | `runs/yolograsp_v2/{v7_v4policy_big, v7_direct_mlp_big, zhou_9d_full_250ep, zhou_9d_pc_only_full_250ep}/` + `deploy2/`(8D v7) + `deploy3/`(9D zhou pending) |
| **paper-figure-author** | **custom (NEW)** | 논문 figure (mathtext, RGB-색칠 PC, mode-balance) + Table 1 (COV/APD/MAE) + 본문 표현 | (직접 `scripts/make_paper_*.py`) | `paper_figs/fig{1,2,3}_*.{png,pdf}` + `table1.{md,json}` |

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

반드시 시작 시 다음을 확인하고 부분 재실행 여부를 결정 (orchestrator-template.md 1.2.1 규칙: 새 실행이면 기존 `_workspace/` 를 `_workspace_{YYYYMMDD_HHMMSS}/` 로 보관 후 새로 생성):

1. `_workspace/` 존재 여부
2. **모든 단계 1–12 완료**. 산출물 존재 확인:
   - YOLO: `img_dataset/yolo_cache_v3/detections.h5` (3242 passed)
   - ICP v3: `img_dataset/icp_cache/poses.h5` (2,485 stable)
   - Grasp v4: `img_dataset/grasp_cache/grasps.h5` (40,984)
   - Dataset v2: `datasets/grasp_v2.h5` (33,744 train + 7,240 val)
   - **Flow v7**: `runs/yolograsp_v2/v7_v4policy_big/.../best.pt` (ep226, val_flow=0.3676)
   - **Direct MLP**: `runs/yolograsp_v2/v7_direct_mlp_big/.../best.pt` (ep58, val_grasp=0.1118)
   - ONNX: `deploy2/onnx/{encoder,velocity}.onnx + meta.json`
   - **Paper**: `paper_figs/fig{1,2,3}_*.{png,pdf}` + `table1.{md,json}`
3. 사용자 요청 파싱 (재실행 트리거):
   - "YOLO 재추론" → 단계 3부터
   - "ICP threshold 조정" → 단계 4부터
   - "Grasp 정책 변경" → 단계 5부터
   - "Dataset 재빌드" → 단계 6부터
   - "모델 재학습" → 단계 7+11 (Direct 도 함께)
   - "**v8 cross-attn 업그레이드**" → 단계 7→8→11→12 (메모리 `model_upgrade_v8_plan.md` 참조)
   - "ONNX 재export" → 단계 8만
   - "**Paper figure 갱신**" → 단계 12만 (paper-figure-author 단독 호출)
   - "**Metric 임계값 변경 / 표 재계산**" → 단계 12만 (`make_paper_table1.py` 만)
   - "처음부터" → `_workspace_{ts}/` 보관 후 Phase 1
4. 변경 감지:
   - **모델·ONNX·논문 figure 모두 완료 (2026-04-27)**. 명시 요청 없으면 재호출 금지.
   - **현재 기본 경로**: 논문 figure / table 미세 조정 + 모델 v8 트리거 대기.

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
- **논문 figure 갱신 / Fig 1 / Fig 2 / Fig 3 / Table 1**
- **mode coverage 다시 / COV / APD / 임계값 변경**
- **standing 모드 표현 늘려 / lying outlier 제거**
- **paper banana / Gemini image / 모델 그림 다시**
- **Direct MLP baseline 재학습 / ablation 추가**
- **v8 cross-attention / multi-scale crop / 모델 업그레이드**
- **시뮬레이션 환경 표기 / Simulink 3D Animation / Unreal**
- **v8 sweep / wandb sweep / variant ablation**
- **standing mode collapse / top-down 부족 / stratified noise**
- **PixArt-α / Hunyuan-DiT / PointNet++ MSG (인용 출처)**

## 핵심 산출물 경로 (논문화)

| 산출물 | 경로 | 갱신 스크립트 |
|---|---|---|
| Fig 1 (모드별 GT) | `paper_figs/fig1_gt_synthesis.{png,pdf}` | `scripts/make_paper_fig1_gt.py` |
| Fig 2 (architecture) | `paper_figs/fig2_architecture.{png,pdf}` | `scripts/make_paper_fig2_arch.py` |
| Fig 3 (Direct vs Flow) | `paper_figs/fig3_compare.{png,pdf}` | `scripts/make_paper_fig3_compare.py` |
| Table 1 (COV/APD/MAE) | `paper_figs/table1.{md,json}` | `scripts/make_paper_table1.py --max_objs 400` |
| 후보 탐색 그리드 | `paper_figs/scan_can_candidates.png` | `scripts/scan_can_candidates.py` |
