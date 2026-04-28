# Dataset Transformation & Usage Strategy — 검증 완료판

**작성일**: 2026-04-28 (2026-04-28 19:00 갱신: graspnet-baseline 라이선스 정정 + AP 메트릭 정의)
**검증 방식**: 4개 전략 에이전트 + 3개 검증 에이전트 + 1회 직접 LICENSE/README curl = 총 8회 병렬 fact-check (paper PDF 정독 + GitHub source code 라인별 확인 + 라이선스 파일 verbatim)

---

## 🚨 0. 검증 결과 — 즉시 알아야 할 4 가지 중대 오류 (2026-04-28 갱신)

### ❌ 오류 0 (신규 2026-04-28 19:00): graspnet-baseline 라이선스 — Apache 2.0 → Non-Commercial 정정

**잘못된 표기** (이전 EXPERIMENTAL_SETUP.md): *"GraspNet baseline — Apache-2.0 (라이선스 깨끗)"*
**검증된 사실**: graspnet-baseline LICENSE 파일 첫 줄 verbatim:
> *"GRASPNET-BASELINE SOFTWARE LICENSE AGREEMENT — ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY"*

⚠️ **graspnet-baseline 코드 = Non-Commercial Research Only**.
✅ graspnetAPI 코드 = MIT (별개 리포)
✅ GraspNet-1B 데이터셋 = CC BY-NC-SA 4.0 (학술 OK)

**영향**: 학술 paper 사용은 OK 단 footnote 의무. 상용 RoboCup ARM 2026 실전 deploy 에 graspnet-baseline ckpt 사용 금지 (자체 모델만 OK).
**조치**: paper Table 2 footnote: *"GraspNet baseline used under non-commercial research license."*

### ❌ 오류 1: ACRONYM gripper convention — origin 위치 (PAPER-RISK 최고)

**잘못된 가정**: *"+Z = approach + 원점 = 손가락 끝 사이"*
**검증된 사실**: **+Z = approach 맞음**, BUT **원점은 그리퍼 BASE (mount)**, 손가락 끝이 아님.

**증거** (`acronym_tools/acronym.py:404-443`, `create_gripper_marker`):

### ❌ 오류 1: ACRONYM gripper convention — origin 위치 (PAPER-RISK 최고)

**잘못된 가정**: *"+Z = approach + 원점 = 손가락 끝 사이"*
**검증된 사실**: **+Z = approach 맞음**, BUT **원점은 그리퍼 BASE (mount)**, 손가락 끝이 아님.

**증거** (`acronym_tools/acronym.py:404-443`, `create_gripper_marker`):
```python
cb1 = trimesh.creation.cylinder(  # gripper "stem"
    segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]   # 0 → 6.6cm
)
cfl = trimesh.creation.cylinder(  # left finger
    segment=[[ 4.10e-02, 0, 6.60e-02], [ 4.10e-02, 0, 1.1217e-01]]
)
```
→ 원점 (0,0,0) 은 mount 위치, **손가락 끝은 z = +0.1122 m (~11.2 cm 앞)**

**영향**: 손가락 끝 origin 가정으로 코딩하면 모든 grasp 가 approach 방향으로 **11.2 cm 빗나감** → 데이터셋 전체 무효
**필수 조치**: ACRONYM 변환 시 우리 grasp_v2 convention 과 비교 → TCP origin 위치 통일. 첫 10 grasp `create_gripper_marker()` 로 시각 검증 후 스케일

### ❌ 오류 2: GraspNet-1B 카메라 K — D435 vs Kinect 혼동

**일부 에이전트 주장**: *"D435 K ≈ (631, 631, 638, 366) at 1280×720"*
**검증된 사실**: 그 K 는 **Kinect Azure** 의 것임. D435 는 다른 값.

**증거** (`graspnetAPI/utils/utils.py:34-37`):
```python
if camera == 'kinect':
    param.intrinsic.set_intrinsics(1280,720,631.55,631.21,638.43,366.50)
elif camera == 'realsense':
    param.intrinsic.set_intrinsics(1280,720,927.17,927.37,651.32,349.62)
```

| 카메라 | fx | fy | cx | cy | 우리 (1109, 1109, 640, 360) 와 비교 |
|---|---|---|---|---|---|
| **Kinect Azure** | 631.55 | 631.21 | 638.43 | 366.50 | 우리보다 **1.76× 광각** |
| **RealSense D435** | 927.17 | 927.37 | 651.32 | 349.62 | 우리보다 **1.20× 광각** |

**영향**: K-warp 식이 카메라마다 다름. 잘못 적용 시 (u,v) 가 다른 위치로 매핑 → 마스크-grasp 불일치
**필수 조치**: 캡처 카메라별로 K 분기 처리 (`scripts/graspnet1b/build_graspnet_h5.py`)

### ❌ 오류 3: Metric 인용 (CLAUDE.md 와 paper 본문 직접 영향)

**현재 `make_paper_table1.py` 인용**:
- *"COV (Achlioptas ICML 2018)"*
- *"5cm/30° threshold (Sundermeyer ICRA 2021)"*

**검증된 사실**:
- **Achlioptas (arxiv 1707.02392) 의 Coverage 정의에 5cm/30° threshold 없음**. Achlioptas 정의:
  > *"For each point cloud in A we first find its closest neighbor in B. Coverage is measured as the fraction of the point clouds in B that were matched to point clouds in A. Closeness can be computed using either the CD or EMD point-set distance"*

  → nearest-neighbor matching 기반, fixed threshold 없음, **rotation component 없음** (point cloud 작업)

- **Sundermeyer Contact-GraspNet (arxiv 2103.14127) 에 5cm/30° threshold 없음**. 사용한 기준:
  - Simulator success: gripper-collision-free + object-still-in-gripper-after-shaking
  - Grasp selection: "contact confidence threshold of 0.23"

→ **두 인용 모두 잘못됨**. 5cm/30° threshold 는 우리가 자체적으로 채택한 기준 (또는 Mousavian 2019 / ACRONYM 2021 어딘가에 사용된 ad-hoc 기준).

**영향**: 논문 reviewer 가 paper PDF 들을 직접 확인하면 **misattribution 으로 reject** 가능
**필수 조치**:
1. paper 본문에서 "5cm/30° threshold" 인용 출처 재검증 — 가장 가까운 prior 는 **Mousavian ICCV 2019** 이거나 **자체 정의** 로 frame
2. `make_paper_table1.py` 에서 metric 인용 코멘트 정정
3. 메모리 `paper_metrics_official.md` 정정 (해당 메모리는 2026-04-27 작성됨)

---

## 1. ✅ 검증 완료 사실 (paper / strategy 의 신뢰도 99%)

### GraspNet-1B (graspnetAPI 라인 단위 검증)

| 항목 | 값 | 출처 (verbatim) |
|---|---|---|
| `offsets` shape | `(N_points, 300, 12, 4, 3)` | `graspnet.py:615-636` `num_views, num_angles, num_depths = 300, 12, 4` |
| `offsets` 마지막 dim | `(angle, depth, width)` | `graspnet.py:633-636` `angles=offsets[...,0]; depths=offsets[...,1]; widths=offsets[...,2]` |
| `meta['poses']` shape | `(3, 4, n)` (NOT 4×4 — `[0,0,0,1]` 직접 append) | `graspnet_dataset.py:162` `poses[:,:,i]` |
| 6 friction thresholds | `[0.2, 0.4, 0.6, 0.8, 1.0, 1.2]`, TOP_K=50 | `graspnet_eval.py:94, 117` |
| 88 객체, **id 18 제외** → 87 valid | — | `graspnet_dataset.py:251-256` `if i==18: continue` |
| Splits | train 0-99 / test_seen 100-129 / test_similar 130-159 / test_novel 160-189 | `graspnet.py:91-101` |
| Friction 규칙 | **Lower 가 더 좋음**, `score=-1` 은 invalid | `eval_utils.py:377-382` + `graspnet_eval.py:194` `(score_list<=fric) & (score_list>0)` |
| GraspGroup | 17 columns: `[score, width, height, depth, R(9), t(3), obj_id]` | `grasp.py:10-34` |
| 300 view 템플릿 | `generate_views(300)` Fibonacci sphere | `utils/utils.py:54-78` |
| factor_depth | per-scene metadata (소스에서 1000 hard-code 안됨) | `graspnet_dataset.py:111-116` |

### ACRONYM (paper § + GitHub source verbatim 검증)

| 항목 | 값 | 출처 |
|---|---|---|
| 데이터셋 규모 | 17.7M grasp / 8,872 객체 / 262 카테고리 | README + paper §II Table I |
| Per-object grasp 수 | 2,000 (antipodal sampling + rejection) | paper §III "Grasp Sampling" |
| 그리퍼 | Franka Panda parallel-jaw, max aperture 8 cm | paper §III |
| 좌표 | object/mesh frame | `acronym.py:381-401` `load_grasps` |
| Approach axis | **+Z** | `create_gripper_marker` 기하학 |
| **Gripper 원점** | **그리퍼 BASE/mount** (NOT 손가락 끝). 손가락 끝 = z + 0.1122 m | (오류 1 참조) |
| Success label | FleX physics: gripper close → shaking → 양 손가락 contact 유지 시 1 | paper §III "Grasp Labelling" |
| Friction 학습 | 1.0 uniform, density 150 kg/m³, **gravity OFF** during labelling | paper §III |
| Renderer | PyRender OffscreenRenderer + PerspectiveCamera(yfov=π/6, default 400×400) | `acronym_render_observations.py:77-104` |
| Per-target mask | depth diff (`is_visible` toggle, abs(obj_d - full_d) < 1e-6`) | `:179-193` |
| Multi-object scene | `Scene.random_arrangement` — `compute_stable_poses` + collision check | `acronym.py:325-351` |
| 라이선스 | code MIT, data CC BY-NC 4.0 | LICENSE + README |
| 메쉬 의존 | ShapeNetSem 별도 + Manifold + simplify 워터타이트 | README:110-114 |

### Baseline metric 인용 (paper PDF 인용)

| 베이스라인 | 보고된 metric | 인용 가능? |
|---|---|---|
| **TOGNet** ✅ | RealSense AP **51.84/46.62/23.74** (mean 40.63) / Kinect **49.60/40.03/19.58** (mean 36.40) — *p.10 Table 1* | "Native dataset (GraspNet-1B) AP" 보조 단락에서 인용 가능. Table 1 row 직접 매칭은 ❌ |
| **Contact-GraspNet** ✅ | 90.20% success in clutter (51 unseen objects, 9 scenes, Franka) — *p.4 Table I* | 동일 |
| **6-DoF GraspNet** ✅ | Per-cat real success: Box 83/Cyl 89/Bowl 100/Mug 86 → overall 88% — *p.7 Table 1* | 동일 |
| **GraspLDM** ✅ | Sim 1C 88% / 63C 78%; Real UR10e 80% / Franka 78.75% — *p.9, p.11 Table 2* | 동일 |
| **SE(3)-DiffusionFields** ✅ | Mug 90개 (~90K grasp), Isaac Gym success rate + EMD — *p.5 Sec IV-A + Fig 4* | 동일 (단 그래프만, 표 없음) |
| **LGD** ✅ | GA-6D 위 CR / EMD / CFR — *p.11* | 동일 |
| **GraspGen** ⚠️ | Real overall **81.3%** (M2T2 52.6%, AnyGrasp 63.7%) — *p.8 Table 1* 검증됨; **sim 48.2/31.3/14.4 NOT FOUND in main text** (likely appendix) | Real 인용 OK; sim 은 PDF appendix 직접 재확인 필요 |
| **EquiGraspFlow** ❓ | PDF fetch 안 됨 — CoRL 2024 OpenReview 또는 PMLR 직접 확인 필요 | 보류 |

### Metric 정의 (인용 정정 필수)

| 우리 metric | 출처 (검증) | 정정 |
|---|---|---|
| COV (5cm/30°) | ❌ Achlioptas 2018 의 정의 아님 | "We adopt a coverage definition based on Achlioptas 2018, modified with a 5cm/30° matching threshold to suit SE(3) grasps" |
| APD (5cm/30°) | ❌ Sundermeyer 2021 의 정의 아님 | "We define average pairwise distance over generated grasps; the 5cm/30° matching threshold is our choice for fair multi-modal evaluation" |
| 또는 | — | Mousavian ICCV 2019 / ACRONYM 2021 PDF 재검증 후 정확한 출처 명시 |

---

## 2. GraspNet-1B 변형 + 활용 전략

### 2-0. 코드 + 라이선스 + 즉시 실행 자원 (2026-04-28 직접 검증)

| 항목 | 값 | 출처 |
|---|---|---|
| 데이터셋 라이선스 | CC BY-NC-SA 4.0 (학술 OK, 상용 X) | graspnet.net |
| **graspnetAPI 라이선스** | **MIT** ✅ (데이터 로드 + 공식 AP eval) | [LICENSE](https://github.com/graspnet/graspnetAPI/blob/master/LICENSE) |
| **graspnet-baseline 라이선스** | ⚠️ **Non-Commercial Research Only** (verbatim: *"ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY"*) | [LICENSE](https://github.com/graspnet/graspnet-baseline/blob/main/LICENSE) — **이전 "Apache 2.0" 표기 오류 정정** |
| 활성도 | graspnet-baseline 마지막 push 2026-04-26 (28 issue), graspnetAPI 2026-04-23 (15 issue) | GitHub API |
| **사전학습 ckpt** | ✅ `checkpoint-rs.tar` (RealSense) + `checkpoint-kn.tar` (Kinect) — Drive/Baidu 미러 | repo README |
| 즉시 추론 | `command_demo.sh` — 임의 RGB-D + K → 6-DoF grasp 출력 | repo README |
| 평가 | `command_test.sh` + `graspnetAPI.GraspNetEval` 자동 AP | repo README |
| Tolerance label | 추가 생성 필요 (`generate_tolerance_label.py`) 또는 Drive 다운 | repo README |

**의의**:
1. ✅ **자체 데이터에 GraspNet 모델 적용 가능** — `command_demo.sh` 가 임의 RGB-D 받음 (재학습 불필요)
2. ✅ **GraspNet-1B 위에 다른 베이스라인 적용 가능** — 표준 RGB-D + 마스크 + 6-DoF annotation
3. ✅ **공식 AP 평가 즉시 가능** — graspnetAPI.GraspNetEval 1줄 호출
4. ⚠️ **graspnet-baseline 모델 weights 는 Non-Commercial** — 학술 paper footnote: *"GraspNet baseline used under non-commercial research license"*

### 2-0a. 공식 메트릭 — AP (Average Precision)

graspnetAPI 소스 라인 단위 검증 (`graspnet_eval.py:94, 117`):
```python
TOP_K = 50                                              # 상위 50 grasp 만
list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]   # 6 friction 임계값
```

**AP 산출 절차**:
1. 모델이 한 씬 → grasp 후보 N개 출력 (점수 sort, 상위 50개만)
2. 각 grasp 에 *force-closure analysis* 적용 → 통과하는 friction coefficient 계산
3. 6 임계값마다 precision 계산 → 평균 → **AP**
4. 3 splits 별 보고: **AP_seen / AP_similar / AP_novel**

**점수 규칙** (`eval_utils.py:377` + `graspnet_eval.py:194`):
- `friction <= threshold AND friction > 0` 만 유효
- friction = -1 은 collision/invalid (제외)
- **lower friction = better grasp**

**TOGNet 이 보고한 실제 AP** (검증됨, ECCV-W'24 *p.10 Table 1*):
| 카메라 | seen | similar | novel | mean |
|---|---|---|---|---|
| RealSense | 51.84 | 46.62 | 23.74 | **40.63** |
| Kinect | 49.60 | 40.03 | 19.58 | **36.40** |

**우리 vs GraspNet metric 차이**:
| metric | 우리 자체 (Table 1) | GraspNet 공식 (Table 2) |
|---|---|---|
| Pos MAE / Ang Err | ✅ | ❌ |
| COV (5cm/30°) | ✅ | ❌ |
| APD | ✅ | ❌ |
| **AP** | ❌ | ✅ |

→ **Table 2 (GraspNet-1B) 는 우리 metric + 공식 AP 둘 다 보고** 권장. 출력 grasp 을 17-col GraspGroup 으로 변환하면 graspnetAPI 가 AP 자동 계산.

### 2-1. 변환 파이프라인 (검증된 데이터 구조 기반)

새 디렉터리 `scripts/graspnet1b/` (총 ~780 LoC, ~20 dev hours):

| 스크립트 | 역할 | 핵심 코드 (검증된 포맷 기반) |
|---|---|---|
| `download_subset.py` | 1-2 scene sanity check | wget per-scene Drive |
| `extract_uv_per_object.py` | `label/*.png` per-pixel mask → centroid | `seg = Image.open(label_png); for inst_id, oid in enumerate(meta['cls_indexes'].flatten(), 1): mask=(seg==inst_id); u,v = xs.mean(), ys.mean()` |
| `flatten_grasp_tensor.py` | `(N_pt, 300, 12, 4, 3)` flatten | `views = generate_views(300); angles = offsets[...,0]; depths = offsets[...,1]; widths = offsets[...,2]; mask = (fric>0)&(fric<=0.4)&(~coll)` |
| `obj_to_cam_frame.py` | `meta['poses']` (3×4) → 4×4 promote → `T_cam = T_obj_to_cam @ T_grasp_obj` | `T = np.eye(4); T[:3,:] = poses[:,:,i]` |
| `build_graspnet_h5.py` | grasp_v2 schema 매칭 | per-row K 저장 (per-scene 변동), mode_id=-1 (unknown bucket) |
| `split_by_scene.py` | 공식 split | scene_id < 100 → train, 100-129 → test_seen, … |

**필수 K 처리** — RealSense vs Kinect 분기:
```python
# build_graspnet_h5.py 안
K_src = meta['intrinsic_matrix']
camera = 'kinect' if 'kinect' in scene_path else 'realsense'
# K_src 가 Kinect (631) 인지 D435 (927) 인지 detect
K_dst = np.array([[1109, 0, 640], [0, 1109, 360], [0, 0, 1]])  # 우리 K
sx, sy = 1109/K_src[0,0], 1109/K_src[1,1]
tx = 640 - sx*K_src[0,2]; ty = 360 - sy*K_src[1,2]
M = np.float32([[sx, 0, tx], [0, sy, ty]])
depth_canon = cv2.warpAffine(depth_mm, M, (1280, 720), flags=cv2.INTER_NEAREST)
u_c = sx*u + tx; v_c = sy*v + ty
# 3D grasp t/q: K-independent, 변경 없음
```

### 2-2. 모델 변경 — 1-line `mode_id` 확장

```python
# src/flow_model.py — Embedding 3 → 4
self.mode_emb = nn.Embedding(4, mode_dim)  # 0=lying, 1=standing, 2=cube, 3=unknown(GraspNet)
# scripts/train_flow.py collate
mode_id_safe = torch.where(mode_id == -1, 3, mode_id)
```

### 2-3. 활용 전략 — 3 가지 학습 변형

| 변형 | 학습 시간 (24GB) | 논문 narrative |
|---|---|---|
| (1) **GraspNet-only** | ~22h (12M rows × 250ep) | "Our framework scales to 88-cat real data" |
| (2) **Joint with dataset_id FiLM** | ~24h | "Single CFM model, two benchmarks" |
| (3) **Pretrain GraspNet → finetune grasp_v2** ⭐ | 18h + 3h | "Real-world pretraining transfers; +X% COV on our 5-class val" |

**권고**: (3) 헤드라인 + (1) ablation row

### 2-4. 활용 실험 — 4 paper experiment

| Experiment | 프로토콜 | 예상 결과 | reviewer impact |
|---|---|---|---|
| **D1. Real-world generalization (zero-shot)** | grasp_v2 학습 → GraspNet test_* eval (label/*.png 마스크 centroid (u,v)). 공식 AP via `GraspNetEval` | AP_seen 0.15-0.25 worst, 0.30+ best | "sim-only" 반박 |
| **D2. Class-agnostic (test_novel)** | D1 의 test_novel만. 비율 `AP_novel / AP_seen ≥ 0.85` 가 claim 입증 | top-tier claim | "5 classes 만 학습" 반박 |
| **D3. Official AP leaderboard** | (1) GraspNet-only 변형 학습 → AP_seen/similar/novel | Top-10 plausible (single-view depth) | absolute number 제공 |
| **D4. Mode-coverage transfer** | test_seen 위 N=64 sample, APD/COV. Direct vs Flow | 우리 sim 의 20× APD ratio 가 real 에도 유지 | "mode collapse 가 sim 만의 문제" 반박 |

### 2-5. Risk register (검증된 함정)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | `poses` 가 (3,4,n) 인데 (4,4,n) 으로 가정 → t_cam translation 빠짐 | **HIGH** | 단위테스트: `assert poses.shape[0]==3 and poses.shape[1]==4`; 명시적 promote |
| R2 | RealSense vs Kinect K 혼동 | **HIGH** | `camera` flag detect, K assertion |
| R3 | factor_depth 카메라마다 다를 가능성 | **HIGH** | per-frame 로드 필수, hard-code 금지 |
| R4 | 점유율 낮은 객체 (occlusion) 의 centroid noise | MED | `mask_px ≥ 200` 필터, distance transform peak 사용 |
| R5 | mode_id 결손 (88 객체에 우리 taxonomy 없음) | MED | "unknown" bucket (id=3) + aux mode head 는 마스킹 |

---

## 3. ACRONYM 변형 + 활용 전략

### 3-1. ⚠️ Gripper convention 처리 (오류 1 정정)

**필수 첫 단계**: 우리 grasp_v2 convention 과 ACRONYM convention 비교. 우리 `grasps_cam[xyz]` 위치가 TCP (손가락 끝) 인지 base 인지 명확히. ACRONYM 은 base.

```python
# scripts/acronym/normalize_origin.py
# ACRONYM grasp T_obj 의 origin 은 mount (0,0,0)
# 우리 grasps_cam 의 origin 이 TCP 라면:
T_tcp_offset = np.eye(4); T_tcp_offset[2,3] = 0.1122  # +Z 11.2cm
T_obj_tcp = T_obj_grasps @ T_tcp_offset  # 손가락 끝 위치로 변환
# 시각 검증: trimesh 으로 그리퍼 + 객체 + grasp 그려서 손가락이 객체 표면에 닿는지
```

### 3-2. 변환 파이프라인 (~900 LoC, ~30 dev hours + ~6h compute)

`scripts/acronym/`:

| 스크립트 | 핵심 코드 |
|---|---|
| `install_check.py` | grasps/*.h5 키 확인, ShapeNetSem 경로, Manifold 빌드 sanity |
| `patch_renderer.py` | `pyrender.IntrinsicsCamera(fx=fy=1109, cx=640, cy=360)` + 1280×720 + per-target instance mask 루프 |
| `render_scenes.py` | 5,000 scene × 2 view (top-down 50° + oblique 30°). 5.5h on 1 GPU |
| `extract_uv_and_grasps.py` | mask centroid → (u,v); FleX `success==1` 필터; `T_cam = inv(T_w_cam) @ T_w_obj @ T_obj_grasp @ T_tcp_offset` |
| `build_acronym_h5.py` | grasp_v2 schema 매칭. mode_id 는 PCA cluster (lying/standing/cube/other) |

### 3-3. 카테고리 subset 전략

**권고**: 매칭 5-class subset (~600 객체) primary + held-out 4,500 객체 generalization eval

| 우리 클래스 | ACRONYM 카테고리 (ShapeNetSem) | 약 객체 수 |
|---|---|---|
| bottle | `Bottle`, `WineBottle` | 250 |
| can | `Can`, `SodaCan`, `BeerCan` | 90 |
| marker | `Marker`, `Pen`, `Pencil` | 80 |
| spam | `FoodItem`, `CannedFood`, `TinCan` | 110 |
| cube | `Box`, `Cube`, `Dice` | 70 |

총 **~600 객체 → ~12,000 grasp** (FleX-success 22% × 2,000)

### 3-4. mode_id 처리 — PCA cluster

```python
# 객체별: PCA on grasp approach axes a_i = T_obj_grasps[:, :3, 2]
# eigenvalue ratio:
#   λ1/λ3 > 8       → standing  (1)  단일 dominant approach
#   λ1/λ3 < 1.5     → cube-like (2)  isotropic
#   else            → lying     (0)
# rare cat (<30 obj) → 3 (other)
```

### 3-5. 활용 실험 — 4 paper experiment

| Experiment | 프로토콜 | reviewer impact |
|---|---|---|
| **F1. Zero-shot generalization to 250 unseen ACRONYM categories** | grasp_v2 + matched ACRONYM subset 학습 → held-out 4,500 객체 eval. COV/APD/FleX-nearest-success. 기대: Direct 5%, Ours 60% | "(u,v) class-agnostic" 강력 입증 |
| **F2. Mode coverage on physics-validated GT** | FleX `success==1` 만 GT pool 로 사용. 우리 metric 이 heuristic policy 와 일치 | "COV 가 우리 정책 cherry-pick" 반박 |
| **F3. Baseline replication on (u,v)-conditioned ACRONYM** | SE(3)-DF / GraspLDM / EquiGraspFlow 의 unconditional 출력을 noise floor 로, (u,v) injected 출력과 우리 비교 | 다 modality 이득의 conditioning 격리 |
| **F4. Sim-to-sim transfer (PyRender → MATLAB Simulink)** | (1) ACRONYM-only 학습 → grasp_v2 val eval. depth-domain MAE 측정 | sim-to-sim gap 정량화 |

### 3-6. Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **Gripper origin 위치 오해** (오류 1) → 모든 grasp 11.2cm 빗나감 | **CRITICAL** | (3-1) 첫 단계 시각 검증 필수 |
| R2 | PyRender EGL headless 크래시 | HIGH | `PYOPENGL_PLATFORM=egl` env var |
| R3 | ShapeNetSem 메쉬 non-watertight (5-10%) | HIGH | Manifold 사전 처리, fail object drop |
| R4 | 17GB raw + HDF5 60GB | MED | gzip lvl4, depth uint16 quantize, shard |
| R5 | `random_arrangement` collision 으로 객체 drop 과다 | MED | retry 5x, log retention |
| R6 | License CC BY-NC | MED | 학술 paper 만, 코드 redistribute X |

---

## 4. 멀티 데이터셋 학습/평가 통합 전략

### 4-1. 권고 학습 Regime — 3-stage progressive

```
Stage 1 (22h): ACRONYM pretrain — 12k row, 80ep
            ↓
Stage 2 ( 8h): GraspNet-1B finetune — 12M row, 40ep, lr×0.1
            ↓
Stage 3 ( 5h): grasp_v2 finetune — 33k row, 60ep, lr×0.01, mode_id 활성화
```

**총 ~35h compute, single 24GB GPU**. 각 stage 가 paper Fig 4 ablation row 1개 산출.

### 4-2. 베이스라인 (u,v) adapter — 핵심 6시간 작업

`scripts/baseline_uv_adapter.py` (~70 LoC):
```python
def uv_to_segmented_pc(depth_mm, uv, K, yolo_mask=None, radius_px=80):
    if yolo_mask is None:
        u, v = uv
        ys, xs = np.mgrid[0:depth_mm.shape[0], 0:depth_mm.shape[1]]
        mask = (xs-u)**2 + (ys-v)**2 < radius_px**2
    else:
        mask = yolo_mask
    z = depth_mm[mask] / 1000.0
    x = (xs[mask] - K[0,2]) * z / K[0,0]
    y = (ys[mask] - K[1,2]) * z / K[1,1]
    pc = np.stack([x, y, z], axis=-1)
    pc = pc[(pc[:,2] > 0.2) & (pc[:,2] < 1.5)]
    return pc  # (N, 3)
```

각 베이스라인 wrapper (`eval_contact_graspnet_uv.py`, ...). 1h × 6 = **6h**.

### 4-3. 카메라 K 통일 — 권고 옵션 1 resample

데이터-time 에 **모든 depth + (u,v) 를 우리 K=(1109, 1109, 640, 360) 로 warp**. 모델 변경 0줄, ONNX 인터페이스 보존.

### 4-4. Mode taxonomy 통일 — 권고 Drop+Embedding

`mode_id ∈ {0,1,2,3}` 4-bucket. Stage 1/2 는 `mode_id=3` (unknown) 만. Stage 3 만 0/1/2 active. Aux mode head 는 `mode_id < 3` 일 때만 loss.

---

## 5. Paper narrative — 5-section experiment chapter

### 5.1 Setup & Metrics
- 데이터셋 (own val 1× / GraspNet-1B 3 splits / ACRONYM 600 + 4,500 obj)
- Metric: **COV (Achlioptas 2018, modified with 5cm/30° matching threshold)**, **APD**, **Pos MAE / Ang Err**, **AP (GraspNet 공식)**, **FleX success (ACRONYM)**
- Inference: N=32, CFG=2.5, single-step Euler

### 5.2 Main result — own val (Table 1, Fig 3)
- **Headline**: Standing COV 33.3% (Direct) → 99.2% (Ours), APD 20× ↑

### 5.3 Real-world transfer — GraspNet-1B (Table 2, Fig 5)
- **Headline**: AP_seen ≥ 40 with depth+(u,v) only, ≥ Contact-GraspNet PC oracle

### 5.4 Large-scale generalization — ACRONYM (Table 3, Fig 4)
- **Headline**: Pos MAE 2.0 cm (5 obj) → 1.1 cm (8.8k obj) log-linear, no architecture change

### 5.5 Ablation (Table 4)
- Direct MLP / w/o cross-attn / w/o multi-scale / DDPM 25 step / RF 1 step

### 표 layout

| Table | 행 | 열 | 1-line story |
|---|---|---|---|
| **T1** | Direct / 6 baseline / Ours v7 / Ours v8 | Pos MAE / Ang Err / COV / APD per-mode | "Ours 만 모든 모드 COV >95%" |
| **T2** | 베이스라인 + Ours | AP, AP_0.4, AP_0.8 × test_seen/similar/novel | "depth+(u,v) 가 PC oracle 매칭" |
| **T3** | training pool {5, 100, 1k, 8.8k} | Pos MAE / Ang Err / COV | "log-linear 확장" |
| **T4** | 5 ablation 변형 | val_flow / standing-COV / NFE | "각 컴포넌트 필수, RF 25× 빠름" |

### 그림

| F | 상태 | story |
|---|---|---|
| F1 mode-wise GT | done | "GT 가 본질적 multi-modal" |
| F2 architecture | done | "픽셀 feature via cross-attn + multi-scale crop" |
| F3 Direct vs Flow qual | done — extend with GraspNet real | "Direct collapse, Ours coverage" |
| F4 ACRONYM scaling curve | NEW | "log-linear, no plateau yet" |
| F5 real UR5e | optional | "30 Hz ONNX 실 로봇 동작" |

---

## 6. 일정 (검증된 시간 견적)

| Week | 작업 | 시간 |
|---|---|---|
| W1 | GraspNet-1B 변환 (스크립트 6개, K-warp, sanity 1 scene) | 30~40h |
| W2 | 베이스라인 6개 ckpt 우리 val 재실행 (uv_adapter) → Table 1 | 30h |
| W3 | ACRONYM 렌더링 + 학습 stage 1+2+3 → Table 3 + Fig 4 | 35h compute + 30h dev |
| W4 | GraspNet test_* 평가 → Table 2 + 논문 작성 | 25h |

**총 ~3.5주 = 17 working days**. De-scope 시 ACRONYM scaling curve (Fig 4) 가 가장 먼저 cut.

---

## 7. 즉시 행동 (검증된 사실 기반 우선순위)

### 🔴 긴급
- [ ] **Metric 인용 정정** ([scripts/make_paper_table1.py](scripts/make_paper_table1.py) + 메모리 `paper_metrics_official.md` + 논문 본문) — Achlioptas/Sundermeyer 5cm/30° misattribution 제거
- [ ] **EquiGraspFlow PDF 직접 fetch** (CoRL 2024 OpenReview) — fact-check 미완료

### 🟠 W1 시작 전
- [ ] GraspNet-1B 1 scene 다운 + `loadGrasp` API 출력 확인 → `scripts/graspnet1b/flatten_grasp_tensor.py` 의 layout 가정 검증
- [ ] ACRONYM 객체 1개로 grasp 시각화 → gripper origin 위치 우리 convention 과 비교 (오류 1 검증)

### 🟢 W1 작업
- [ ] Phase 1 시작 — GraspNet-1B `download_subset.py` + `extract_uv_per_object.py`

---

## 부록 A. 검증 출처 모음

### GraspNet 검증 출처 (라인 단위)
- `graspnetAPI/graspnet.py:91-101` (splits), `:142` (88 obj), `:615-636` (offsets/views/angles/depths/widths)
- `graspnetAPI/grasp.py:10-34` (GraspGroup 17-col)
- `graspnetAPI/graspnet_eval.py:94, 117, 194-197` (AP eval params)
- `graspnetAPI/utils/eval_utils.py:377-382` (-1 invalid)
- `graspnetAPI/utils/utils.py:34-37` (camera K), `:54-78` (`generate_views`)
- `graspnet-baseline/dataset/graspnet_dataset.py:108-256` (pose load, factor_depth, obj 18 skip)
- **graspnet-baseline LICENSE** (curl verbatim 2026-04-28): *"GRASPNET-BASELINE SOFTWARE LICENSE AGREEMENT — ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY"*
- **graspnetAPI LICENSE** (curl 2026-04-28): "MIT License, Copyright (c) 2025 GraspNet"
- **graspnet-baseline README** (curl 2026-04-28): `command_demo.sh` (RGB-D 임의 입력 지원), `command_test.sh` (eval), `checkpoint-rs.tar` + `checkpoint-kn.tar` 사전학습 ckpt 미러
- **GitHub API** (2026-04-28): graspnet-baseline 마지막 push 2026-04-26, 28 issues; graspnetAPI 마지막 push 2026-04-23, 15 issues (둘 다 active)

### ACRONYM 검증 출처
- `acronym_tools/acronym.py:325-351` (`random_arrangement`), `:381-401` (`load_grasps`), `:404-443` (`create_gripper_marker` — gripper geometry)
- `scripts/acronym_render_observations.py:77-104` (camera), `:179-193` (per-target mask)
- `README.md:1, 8, 109, 110-114` (사이즈 / 라이선스 / 다운 / mesh prep)
- Eppner 2021 (arxiv 2011.09584) §III "Grasp Sampling", "Grasp Labelling", page 2 Table I

### Baseline metric 검증 출처
- TOGNet (arxiv 2408.11138) p.10 Table 1
- Contact-GraspNet (arxiv 2103.14127) p.4 Table I, abstract
- 6-DoF GraspNet (arxiv 1905.10520) p.7 Table 1
- GraspLDM (arxiv 2312.11243) p.9, p.11 Table 2
- SE(3)-DiF (arxiv 2209.03855) p.5 Sec IV-A + Fig 4
- LGD (arxiv 2407.13842) p.11
- GraspGen (arxiv 2507.13097) p.6 Fig 4 + p.8 Table 1
- Achlioptas (arxiv 1707.02392) Section 2.3 (no threshold!)

### 검증 미완료 (TODO)
- EquiGraspFlow CoRL 2024 — 카테고리 / per-cat 수치
- GraspGen sim 수치 (48.2 / 31.3 / 14.4 / 40.4 / 23.5) — appendix table 직접 확인 필요
