# deploy/viz — MATLAB 참조 구현

**대상**: RoboCup_ARM `matlab-pipeline` 에이전트.
**목적**: Python에서 이미 동작하는 full inference + 필터 파이프라인을 **그대로 MATLAB로 포팅**. HTML은 "올바르게 동작했을 때 이런 그림이 나와야 한다"의 ground truth 이미지.

---

## 파일

| 파일 | 역할 |
|---|---|
| `demo_inference.py` | **참조 Python 구현** — MATLAB이 함수 단위로 1:1 매칭해서 구현할 것 |
| `index.html` | 7 카테고리 탭 네비게이터 |
| `infer_{cat}.html` | 카테고리별 3D 시각화 (7개) — 각 카테고리 2 case × N=32 grasp |

카테고리: `standing_bottle, standing_can, lying_bottle, lying_can, lying_marker, lying_spam, cube`

---

## 포팅 매핑 (Python → MATLAB)

| Python 함수 / 블록 | MATLAB 대응 | 비고 |
|---|---|---|
| `model.encode(depth, uv)` (torch) | `predict(net_enc, depth_in, uv_in)` | `../onnx/encoder.onnx` 사용. scene당 **1회** 호출 |
| `flow_sample()` Euler + CFG 루프 | `predict(net_vel, ...)` × `T_euler × 2` | `../onnx/velocity.onnx`. cond_on + cond_off 두 번 |
| `g8_to_components(g)` | `decode_grasp_8d(g)` | 8D → (pos, approach_unit, yaw) |
| `gripper_u(pos, approach, yaw)` | `grip = grip_points(pos, R)` | wrist/tip/palm 등 기하 포인트 재구성 |
| `back_project(depth, stride)` | `pc = backProject(depth, K, stride)` | scene point cloud |
| `build_scene_kdtree` | `Mdl = KDTreeSearcher(pc)` | MATLAB 내장 |
| `grasp_filter_single(w, tree)` | `[flag, dsweep] = filterGrasp(w, Mdl)` | body<5mm + tip→TCP sweep<15mm |
| `filter_grasps(grasps, depth)` | loop over N + aggregate flags | `kept/body_collision/no_contact` 분류 |
| `_side_topdown_split(...)` | `[nt, ns] = sideTopSplit(...)` | standing 카테고리 검증용 |
| `add_scene(...)` plotly | **구현 생략 가능** | MATLAB 시각화는 선택. 숫자 검증만 해도 됨 |

---

## 필수 재현 체크

아래 숫자가 HTML에서 확인되므로 MATLAB 결과와 대조:

| Case | 기대 결과 (v6 best.pt, ep=118) |
|---|---|
| standing_bottle × 2 case | kept=32/32, 32/32 (top9/side23, top9/side23) |
| standing_can × 2 case | kept=31/32 (top5/side26), 30/32 (top4/side26) |
| lying_bottle × 2 | kept=25/32, 31/32 |
| lying_can × 2 | kept=29/32, 31/32 |
| lying_marker × 2 | kept=31/32, 26/32 |
| lying_spam × 2 | kept=29/32, 30/32 |
| cube × 2 | kept=32/32, 32/32 |

**kept 개수가 ±3 이내로 일치하면 포팅 성공**. 정확히 일치할 수는 없음 — noise 초기값이 `randn`이므로 MATLAB RNG seed를 맞춰도 torch와 다름. 분포 통계가 맞는지 확인.

---

## 추론 하이퍼파라미터 (고정)

`../onnx/meta.json` 과 동일:

```
N_samples  = 32
T_euler    = 32
noise_temp = 0.8         # g_t ~ N(0, 0.64 I)
cfg_w      = 2.0
```

필터:

```
BODY_MARGIN        = 0.005 m   # wrist+stem 5샘플, PC에서 5mm 이내면 reject
TIP_SWEEP_RADIUS   = 0.015 m   # 각 tip→TCP 6샘플, 어느 하나도 15mm 이내 없으면 reject
STEM_SAMPLES       = 4
SWEEP_SAMPLES      = 6
PC stride          = 3         # back-project에서 3픽셀 건너뛰기
```

---

## 실행 방법 (참조 Python, 재생성용)

```bash
cd /home/robotics/Competition/YOLO_Grasp
YGRASP_CKPT=runs/yolograsp_v2/v6_150ep/adaln_zero_lr0.001_nb8_h768/checkpoints/best.pt \
  /home/robotics/anaconda3/bin/python deploy/viz/demo_inference.py
xdg-open deploy/viz/index.html
```

v6 체크포인트를 쓰면 위 기대 결과가 재현됩니다. 다른 체크포인트면 숫자가 달라짐.

---

## 좌표/단위 규약 (MATLAB 구현 시 주의)

- **카메라 프레임**: x 오른쪽 / y 아래 / z 앞 (OpenCV/ROS 규약). `K = [1109 0 640; 0 1109 360; 0 0 1]`
- **depth**: Simulink는 uint16 mm → `/1000.0` 해서 float meter로 ONNX에 전달
- **uv**: 1-indexed MATLAB 좌표면 `u_onnx = u_matlab - 1`
- **quaternion**: 모델 출력은 8D (approach + sincos yaw). 7D `[x y z qw qx qy qz]` 로 변환 시 `rotm2quat(R)` (**wxyz** 순서)
- **Base frame 변환**은 필터 이후에 수행 (필터는 camera frame PC로 동작). `H_base = CameraTform * H_cam`

---

## 관련 문서

- **ONNX I/O 스펙 + 통합 가이드**: [../onnx/README.md](../onnx/README.md) ← **먼저 읽을 것**
- **Meta 상수**: [../onnx/meta.json](../onnx/meta.json)
- **모델 정의 (디코딩 규약 원본)**: [../../src/flow_model.py](../../src/flow_model.py), [../../src/flow_dataset.py](../../src/flow_dataset.py) (`_build_R_tool`)
