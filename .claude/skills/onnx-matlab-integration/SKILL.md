---
name: onnx-matlab-integration
description: "Flow Matching **velocity MLP**를 ONNX로 export하고 MATLAB으로 로드해 Simulink 파이프라인 통합. 추론 시 MATLAB에서 **N개 노이즈 샘플링 → 1-step Euler → 충돌 필터 → 최적 grasp 선정 → CameraTform 곱해 base frame 7D pose** 생성. ONNX export·MATLAB 배포·Flow Matching 추론·충돌 필터 작업에 반드시 사용."
---

# ONNX → MATLAB 통합 (Flow Matching Velocity Net + Runtime Sampling/Filter)

## ⚡ 현재 아티팩트 (2026-04-21, v6_150ep)

**Canonical spec**: [`deploy/onnx/README.md`](../../../deploy/onnx/README.md) — 항상 이 문서를 먼저 읽기.

```
deploy/onnx/
├── encoder.onnx    (depth, uv) → cond(256)            scene당 1회
├── velocity.onnx   (g_t, cond, t, uv_norm) → v(8)     T × 2회 (CFG on/off)
│                   sinusoidal time embed 내장
├── meta.json       pos_mean/std, filter 상수, 권장 하이퍼
└── README.md       MATLAB 통합 스펙 (357줄)
```

- source ckpt: `runs/yolograsp_v2/v6_150ep/.../best.pt` (ep 118, val_flow=0.3640)
- onnxruntime round-trip max\|Δ\| = 6.2e-06
- 스펙 변경: 과거 "velocity 만 export" → **encoder 도 별도 ONNX**. conv 재계산 방지

### 필터 + best-pick 상수 (확정)

| 단계 | 상수 | 비고 |
|---|---|---|
| body collision | `wrist + stem 5샘플 < 5mm` | palm 제외 (narrow bottle 측면 파지 false positive 방지) |
| tip-sweep contact | `각 tip→TCP 6샘플 중 any < 15mm` | closure path 가 scene PC 훑어야 pass. 단순 tip 거리는 과탈락 |
| best-pick | `\|a_z\| > 0.7 우선 → uv 3D 최단 거리` | 생존 중 1개 선정. 색 #ffd700 |

### 라이브 추론 파이프

- `scripts/ros_capture_once.py` — python3 (isaacsim kit + ROS), DOMAIN_ID=13
- `scripts/infer_live.py` — conda base python, YOLO → Flow → filter → best → HTML
- `scripts/demo_inference.py` — 캐시 val scene (random6) 7 카테고리
- `deploy/viz/` — 결과 HTML 탭 네비 (캐시 + live + gt)

---

PyTorch 모델은 **velocity MLP** 만 export. MATLAB 런타임이 노이즈 N개 생성 → 1-step Euler → 8D g_1 → 7D pose_cam (quat) → 충돌 필터 → top-k → `CameraTform` → base pose → `move_to_grasp_pose.m`.

## 전체 흐름

```
[PyTorch]  velocity MLP:  (depth, uv, g_t, t) → v_θ ∈ ℝ^8
    │
    ↓ torch.onnx.export (opset 17)
[ONNX]  velocity.onnx
    │
    ↓ MATLAB importONNXNetwork
[MATLAB runtime]  predict_grasp.m:
  1. 32~64 개 g_0 ~ N(0, I) 노이즈 생성
  2. N 개 sample 에 대해 batch forward → v_θ → g_1 = g_0 + v (1-step)
  3. g_1 (8D) → pose_cam (7D quat)  [x, y, z, qw, qx, qy, qz]
  4. 충돌 필터 (depth swept-volume, workspace bound, gripper width)
  5. 생존자 중 implicit score 최고 → 1개 선택
  6. T_cam 4x4 구성 → T_base = CameraTform * T_cam
  7. pose_base 7D → move_to_grasp_pose(pose_base)
```

## Export 절차

```python
import torch

model.eval()
dummy_depth = torch.randn(1, 1, 720, 1280)
dummy_uv    = torch.randn(1, 2)
dummy_gt    = torch.randn(1, 8)                  # g_t (8D parameterization)
dummy_t     = torch.zeros(1)                      # flow time

torch.onnx.export(
    model,
    (dummy_depth, dummy_uv, dummy_gt, dummy_t),
    "runs/yolograsp_v2/onnx/velocity.onnx",
    input_names=["depth", "uv", "g_t", "t"],
    output_names=["v_theta"],                     # (B, 8) velocity
    dynamic_axes={
        "depth": {0: "batch"}, "uv": {0: "batch"},
        "g_t":   {0: "batch"}, "t":  {0: "batch"},
        "v_theta": {0: "batch"},
    },
    opset_version=17,
)
```

**참고:** ODE 루프·노이즈 생성·충돌 필터는 ONNX에 포함하지 않는다. MATLAB 런타임이 반복/확률 단계를 수행.

## PT vs ONNX 수치 검증 (필수)

```python
# verify.py
import torch, onnxruntime as ort, numpy as np
pt_model = load_pt_model()
pt_model.eval()
session = ort.InferenceSession("velocity.onnx")

for _ in range(10):
    d = torch.randn(1, 1, 720, 1280); u = torch.randn(1, 2)
    g = torch.randn(1, 8);            t = torch.rand(1)
    with torch.no_grad():
        v_pt = pt_model(d, u, g, t).numpy()
    v_onnx = session.run(None, {
        "depth": d.numpy(), "uv": u.numpy(),
        "g_t": g.numpy(), "t": t.numpy(),
    })[0]
    assert np.allclose(v_pt, v_onnx, atol=1e-4)
```

**통과 기준:** diff < 1e-4 (FP32) 또는 < 1e-3 (FP16 TensorRT).

## MATLAB predict_grasp.m

```matlab
function [grasp_pose_base] = predict_grasp(depth_img, uv_target, CameraTform, scene_pc)
% depth_img: (720, 1280) float32 meter
% uv_target: (1,2) pixel from YOLO
% CameraTform: 4x4 cam → base
% scene_pc: Nx3 scene point cloud for collision filter

net = importONNXNetwork("velocity.onnx", "InputDataFormats", ...);
N = 32;                                    % 샘플 개수

% 1) 노이즈 생성
g0 = randn(N, 8, 'single');
t0 = zeros(N, 1, 'single');

% 2) batch forward (MATLAB은 batch 확장 수동)
depth_batch = repmat(depth_img, [1, 1, 1, N]);   % N 복제
uv_batch    = repmat(uv_target, [N, 1]);
v = predict(net, depth_batch, uv_batch, g0, t0); % (N, 8)

% 3) 1-step Euler
g1 = g0 + v;                                 % (N, 8)

% 4) 8D → 7D pose_cam
pose_cam = zeros(N, 7, 'single');
for i = 1:N
    pos = g1(i, 1:3);
    app = g1(i, 4:6) / norm(g1(i, 4:6));
    yaw_app = atan2(g1(i, 7), g1(i, 8));
    R = build_rotation_tool(app, yaw_app);    % [Tool_X, Tool_Y, Tool_Z]
    q = rotm_to_quat(R);                       % [qw, qx, qy, qz]
    pose_cam(i, :) = [pos, q];
end

% 5) 충돌 필터
ok_mask = false(N, 1);
for i = 1:N
    ok_mask(i) = check_collision(pose_cam(i,:), scene_pc) ...
              && check_gripper_width(pose_cam(i,:)) ...
              && check_workspace(pose_cam(i,:));
end
survivors = find(ok_mask);
assert(~isempty(survivors), "모든 grasp 후보 충돌 reject");

% 6) 생존자 중 ICP fitness-like 스코어 높은 것 선택 (Flow 예측 자체는 무순위 → 다른 기준 적용 가능)
best_idx = survivors(1);                    % 간이: 첫 생존자
% 또는 reachability heuristic 등

% 7) Cam → Base 변환
T_cam = pose7_to_tform(pose_cam(best_idx, :));
T_base = CameraTform * T_cam;
grasp_pose_base = tform_to_pose7(T_base);

% 8) 기존 파이프라인 연결
% q_target = move_to_grasp_pose(grasp_pose_base);
end
```

## 충돌 체크 함수 (MATLAB)

```matlab
function ok = check_collision(pose_cam7, scene_pc)
% pose_cam7: [x y z qw qx qy qz]
% scene_pc: Nx3 (camera frame)
GRIPPER_LEN = 0.14; GRIPPER_HALF_W = 0.0425;
FINGER_LEN = 0.04;

R = quat_to_rotm(pose_cam7(4:7));
TCP = pose_cam7(1:3);
approach = R(:, 3);                          % Tool Z
binormal = R(:, 2);                          % Tool Y

% 핑거 swept volume (두 개 직육면체) 점수
finger_base = TCP - approach' * FINGER_LEN;
tip1 = TCP + binormal' * GRIPPER_HALF_W;
tip2 = TCP - binormal' * GRIPPER_HALF_W;

% scene_pc 중 swept volume 내부 점 개수 — 임계 이하면 pass
n_intrusion = count_points_in_volume(scene_pc, ...);
ok = n_intrusion < 5;                        % 임계 튜닝
end
```

## Simulink 통합

```
input: /camera/depth, /camera/rgb (우리는 depth만 사용), /tf
MATLAB Function Block:
  [q_target] = step_compute_target(depth, uv, CameraTform, scene_pc)
    → predict_grasp → move_to_grasp_pose
output: /joint_command_target
```

## 검증 체크리스트

- [ ] ONNX input/output 개수 = 4/1 (depth, uv, g_t, t → v_theta)
- [ ] PT vs ONNX diff < 1e-4
- [ ] MATLAB 측 랜덤 seed 고정으로 재현 가능 검증 케이스
- [ ] 충돌 필터가 최소 1개 grasp 통과 (`survivors` 비어있지 않음)
- [ ] CameraTform × T_cam = T_base 수치 일치 (Python·MATLAB 교차 검증)
- [ ] 최종 pose_base의 quaternion norm ≈ 1

## 흔한 실패

| 증상 | 원인 | 대응 |
|---|---|---|
| ONNX export 실패 | 커스텀 op 잔존 | `torch.linalg.solve`, `grid_sample` 등 수동 치환 |
| PT vs ONNX diff > 1e-3 | BN `model.eval()` 누락 | 재 export, BN running_stats 확인 |
| MATLAB Forward 불일치 | 입력 shape order (NCHW vs NHWC) | importONNXNetwork `InputDataFormats` 지정 |
| 모든 샘플 충돌 reject | 학습 데이터에 충돌 grasp 혼입 | grasp-synthesizer collision 필터 강화 |
| Diversity 낮음 (N=32 중 몇 개만 유효) | condition encode가 과도하게 결정적 | cond dropout, g_0 분산 ↑ |
