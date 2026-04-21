# YOLO_Grasp Flow Matching — MATLAB 통합 스펙

**대상 독자**: RoboCup_ARM 의 `matlab-pipeline` 에이전트.
**목적**: 카메라 프레임 Depth + YOLO (u,v) → 6-DoF grasp pose N개를 `ICPGraspPlanner`의 ICP 자리에 드롭인 교체.

---

## 0. TL;DR

```
Input  : depth (720,1280) float32 meter,  uv (2,) pixel
Output : grasps_cam (N,7)  = [x,y,z, qw,qx,qy,qz]  (카메라 프레임)
         + 신뢰도 점수 (충돌필터 통과 여부)
```

체크포인트: `v6_150ep/best.pt`, ep=118/150, val_flow=**0.3640**
내보낸 ONNX: [encoder.onnx](encoder.onnx) (2.38 MB), [velocity.onnx](velocity.onnx) (57.22 MB), [meta.json](meta.json)
수치 검증: onnxruntime round-trip max|Δ| = 6.2e-06 (float32 허용 범위).

---

## 1. 파일 구성

| 파일 | 크기 | 역할 | 호출 빈도 (scene당) |
|---|---|---|---|
| `encoder.onnx` | 2.4 MB | depth + uv → cond (256-d) | **1회** |
| `velocity.onnx` | 57 MB | g_t, cond, t, uv_norm → v (8-d) | **T_steps × 2회** (CFG on/off) |
| `meta.json` | <2 KB | pos 정규화 통계, 그리퍼 상수, 추천 하이퍼 | (로드시 1회) |

분리 이유: encoder는 무거운 conv 스택(720×1280 depth 처리) → 한 scene당 1회. velocity는 가벼운 MLP → Euler 32스텝 × CFG 2회 = **64회** 호출. 합쳐서 export하면 conv를 64번 반복 → 비효율.

---

## 2. Inference 흐름 (권장 하이퍼파라미터)

```
N_samples    = 32          # noise 개수 = 최종 candidate 개수
T_euler      = 32          # Euler 스텝 수
noise_temp   = 0.8         # g_0 ~ N(0, temp²·I)  (학습은 temp=1.0, 추론은 0.8 권장)
cfg_w        = 2.0         # Classifier-Free Guidance 계수
```

**Pseudo-code**:

```
cond_on  = encoder(depth, uv)                    # (1, 256)      — 1회
cond_on  = repmat(cond_on, N, 1)                 # (N, 256)
cond_off = zeros(N, 256)                         # CFG의 "null" condition
uv_norm  = [u/W, v/H] repeated N                 # (N, 2)

g_t = randn(N, 8) * noise_temp                   # 초기 노이즈

for k = 0 : T_euler-1
    t     = (k / T_euler) * ones(N, 1)           # (N,) float32
    v_on  = velocity(g_t, cond_on,  t, uv_norm)  # (N, 8)
    v_off = velocity(g_t, cond_off, t, uv_norm)  # (N, 8)
    v     = v_off + cfg_w * (v_on - v_off)
    g_t   = g_t + v / T_euler                    # Euler step
end

% (pos만) 역정규화
g_t(:, 1:3) = g_t(:, 1:3) .* pos_std + pos_mean  # meta.json에서 로드

% 8D → SE(3) 7D
for i = 1:N
    [pos, R]    = decode_8d(g_t(i, :))           # §4 참조
    q_wxyz(i,:) = rotm2quat(R)                   # MATLAB: [w,x,y,z]
    grasps(i,:) = [pos, q_wxyz(i,:)]
end

% §5 필터 → best 선택
```

---

## 3. I/O 스키마 (ONNX 레벨)

### 3-1. `encoder.onnx`

| 이름 | shape | dtype | 단위/의미 |
|---|---|---|---|
| **in** `depth` | `(B, 1, 720, 1280)` | float32 | **meter** (0.3m 미만/1.5m 초과는 모델 내부에서 clip+scale. 0-값 = invalid) |
| **in** `uv` | `(B, 2)` | float32 | 픽셀 좌표 (u, v). origin = 이미지 좌상단, u→오른쪽, v→아래. YOLO centroid 그대로. |
| **out** `cond` | `(B, 256)` | float32 | scene/object 조건 벡터. velocity에 그대로 전달. |

B는 동적. 일반적으로 scene당 B=1.

### 3-2. `velocity.onnx`

| 이름 | shape | dtype | 의미 |
|---|---|---|---|
| **in** `g_t` | `(N, 8)` | float32 | 현재 flow 상태. 초기값 `randn(N,8)*temp`, Euler로 업데이트. |
| **in** `cond` | `(N, 256)` | float32 | encoder 출력을 N개로 broadcast한 값. CFG의 unconditional에서는 **zeros(N,256)**. |
| **in** `t` | `(N,)` | float32 | Euler 시간 `k/T_euler ∈ [0,1]`. 모든 샘플 같은 값 전달 OK. |
| **in** `uv_norm` | `(N, 2)` | float32 | `[u/1280, v/720]` 을 N개 broadcast. |
| **out** `v` | `(N, 8)` | float32 | velocity 벡터. `g_t += v/T_euler` 로 전진. |

sinusoidal time embedding은 **velocity.onnx 내부에 포함**되어 있음 (MATLAB이 별도 계산할 필요 없음).

### 3-3. `meta.json` 주요 필드

```json
{
  "normalize_pos": true,
  "pos_mean": [0.0244, -0.0056, 0.5636],      // camera frame, meter
  "pos_std":  [0.1108,  0.0719, 0.0995],
  "image": {"H": 720, "W": 1280,
            "camera_K": {"fx":1109, "fy":1109, "cx":640, "cy":360}},
  "depth": {"clip_min": 0.3, "clip_max": 1.5, "scale_div": 1.5},
  "recommended_inference": {
      "N_samples": 32, "T_euler_steps": 32,
      "noise_temp": 0.8, "cfg_guidance": 2.0},
  "gripper": {"GRIPPER_HALF": 0.0425, "FINGER_LEN": 0.040,
              "PALM_BACK": 0.025, "TCP_to_flange_offset_z_tool": 0.14}
}
```

---

## 4. 8D → SE(3) 디코딩

모델 출력 `g = [x, y, z, a_x, a_y, a_z, sin_yaw, cos_yaw]`.

```matlab
function [pos, R, yaw] = decode_grasp_8d(g)
    pos = g(1:3);                                 % TCP, camera frame, meter
    a   = g(4:6) / norm(g(4:6));                  % Tool Z (approach, unit)
    yaw = atan2(g(7), g(8));                      % 그리퍼 roll around approach

    % Gram-Schmidt로 b axis 복원 (학습 시 규약과 동일)
    if abs(a(1)) < 0.95, ref = [1;0;0]; else, ref = [0;1;0]; end
    b0 = ref - (ref.' * a) * a;   b0 = b0 / norm(b0);
    n0 = cross(a, b0);
    b  = b0 * cos(yaw) + n0 * sin(yaw);           % Tool Y (opening dir)
    x  = cross(b, a);                             % Tool X
    R  = [x, b, a];                                % 3x3, camera frame
end
```

- **Tool Z (R(:,3))** = approach direction. 그리퍼가 물체로 전진하는 방향.
- **Tool Y (R(:,2))** = finger 개폐 방향. 두 핑거 tip은 `pos ± b·0.0425` 에 위치.
- **Tool X (R(:,1))** = Y × Z (동차 왼손 × 오른손 구성은 `_build_R_tool` 와 동일).

### 4-1. 카메라 프레임 규약

- x → 이미지 오른쪽
- y → 이미지 아래
- z → 이미지 속 (깊이 방향, depth 값이 곧 z)

K = `[1109 0 640; 0 1109 360; 0 0 1]`. Back-project: `X = (u-cx)·z/fx`, `Y = (v-cy)·z/fy`.

### 4-2. Base frame 변환

**ICPGraspPlanner.m 기존 규약 그대로**:

```matlab
H_cam = eye(4);
H_cam(1:3, 1:3) = R;
H_cam(1:3, 4)   = pos;
H_base = CameraTform * H_cam;        % CameraTform: 4x4 camera→base
pos_base   = H_base(1:3, 4);
R_base     = H_base(1:3, 1:3);
q_wxyz     = rotm2quat(R_base);      % MATLAB: [w x y z]
```

**주의**: `rotm2quat` 는 wxyz 순서. ROS2 `geometry_msgs/Pose` 는 **xyzw** 순서. 발행 시 매핑 필수.

### 4-3. TCP → Flange offset (중요, 기존 코드와 차이점)

`move_to_grasp_pose.m` 은 `pos_flange = pos_tcp + [0, 0, 0.14]` (base frame world-Z) 로 계산하는데, 이는 **top-down grasp만 올바름**. 측면 파지(standing bottle side grasp)는 Tool Z ≠ world -Z 이므로 Tool-frame 기준 offset이 필요:

```matlab
pos_flange = pos_tcp - R_base(:, 3) * 0.14;   % Tool Z 반대 방향으로 0.14m
```

`R(:,3)` 은 Tool Z = approach direction. 플랜지는 TCP에서 approach 반대쪽으로 그리퍼 길이만큼 뒤에 있다. 모든 자세(top-down + side)에서 일관되게 동작.

---

## 5. 충돌/Contact 필터 (N=32 중 best 선택 기준)

Python 참조 구현: `scripts/demo_inference.py::grasp_filter_single`.

### 5-1. Scene point cloud 준비

```matlab
% depth → camera-frame PC (stride=3으로 서브샘플, ~100k points)
[u_grid, v_grid] = meshgrid(1:3:1280, 1:3:720);
z   = depth(sub2ind([720 1280], v_grid, u_grid));
val = z > 0.1 & z < 2.0;
X = (u_grid(val) - 640) .* z(val) / 1109;
Y = (v_grid(val) - 360) .* z(val) / 1109;
Z = z(val);
pc = [X(:), Y(:), Z(:)];
Mdl = KDTreeSearcher(pc);
```

### 5-2. 각 grasp에 대해 두 조건 검사

**(a) Body collision** — wrist + approach-stem 5개 샘플이 scene PC에서 `BODY_MARGIN = 0.005 m` 이내면 **reject**.

- 샘플 포인트:
  - `wrist   = pos - R(:,3) * (FINGER_LEN + PALM_BACK) = pos - R(:,3) * 0.065`
  - `app_end = wrist - R(:,3) * 0.050`
  - `stem_samples = linspace(wrist, app_end, 4)` (총 5점)
- **palm은 제외**: 좁은 병 측면 파지 시 palm이 실린더 표면 ~5mm 바깥에 오므로 false positive 유발.

**(b) Tip-sweep contact** — 각 fingertip에서 TCP까지의 **닫힘 경로**를 6점 샘플, scene PC에서 `TIP_RADIUS = 0.015 m` 이내에 하나라도 있어야 **keep** (없으면 "허공 파지" reject).

```
tip1 = pos + R(:,2) * 0.0425
tip2 = pos - R(:,2) * 0.0425
for tip in (tip1, tip2):
    samples = linspace(tip, pos, 6)     % 닫힘 경로
    d_min   = min over samples of  min distance to pc
if min(d_min_tip1, d_min_tip2) > TIP_RADIUS:  reject
```

**왜 단순 tip→PC 거리가 아닌 sweep인가**: Robotiq 2F-85 (half=4.25cm) 가 반경 1.5cm 병을 감쌀 때, 열린 상태의 tip은 표면에서 2.75cm 떨어져 있다. 닫힘 경로를 따라가야 표면과 교차. 단순 거리는 narrow bottle side grasp를 전부 오탈락시킴 (실측: standing_bottle case 32/32 reject).

### 5-3. 최종 선택

필터 통과한 grasp 중:
1. (권장) **object mask와의 접촉점 수**가 많은 순으로 정렬 → IK가 풀리는 첫 후보 선택
2. 간단 버전: 필터 통과 중 **무작위/첫째** 선택
3. 옵션: Reachability (기존 UR5e IK) 실패 시 다음 후보로 폴백

---

## 6. ICPGraspPlanner.m 교체 가이드

| 단계 | 기존 동작 | 교체 |
|---|---|---|
| Step 1 YOLO seg | `importNetworkFromONNX(best.onnx)` → mask + class | **유지**. centroid (u,v) 추출. |
| Step 2 3D 역투영 | pcfromdepth | **유지**, 그러나 Flow Matching 경로에서는 필터용 PC로만 사용 |
| Step 3 PLY 로드 | `loadModel()` | **제거** (Flow Matching은 PLY 불필요) |
| Step 4 ICP | `estimatePoseTrimmed()` → R, t, fitness | **제거 → `predictGraspFlow()` 호출**. 반환: grasps(N,7), 필터결과 |
| Step 5 normalize / grasp point | `normalizeGraspPose`, `calculateGraspPoint` | **제거**. Flow Matching 출력이 이미 TCP grasp pose. |
| Step 6 CameraTform 적용 | `H_base = CameraTform * H_cam` | **유지**. grasps(:,1:3) 와 R_base 각각에 적용. |
| Output | `[GraspPose(7), routeCode, fitness]` | **호환 유지**: GraspPose = best candidate, fitness → filter pass 개수 / N 또는 sweep 거리 inv. |

Cube 클래스 (`cube_blue/green/p/red`)는 Flow Matching 내부에서 `cube` 하나로 학습됨 → YOLO class name을 Flow Matching에 전달할 필요 **없음** (조건 입력 아님). 단 bin 매핑은 기존처럼 class name 기반 처리.

---

## 7. MATLAB 구현 템플릿

```matlab
function [best, grasps, flags] = predictGraspFlow(depth_m, uv_px, meta, net_enc, net_vel)
% depth_m (720,1280) float32 meter
% uv_px  [u; v] pixel
% meta   struct from meta.json
% net_enc/net_vel: dlnetwork (from importNetworkFromONNX)

    H = 720; W = 1280;
    N = meta.recommended_inference.N_samples;
    T = meta.recommended_inference.T_euler_steps;
    temp = meta.recommended_inference.noise_temp;
    w    = meta.recommended_inference.cfg_guidance;
    pos_mu    = meta.pos_mean(:).';
    pos_sigma = meta.pos_std(:).';

    % --- encoder ---
    depth_in = dlarray(reshape(single(depth_m), 1,1,H,W), 'SSCB');
    uv_in    = dlarray(single(uv_px(:).'), 'CB');
    cond1    = extractdata(predict(net_enc, depth_in, uv_in));   % (256,1)
    cond_on  = repmat(cond1(:).', N, 1);                          % (N,256)
    cond_off = zeros(N, 256, 'single');
    uv_norm  = repmat(single([uv_px(1)/W, uv_px(2)/H]), N, 1);

    % --- Euler + CFG ---
    g_t = randn(N, 8, 'single') * single(temp);
    for k = 0 : T-1
        t_vec = single(k/T) * ones(N, 1, 'single');
        v_on  = runVelocity(net_vel, g_t, cond_on,  t_vec, uv_norm);
        v_off = runVelocity(net_vel, g_t, cond_off, t_vec, uv_norm);
        v     = v_off + single(w) * (v_on - v_off);
        g_t   = g_t + v / single(T);
    end
    g_t(:, 1:3) = g_t(:, 1:3) .* pos_sigma + pos_mu;   % denormalize

    % --- 8D → 7D (camera frame) ---
    grasps = zeros(N, 7, 'single');
    R_cam  = zeros(3, 3, N, 'single');
    pos_cam= zeros(N, 3, 'single');
    for i = 1:N
        [p, R] = decode_grasp_8d(g_t(i, :));
        grasps(i, :) = [p, rotm2quat(R)];   % rotm2quat returns wxyz
        R_cam(:,:,i) = R;  pos_cam(i,:) = p;
    end

    % --- 필터 ---
    Mdl   = sceneKDTree(depth_m);
    flags = strings(N, 1);
    scores = -inf(N, 1);
    for i = 1:N
        [flag, dsweep] = filterGrasp(pos_cam(i,:), R_cam(:,:,i), Mdl, meta);
        flags(i) = flag;
        if flag == "kept", scores(i) = -dsweep; end   % 작을수록 좋음
    end
    [~, best_i] = max(scores);
    best = grasps(best_i, :);
end

function v = runVelocity(net_vel, g_t, cond, t, uv_norm)
    v = extractdata(predict(net_vel, ...
        dlarray(single(g_t)    ,'CB'), ...
        dlarray(single(cond)   ,'CB'), ...
        dlarray(single(t)      ,'CB'), ...
        dlarray(single(uv_norm),'CB')));
    v = v.';   % (N,8)
end
```

`decode_grasp_8d`, `sceneKDTree`, `filterGrasp` 는 §4~§5 규약대로 구현.

---

## 8. 검증 체크리스트

- [ ] **Round-trip**: onnxruntime / MATLAB 로드 후 입력을 고정하고 Python 출력과 max|Δ| < 1e-4 확인 (Python 스크립트: `scripts/export_onnx.py` 이미 onnxruntime 확인 완료).
- [ ] **단위 정합**: depth 입력이 meter인지 재확인 (Simulink에서 uint16 mm → /1000.0 변환 경로 체크).
- [ ] **uv 좌표계**: MATLAB 1-indexed 이미지 좌표를 ONNX 0-indexed 픽셀로 변환 `u_onnx = u_matlab - 1`.
- [ ] **CFG 2배 호출 성능**: N=32, T=32, CFG on/off = 2048회 velocity 호출 → MATLAB에서 `predict` mini-batch 묶음 권장.
- [ ] **측면 파지**: standing bottle case에서 filter 통과 `side` grasp 개수가 0이 아닌지 확인 (Python 참조값: 23/32, 23/32).
- [ ] **Flange offset**: §4-3의 Tool-frame offset을 `move_to_grasp_pose.m` 에 적용. 기존 `+[0,0,0.14]` 은 측면 파지 시 오류.
- [ ] **fitness 대체**: 기존 ROS2 토픽 `/icp/fitness` 의미를 "filter pass ratio" 혹은 "mean tip-sweep dist" 로 재정의. rulebook/task-sequencer 에이전트에 알림.
- [ ] **Cube class 통합**: YOLO 에서 `cube_blue/green/p/red` 전부 `cube` 로 합쳐 전달. bin 매핑만 class name 원본 유지.

---

## 9. 알려진 제약

- **학습 씬과 다른 카메라 pose**: 학습 데이터는 실제 대회 fixed top-down camera 기준. 카메라 위치/각도가 바뀌면 성능 저하. (룰북상 카메라 수정 금지이므로 영향 없음)
- **조명 일반화**: depth-only 입력이라 조명 robust. YOLO seg 쪽이 취약.
- **물체 크기 가정**: 훈련 데이터가 RoboCup 8종 (bottle/can/marker/spam/cube 4색). 생소한 물체는 unlabeled.
- **ONNX opset**: 17 사용. MATLAB R2023a+ 호환. R2025b는 OK.

---

## 10. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-04-20 | 초기 문서 (v6_150ep/best.pt, ep=118, val_flow=0.3640 기반) |

---

## 11. 관련 파일

- 체크포인트: [../../runs/yolograsp_v2/v6_150ep/adaln_zero_lr0.001_nb8_h768/checkpoints/best.pt](../../runs/yolograsp_v2/v6_150ep/adaln_zero_lr0.001_nb8_h768/checkpoints/best.pt)
- Export 스크립트: [../../scripts/export_onnx.py](../../scripts/export_onnx.py)
- Python 추론 + 필터 참조 구현: [../../scripts/demo_inference.py](../../scripts/demo_inference.py)
- 모델 정의: [../../src/flow_model.py](../../src/flow_model.py)
- 데이터 규약: [../../src/flow_dataset.py](../../src/flow_dataset.py)
- YOLO_Grasp 전체 CLAUDE: [../../CLAUDE.md](../../CLAUDE.md)
- RoboCup_ARM 통합 대상: `/home/robotics/Competition/RoboCup_ARM/scripts/ICPGraspPlanner.m` (Step 4 교체), `/home/robotics/Competition/RoboCup_ARM/scripts/move_to_grasp_pose.m` (§4-3 offset 수정)
