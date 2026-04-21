---
name: top-down-grasp-synthesis
description: "ICP 카메라 프레임 pose + PLY + mask + depth로부터 mode별 grasp 후보를 합성한다. Standing bottle/can은 top-down 8 + 측면 horizontal 8 = **16개 (90° 직각)**, lying은 긴축 N 샘플 × 180° 대칭, cube는 면 edge 정렬 2개. 출력은 **6-DoF SE(3)** `pose_cam(7,)` = [x,y,z, qw,qx,qy,qz]. 카메라 프레임 전용. Grasp 합성·multi-mode·SE(3) 작업에 반드시 사용."
---

# Grasp Synthesis — 6-DoF Multi-Mode (카메라 프레임)

물체 pose에서 UR5e + Robotiq 2F-85 규약과 호환되는 grasp 다수를 mode별로 합성한다. **카메라 프레임에서만 계산·저장**하고 base 변환은 MATLAB 런타임 담당.

## 왜 6-DoF인가 (이전 4-DoF 가정 폐기)

Standing bottle/can에 대해 "측면 horizontal 접근" 정책이 추가됨 → approach 방향이 cam +Z(수직)과 cam XY(수평) 둘 다 필요. 이로 인해 full SE(3) 표현이 불가피. 결국 저장은 **quaternion 7D**.

그 외(lying, cube)는 여전히 approach = cam +Z 지만, 파이프라인·모델 일관성을 위해 동일한 7D 스키마 사용.

## 환경 상수

- 해상도 1280×720, Camera K: fx=fy=1109, cx=640, cy=360
- 좌표 프레임: 카메라 프레임 전용
- 그리퍼: Robotiq 2F-85 (TCP→flange 0.14m, 최대개폐폭 0.085m, finger length ≈ 0.04m)
- Python: conda base + open3d + numpy

## Gripper 좌표계 (MATLAB 호환)

Tool frame 축 정의:
- **Tool Z**: approach 방향 (손가락이 물체로 향하는 방향)
- **Tool Y**: gripper open 방향 (binormal, 양 핑거 사이)
- **Tool X**: Tool Y × Tool Z (오른손 좌표계)

Rotation matrix `R_cam_in_tool = [Tool_X, Tool_Y, Tool_Z]` (열 벡터). 이 R 을 quaternion `[qw,qx,qy,qz]`로 변환해 저장.

## Mode 분기

ICP `R_cam`에서 PLY 긴축(`long_axis_idx` 메타)이 카메라에서 가리키는 방향으로 판정:

```python
e_long = np.zeros(3); e_long[long_axis_idx] = 1.0
long_cam = R_cam @ e_long
cos_z = abs(long_cam[2])

if cls_name.startswith("cube"):         mode = "cube"
elif cos_z > 0.7:                        mode = "standing"   # 긴축이 cam Z에 평행
else:                                    mode = "lying"       # 긴축이 cam XY 평면 근접
```

## 모드별 생성 규칙 (최종 정책)

### Standing (bottle / can / marker / spam, 세워짐) — **16개**

긴축 방향 중 **카메라 쪽(smaller z_cam) 끝이 cap/top**.

```python
sign_toward_cam = -1 if long_cam[2] > 0 else +1
cap_off = sign_toward_cam * (L * 0.5) * e_long
p_cap_top = t_icp + R_icp @ cap_off          # 뚜껑 표면 상단
p_cap_top[2] += 0.003                        # 3mm margin
p_cap_side = t_icp + R_icp @ (sign_toward_cam * (L/2 - 0.015) * e_long)  # 뚜껑 밴드
```

**(A) Top-down 8개** — approach `(0,0,1)`, yaw 8방향 (45° 간격)
```python
for yaw in np.linspace(0, 2*np.pi, 8, endpoint=False):
    R = build_R(approach=(0,0,1), yaw=yaw)
    grasps.append((p_cap_top, R, "top-down"))
```

**(B) Side-cap 8개** — 수평 8방위 (cam Z에 **정확히 90° 직각**)
```python
for az in np.linspace(0, 2*np.pi, 8, endpoint=False):
    approach = (cos(az), sin(az), 0.0)        # horizontal, 수평
    R = build_R(approach=approach, yaw=0.0)   # binormal = 수평 perpendicular
    grasps.append((p_cap_side, R, "side-cap"))
```

### Lying (bottle / can / marker / spam, 누움) — **8 or 6개**

긴축 따라 N등분 × 180° 대칭. Approach = cam +Z (top-down 고정).

```python
N = 3 if cls_name in ("marker","spam") else 4    # 3 × 2 = 6  or  4 × 2 = 8
ss = np.linspace(-0.35*L, 0.35*L, N)
short = [extent[i] for i in range(3) if i != long_axis_idx]
short_r = mean(short) / 2
long_xy = long_cam[:2] / ||long_cam[:2]||
base_yaw = atan2(long_xy[1], long_xy[0]) + np.pi/2   # 긴축에 수직 open

for s in ss:
    off = zeros(3); off[long_axis_idx] = s
    pos = t_icp + R_icp @ off
    pos[2] -= short_r                     # 물체 표면 위 margin
    for sym in (0, np.pi):
        R = build_R(approach=(0,0,1), yaw=base_yaw + sym)
        grasps.append((pos, R, "lying"))
```

### Cube (4종) — **2개, 면 edge 정렬 필수**

대각선 금지. R_icp 3 column 중 카메라 수직축 제외한 나머지 2 column = top face edge 방향.

```python
cam_z_proj = R_icp[2, :]                                   # 각 local 축의 z cam 성분
vertical_col = argmax(|cam_z_proj|)
edge_cols = [i for i in 0,1,2 if i != vertical_col]
cube_half = extent[vertical_col] / 2
pos = [t_icp[0], t_icp[1], t_icp[2] - cube_half + 0.002]   # top face + 2mm

for ec in edge_cols:
    edge_cam = R_icp[:, ec]
    yaw = atan2(edge_cam[1], edge_cam[0])
    R = build_R(approach=(0,0,1), yaw=yaw)
    grasps.append((pos, R, "cube"))
```

## build_R(approach, yaw) 헬퍼

```python
def build_R(approach, yaw):
    """Tool R in cam frame, 열: [Tool_X, Tool_Y, Tool_Z]."""
    a = approach / np.linalg.norm(approach)               # Tool Z
    ref = np.array([1,0,0.0]) if abs(a[0]) < 0.95 else np.array([0,1.0,0])
    b0 = ref - (ref @ a) * a; b0 /= np.linalg.norm(b0)
    n0 = np.cross(a, b0)
    b = b0 * np.cos(yaw) + n0 * np.sin(yaw)               # Tool Y (open dir)
    x = np.cross(b, a)                                     # Tool X
    return np.column_stack([x, b, a])
```

## Rotation → Quaternion 변환

Shepperd method (수치 안정) 권장:
```python
def R_to_quat_wxyz(R): ...    # [qw, qx, qy, qz], qw ≥ 0 normalize
```

## 충돌 체크

1. **테이블 관통**: `tcp.z + gripper_len(0.14) > z_table - ε` 이면 reject
   - `z_table`: scene PC z 95-percentile (경험적 추정)
   - 단, 측면 approach는 finger가 수평이므로 `tcp.z + 0` (TCP 자체 높이만) 체크
2. **그리퍼 폭**: 물체 단면 + 5mm > 0.085m 이면 reject
3. **Swept volume** (선택, 후기): finger 궤적에 scene PC 침투 체크 — 초기엔 생략

## (u,v) 페어링 규약

**학습 입력의 uv = YOLO centroid 고정**. 각 grasp 별 uv가 아니라 **object-level 공통 uv** 저장. 이유: 추론 시 YOLO가 centroid 하나만 주고, 모델이 다봉 분포를 출력해 여러 grasp 생성.

```python
uv_centroid = yolo_cache[sample][obj]["uv_centroid"]    # (2,) float
# grasps.h5 에 object 단위로 저장
```

## 출력 스키마

```
img_dataset/grasp_cache/
  ├── grasps.h5
  │    └── /sample_{sid}/object_{k}/
  │         ├── class_id          int32  (YOLO 원본, curator가 통합)
  │         ├── class_name        str
  │         ├── mode              str    ("standing"/"lying"/"cube")
  │         ├── n_grasps          int32
  │         ├── uv_centroid       (2,) float32   # 공통 입력
  │         ├── grasps_cam        (n, 7) float32 # [x,y,z, qw,qx,qy,qz]
  │         ├── approach_vec      (n, 3) float32 # 편의 (Tool Z)
  │         ├── yaw_around_app    (n,) float32   # 편의
  │         ├── grasp_group       (n,) int32     # 0=top-down,1=side-cap,2=lying,3=cube
  │         ├── collision_ok      (n,) bool
  │         └── fitness_src       float32        # 상위 ICP fitness
  └── synthesis_report.json
```

**금지 필드:** `grasps_base`, 7D pose_base. 카메라 프레임만.

## 예상 개수 (2,238 object 기준)

| mode | 클래스 | 객체 수 (대략) | grasp/object | 총 grasp |
|---|---|---|---|---|
| standing | bottle/can/marker/spam | ~500 | 16 | ~8,000 |
| lying (bottle/can) | bottle/can | ~700 | 8 | ~5,600 |
| lying (marker/spam) | marker/spam | ~300 | 6 | ~1,800 |
| cube | cube_* | ~530 | 2 | ~1,060 |
| **합계** | | 2,030~ | avg ~8 | **~16,500** |

## synthesis_report.json

```json
{
  "total_objects": 2238,
  "per_mode": {"standing": N, "lying": N, "cube": N},
  "total_grasps_generated": N,
  "total_grasps_collision_reject": N,
  "per_group": {"top-down": N, "side-cap": N, "lying": N, "cube": N},
  "reject_rate": {"table": f, "grip_width": f, "swept": f}
}
```

## 검증 체크리스트

- [ ] Standing bottle/can grasp 수 = 16 (8 top + 8 side), side approach의 z 성분 절대값 < 0.01
- [ ] Lying grasp 수 = 8 (bottle/can) 또는 6 (marker/spam), 모두 approach=(0,0,1)
- [ ] Cube grasp 수 = 2, yaw가 R_icp edge column 방위각과 일치 (대각선 ❌)
- [ ] 재투영 uv_centroid가 이미지 범위 내, YOLO 마스크 내부
- [ ] quaternion norm ≈ 1 (±1e-5)
- [ ] grasps_base / pose_base 필드 저장 안 됨 (스키마 위반 ❌)
