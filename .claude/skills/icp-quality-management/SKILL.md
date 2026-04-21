---
name: icp-quality-management
description: "Python + open3d로 YOLO mask + depth + PLY 모델 정합을 배치 실행하여 물체별 6D pose(카메라 프레임)를 산출하는 절차. MATLAB ICPGraspPlanner.m의 multi-scale ICP 파이프라인을 포팅. fitness threshold·inlier RMSE·normalizeGraspPose로 고품질 pseudo-GT만 통과시킨다. CameraTform 변환은 하지 않는다 (카메라 프레임 전용). ICP·fitness·정합 품질·pseudo-GT 관련 작업에 반드시 사용."
---

# ICP Quality Management — 정합 품질 관리 (Python/open3d)

YOLO mask + depth + PLY 모델로 물체 6D pose를 추정하고, 고품질 샘플만 통과시켜 pseudo-GT 노이즈를 최소화한다. **카메라 프레임에서만 작업**한다.

## 왜 Python/open3d인가

| 기준 | MATLAB 기존 | **Python(open3d, 선택)** |
|------|--------------|-----------------|
| 배치 속도 | 샘플당 1~3s | 샘플당 0.3~1s |
| 환경 통합 | Simulink GUI 필요 | conda base 스크립트로 단독 실행 |
| 디버깅 | MATLAB GUI | Python 표준 |
| 포팅 노력 | 0 | 중 (이 스킬의 대상) |

open3d 0.19.0 설치 확인됨 (conda base). MATLAB ICPGraspPlanner 로직을 Python으로 옮긴다.

## 환경 상수

- 해상도: 1280×720
- Camera K: `fx=fy=1109, cx=640, cy=360`
- Depth: uint16 PNG → `/1000.0`로 meter
- Python: `/home/robotics/anaconda3/bin/python`
- PLY 디렉터리: `/home/robotics/Competition/RoboCup_ARM/models/ply/`

## 카메라 프레임만 — 제거된 단계

MATLAB 원본의 다음 단계는 **하지 않는다**:
- `R_fix = [0 1 0; -1 0 0; 0 0 1]` 좌표 보정 → 필요 없음
- `CameraTform * H_grasp_cam` base 변환 → 필요 없음
- `EMA PoseFilter` → batch 모드라 불필요

ICP 출력을 그대로 카메라 프레임 6D pose로 사용. 기존 MATLAB에서 "CameraTform 미제공 → base frame skip"의 경로와 동일.

## 품질 게이트 3단계

```
[Multi-scale ICP]
   │
   ▼
Gate 1: fitness ≥ τ_f   (기본 0.02, MATLAB 원본 0.005보다 엄격)
   │
   ▼
Gate 2: inlier_rmse ≤ τ_r   (기본 0.005 m)
   │
   ▼
Gate 3: multi-frame 안정성 (선택)
   - 같은 scene·같은 class에서 K(=3~5)프레임 검출
   - 위치 표준편차 < 5mm, 회전 표준편차 < 5°
   │
   ▼
[pose_cam을 poses.h5에 저장]
```

threshold 결정 전략: 초기 `τ_f=0.02`로 돌린 뒤 `quality_report.json`의 통과율 확인. 40~80%면 유지, <10%면 완화(τ_f=0.015), >80%면 상향(τ_f=0.03).

## PLY 매핑 (cube 통합)

| YOLO 클래스 | PLY 파일 | 비고 |
|---|---|---|
| bottle | blueBottle.ply (or red/yellow — 형상 동일) | 색 무관 |
| can | greenCan.ply (or red/yellow — 형상 동일) | 색 무관 |
| marker | marker.ply | — |
| spam | Simsort_SPAM.ply | — |
| cube_blue/green/p/red | **cube.ply 단일** | 하류에서 class=`cube` 1종으로 통합 |

ICP 단계부터 이미 cube는 단일 PLY 사용 (원래 MATLAB도 그렇게 함).

## Multi-scale ICP (MATLAB 설정 이식)

| 단계 | threshold | max_iter |
|---|---|---|
| coarse | voxel × 16 = 0.048 m | 200 |
| medium | voxel × 4 = 0.012 m | 200 |
| fine | voxel × 1 = 0.003 m | 500 |

voxel size: 일반 물체 0.003 m, 작은 물체(marker) 0.002 m. relative_fitness/rmse 수렴 기준 1e-7.

## Python 핵심 코드 스케치

```python
import open3d as o3d
import numpy as np
import h5py, cv2

K = np.array([[1109.,0,640],[0,1109.,360],[0,0,1]])
H, W = 720, 1280

def load_ply_centered(ply_path):
    m = o3d.io.read_triangle_mesh(str(ply_path))
    pc = m.sample_points_poisson_disk(5000)  # or uniform
    # 센터링
    pc.translate(-pc.get_center())
    extent = m.get_axis_aligned_bounding_box().get_extent()
    return pc, extent

def depth_mask_to_pc(depth_m, mask_bool, K, denoise=True):
    ys, xs = np.where(mask_bool & (depth_m > 0.1) & (depth_m < 2.0) & np.isfinite(depth_m))
    Z = depth_m[ys, xs]
    X = (xs - K[0,2]) * Z / K[0,0]
    Y = (ys - K[1,2]) * Z / K[1,1]
    pts = np.stack([X, Y, Z], axis=1).astype(np.float64)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    if denoise:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=4, std_ratio=0.5)
    return pc

def multiscale_icp(source, target, voxel=0.003):
    T = np.eye(4)  # initial guess — PLY centered at scene centroid
    T[:3, 3] = target.get_center()
    for th, it in [(voxel*16, 200), (voxel*4, 200), (voxel, 500)]:
        res = o3d.pipelines.registration.registration_icp(
            source, target, th, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=it))
        T = res.transformation
    return T, res.fitness, res.inlier_rmse
```

## Rotation 정규화 (MATLAB normalizeGraspPose 이식)

ICP 출력 R을 그대로 쓰면 대칭 물체(원통·직육면체)에서 회전 축이 불안정하다. MATLAB 원본 규약 그대로:

- **bottle/can/marker**: PLY 긴축은 Y축 방향. 출력 R의 Y축을 "긴축"으로 정렬
  - bottle/can: 긴축을 출력 Z로 재배치
  - marker: 긴축을 출력 X로 (그리퍼 ⟂ 축)
- **cube/spam**: 180° 대칭 중 이전 프레임과 가까운 쪽 선택 (batch에서는 첫 프레임을 canonical로)
- SVD cleanup: `U @ V.T`, `det<0`이면 마지막 열 부호 반전

**base frame 관련 코드는 전부 제거** (CameraTform 미사용).

## 출력 스키마

```
img_dataset/icp_cache/poses.h5
  (attrs: camera_fx, cy, cx, cy, image_h=720, image_w=1280,
          coord_frame="camera",
          voxel_size, fitness_threshold, rmse_threshold)
  │
  └── /sample_{sid}/object_{k}/
       ├── class_id        int32  (cube는 이미 통합된 ID 쓰거나 원본 유지 후 curator가 통합)
       ├── pose_cam        (7,) float32 [x,y,z,qw,qx,qy,qz]  # 카메라 프레임
       ├── fitness         float32
       ├── inlier_rmse     float32
       ├── scene_pc_npts   int32
       └── stable_flag     bool  (multi-frame 검증, 선택적)
```

## quality_report.json

```json
{
  "total_samples": 588,
  "total_detections_processed": 3242,
  "threshold": {"fitness": 0.02, "rmse": 0.005},
  "gate1_fitness_pass": <int>,
  "gate2_rmse_pass": <int>,
  "gate3_stable_pass": <int>,
  "per_class_pass_rate": {
    "bottle": 0.xx, "can": 0.xx, "cube": 0.xx, "marker": 0.xx, "spam": 0.xx
  }
}
```

## 실패 모드와 대응

| 증상 | 원인 | 대응 |
|------|------|------|
| 전체 fitness 낮음 | Camera K 오류, PLY 스케일 | K 재검증, PLY extent 출력해 확인 |
| 특정 클래스만 낮음 | PLY 축 불일치 | PLY를 meshlab에서 확인, normalize_rotation에서 축 재매핑 |
| ICP 비수렴 | VoxelSize 너무 작음 | 0.002 → 0.003, 또는 coarse 단계 iter 증가 |
| marker가 이상한 자세로 수렴 | 마스크 경계 노이즈 | YOLO mask dilation=3 적용(선택), 또는 `remove_statistical_outlier` 강화 |

## 검증 체크리스트

- [ ] 랜덤 10개 샘플의 ICP 결과를 open3d로 시각화 (scene PC + transformed PLY overlay)
- [ ] fitness 분포 히스토그램 — 대다수가 0.02~0.1 범위인지
- [ ] quaternion norm 모두 1.0±1e-4 (SVD cleanup 확인)
- [ ] 카메라 프레임에서 Z > 0 (모든 물체가 카메라 앞)

## 참고

- MATLAB 원본: [ICPGraspPlanner.m:1009-1024](/home/robotics/Competition/RoboCup_ARM/scripts/ICPGraspPlanner.m#L1009-L1024) (buildIcpConfig), [834-893](/home/robotics/Competition/RoboCup_ARM/scripts/ICPGraspPlanner.m#L834-L893) (normalizeGraspPose)
- open3d docs: https://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html
