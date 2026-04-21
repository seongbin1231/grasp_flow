---
name: hdf5-dataset-management
description: "yolo/icp/grasp 캐시를 통합해 학습용 HDF5(grasp_v1.h5)를 만든다. **Grasp은 6-DoF SE(3) 7D quaternion** [x,y,z, qw,qx,qy,qz]. uv는 per-object YOLO centroid (grasp마다 동일). Unrolled rows (per grasp), scene 단위 train/val 분할, cube 4종→1종 통합, 경계면 QA. 해상도 1280×720, 카메라 프레임 전용."
---

# HDF5 Dataset Management — 학습 데이터셋 큐레이션

파이프라인의 중간 산출물(yolo_cache_v3, icp_cache, grasp_cache)을 학습에 바로 쓸 수 있는 단일 HDF5로 통합. 모든 좌표는 **카메라 프레임**, 해상도 **1280×720**.

## 환경 상수

| 항목 | 값 |
|---|---|
| 해상도 | 1280×720 |
| Camera K | fx=fy=1109, cx=640, cy=360 |
| Depth | float32 meter (uint16 PNG `/1000.0`으로 변환 후 저장) |
| 좌표 프레임 | 카메라 |
| Cube 통합 | yolo 4종 → **단일 `cube`** (class_id 매핑 테이블 유지) |

## 클래스 통합 정책

YOLO 캐시의 원본 class_id (0~7) → 학습용 통합 class_id (0~4):

| YOLO 원본 | 이름 | 통합 ID | 통합 이름 |
|---|---|---|---|
| 0 | bottle | 0 | bottle |
| 1 | can | 1 | can |
| 2 | cube_blue | 2 | cube |
| 3 | cube_green | 2 | cube |
| 4 | cube_p | 2 | cube |
| 5 | cube_red | 2 | cube |
| 6 | marker | 3 | marker |
| 7 | spam | 4 | spam |

curator가 ICP/grasp 캐시 → 최종 HDF5 빌드 시 통합 ID로 변환해 저장. YOLO 캐시는 원본 유지 (참고용).

## 데이터 전개 규약

한 샘플(sample_id)에 물체 M개, 각 물체에 grasp N_i개면 학습 행은 `Σ N_i`개. depth는 이미지 단위 유니크 저장 (`depth_ref` 인덱스 참조) — 메모리 절약.

```
/train/
  depths         (S, 720, 1280) float32     # S = 유니크 depth 이미지
  depth_ref      (N,) int32                 # depths 행 인덱스
  uvs            (N, 2) float32             # per-object YOLO centroid, 픽셀 (1280×720)
                                             # NOTE: 같은 object의 모든 grasp 행에서 동일
  grasps_cam     (N, 7) float32             # [x, y, z, qw, qx, qy, qz] 6-DoF SE(3)
  approach_vec   (N, 3) float32             # Tool Z (= grasp_cam quat의 3번째 열) 편의
  yaw_around_app (N,) float32               # 편의 (binormal yaw 재구성용)
  object_class   (N,) int32                 # 통합 ID (0~4)
  object_mode    (N,) int32                 # 0=lying, 1=standing, 2=cube
  grasp_group    (N,) int32                 # 0=top-down, 1=side-cap, 2=lying, 3=cube
  scene_id       (N,) int32
  sample_ref     (N,) str                   # sample_{sid}
  object_ref     (N,) int32                 # 같은 object의 grasp 그룹핑 키
  fitness        (N,) float32               # ICP fitness 전파
/val/
  ... (동일 구조)
/metadata/
  camera_K        (3,3) float32
  image_size      [720, 1280] int
  coord_frame     "camera"
  grasp_dof       6                          # SE(3)
  class_names     ['bottle','can','cube','marker','spam']
  schema_version  "v2"                       # v1(4-DoF) → v2(6-DoF)
  stats           json
```

**`grasps_base` / `pose_base` 저장 금지** — 카메라 프레임 전용. base 변환은 MATLAB 런타임.

### uv 규약: per-object 공통

하나의 object에 N grasp → N 행, 그러나 `uvs[0..N-1]`는 모두 동일 (YOLO centroid 하나). 이유:
- 추론 시 YOLO는 centroid 1개만 제공
- 모델이 같은 (depth, uv) 입력에 여러 grasp 출력하도록 학습해야 함 (Flow Matching 다봉 분포)
- `object_ref`로 같은 object grasp 들을 묶어 sampler가 한 object 내에서 랜덤 1개 선택 가능

## Train/Val 분할 — 씬 단위 필수

```python
unique_scenes = np.unique(all_scene_ids)  # random1..random6
train_scenes, val_scenes = train_test_split(unique_scenes, test_size=0.2, random_state=42)
```

**같은 scene_id가 train/val 양쪽에 있으면 안 됨** (near-duplicate 누수 방지).

## 경계면 QA (dataset-curator가 책임)

각 캐시 입력 시 다음을 교차 검증하고 실패 시 즉시 에스컬레이션:

| 항목 | 체크 | 실패 처리 |
|---|---|---|
| Depth dtype | float32 | 변환 |
| Depth 범위 | 0 ≤ d ≤ 5.0 m | clamp + warning |
| UV 범위 | 0 ≤ u ≤ 1280, 0 ≤ v ≤ 720 | 경계 클립 + flag |
| Quaternion norm | ‖q‖ ∈ [1-1e-4, 1+1e-4] | 재정규화 or 폐기 |
| qw 부호 | qw ≥ 0 (double-cover 통일) | flip |
| Position 범위 | 카메라 앞 (z_cam > 0.1) | 이상치 폐기 |
| approach_vec 일치 | quat에서 재계산한 Tool Z와 approach_vec 일치 (‖diff‖<1e-3) | 경고 |
| YOLO 캐시 해상도 | `image_h=720, image_w=1280` attrs 일치 | 중단 |
| Camera K | `fx=fy=1109, cx=640, cy=360` 일관 | 중단 |
| 클래스 통합 매핑 | 원본 → 통합 모두 정의 | 중단 |
| Scene leak | train ∩ val scene == ∅ | 중단 |

## 빌드 스크립트 스케치

```python
def build_dataset(yolo_h5, icp_h5, grasp_h5, manifest, out_path):
    rows = []
    depths_unique = []
    depth_index_map = {}
    
    for sid in manifest.sample_ids:
        depth = load_depth(manifest[sid].depth_path)
        if sid not in depth_index_map:
            depth_index_map[sid] = len(depths_unique)
            depths_unique.append(depth)
        dref = depth_index_map[sid]
        
        for k in grasp_h5[f"sample_{sid}"].keys():
            g = grasp_h5[f"sample_{sid}/{k}"]
            n = int(g["n_grasps"][()])
            grasps = g["grasps_cam"][:]       # (n, 4)
            uvs = g["uvs_per_grasp"][:]       # (n, 2)
            cls_raw = int(icp_h5[f"sample_{sid}/{k}/class_id"][()])
            cls_unified = UNIFY_MAP[cls_raw]
            mode = MODE_MAP[str(g["mode"][()])]
            fit = float(icp_h5[f"sample_{sid}/{k}/fitness"][()])
            scene = parse_scene(sid)
            for i in range(n):
                rows.append({
                    "depth_ref": dref,
                    "uv": uvs[i], "grasp_cam": grasps[i],
                    "class": cls_unified, "mode": mode,
                    "scene": scene, "fitness": fit, "sample": sid,
                })
    
    run_integrity_checks(rows, depths_unique)
    train_rows, val_rows = scene_split(rows)
    write_h5(out_path, train_rows, val_rows, depths_unique)
```

## 메모리 용량 고려

Depth (720, 1280) float32 = 3.5 MB per image × 588 = **2 GB**. 모두 로드해 단일 HDF5에 넣으면 큰 편. 대안:
1. 그대로 저장 (현실적, 디스크 OK)
2. uint16 mm으로 유지 후 학습 때 `/1000.0`
3. chunk/compress (gzip=4)로 저장

**권장**: 초기엔 float32 그대로 + gzip chunking `(1, 720, 1280)`.

## 정규화 스펙 — model-engineer와 합의

metadata attrs에 저장해 추론 시 재사용:

```
depth_clip        [0.3, 1.5]
depth_scale       1.0 / 1.5
uv_normalize      false   # 픽셀 그대로
xyz_mean, xyz_std (통계 기반)
```

## Stats 리포트 예시

```json
{
  "schema_version": "v1",
  "coord_frame": "camera",
  "total_rows": <int>,
  "train_rows": <int>, "val_rows": <int>,
  "unique_scenes": {"train": ["random1","random3","random5"], "val": ["random2","random4","random6"]},
  "class_distribution": {"bottle": xx, "can": xx, "cube": xx, "marker": xx, "spam": xx},
  "mode_distribution": {"lying": 0.xx, "standing": 0.xx, "cube": 0.xx},
  "fitness_stats": {"mean": 0.xx, "p10": 0.xx, "p90": 0.xx}
}
```

## 검증 체크리스트

- [ ] `coord_frame=="camera"` metadata 확인
- [ ] `grasps_base` 필드 **부재** 확인
- [ ] 통합 class_names가 5개 (`cube` 포함)인지
- [ ] depth dtype float32, shape (720, 1280)
- [ ] scene leak 없음
