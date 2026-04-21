---
name: yolo-batch-inference
description: "학습된 YOLOv8-seg 모델(runs/yolov8m_seg_v3_1280/weights/best.pt)로 1280×720 이미지를 일괄 추론하여 물체별 (u,v)·마스크·bbox를 HDF5로 캐시하는 절차. centroid/bbox center/distance transform peak 세 가지 (u,v) 전략을 동시 계산해 하류 grasp 샘플링에 활용. YOLO 추론·(u,v) 추출·mask 캐시 작업 시 반드시 사용. **ultralytics에 numpy 전달 시 반드시 BGR — 이게 가장 흔한 함정이다.**"
---

# YOLO Batch Inference — 일괄 (u,v) 추출

학습된 YOLOv8-seg로 캡처 이미지를 처리하고 하류 파이프라인이 필요한 (u,v)·마스크를 표준 스키마로 캐시한다.

## 🚨 가장 중요한 규칙: BGR

`cv2.imread`로 읽은 numpy 배열을 **그대로 `model.predict(source=bgr_numpy)`에 전달**한다. `cv2.cvtColor(BGR→RGB)` 변환 금지.

**이유**: ultralytics의 `predictor.py preprocess()`는 numpy 입력을 **BGR로 가정하고 내부에서 `im[..., ::-1]`로 RGB 변환**한다. 우리가 RGB를 주면 모델에 **R·B 채널이 뒤바뀐 이미지**가 전달되어 confidence가 붕괴한다 (이 프로젝트에서 v3 passed detections가 1351→3242로 벌어진 이유).

```python
# ❌ 틀림 — 지난번 이 버그로 모델 성능을 절반으로 깎아먹음
rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
model.predict(source=rgb, ...)

# ✅ 맞음
bgr = cv2.imread(p)
model.predict(source=bgr, ...)

# ✅ 더 단순 (경로 전달 — ultralytics 내부가 cv2.imread = BGR)
model.predict(source=str(p), ...)
```

PIL 이미지 입력 시에는 ultralytics가 RGB→BGR 변환을 알아서 한다. 이 스킬은 numpy 경로만 다룬다.

## 환경 상수

- 해상도: **1280×720** (Simulink 원본, Roboflow v2도 원본 유지)
- 모델: `runs/yolov8m_seg_v3_1280/weights/best.pt` (imgsz=1280 학습)
- 추론 imgsz: **1280** (학습과 일치)
- 클래스 (8): `bottle, can, cube_blue, cube_green, cube_p, cube_red, marker, spam`
- Python: `/home/robotics/anaconda3/bin/python` (conda base)

## 왜 3종 (u,v)인가

물체별 grasp 샘플링 전략이 다르기 때문이다:

| (u,v) 종류 | 정의 | 용도 |
|-----------|------|------|
| **centroid** | 마스크 픽셀 무게중심 | 누운 실린더의 긴축 기준점. 보간의 시작점 |
| **bbox_center** | bbox 중심 | 대칭 물체(cube)의 안정적 기준 |
| **dt_peak** | 마스크 내부 distance transform의 최댓값 | 부분 폐색 시 안정적 |

하류 grasp-synthesizer가 mode에 따라 선택. aspect ratio(PCA)는 mode 판정 proxy로만 활용 (긴축 4점 좌표는 최종 데이터에 저장하지 않음 — ICP 3D 정보가 이를 대체).

## 실행 흐름

```
Load YOLO model (best.pt)
   │
   ▼
Iterate manifest (588장) — batch or per-image
   │  for each image:
   │    - bgr = cv2.imread(path)        (BGR 유지)
   │    - model.predict(source=bgr, imgsz=1280, conf=0.05, iou=0.7)
   │    - conf>=0.5 통과, 나머지 count만 기록
   │    - per-instance:
   │         three_uv(mask)
   │         pca_aspect + long_axis
   │         polygon
   │
   ▼
Write detections.h5 (sample_id 기준 그룹)
```

`conf=0.05`로 predict한 뒤 `>=0.5` 필터링하는 이유: 낮은 confidence까지 기록해 통계로 활용 가능. 최종 통과는 0.5 기준.

## 핵심 코드 패턴 (Python)

```python
from ultralytics import YOLO
import h5py, numpy as np, cv2
from scipy.ndimage import distance_transform_edt

model = YOLO("runs/yolov8m_seg_v3_1280/weights/best.pt")

def three_uv(mask_bool):
    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        nan = np.array([np.nan, np.nan], dtype=np.float32)
        return nan, nan, nan
    centroid = np.array([xs.mean(), ys.mean()], dtype=np.float32)
    bbox_ctr = np.array([(xs.min()+xs.max())/2, (ys.min()+ys.max())/2], dtype=np.float32)
    dt = distance_transform_edt(mask_bool)
    py, px = np.unravel_index(dt.argmax(), dt.shape)
    return centroid, bbox_ctr, np.array([px, py], dtype=np.float32)

def run_image(path):
    bgr = cv2.imread(path)                  # BGR, 1280×720
    res = model.predict(source=bgr, imgsz=1280, conf=0.05, iou=0.7,
                        verbose=False, device=0)[0]
    # res.boxes.xyxy는 원본 해상도 좌표로 이미 복원됨 → 1280×720 기준
    # res.masks.data는 모델 내부 해상도 → 1280×720으로 INTER_NEAREST resize 필요
    ...
```

## HDF5 캐시 스키마

```
img_dataset/yolo_cache_v3/detections.h5
  (attrs: camera_fx=1109, camera_fy=1109, camera_cx=640, camera_cy=360,
          image_h=720, image_w=1280, coord_frame="native_1280x720",
          conf_threshold=0.5, imgsz=1280, model_path=...,
          class_names=['bottle','can','cube_blue','cube_green','cube_p','cube_red','marker','spam'])
  │
  └── /sample_{sid}/
       ├── (attrs) scene, rgb_path, depth_path, depth_available,
       │           n_above_threshold, n_below_threshold, detected
       ├── classes         (N,) int32
       ├── confidences     (N,) float32
       ├── bboxes          (N,4) float32  [x1,y1,x2,y2] in 1280×720
       ├── uv_centroid     (N,2) float32
       ├── uv_bbox_ctr     (N,2) float32
       ├── uv_dt_peak      (N,2) float32
       ├── pca_aspect      (N,)  float32  (mode hint)
       ├── pca_long_axis   (N,2) float32  (unit vector, image coords)
       └── mask_poly       VLEN float32   (largest contour, flattened)
```

h5py vlen 함정: 빈 배열을 섞으면 `AttributeError: 'float' object has no attribute 'dtype'`. 빈 polygon은 `np.zeros(2, dtype=np.float32)` 등 **dtype 보장된 최소 배열**로 채워야 한다. 또한 `create_dataset(data=list)` 대신 먼저 `shape=(N,)` 빈 dataset 만들고 `ds[k] = arr`로 넣는 게 안전.

## Batch 실행 팁

- `model.predict(source=[img_list], stream=True)` → 메모리 절약. 다만 이미지당 메타(sample_id, depth 경로)를 짝지어야 하니 per-image 루프도 OK.
- detected==0 이미지도 그룹은 생성하고 `g.attrs["detected"] = False`. 데이터셋 결측 추적용.
- CUDA OOM 시 1280 해상도 + batch 1로도 충분. 4070 12GB에서 1장씩 0.4s 정도.

## 🔎 클래스 정책 — 하류 통합

학습 단계에서는 8 클래스 그대로지만, 하류(ICP/grasp/학습 데이터)에서는:
- `cube_blue, cube_green, cube_p, cube_red` → `cube` 단일 (depth 형상 동일, 색은 학습에 불필요)
- 매핑은 dataset-curator가 담당. YOLO 캐시에는 원본 클래스 ID 그대로 저장.

## 검증 체크리스트

- [ ] `summary.json`의 total detections 합이 per-class 합과 일치
- [ ] `uv_centroid`가 NaN인 행이 없는지 (빈 마스크 결함)
- [ ] 랜덤 샘플 10개의 mask polygon을 시각화하여 YOLO 결과와 대조
- [ ] `zero_detect_count`가 ~0 (현재 v3에서 0 달성)
- [ ] class별 mean conf > 0.7 — 낮으면 BGR 버그 재발 의심

## 참고: BGR 버그 재발 감지

이번 프로젝트의 "삽질 교훈": 추론 결과가 어딘가 부자연스럽게 저조하면 (특히 cube_red=0 같은 증상) **가장 먼저 BGR 체크**한다. 같은 이미지를 `source=str(path)` 경로 전달과 `source=numpy` 전달로 각각 돌려 결과가 일치하는지 확인. 다르면 numpy 입력이 RGB인 것.
