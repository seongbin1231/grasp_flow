# YOLOv8-seg 1280 — MATLAB 통합 스펙

**대상**: RoboCup_ARM `matlab-pipeline` 에이전트 (`weights/best.onnx` 사용자)
**목적**: Flow Matching 모델 훈련에 사용된 **동일 YOLO 모델**을 제공해 훈련-추론 uv 좌표 일관성을 보장.

---

## 0. 왜 이 파일이 필요한가

Flow Matching 모델(`deploy/onnx/encoder.onnx` 등)의 **훈련 데이터 (u,v)** 는 본 파일(`yolov8m_seg_1280.onnx`, imgsz=**1280**)로 생성됐습니다.

- Flow 훈련 시: `runs/yolov8m_seg_v3_1280/weights/best.pt` → ICP → grasp 합성 → HDF5 (per-object uv = YOLO centroid)
- 기존 `RoboCup_ARM/weights/best.onnx` 는 imgsz=640. 같은 project best.pt에서 왔을 수 있으나 **해상도/letterbox가 다름** → 마스크 경계/중심점이 최대 수 픽셀 다를 수 있음
- 얇은 물체(marker) 에서는 이 몇 픽셀 차이가 Flow의 conditional signal을 흔듬

**권장**: MATLAB FlowGraspPlanner 경로에서는 **본 파일**을 로드. ICPGraspPlanner 기존 pipeline은 기존 `weights/best.onnx` 유지 가능 (ICP는 mask만 쓰면 되므로 imgsz 무관).

---

## 1. 파일 목록

| 파일 | 크기 | MD5 | 용도 |
|---|---|---|---|
| `yolov8m_seg_1280.onnx` | 105 MB | `1c09763fa3d41c7c09e79a05390c0f36` | **MATLAB 로드 대상** |
| `yolov8m_seg_1280.pt` | 55 MB | `2b5d9b7767375c4f1b3019af05a7566c` | 원본 (ultralytics export source) |

재export 명령:
```bash
/home/robotics/anaconda3/bin/python -c "
from ultralytics import YOLO
YOLO('runs/yolov8m_seg_v3_1280/weights/best.pt').export(
    format='onnx', imgsz=1280, opset=12, simplify=True, dynamic=False)"
```

---

## 2. 모델 스펙

| 항목 | 값 |
|---|---|
| Architecture | YOLOv8m-seg |
| Training imgsz | **1280** (정사각) |
| Params | 27.2M |
| FLOPs | 104.3 GFLOPs |
| Mask mAP50-95 | **0.982** (v3_1280, 50 epoch finetune, Roboflow v2 1870장) |
| ONNX opset | 12 |
| simplify | ✓ (onnxslim) |

### Class 순서 (훈련 시 = 인덱스 순)

```
0 bottle
1 can
2 cube_blue
3 cube_green
4 cube_p
5 cube_red
6 marker
7 spam
```

**기존 `RoboCup_ARM/weights/best.onnx` 와 동일 순서**. `yoloSegONNX.m` 내 class index 하드코딩 변경 불필요.

---

## 3. ONNX I/O 스키마

```
IN  images   shape=[1, 3, 1280, 1280]  dtype=float32    (NCHW, RGB, /255)
OUT output0  shape=[1, 44, 33600]      dtype=float32    (4 bbox + 8 cls + 32 mask coef, per anchor)
OUT output1  shape=[1, 32, 320, 320]   dtype=float32    (mask prototypes)
```

- **44 = 4 + 8 + 32**:
  - 4 = bbox `[cx, cy, w, h]` (letterbox 1280×1280 좌표)
  - 8 = class logits
  - 32 = mask coefficient
- **Mask 복원**: `mask = sigmoid(mask_coef @ proto.reshape(32, -1))` → `(320, 320)` → crop to bbox → resize to original H×W

---

## 4. MATLAB 통합 — `yoloSegONNX.m` 수정 포인트

기존 `scripts/yoloSegONNX.m` (imgsz=640 전제) 를 imgsz=1280 로 **두 줄 수정**:

```matlab
% 기존
NET_SIZE = 640;

% 변경
NET_SIZE = 1280;
```

letterbox / NMS / proto decode 로직은 그대로 재사용. I/O 텐서 이름(`images`, `output0`, `output1`)도 동일. 실행:

```matlab
net = importNetworkFromONNX('/home/robotics/Competition/YOLO_Grasp/deploy/yolo/yolov8m_seg_1280.onnx');
[masks, classes, confs, bboxes] = yoloSegONNX(net, rgb_720x1280);
```

**경고**: imgsz=1280 은 GPU 메모리 사용량이 640 대비 4× (1280²/640²). RTX 4070 에서 초당 추론 수 반감 가능. ICPGraspPlanner 경로에서는 imgsz=640 유지하고, **FlowGraspPlanner 에서만** 1280 ONNX 로드 권장.

---

## 5. 출력 후처리 (Flow 로 uv 전달)

```matlab
% yoloSegONNX 반환
% masks:  (H, W, N)   bool,  원본 이미지 크기 (720×1280)
% classes: (N, 1)     int,   0..7
% confs:   (N, 1)     float
% bboxes:  (N, 4)     [x1 y1 x2 y2]  원본 좌표

for i = 1:size(masks, 3)
    [ys, xs] = find(masks(:,:,i));
    uv = [mean(xs); mean(ys)];    % (u, v) — YOLO centroid, 1-indexed
    % → Flow encoder 전달 시 0-indexed 로 보정
    uv_flow = uv - 1;             % MATLAB 1-index → ONNX 0-index
    cond = predict(net_flow_enc, depth_img, uv_flow);
end
```

**주의**: MATLAB `find` 는 1-indexed. Flow encoder 는 0-indexed 픽셀 좌표 기대. `u_flow = u_matlab - 1`, `v_flow = v_matlab - 1`.

---

## 6. 검증 체크리스트

- [ ] MD5 일치: `md5sum yolov8m_seg_1280.onnx` → `1c09763fa3d41c7c09e79a05390c0f36`
- [ ] class 순서 확인: `{bottle, can, cube_*, marker, spam}` 인덱스 0~7
- [ ] 한 프레임 테스트: ROS `/camera/rgb` capture → `yoloSegONNX` → 8 class detection 수가 기존 best.onnx 와 유사한지 확인
- [ ] uv centroid 좌표: 1280×720 원본 해상도 좌표계인지 확인 (letterbox 역변환 정상)
- [ ] Flow encoder 연결: `uv - 1` 보정 후 `deploy/onnx/encoder.onnx` 호출, cond norm 0.7~0.9 범위 확인
- [ ] 성능: RTX 4070 GPU 기준 1 frame 추론 ~60~150ms 범위 (640→22~100ms 대비 증가)

---

## 7. ICPGraspPlanner 와의 관계

| Pipeline | YOLO 파일 | imgsz | 이유 |
|---|---|---|---|
| **ICPGraspPlanner** (기존, ICP 기반) | `weights/best.onnx` | 640 | 속도 우선, mask 만 사용. 정확도 충분 |
| **FlowGraspPlanner** (신규, Flow 기반) | `deploy/yolo/yolov8m_seg_1280.onnx` | **1280** | Flow 훈련과 uv 일치. 얇은 물체 정밀도 |

두 파일 모두 같은 project data 로 학습된 **동일 weights** (best.pt 동일) 일 가능성 높음 — 단지 export imgsz 가 다름. 인덱스/class 순서 동일.

---

## 8. Known Issues

- **marker uv 정확도 한계**: 1280 이어도 marker 는 ~10mm 너비라 mask 수백 픽셀. centroid 는 수 픽셀 흔들림. Flow 측 마커 localization 실패의 원인 중 하나 (주 원인은 훈련 분포 6.6%).
- **opset 12 고정**: MATLAB `importNetworkFromONNX` 호환성 때문. 상위 opset 필요 시 재export 전 호환성 확인.
- **simplify=True**: onnxslim 으로 단순화됨. 노드 수/이름이 원본과 다를 수 있으나 I/O 동일.

---

## 9. 관련 문서

- **Flow 통합 스펙**: [../onnx/README.md](../onnx/README.md) ← uv 연결 규약
- **참조 Python 파이프라인**: [../viz/demo_inference.py](../viz/demo_inference.py) (YOLO → Flow → filter 전체 흐름)
- **Live ROS 추론 예**: [../../scripts/infer_live.py](../../scripts/infer_live.py) (ROS 프레임 → YOLO → Flow)
- **YOLO 훈련 로그**: `runs/yolov8m_seg_v3_1280/` (confusion matrix, val metrics)

---

## 10. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-04-21 | 초기 export (v3_1280 best.pt → imgsz=1280 opset=12 simplify) |
