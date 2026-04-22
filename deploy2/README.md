# YOLO_Grasp Deploy — **v7 / v4policy** (신규 배포)

**날짜**: 2026-04-22
**대상**: RoboCup_ARM 2026 MATLAB 팀

---

## 이전 배포 (deploy/) 와의 차이

| 항목 | `deploy/` (v6) | **`deploy2/` (v7)** |
|---|---|---|
| Flow ckpt | v6_150ep ep118 | **v7_v4policy_big ep226** |
| val_flow | 0.3640 | **0.3676** (동등) |
| 아키텍처 | h=768, n_blocks=8 (26M) | **h=1024, n_blocks=12 (35.28M)** |
| Grasp 정책 | 6dof-v1 (총 17,370) | **6dof-v4 (총 40,984)** |
| lying tilt | 없음 (top-down only) | **bottle/can ±30°, marker/spam ±15°** |
| standing tilt | top-down 8 + side 8 = 16 | **top-down 8 + side-45° 8 + side 8 = 24** |
| 정책 제외 | 없음 | **marker_standing, spam_standing, spam_lying(뚜껑 측면)** |
| YOLO ONNX | 동일 | 동일 (재사용) |

---

## 디렉토리 구성

```
deploy2/
├── README.md            ← 이 파일
├── onnx/
│   ├── encoder.onnx     (2.4 MB)   depth + uv → cond(256)       scene당 1회
│   ├── velocity.onnx    (139 MB)   g_t + cond + t + uv → v(8)   T × 2회 (CFG)
│   ├── meta.json        v4 정책 메타 포함
│   └── README.md        MATLAB 통합 스펙 (357줄)
│
├── yolo/
│   ├── yolov8m_seg_1280.onnx (110 MB)  YOLOv8m-seg, imgsz=1280
│   ├── yolov8m_seg_1280.pt   (55 MB)   PyTorch 원본 (검증용)
│   └── README.md              YOLO 사용 가이드
│
└── viz/
    ├── index.html              탭 네비 (v7 추론 7 카테고리)
    ├── infer_*.html            각 카테고리 viz
    ├── gt_policy/index.html    v4 정책 시각화 (10 case + side-45 단독)
    ├── icp/index.html          ICP 정합 품질 (클래스별 3샘플)
    └── live/index.html         라이브 ROS 캡처
```

---

## MATLAB 팀 integration 포인트 (불변)

인터페이스는 **v6 와 동일**. 기존 MATLAB 코드 변경 불필요.

```
Input  : depth (720,1280) float32 meter,  uv (2,) pixel
Output : grasps_cam (N,7) = [x,y,z, qw,qx,qy,qz]  카메라 프레임
```

변경된 것은 **모델 성능 & 다양성**:
- standing 에 **45° 기울어진 중간 층** grasp 추가 → 실제 로봇 환경에서 접근 가능한 각도 많아짐
- lying 에 **±15~30° tilt** 추가 → reachability 개선
- spam/marker 의 비현실적 pose (standing) 제거 → 학습 노이즈 감소

---

## 추천 인퍼런스 하이퍼 (동일)

```
N_samples     = 32
T_euler       = 32
noise_temp    = 0.8
cfg_w         = 2.0
BODY_MARGIN   = 5mm
TIP_SWEEP_R   = 15mm
best_pick     = top-down(|a_z|>0.7) 우선 → uv 3D 최단
```

---

## 검증 결과 (random6 val scene)

| case | kept/N | best \|a_z\| | Δuv 3D |
|---|---|---|---|
| standing_bottle | 32/32 | 0.99 | 1.8cm |
| standing_can | 32/32 | 1.00 | 1.2cm |
| lying_bottle | 28/32 | 0.99 | 0.2~1.3cm |
| lying_can | 28/32 | 1.00 | 0.3~0.8cm |
| lying_marker | 28~32/32 | 0.97~0.98 | 0.5~3.9cm *얇은물체 약점* |
| lying_spam | 32/32 | 1.00 | 0.4~0.5cm |
| cube | 32/32 | 1.00 | 0.7cm |

Standing 추론에서 `kept split: topdown=X side=Y mid=Z` 로그 확인 시 **3-layer 모두 학습됨** 검증 가능.

---

## 이전 배포 유지

`deploy/` 는 v6 상태로 그대로 보존. 문제 발견 시 rollback 가능.

## 문의

- 수치 검증 실패 시: `/home/robotics/Competition/YOLO_Grasp/scripts/export_onnx.py` 재실행
- MATLAB 통합 상세: `deploy2/onnx/README.md` 참조
- 정책 의도 확인: `deploy2/viz/gt_policy/index.html` 시각 참조
