---
name: matlab-data-collection
description: "Simulink 3D Animation 캡처 → RGB PNG + Depth uint16 PNG(mm) 쌍으로 저장하는 데이터 수집 규약. 현 상태 img_dataset/captured_images에 588장이 이미 확보되어 있고, 추가 수집 필요 시 이 스킬을 따른다. 해상도 1280×720, Camera K 고정값 사용. CameraTform은 수집할 필요 없음 (카메라 프레임 전용 파이프라인). Simulink 데이터 로깅·추가 수집 시 사용."
---

# MATLAB Data Collection — RGB/Depth 캡처 규약

Simulink 3D Animation에서 생성된 top-view RGB/Depth 이미지를 파일 시스템에 저장하는 표준 방식.

## 현재 상태 (2026-04-18)

| 항목 | 값 |
|---|---|
| 총 샘플 수 | **588장** (6 scene × ~98 frames) |
| 저장 위치 | [img_dataset/captured_images/](/home/robotics/Competition/YOLO_Grasp/img_dataset/captured_images/) + `captured_images_depth/` |
| 해상도 | **1280×720** RGB, **1280×720** Depth (uint16 PNG, mm) |
| 씬 구분 | `random1 ~ random6` (폴더 단위) |
| 파일 네이밍 | `random{scene}_{frame}.png` / `random{scene}_depth_{frame}.png` |

**이미 확보된 데이터로 파이프라인 전체 iteration이 가능**. 추가 수집은 YOLO/ICP 단계 완료 후 필요할 때 결정.

## 환경 상수

| 항목 | 값 |
|---|---|
| Camera K | `fx=fy=1109, cx=640, cy=360` (image 1280×720 기준) |
| Depth 단위 | uint16 PNG, 밀리미터 → Python에서 `/1000.0` |
| 좌표 프레임 | **카메라 프레임만 다룬다**. base frame 변환은 MATLAB 런타임(Simulink)에서 CameraTform으로 처리 |

## 파일 레이아웃

```
img_dataset/
  captured_images/
    random1/
      random1_0.png     # RGB, 1280×720, 3ch
      random1_1.png
      ...
  captured_images_depth/
    random1_dep/
      random1_depth_0.png   # uint16, 1280×720, mm
      random1_depth_1.png
      ...
  check_saved_depth_img.m   # depth 검증 헬퍼
```

RGB와 Depth 페어링은 파일명 index로: `random1/random1_10.png` ↔ `captured_images_depth/random1_dep/random1_depth_10.png`.

## Depth 파일 규약

- 포맷: uint16 PNG (16-bit single channel)
- 단위: **밀리미터 (mm)**. Python에서 로드 후 `/1000.0`로 meter 변환.
- 무효 depth: 0 (배경/범위 밖)
- 유효 범위: 0.3~1.5 m (top-view 테이블 기준)

```python
# Python 로드 예시
d_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16
depth_m = d_raw.astype(np.float32) / 1000.0           # meter
depth_m[~np.isfinite(depth_m)] = 0                     # 안전망
mask_valid = (depth_m > 0.3) & (depth_m < 2.0)
```

## 왜 CameraTform이 없어도 되는가

이 프로젝트는 **카메라 프레임 전용 파이프라인**으로 단순화됐다:
- ICP 출력: 카메라 프레임 6D pose (`objectPose_cam`)
- Grasp 합성: 카메라 프레임 `(x, y, z, yaw)`
- 모델 학습/예측: 카메라 프레임 출력
- **배포 시** MATLAB `predict_grasp.m`이 Simulink에서 받은 `/camera/tf` 토픽으로 base frame 변환 후 IK

즉 "학습 데이터 수집 시에는 CameraTform 필요 없고 MATLAB 런타임에만 필요"가 대원칙.

## 추가 수집 시 체크리스트 (향후)

Simulink에서 새 데이터 수집하게 되면:

- [ ] RGB PNG 저장 시 **해상도 1280×720 유지** (축소 금지)
- [ ] Depth 저장 시 uint16 mm 유지 (float 변환하지 않음)
- [ ] scene 폴더 이름 규약 따르기 (`random{N}/random{N}_{frame}.png`)
- [ ] 파일명 index가 RGB와 Depth에서 **일치하는지** 검증
- [ ] Top-view 확인: Simulink의 CameraTform의 Z축이 `[0,0,-1]`에 근접 (topness > 0.95)
- [ ] [check_saved_depth_img.m](/home/robotics/Competition/YOLO_Grasp/img_dataset/check_saved_depth_img.m)로 random 10장 확인

## 수집 시 Simulink 설정 (참고)

- Simulink 3D Animation 카메라: FOV, 해상도 = 1280×720
- Camera intrinsics가 `fx=fy=1109, cx=640, cy=360`에 맞게 설정되어야 함
- 실제 물체 배치는 테이블 위 0.3~0.6m 범위가 가장 안정적

## MATLAB 코드 참고

기존 [RoboCup_ARM/scripts/test.m](/home/robotics/Competition/RoboCup_ARM/scripts/test.m) 이 ROS2 구독 기반. 파일로 저장하려면:
```matlab
rgbMsg = subRgb.LatestMessage;
rgb = rosReadImage(rgbMsg);   % HxWx3 uint8
imwrite(rgb, sprintf('img_dataset/captured_images/random%d/random%d_%d.png', ...
        sceneId, sceneId, frameId));

depthMsg = subDepth.LatestMessage;
depth_m = rosReadImage(depthMsg);    % HxW float32 meter
depth_mm = uint16(depth_m * 1000);   % ← mm로 변환해 저장
imwrite(depth_mm, sprintf('img_dataset/captured_images_depth/random%d_dep/random%d_depth_%d.png', ...
        sceneId, sceneId, frameId));
```
