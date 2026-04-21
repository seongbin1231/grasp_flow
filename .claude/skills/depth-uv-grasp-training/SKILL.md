---
name: depth-uv-grasp-training
description: "Depth(720,1280) + per-object (u,v) 입력으로 **6-DoF SE(3) grasp** [x,y,z, qw,qx,qy,qz]을 카메라 프레임에서 생성하는 **Conditional Rectified Flow (Flow Matching)** 학습·평가. 다봉 분포 샘플링 지원 (노이즈 N개 → grasp N개), 충돌 필터와 결합. GraspFlow 참조 코드 차용, ONNX-safe, MATLAB 배포."
---

# Depth+UV Grasp Training — Conditional Rectified Flow (6-DoF)

`(depth, uv) → N개의 6-DoF grasp` 다봉 분포 모델. **모든 출력은 카메라 프레임**, base 변환은 배포 시 MATLAB.

## 왜 Flow Matching인가

단일 회귀는 mode collapse (lying bottle의 8 GT grasp 평균 → 어느 GT와도 안 맞음). Set prediction은 고정 K 슬롯 제약. **Rectified Flow**는:
- 학습: `‖v_θ(g_t, t, c) − (g_1 − g_0)‖²` — 안정적 MSE 회귀
- 추론: 1-step (Reflow 후) 또는 몇 step ODE. **샘플 N개 병렬**로 다봉 분포 cover
- ONNX 배포: velocity MLP 단일 export, scheduler·반복 loop 불필요

참조: Lipman 2023 (arXiv:2210.02747), Liu 2022 (arXiv:2209.03003). 로컬 구현체 `/home/robotics/Competition/grasp_model/graspflow/` 상당 부분 재사용 가능.

## 환경 상수

- 입력: depth (1, 720, 1280) float32, uv (2,) pixel
- 출력: g = (x, y, z, qw, qx, qy, qz) ∈ ℝ^7 (6-DoF SE(3))
  - 간이 파라미터화 (학습용): `g_train = (x, y, z, a_x, a_y, a_z, sin(yaw_app), cos(yaw_app))` ∈ ℝ^8
    - approach unit vector + yaw-around-approach sin/cos. quat으로 재변환은 loss 외부에서.
    - Quat을 직접 학습하면 double-cover / unit 제약 문제 번거로움
- GPU: RTX 4070 12GB

## 아키텍처

```
Depth (1, H, W)                        
   ↓ clip [0.3, 1.5]/1.5               
   ↓ ResNet-18 encoder → feat_img (C,H',W')
   ↓ Gaussian pool @ (u,v) 5×5         → feat_uv (C,)
   ↓ MLP 2→64 uv Fourier                → feat_uv' (C',)
   ↓ cat → cond (C + C',)

Flow time t ~ U(0,1), noise g_0 ~ N(0, I) ∈ ℝ^8
   ↓ interpolate g_t = (1-t)·g_0 + t·g_1  (Euclidean for 간이 파라미터화)
   ↓ MLP [g_t (8) + cond + t_emb (64)] 
      → Linear 512 → 512 → 512 → 8
   ↓ Output v_θ  (predicted velocity at g_t)

Loss: ‖v_θ − (g_1 − g_0)‖²
```

**왜 Riemannian Flow Matching (SE(3) manifold) 아님?** Quaternion 대신 (approach unit + sincos yaw)로 파라미터화하면 수학이 단순 Euclidean에 가까워짐. unit 제약은 output에서 normalize로 강제. SO(3) geodesic은 후속 개선 대상.

**참고**: GraspFlow 원본은 SE(3) Riemannian. 우리는 정책 단순화로 (approach = (0,0,1) or horizontal) 때문에 간이 파라미터화가 효율적. 성능 아쉬우면 Riemannian으로 전환.

## 입력 해상도

720×1280 유지 (ResNet-18 CNN은 충분 여유). OOM이면 360×640 bilinear.

## Loss (Rectified Flow, 8D 파라미터화)

```python
def flow_matching_loss(model, depth, uv, g1_8d):
    """
    g1_8d: (B, 8) = [x, y, z, a_x, a_y, a_z, sin_yaw, cos_yaw] from GT
    """
    B = g1_8d.size(0)
    g0 = torch.randn(B, 8, device=g1_8d.device)      # 초기 분포
    t = torch.rand(B, device=g1_8d.device)            # U(0,1)
    g_t = (1 - t[:, None]) * g0 + t[:, None] * g1_8d  # 선형 내삽
    v_target = g1_8d - g0                             # 직선 벡터
    v_pred = model(depth, uv, g_t, t)                 # (B, 8)
    return F.mse_loss(v_pred, v_target)
```

**Soft augmentation**: `g1_8d`에 `+ N(0, 0.003)` (xyz) + `yaw ± U(-3°, 3°)`로 데이터 증강 8× → ~130k 유효 샘플.

## Multi-modal 처리 = Unrolled GT

데이터셋에 한 object의 N grasp이 모두 개별 row로 저장. 매 epoch 같은 (depth, uv) 입력에 대해 **매번 다른 GT grasp이 페어링**되도록 sampler 구성 → 모델이 자연스럽게 다봉 분포 학습.

```python
# same object의 grasp들은 object_ref로 묶임. 배치 구성 시 shuffle해 mode 편향 방지.
```

## 추론 (1-step Reflow)

```python
def sample_grasps(model, depth, uv, N=32):
    """32개 다양한 grasp 후보 병렬 생성."""
    g0 = torch.randn(N, 8, device=depth.device)       # 다른 noise
    # 1-step: 실험적으로 충분. 부족하면 Euler 4-step
    v = model(depth.expand(N, -1, -1, -1), uv.expand(N, -1), g0, torch.zeros(N, device=...))
    g1 = g0 + v
    return g1    # (N, 8) — 추후 쿼터니언 + filter
```

## 7D 변환 (pose_cam)

```python
def to_pose_cam(g_8d):
    """(x,y,z,ax,ay,az,sy,cy) → (x,y,z,qw,qx,qy,qz)"""
    pos = g_8d[..., :3]
    approach = g_8d[..., 3:6] / g_8d[..., 3:6].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    yaw = torch.atan2(g_8d[..., 6], g_8d[..., 7])
    R = build_rotation_matrix(approach, yaw)   # tool Z = approach, Y = open, X = Y×Z
    q = rotation_matrix_to_quat_wxyz(R)
    return torch.cat([pos, q], dim=-1)
```

## Dataset / DataLoader

```python
class GraspDataset(Dataset):
    def __init__(self, h5_path, split="train"):
        self.f = h5py.File(h5_path, "r")
        self.g = self.f[f"/{split}"]
        self.depths = self.g["depths"]
        self.depth_ref = self.g["depth_ref"][:]
        self.uvs = self.g["uvs"][:]
        self.grasps = self.g["grasps_cam"][:]            # (N, 7) quat
        self.approach = self.g["approach_vec"][:]         # (N, 3)
        self.yaw = self.g["yaw_around_app"][:]            # (N,)
        self.modes = self.g["object_mode"][:]

    def __getitem__(self, i):
        depth = np.clip(self.depths[self.depth_ref[i]], 0.3, 1.5) / 1.5
        uv = self.uvs[i].astype(np.float32)              # pixel, 정규화는 모델 내부
        g1_8d = np.concatenate([
            self.grasps[i, :3],                          # xyz
            self.approach[i],                             # approach 3D
            [np.sin(self.yaw[i]), np.cos(self.yaw[i])],
        ]).astype(np.float32)
        return (
            torch.from_numpy(depth)[None].float(),
            torch.from_numpy(uv).float(),
            torch.from_numpy(g1_8d).float(),
            int(self.modes[i]),
        )
```

## 평가 메트릭

- **Position MAE** (mm)
- **Rotation geodesic MAE** (deg): GT R vs pred R의 SO(3) 거리
  - `angle = arccos((trace(R_gt^T R_pred) - 1) / 2)`
- **Grasp Success Proxy**: predicted sample이 GT set 중 하나에 근접 (pos<5mm, rot<10°)
- **Mode coverage**: 32 sample 중 각 GT mode(top-down/side-cap/lying/cube)에 매칭되는 비율
  - Mode 다봉 학습이 제대로 됐는지 확인용

## 학습 루프

```python
model = GraspFlowNet(backbone="resnet18", cond_dim=512, g_dim=8).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

for epoch in range(100):
    model.train()
    for depth, uv, g1, mode in train_loader:
        depth, uv, g1 = depth.cuda(), uv.cuda(), g1.cuda()
        loss = flow_matching_loss(model, depth, uv, g1)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    scheduler.step()
    
    if epoch % 5 == 0:
        val_metrics = evaluate_multi_sample(model, val_loader, N_samples=32)
        save_if_best(model, val_metrics)
```

## 재현성·로그

- seed 고정, config.yaml (schema_version=v2, flow_matching_rev=1, etc.)
- JSONL per-epoch (loss mean, val metrics per-mode)

## ONNX 호환

- velocity MLP만 export (convnet + flow head 묶음)
- 커스텀 CUDA op 금지
- `torch.randn` (노이즈) 과 step 루프는 **MATLAB에서 실행** (ONNX는 v_θ forward만)

## 흔한 실패

| 증상 | 원인 | 대응 |
|---|---|---|
| 모든 sample 똑같음 (diversity 없음) | condition이 과도하게 dominate | cond MLP 축소, dropout, g_0 분산 ↑ |
| GT와 동떨어진 sample | Flow 수렴 부족 | epoch ↑, lr ↓, Reflow 2차 시도 |
| approach unit norm 이탈 | 정규화 누락 | model output에 normalize layer |
| val ≫ train | scene leak or 8-aug 편향 | curator에 scene_split 재확인 |
| quaternion norm 이탈 (후처리) | R→quat 변환 수치 오류 | Shepperd method, SVD cleanup |
