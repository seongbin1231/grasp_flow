# Experiment Playbook — IEEE 급 실험 즉시 실행 가이드

**작성일**: 2026-04-28 (2026-04-28 20:30 갱신: 3개 fact-check 에이전트로 install/dataset/training 명령 검증)
**목적**: 사용자가 단일 24GB GPU 환경에서 IEEE 급 paper 에 부합하는 실험을 단계별로 즉시 실행 가능하도록 모든 명령·셋팅·일정 정리
**전제**: Linux Ubuntu / Python 3 conda base / CUDA 12.x / 자체 grasp_v2.h5 보유 / v7_v4policy_big 학습 완료

---

## 🚨 ERRATA — 2026-04-28 검증으로 발견된 오류 (이 섹션 먼저 읽기)

이 playbook 본문에 다수 오류가 있어 직접 copy-paste 시 작동 안 함. 아래 정정사항 **반드시 먼저 적용**.

### 🔴 베이스라인 인스톨 오류 (8건)

| # | 오류 | 정확한 사실 | 출처 |
|---|---|---|---|
| 1 | TOGNet 리포 URL `liuliu66/TOGNet` | **존재하지 않음** (GitHub 404). 사용자 `liuliu66` 의 8개 리포 중 TOGNet 없음 | GitHub API 404 |
| 2 | Contact-GraspNet 라이선스 = NVIDIA NC | **MIT** (LICENSE 파일 verbatim "Permission is hereby granted, free of charge…") | [LICENSE](https://github.com/NVlabs/contact_graspnet/blob/main/LICENSE) |
| 3 | TF 1.15 + CUDA 11.0 | **TF 2.2 + CUDA 10.1/11.1** (env.yml: `python=3.7.9, tensorflow-gpu==2.2.0`) | env.yml |
| 4 | Contact-GraspNet Docker 제공 | **Docker 없음** (conda env 만) | README |
| 5 | Contact-GraspNet `--segmap_id` 플래그 | **존재하지 않음**. segmap 은 `.npz/.npy` 의 `'segmap'` 키에서 읽음. 실제: `python contact_graspnet/inference.py --np_path=test_data/*.npy --local_regions --filter_grasps` | README |
| 6 | GraspGen ckpt = GitHub releases | **HuggingFace** [`adithyamurali/GraspGenModels`](https://huggingface.co/adithyamurali/GraspGenModels) | README |
| 7 | SE(3)-DiF ckpt = Google Drive | **HuggingFace** [`camusean/grasp_diffusion`](https://huggingface.co/camusean/grasp_diffusion). Drive 는 데이터만 | README |
| 8 | GraspLDM 인스톨 / 추론 스크립트 | 추론 = `tools/generate_grasps.py` (not `infer.py`). `--partial` 플래그 없음 — 실험 폴더명으로 분기 | README |

### 🔴 데이터셋 명령 오류 (5건)

| # | 오류 | 정확한 사실 |
|---|---|---|
| D1 | GraspNet-1B 97 GB | **~131 GB core** (train 66.3 GB + test 59 GB + labels 2.3 GB + models 4.3 GB). 옵션 (rect 12.5 + dex 8.9) 추가 시 ~152 GB |
| D2 | scene_0000 1 씬만 다운 가능 | **불가**. 최소 단위 = `train_1.zip 20 GB` (그 안에 scene_0000 포함). per-scene zip 미존재 |
| D3 | ShapeNetSem 30 GB | **12.2 GB** (HF mirror `ShapeNet/ShapeNetSem-archive`, 12,000 model) |
| D4 | tolerance label `python dataset/generate_tolerance_label.py --dataset_root X` | **`cd dataset && sh command_generate_tolerance_label.sh`** (shell 래퍼 사용) |
| D5 | graspnetAPI 호출 `GraspNet(camera, split)` | **`GraspNet(root, camera, split)`** — `root` 가 첫 인자 (필수) |

### 🔴 학습/추론 코드 오류 (10건)

| # | 오류 | 정확한 사실 (소스 라인 검증) |
|---|---|---|
| T1 | `--train_h5 datasets/...` | **`--h5 datasets/...`** ([train_flow.py:43](scripts/train_flow.py#L43)) |
| T2 | `--pretrained <path>` | ✅ **해결됨 (2026-04-29)** — `train_flow.py` 에 추가 완료. v7 ckpt warm-start 검증 (missing=0/unexpected=0). |
| T3 | `--multiscale_local` 플래그 | **값을 받는 STRING**: `--multiscale_local 96,192,384` ([train_flow.py:74](scripts/train_flow.py#L74)) |
| T4 | `--lr` 기본 1e-3 | **3e-4** ([train_flow.py:48](scripts/train_flow.py#L48)) |
| T5 | `--epochs` 기본 250 | **30** ([train_flow.py:45](scripts/train_flow.py#L45)) |
| T6 | `--ema_decay` 기본 0.9998 | **0.999** ([train_flow.py:67](scripts/train_flow.py#L67)) — 사용자 0.9998 설정 OK |
| T7 | `mode_emb = nn.Embedding(4, dim)` 변경 | **현재 모델에 mode_emb 없음**. `aux_mode_head=nn.Linear(cond_dim, 3)` 만 존재 ([flow_model.py:314](src/flow_model.py#L314)). mode_id 는 model `forward()` 입력 아님 — sampler/aux loss 에만 사용 |
| T8 | collate `mode_id_safe` 처리 | **존재하지 않음**. flow_dataset.py 에 `-1` sentinel / torch.where 없음 |
| T9 | graspnetAPI eval `ap = ge.eval_seen(...)` | **튜플 반환**: `res, ap = ge.eval_seen(dump_folder, proc=8)` |
| T10 | `GraspGroup.dump(path)` | **존재 안 함**. `gg.save_npy(path)` 사용. 로드 = `GraspGroup.from_npy(path)` |

### ✅ 검증 완료 (이 항목들은 그대로 사용 OK)
- graspnet-baseline 라이선스 Non-Commercial Research Only — 정확
- graspnetAPI 라이선스 MIT — 정확
- 사전학습 ckpt `checkpoint-rs.tar` + `checkpoint-kn.tar` Drive URL — 정확
- `command_demo.sh`, `command_test.sh`, `command_generate_tolerance_label.sh` — 정확
- pointnet2 + knn 별도 setup — 정확
- ACRONYM `acronym.tar.gz` 1.6 GB Drive — 정확
- `Scene.random_arrangement(meshes, support, distance_above_support, gaussian)` — 정확
- `load_grasps()` 반환 (2000,4,4) + (2000,) — 정확
- `meta['poses']` (3,4,n) — 정확
- `--xattn`, `--cond_dropout`, `--marker_boost`, `--spam_boost`, `--rot_loss_weight`, `--symmetry_loss` 모두 존재 — 정확
- v7 ckpt 경로 `runs/yolograsp_v2/v7_v4policy_big/adaln_zero_lr0.001_nb12_h1024/checkpoints/best.pt` — 정확
- `scripts/export_onnx.py` 존재, `--ckpt` 플래그 사용 — 정확

### 🟡 v9 ONNX export 추가 주의

`scripts/export_onnx.py` 가 v6 (cond_dim=256) 가정으로 작성됐을 가능성. v8/v9 (xattn + multiscale_local) 사용 시 **cond_dim = 128 + 128×3 = 512** 로 바뀜. export 전 [export_onnx.py:95-155](scripts/export_onnx.py#L95) 의 FlowGraspNet 재구성 부분에 `use_xattn`, `multiscale_local_scales` flag 가 cfg 에서 round-trip 되는지 확인 필수.

### 🔴 정정된 Stage 1 학습 명령 (실제 동작 버전)

```bash
# Stage 1 학습 (단, --pretrained 는 train_flow.py 에 직접 추가해야 Stage 2/3 가능)
/home/robotics/anaconda3/bin/python scripts/train_flow.py \
    --run runs/yolograsp_v2/v9_stage1_acronym \
    --h5 datasets/acronym_v1.h5 \
    --epochs 80 --batch_size 16 --lr 1e-3 \
    --warmup_frac 0.04 --ema_decay 0.9998 --cond_dropout 0.5 \
    --block adaln_zero --n_blocks 12 --hidden 1024 \
    --xattn --multiscale_local 96,192,384 --scale_dropout 0.5 \
    --weight_decay 0.05 \
    --seed 42 --wandb --wandb_project yolograsp-v2-multidata
```

### 🔴 정정된 graspnetAPI eval 호출

```python
from graspnetAPI import GraspNetEval, GraspGroup
import numpy as np

# 추론 결과 저장 — 17-col GraspGroup, save_npy 사용
arr_17col = np.zeros((N, 17), dtype=np.float32)
arr_17col[:, 0]    = scores       # score
arr_17col[:, 1]    = widths       # width
arr_17col[:, 2]    = 0.02         # height (fixed)
arr_17col[:, 3]    = 0.02         # depth  (fixed)
arr_17col[:, 4:13] = R_flat       # 9D rotation matrix
arr_17col[:, 13:16]= t_xyz        # translation
arr_17col[:, 16]   = obj_id
gg = GraspGroup(arr_17col)
gg.save_npy(f'runs/v9_predictions/scene_{sid:04d}/realsense/{ann:04d}.npy')

# 평가 — 튜플 반환
ge = GraspNetEval(root='/data/graspnet1b', camera='realsense', split='test')
res_seen,    ap_seen    = ge.eval_seen(   dump_folder='runs/v9_predictions/', proc=8)
res_similar, ap_similar = ge.eval_similar(dump_folder='runs/v9_predictions/', proc=8)
res_novel,   ap_novel   = ge.eval_novel(  dump_folder='runs/v9_predictions/', proc=8)
print(f"AP: seen={ap_seen:.2f}, similar={ap_similar:.2f}, novel={ap_novel:.2f}")
```

### 🔴 정정된 첫 1주 첫째날 명령

```bash
# ① 환경 OK
df -h /home/robotics; nvidia-smi
mkdir -p paper_notes datasets/external img_dataset/external scripts/{graspnet1b,acronym,baselines}

# ② Metric 인용 정정 (1h)
# scripts/make_paper_table1.py + paper 본문: "5cm/30° threshold inspired by Achlioptas 2018"

# ③ GraspNet-1B sanity — train_1.zip 20 GB 다운 (per-scene 다운 불가)
mkdir -p datasets/external/graspnet1b
cd datasets/external/graspnet1b
# graspnet.net/datasets.html 에서 train_1.zip Drive URL 받아서 wget
unzip train_1.zip   # 그 안에 scene_0000 ~ 0024
# graspnetAPI 로 sanity load (정정된 호출):
/home/robotics/anaconda3/bin/python -c "
from graspnetAPI import GraspNet
g = GraspNet(root='datasets/external/graspnet1b', camera='kinect', split='train')
grasps = g.loadGrasp(sceneId=0, annId=0, format='6d')
print('OK', len(grasps))
"

# ④ ACRONYM gripper 시각 검증
mkdir -p /tmp/acronym_check
cd /tmp/acronym_check
git clone https://github.com/NVlabs/acronym
cd acronym && pip install -e .
# acronym.tar.gz 1.6 GB Drive 다운 후
tar xzf acronym.tar.gz
/home/robotics/anaconda3/bin/python -c "
from acronym_tools import load_grasps, create_gripper_marker
import trimesh
T, q = load_grasps('grasps/Bottle_<sha>_0.0095.h5')
gripper = create_gripper_marker()
for g in T[q.astype(bool)][:5]:
    g_t = gripper.copy().apply_transform(g)
    trimesh.Scene([g_t]).show()
"
# 손가락 끝이 z=+0.1122 위치인지 시각 확인
```

---

본문은 historical 기록으로 유지. **위 ERRATA 가 우선**.

---

## 0. IEEE 급 실험 기준 (이 playbook 이 만족하는 것)

| 기준 | 충족 방법 |
|---|---|
| 다중 데이터셋 | 자체 + GraspNet-1B + ACRONYM (3개) |
| 충분한 베이스라인 | Direct + 6 외부 모델 (3 family 균형) |
| 표준 metric | Pos MAE / Ang Err / **COV** / **APD** + 공식 **AP** |
| Ablation | 컴포넌트별 (Cross-Attn / Multi-Scale / RF vs DDPM) |
| 통계적 유의성 | seed 3개 평균 ± std (시간 허용 시) |
| Real-world | UR5e + RealSense 정성 (옵션) |
| 재현성 | 모든 ckpt + 메트릭 스크립트 + Docker 공개 |

---

## 1. 환경 셋업 (30분, 1회)

```bash
cd /home/robotics/Competition/YOLO_Grasp

# 디스크 ~150GB 여유 확인
df -h /home/robotics

# 작업 디렉터리
mkdir -p paper_notes datasets/external img_dataset/external
mkdir -p scripts/{graspnet1b,acronym,baselines}
mkdir -p runs/baseline_eval

# Docker 권장 (TF1.15 + theseus 빌드 회피)
docker --version || sudo apt install docker.io
nvidia-smi  # 24GB 확인
```

### 공통 conda env (자체 모델 + uv_adapter 용)
이미 보유 (conda base). 신규 의존성:
```bash
/home/robotics/anaconda3/bin/pip install graspnetAPI open3d trimesh pyrender h5py scipy
```

---

## 2. 데이터셋 셋업 (3개)

### 2-1. 자체 grasp_v2.h5 — ✅ done

이미 학습/평가 완료. 추가 작업 없음.

### 2-2. GraspNet-1B (97 GB, ~6h 다운 + ~30h 변환)

#### Step A — 1 씬 sanity check (4시간)
```bash
# 작은 1 씬만 먼저 받아서 데이터 구조 확인
mkdir -p datasets/external/graspnet1b/scenes
cd datasets/external/graspnet1b
# wget URL 은 https://graspnet.net/datasets.html 에서 scene_0000 zip 만
# (Google Drive 또는 Baidu Pan 경로)
```

#### Step B — 변환 스크립트 6개 작성 (5일)

| 스크립트 | 핵심 코드 | LoC |
|---|---|---|
| `scripts/graspnet1b/extract_uv.py` | `seg = Image.open(label_png); for inst_id, oid in enumerate(meta['cls_indexes'].flatten(), 1): mask = (seg==inst_id); u,v = xs.mean(), ys.mean()` | 120 |
| `scripts/graspnet1b/flatten_grasps.py` | `views = generate_views(300); angles = offsets[...,0]; depths = offsets[...,1]; widths = offsets[...,2]; mask = (fric>0)&(fric<=0.4)&(~coll)` | 180 |
| `scripts/graspnet1b/obj_to_cam.py` | `T = np.eye(4); T[:3,:] = poses[:,:,i]; t_cam = T[:3,:3]@t_obj.T + T[:3,3]; q_cam = mat2quat_wxyz(T[:3,:3]@R_obj)` | 110 |
| `scripts/graspnet1b/k_warp.py` | `if 'kinect' in path: K_src = (631.55, 631.21, 638.43, 366.50) else: K_src = (927.17, 927.37, 651.32, 349.62); cv2.warpAffine for canonical K=(1109,1109,640,360)` | 80 |
| `scripts/graspnet1b/build_h5.py` | unrolled rows, `mode_id=-1` (unknown bucket), per-row K, dataset_tag | 250 |
| `scripts/graspnet1b/split.py` | scene 0-99 train, 100-129 seen, 130-159 similar, 160-189 novel | 40 |

#### Step C — 모델 1줄 변경 (5분)
```python
# src/flow_model.py
self.mode_emb = nn.Embedding(4, mode_dim)  # 0=lying, 1=standing, 2=cube, 3=unknown(GraspNet)
# scripts/train_flow.py collate
mode_id_safe = torch.where(mode_id == -1, 3, mode_id)
```

#### 산출물
- `datasets/graspnet1b_v1.h5` — ~12M rows, ~50GB
- 함정: `meta['poses']` 가 (3,4,n) 모양 — `[0,0,0,1]` 직접 append 필수

### 2-3. ACRONYM (1.6 GB grasp + 30 GB ShapeNetSem, ~3~5일 dev + 6h render)

#### Step A — 다운 (1시간)
```bash
mkdir -p datasets/external/acronym
cd datasets/external/acronym
# acronym.tar.gz 1.6 GB from Drive (https://github.com/NVlabs/acronym README)
tar xzf acronym.tar.gz
# ShapeNetSem 별도 등록: https://www.shapenet.org → academic free
```

#### Step B — Manifold 워터타이트 메쉬 (4시간 1회)
```bash
git clone https://github.com/hjwdzh/Manifold && cd Manifold && mkdir build && cd build
cmake .. && make
# 각 ShapeNetSem 메쉬에 적용:
#   manifold model.obj temp.watertight.obj -s
#   simplify -i temp.watertight.obj -o model.obj -m -r 0.02
```

#### Step C — 변환 스크립트 (3일)

| 스크립트 | 핵심 |
|---|---|
| `scripts/acronym/install_check.py` | grasp h5 키 (`grasps/transforms`, `grasps/qualities/flex/object_in_gripper`, `object/file`, `object/scale`) 검증 + Manifold 동작 확인 |
| `scripts/acronym/patch_renderer.py` | `pyrender.IntrinsicsCamera(fx=1109, fy=1109, cx=640, cy=360)` + 1280×720 + per-target instance mask 루프 |
| `scripts/acronym/render_scenes.py` | 5,000 씬 × 2 view (top-down 50° + oblique 30°) — 5.5h on 1 GPU |
| `scripts/acronym/extract_grasps.py` | `T_cam = inv(T_w_cam) @ T_w_obj @ T_obj_grasp`. **⚠️ gripper 원점 보정**: ACRONYM origin 은 mount, 손가락 끝은 z+0.1122m. 우리 grasp_v2 convention 과 맞추기 |
| `scripts/acronym/build_h5.py` | grasp_v2 schema 매칭. mode_id 는 PCA cluster (lying/standing/cube/other) |

#### Step D — 카테고리 subset

```python
# 매칭 5-class (Table 1 비교용)
CATEGORIES = {
    'bottle': ['Bottle', 'WineBottle'],          # ~250 obj
    'can':    ['Can', 'SodaCan', 'BeerCan'],     # ~90
    'marker': ['Marker', 'Pen', 'Pencil'],       # ~80
    'spam':   ['FoodItem', 'CannedFood', 'TinCan'],  # ~110
    'cube':   ['Box', 'Cube', 'Dice']            # ~70
}
# 합 ~600 obj × ~22% FleX-success × 2000/obj ≈ 12,000 grasp

# Generalization eval (Table 3 unseen)
HELDOUT = ['Mug', 'Bowl', 'Camera', 'Phone', 'ToyCar', 'Vase', ...]  # top-20 by count
```

#### 함정 (paper-killer)
- ⚠️ **gripper 원점**: mount 위치임, 손가락 끝 가정 시 11.2cm 빗나감
- PyRender headless: `PYOPENGL_PLATFORM=egl` env var 필수
- ShapeNetSem 메쉬 5-10% non-watertight → fail object skip

---

## 3. 베이스라인 셋업 (7개)

각 베이스라인은 **별도 conda env / Docker container** 권장 (의존성 충돌 회피).

### 3-0. 공통 — `(u,v) adapter` (1일)
```python
# scripts/baselines/uv_adapter.py — 모든 PC 입력 베이스라인이 사용
def uv_to_segmented_pc(depth_mm, uv, K, yolo_mask=None, radius_px=80):
    if yolo_mask is None:
        u, v = uv
        ys, xs = np.mgrid[0:depth_mm.shape[0], 0:depth_mm.shape[1]]
        mask = (xs - u)**2 + (ys - v)**2 < radius_px**2
    else:
        mask = yolo_mask
    z = depth_mm[mask].astype(np.float32) / 1000.0
    x = (xs[mask] - K[0,2]) * z / K[0,0]
    y = (ys[mask] - K[1,2]) * z / K[1,1]
    pc = np.stack([x, y, z], axis=-1)
    pc = pc[(pc[:,2] > 0.2) & (pc[:,2] < 1.5)]
    return pc  # (N, 3) — 각 베이스라인 입력
```

### 3-1. Direct MLP (자체) — done
[src/direct_model.py](src/direct_model.py) + [runs/yolograsp_v2/v7_direct_mlp_big](runs/yolograsp_v2/v7_direct_mlp_big) ckpt 사용. 0h.

### 3-2. GraspLDM ⭐ 가장 쉬움 (1.5일)
**License**: Apache 2.0 ✅ | **Repo**: [kuldeepbrd1/graspLDM](https://github.com/kuldeepbrd1/graspLDM)
```bash
git clone https://github.com/kuldeepbrd1/graspLDM
cd graspLDM
docker build -t graspldm .   # Devcontainer 제공
docker run --gpus all -v $(pwd):/workspace -v /home/robotics/Competition/YOLO_Grasp/img_dataset:/data graspldm

# HuggingFace 사전학습 ckpt
huggingface-cli download kuldeepbarad/GraspLDM --local-dir checkpoints

# 우리 val 추론 (uv_adapter 사용)
python scripts/eval_on_yolograsp.py \
    --val_h5 /data/grasp_v2_val.h5 \
    --ckpt checkpoints/partial_pc.ckpt \
    --num_samples 32 --output runs/graspldm_eval.json
```
**예상 시간**: install 3h + eval 3h = **6h**

### 3-3. TOGNet ⭐ 픽셀 prompt 직접 비교 (2일)
**License**: MIT ✅ | **Repo**: [liuliu66/TOGNet](https://github.com/liuliu66/TOGNet)
```bash
git clone https://github.com/liuliu66/TOGNet
cd TOGNet
conda create -n tognet python=3.8 && conda activate tognet
pip install -r requirements.txt
# graspnetAPI + PointNet2 ops
cd pointnet2 && python setup.py install
# 사전학습 ckpt 다운 (repo README → Drive)

# 우리 val 추론 — TOGNet 은 (u,v) prompt 받음, adapter 불필요
python infer.py --depth_dir img_dataset/captured_images_depth/random6_dep \
                --uv_csv runs/yolo_uv_random6.csv \
                --ckpt checkpoints/tognet.pth
```
**예상 시간**: 14h (PointNet2 빌드 포함)

### 3-4. SE(3)-DiffusionFields (2일)
**License**: MIT ✅ | **Repo**: [robotgradient/grasp_diffusion](https://github.com/robotgradient/grasp_diffusion)
```bash
git clone https://github.com/robotgradient/grasp_diffusion
cd grasp_diffusion
# theseus 빌드 어려움 → Docker 권장
docker build -t se3diff .   # 기존 Dockerfile 있음
# 사전학습 ckpt: Google Drive (repo issue #19 미러 끊김 가능, fork 미러 사용)

# (u,v) → segmented PC → 추론
python scripts/sample.py --pc_input <segmented_pc> --num_samples 32
```
**함정**: theseus CUDA 12 빌드, headless EGL → Docker 강제
**예상 시간**: 16h

### 3-5. 6-DoF GraspNet (PyTorch port, 2일)
**License**: NVIDIA NC ★ | **Repo**: [jsll/pytorch_6dof-graspnet](https://github.com/jsll/pytorch_6dof-graspnet) (TF1.12 dodge)
```bash
git clone https://github.com/jsll/pytorch_6dof-graspnet
cd pytorch_6dof-graspnet
conda create -n grasp6dof python=3.8 && conda activate grasp6dof
pip install -r requirements.txt
# 사전학습 ckpt: ACRONYM 학습된 것 repo README

python eval.py --pc <segmented_pc> --num_grasps 32 --score_thresh 0.5
```
**Footnote 의무**: NVIDIA Source Code License
**예상 시간**: 12h

### 3-6. Contact-GraspNet (TF1.15 Docker, 3일)
**License**: NVIDIA NC ★ | **Repo**: [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet)
```bash
git clone https://github.com/NVlabs/contact_graspnet
cd contact_graspnet
docker build -t contactgrasp -f docker/Dockerfile .   # TF1.15 + CUDA 11
docker run --gpus all -v $(pwd):/work -v /home/robotics/Competition/YOLO_Grasp:/yolograsp contactgrasp

# 사전학습 ckpt: scene_2048_bs3_hor_sigma_001 등 (repo README)
python contact_graspnet/inference.py \
    --np_path /yolograsp/img_dataset/captured_images_depth/sample.npy \
    --K "[[1109,0,640],[0,1109,360],[0,0,1]]" \
    --segmap_id 1 \
    --forward_passes 5
```
**함정**: TF1.15 / protobuf / CUDA 11.0 → Docker 필수
**예상 시간**: 18h

### 3-7. GraspGen (NVIDIA 2025, 신 SOTA, 2일)
**License**: ⚠️ NVIDIA NC ★ | **Repo**: [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
```bash
git clone https://github.com/NVlabs/GraspGen
cd GraspGen
pip install -r requirements.txt   # 모던 PyTorch, CUDA 12 OK
# 사전학습 ckpt: repo releases 또는 project page

python infer.py --pc <segmented_pc> --num_samples 32
```
**예상 시간**: 12h

### 3-8. graspnet-baseline (T2 평가용 보조, 2일)
**License**: ⚠️ Non-Commercial Research Only | **Repo**: [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
```bash
git clone https://github.com/graspnet/graspnet-baseline
cd graspnet-baseline
pip install -r requirements.txt
cd pointnet2 && python setup.py install && cd ..
cd knn && python setup.py install && cd ..

# 사전학습 ckpt
wget <Drive_URL_for_checkpoint-rs.tar>   # README 참조
wget <Drive_URL_for_checkpoint-kn.tar>

# Tolerance label
python dataset/generate_tolerance_label.py --dataset_root /data/graspnet1b --num_workers 16

# 임의 RGB-D 추론 데모
bash command_demo.sh   # checkpoint_path 만 수정
```
**용도**: Table 2 (GraspNet-1B test_*) 의 reference 베이스라인 — 우리 모델과 동일 데이터에서 비교
**예상 시간**: 16h

### 3-9. 베이스라인 통합 결과 포맷
모든 베이스라인 출력 → 통일 JSON:
```json
{
  "scene_id": "random6_31",
  "object_ref": 2,
  "uv": [640, 360],
  "grasps_cam": [[x,y,z, qw,qx,qy,qz], ...],   // N=32
  "scores": [...]
}
```

---

## 4. 학습 계획 (3-stage progressive)

### Stage 1 — ACRONYM pretrain (~22h, 80 ep)
```bash
/home/robotics/anaconda3/bin/python scripts/train_flow.py \
    --run runs/yolograsp_v2/v9_stage1_acronym \
    --train_h5 datasets/acronym_v1.h5 \
    --epochs 80 --batch_size 16 --lr 1e-3 \
    --warmup_frac 0.04 --ema_decay 0.9998 --cond_dropout 0.5 \
    --block adaln_zero --n_blocks 12 --hidden 1024 \
    --xattn --multiscale_local --scale_dropout 0.5 \
    --weight_decay 0.05 \
    --seed 42 --wandb --wandb_project yolograsp-v2-multidata
```

### Stage 2 — GraspNet-1B finetune (~8h, 40 ep, lr×0.1)
```bash
/home/robotics/anaconda3/bin/python scripts/train_flow.py \
    --run runs/yolograsp_v2/v9_stage2_graspnet \
    --train_h5 datasets/graspnet1b_v1.h5 \
    --pretrained runs/yolograsp_v2/v9_stage1_acronym/best.pt \
    --epochs 40 --batch_size 16 --lr 1e-4 \
    --cond_dropout 0.2 \
    --seed 42 --wandb
```

### Stage 3 — grasp_v2 finetune (~5h, 60 ep, lr×0.01, mode_id 활성화)
```bash
/home/robotics/anaconda3/bin/python scripts/train_flow.py \
    --run runs/yolograsp_v2/v9_stage3_yolograsp \
    --train_h5 datasets/grasp_v2.h5 \
    --pretrained runs/yolograsp_v2/v9_stage2_graspnet/best.pt \
    --epochs 60 --batch_size 16 --lr 1e-5 \
    --marker_boost 1.5 --spam_boost 2.5 \
    --symmetry_loss --rot_loss_weight 2.0 \
    --seed 42 --wandb
```

### 추가 학습 변형 (ablation 행)
```bash
# Variant A: ACRONYM-only (no Stage 2/3) — generalization 단독 측정
# Variant B: grasp_v2-only (= 기존 v7) — single-dataset baseline
# Variant C: Joint training with dataset_id FiLM (옵션)
```

총 컴퓨팅: **~35h GPU** (single 24GB)

---

## 5. 평가 프로토콜

### 5-1. 통합 metric 스크립트 (`scripts/eval_table_unified.py`)

```python
# 입력: model_output (N=32 grasps), GT_grasps_for_uv
# 출력: Pos MAE / Ang Err / COV / APD per-mode

def compute_metrics(pred_grasps_N7, gt_grasps_M7, pos_thresh_m=0.05, ang_thresh_deg=30):
    # 1) 가장 가까운 GT 찾기 (per pred)
    # 2) Pos MAE = mean nearest distance (cm)
    # 3) Ang Err = mean nearest geodesic angle (deg)
    # 4) COV = % of GT for which any pred is within (5cm, 30°) — Achlioptas-modified
    # 5) APD = mean pairwise distance among preds (cm) — diversity
    return {'pos_mae_cm': ..., 'ang_err_deg': ..., 'cov': ..., 'apd_cm': ...}
```

### 5-2. 4 Table 정의

#### Table 1 — 자체 val (98 씬, 5 클래스, primary)
| 행 | 열 |
|---|---|
| Direct MLP / GraspLDM / TOGNet / SE(3)-DF / 6-DoF GN / Contact-GN / GraspGen / **Ours v7** / **Ours v9 (Stage 3)** | Pos MAE / Ang Err / COV / APD × per-mode (lying/standing/cube) |

#### Table 2 — GraspNet-1B 실세계 (test_seen/similar/novel)
| 행 | 열 |
|---|---|
| Contact-GN / 6-DoF GN / **graspnet-baseline** / GraspGen / **Ours v9** | **공식 AP** (3 splits) + Pos MAE + COV |

공식 AP 평가:
```python
from graspnetAPI import GraspNetEval
ge = GraspNetEval(graspnet_root, camera='realsense', split='test_seen')
ap = ge.eval_seen(dump_folder='runs/v9_graspnet_predictions/', proc=8)
```

#### Table 3 — ACRONYM 일반화
| 행 (학습 pool) | 열 |
|---|---|
| 5 obj / 100 obj / 1k obj / 8.8k obj | Pos MAE / Ang Err / COV / FleX-success |

#### Table 4 — Ablation
| 행 | 열 |
|---|---|
| Direct / w/o cross-attn / w/o multi-scale / DDPM 25-step / **RF 1-step (Ours)** | val_flow / standing-COV / NFE |

### 5-3. Figure 5개

| Fig | 내용 | 상태 |
|---|---|---|
| F1 mode-wise GT | 자체 데이터 mode 분포 | done |
| F2 architecture | encoder + cross-attn + velocity MLP | done |
| F3 Direct vs Flow qualitative | seen 씬 N=16 시각화 | done — extend with GraspNet real |
| F4 ACRONYM scaling curve | X=#train obj log scale, Y=Pos MAE | NEW |
| F5 real-world UR5e | 4 success case 정성 | optional |

---

## 6. 5주 일정 (calendar)

| 주 | 작업 | 컴퓨팅 | dev |
|---|---|---|---|
| **W0** | 환경 셋업 + metric 인용 정정 + 1 씬 sanity (GraspNet, ACRONYM) | 0h | 8h |
| **W1** | GraspNet-1B 변환 6 스크립트 + GraspLDM 시범 1 씬 | 0h | 35h |
| **W2** | 베이스라인 6개 인스톨 + 우리 val 재실행 → **Table 1** | 12h | 25h |
| **W3** | ACRONYM 렌더링 + Stage 1 학습 + GraspNet test_* 평가 → **Table 2** | 28h | 25h |
| **W4** | Stage 2/3 학습 + Table 3 ACRONYM 일반화 + Table 4 ablation | 13h | 20h |

총 **~150h dev + ~53h compute**, single engineer 풀타임 5주.

### De-scoping (시간 부족 시 cut 우선순위)
1. ❌ **F4 ACRONYM scaling curve** — 4 학습 추가 비용
2. ❌ **F5 real UR5e** — 옵션
3. ❌ **GraspGen** — 새 SOTA 지만 NVIDIA NC, 다른 베이스라인 충분
4. ❌ **Contact-GraspNet** — TF1.15 빌드 어려움 시 graspnet-baseline 으로 swap
5. ⚠️ **ACRONYM 전체** — Table 3 cut 시 paper 는 Table 1+2+4 로 IEEE 급 충분

최소 ship 셋: Table 1 + Table 2 + Table 4 = **3주 (~85h dev + 25h compute)**

---

## 7. 함정 + 사전 점검 체크리스트

### 🔴 Paper-killer (반드시 확인)
- [ ] **ACRONYM gripper 원점**: mount 위치 vs 우리 grasp_v2 의 TCP 위치 — `trimesh.show()` 로 시각 검증 (z 차이 11.2cm 보정 필요한지)
- [ ] **GraspNet K**: RealSense (927) vs Kinect (631) 카메라별 분기 처리
- [ ] **`meta['poses']`**: (3,4,n) 모양, `[0,0,0,1]` append 필수
- [ ] **Metric 인용**: "5cm/30° threshold inspired by Achlioptas 2018" — Sundermeyer 2021 misattribution 제거

### 🟡 인스톨 함정
- [ ] PyRender headless: `export PYOPENGL_PLATFORM=egl`
- [ ] theseus (SE(3)-DF): Docker 필수 — CUDA 12 직접 빌드 X
- [ ] TF1.15 (Contact-GN): Docker 필수 — protobuf / numpy 충돌
- [ ] PointNet2 ops (TOGNet, graspnet-baseline): CUDA 11/12 fork 사용

### 🟢 진행 점검
- [ ] 환경 셋업 완료
- [ ] 자체 metric 스크립트 통합 + 정정 표현 적용
- [ ] GraspNet 1 씬 변환 완료 + `graspnetAPI.loadGrasp` 출력 일치 확인
- [ ] ACRONYM 1 객체 grasp 시각화 완료
- [ ] 베이스라인 1개 (GraspLDM) eval 결과 1줄 출력
- [ ] Table 1 (8 행) 빈 셀 0개
- [ ] Stage 1/2/3 ckpt 3개 보관
- [ ] AP eval 자동화 1줄 호출 가능

---

## 8. 참고 — 모든 라이선스 / 사이즈 / 의존성 한눈에

### 데이터셋
| Dataset | 라이선스 | 크기 | 등록 |
|---|---|---|---|
| 자체 grasp_v2.h5 | — | 5 GB | — |
| GraspNet-1B | CC BY-NC-SA 4.0 | 97 GB | 무 |
| ACRONYM | code MIT, data CC BY-NC | 1.6 + 30 GB | ShapeNetSem 학술 |

### 베이스라인 코드
| 모델 | 라이선스 | 사전학습 ckpt | 인스톨 난이도 |
|---|---|---|---|
| Direct MLP (자체) | — | ✅ | 0 |
| GraspLDM | Apache 2.0 ✅ | HF | ⭐ |
| TOGNet | MIT ✅ | Drive | ⭐⭐ |
| SE(3)-DiF | MIT ✅ | Drive (mirror) | ⭐⭐⭐ (theseus) |
| 6-DoF GN (PyTorch port) | MIT-ish (port) / NVIDIA NC (orig) | Drive | ⭐⭐ |
| Contact-GN | NVIDIA NC | Drive | ⭐⭐⭐ (TF1.15) |
| GraspGen | NVIDIA NC | repo releases | ⭐⭐ |
| graspnet-baseline | **Non-Commercial Research Only** | Drive (rs/kn) | ⭐⭐ |

### Footnote 의무 (paper)
> *"GraspNet baseline, Contact-GraspNet, 6-DoF GraspNet, and GraspGen are used under their respective non-commercial research licenses (NVIDIA Source Code License / GraspNet-Baseline Software License Agreement). All comparisons are performed in a non-commercial academic research setting."*

---

## 9. 즉시 시작 — 첫 1주 첫째날

```bash
# ① 환경 (30분)
df -h /home/robotics
nvidia-smi
mkdir -p paper_notes datasets/external img_dataset/external scripts/{graspnet1b,acronym,baselines}

# ② Metric 인용 정정 (1시간)
# scripts/make_paper_table1.py 의 "Achlioptas/Sundermeyer 5cm/30°" 표현 → "inspired by Achlioptas 2018, with our 5cm/30° matching threshold"

# ③ GraspNet-1B 1 씬 sanity (4시간)
# graspnet.net 에서 scene_0000 (kinect) zip 다운 → graspnetAPI 로 (label, depth, meta, grasp) 로드 → 시각화

# ④ ACRONYM gripper 시각 검증 (3시간)
git clone https://github.com/NVlabs/acronym /tmp/acronym
cd /tmp/acronym && pip install -e .
python -c "
from acronym_tools import load_grasps, create_gripper_marker
import trimesh
T, q = load_grasps('grasps/Bottle_<sha>_0.0095.h5')
gripper = create_gripper_marker()
for g in T[q.astype(bool)][:5]:
    gripper_g = gripper.copy().apply_transform(g)
    trimesh.Scene([gripper_g]).show()
"
# 손가락 끝 z=0.1122 위치 확인
```

이 4가지가 W0 의 deliverable. 끝나면 W1 시작.

---

## 부록 — 명령어 요약 카드

```bash
# 학습 1 줄
python scripts/train_flow.py --run <dir> --train_h5 <h5> --pretrained <prev_best.pt> --epochs N --lr LR

# 자체 val 평가 1 줄
python scripts/eval_table_unified.py --ckpt <best.pt> --val_h5 datasets/grasp_v2.h5 --output runs/eval.json

# GraspNet AP 1 줄
python -c "from graspnetAPI import GraspNetEval; GraspNetEval('<root>', 'realsense', 'test_seen').eval_seen('runs/v9_predictions/', proc=8)"

# 베이스라인 (예: GraspLDM)
docker run --gpus all -v $(pwd):/workspace graspldm python scripts/eval.py --ckpt <hf_ckpt> --val_h5 /data/grasp_v2_val.h5
```
