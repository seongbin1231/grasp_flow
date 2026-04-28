# Experimental Setup — 데이터셋 · 베이스라인 · VLM PoC 사용가능성 검증

**작성일**: 2026-04-28
**조사 방식**: 5인 병렬 에이전트 + WebSearch + WebFetch — GitHub 리포지토리 라이선스 / README / requirements / open issues 직접 확인
**사용자 환경**: Linux Ubuntu, conda base Python (`/home/robotics/anaconda3/bin/python`), 단일 GPU 24GB, CUDA 12.x, MATLAB R2025b + Simulink 3D Animation, UR5e + RealSense, 588 sim scenes / 40,984 grasps

---

## ⚠️ 0-A. ERRATA — 2026-04-28 추가 검증 결과 (이 섹션 먼저 읽기)

4 인 병렬 에이전트 추가 검증 (paper PDF + GitHub README + HF dataset card 직접 확인) 결과 **두 가지 중대한 정정** 필요:

### 정정 1: Grasp-Anything-6D 주장 일부 거짓 — primary 에서 제외

**이전 주장 (오류)**:
> "GA-6D 가 단일 픽셀 (u,v) prompt 라벨을 native 로 가지고 있음"
> "라이선스 CC BY-NC"

**정확한 사실** (LGD paper arxiv 2407.13842 + HF dataset card 인용):
- GA-6D 는 **언어 prompt + 3D 마스크** 만 보유, **(u,v) 라벨 없음**. mask centroid 합성 필요 (= native 가 아님)
- 라이선스는 **MIT** (HF card)
- Depth 는 **Stable Diffusion 이미지 + ZoeDepth 추정** (RealSense 와 도메인 갭 큼, paper 정량 검증 X)
- Camera K: 55° FoV / central principal point / **해상도 미명시** — 우리 1280×720/fx=1109 와 안 맞음
- LGD metric (CR / EMD / Collision-Free Rate) ≠ 우리 COV/APD

**조치**: GA-6D **primary benchmark 제외**, **Related Work cite-only**. 차별화 narrative 강화 (오히려 좋음):
> *"Existing large-scale 6-DoF synthetic datasets (Grasp-Anything-6D) provide language prompts but **no per-grasp pixel-level conditioning labels**; we are the first to release pixel-prompted 6-DoF grasp data on metric-depth scenes."*

### 정정 2: Table 1 직접 인용 ❌ 100% 불가능 — 모든 베이스라인 재실행 필수

**이유 (검증 완료)**:
1. **데이터셋 다름**: 베이스라인마다 ACRONYM / GraspNet-1B / FetchBench / GA-6D / custom 각각 다름
2. **Metric 정의 다름**:
   - Achlioptas COV (5cm/30°) ≠ GraspGen "1cm coverage AUC" ≠ LGD "Coverage Rate"
   - 각 paper 의 success rate = 각자의 physics simulator 결과 (FleX vs Isaac Gym vs PyBullet)
   - 우리 Pos MAE / Ang Err / COV / APD 는 베이스라인 paper 들이 **거의 보고 안 함**
3. **입력 modality 다름**: 모두 point cloud, 우리 depth+(u,v) 와 직접 비교 불가
4. **Conditioning 다름**: LGD 는 언어 — incompatible

**Contact-GraspNet 흔한 오해 정정**: 원본 paper 는 GraspNet-1B 에서 평가 안 함 (custom clutter scenes 만). 후속 논문이 사후 측정한 것.

**필수 조치**: **6 개 베이스라인 모두 사전학습 ckpt 로 우리 588 val scene 위에서 재실행** → 같은 metric 계산. 약 **2~3주** 작업.

### 정정 3: 데이터셋별 (u,v)+학습 가능성 정밀화

| Dataset | (u,v) 추출 | 학습 | 변경 |
|---|---|---|---|
| **GraspNet-1B** | ✅ 100% (`label/*.png` per-pixel instance mask) | ✅ 30~50h 변환 | 이전과 동일 — primary benchmark |
| **ACRONYM** | ✅ 단 **렌더링 직접 1~2일** (mesh + grasp pair 만 제공, 이미지 0장) | ✅ 3~5일 변환 (5 신규 스크립트 ~580 LOC) | **Secondary 로 강등** — 표준화된 multi-object eval split 없음, baseline 들 (u,v) 안 받음 |
| ~~**Grasp-Anything-6D**~~ | ❌ native 없음 | ❌ 도메인 갭 검증 안 됨 | **제외**, cite-only |

→ **새 권고**: Primary = 자체 grasp_v2.h5 + GraspNet-1B 두 데이터셋, ACRONYM 은 generalization 보조 study

### 정정 4: 일정 재산정

이전: 5주 (Phase 1~5)
정정: **3~4주** (GA-6D 제거 + 베이스라인 재실행 시간 정확화)

| 주차 | 작업 |
|---|---|
| W1 | GraspNet-1B 다운 + 변환 + 1 씬 sanity check |
| W2 | 베이스라인 ckpt 6개 환경 + 우리 val 재실행 (Direct MLP 자체 + Contact-GraspNet + 6-DoF GraspNet + SE(3)-DF + GraspLDM + GraspGen + TOGNet) |
| W3 | Table 1 통합 메트릭 + (옵션) ACRONYM 렌더링 generalization study |
| W4 | EquiGraspFlow retrain (필요 시) + 논문 Table 갱신 |

### 행동
이 ERRATA 가 [§7 통합 권고](#7-통합-권고-table-1--데이터셋--일정) 와 [§9 실행 todo](#9-실행-todo-단계별-effort) 보다 우선. 본문은 historical 기록으로 유지하되 GA-6D 관련 라인은 무시하고 위 ERRATA 따르기.

---

## 0. 목차

- [1. TL;DR — 권고 셋업](#1-tldr--권고-셋업)
- [2. 데이터셋 — 사용가능성 검증](#2-데이터셋--사용가능성-검증)
- [3. 베이스라인 — Generative family](#3-베이스라인--generative-family)
- [4. 베이스라인 — Direct regression / Sample-and-score](#4-베이스라인--direct-regression--sample-and-score)
- [5. 베이스라인 — Pixel-prompted (가장 중요)](#5-베이스라인--pixel-prompted-가장-중요)
- [6. VLM future-work PoC stack](#6-vlm-future-work-poc-stack)
- [7. 통합 권고: Table 1 + 데이터셋 + 일정](#7-통합-권고-table-1--데이터셋--일정)
- [8. 라이선스 위험 / 실행 불가 항목 (블랙리스트)](#8-라이선스-위험--실행-불가-항목-블랙리스트)
- [9. 실행 todo (단계별 effort)](#9-실행-todo-단계별-effort)

---

## 1. TL;DR — 권고 셋업

### 데이터셋 (모두 사용 가능)
| 우선순위 | 데이터셋 | 핵심 | 적용 영역 |
|---|---|---|---|
| 🥇 1 | **GraspNet-1Billion** (CVPR'20) | RealSense + Kinect, 6-DoF, 마스크, 캠프레임, 97GB | 메인 벤치마크 (현재 논문) |
| 🥈 2 | **Grasp-Anything-6D** (ECCV'24) | **단일 픽셀 prompt 라벨 native 보유** — 우리 paradigm 정확 일치 | 픽셀-컨디셔닝 차별화 |
| 🥉 3 | **ACRONYM** (ICRA'21) | mesh + FleX, 자체 렌더로 fx=1109 정확 매칭 가능 | 우리 환경 재현 + 물리 검증 |

### 베이스라인 (Top-5, 모두 코드 공개 + 우리 GPU 호환)
| # | 베이스라인 | family | license | 학습/추론 | effort |
|---|---|---|---|---|---|
| 1 | **Direct MLP (자체)** | Direct, pixel-cond | — | done | 0h |
| 2 | **Contact-GraspNet** (ICRA'21) | Direct, scene 6-DoF | NVIDIA NC ★ | 사전학습 ckpt | ~12h |
| 3 | **6-DoF GraspNet** (Mousavian ICCV'19) | Sample-and-score | NVIDIA NC ★ | 사전학습 ckpt (PyTorch port 추천) | ~10h |
| 4 | **SE(3)-DiffusionFields** (ICRA'23) | Generative diffusion | MIT | 사전학습 ckpt | ~10h |
| 5 | **EquiGraspFlow** (CoRL'24) | Generative flow (CNF) | MIT | 사전학습 ckpt (4 cat) | ~10h |
| **(+1)** | **TOGNet** (ECCV-W'24) | **Pixel-prompted 6-DoF** | MIT | 사전학습 ckpt | ~14h ← **반드시 포함** |

★ = NVIDIA Source Code License (academic use OK, footnote 필수)

### VLM PoC (future paper)
- **VLM**: **Molmo-7B-D** (AI2, Apache-2.0) — *native pixel point output* `<point x= y=/>` 1-call, 16GB VRAM
- **Fallback**: **GroundingDINO** (Apache-2.0, 6GB) — bbox center → (u,v)
- **데이터셋**: **OCID-Ref** (NAACL'21, 305K expressions, RGBD + 마스크) primary + **RoboRefIt** (IROS'23) secondary
- **Effort**: ~1.5 일 PoC

---

## 2. 데이터셋 — 사용가능성 검증

### 2-1. 메인 벤치마크 — 사용 가능 ✅

| Field | **GraspNet-1Billion** | **Grasp-Anything-6D** | **ACRONYM** |
|---|---|---|---|
| URL | [graspnet.net](https://graspnet.net) · [API](https://github.com/graspnet/graspnetAPI) | [airvlab.github.io](https://airvlab.github.io/grasp-anything/) | [NVlabs/acronym](https://github.com/NVlabs/acronym) |
| License | **CC BY-NC-SA 4.0** (학술 OK) | **CC BY-NC 4.0** | code MIT, 데이터 **CC BY-NC** |
| 등록 | 무료 (Google Drive 직접) | **Google Forms 필요** | 무료 (1.6GB) |
| 크기 | **190 scene / 97,280 frame / 88 obj / 1.1B grasp** (97GB) | **1M scenes / 200M+ grasp** | **17.7M grasp / 8,872 obj (mesh)** |
| 모달리티 | RGB-D (RealSense D435 + Azure Kinect 두 개) | RGB + depth + PC + **point-prompt 라벨** + language | mesh + grasp (자체 렌더 필요) |
| Camera K | 씬별 `camK.npy` 제공 (D435 ~927 px). **우리 1109 와 다름** | 다중 (synthetic) | **사용자 지정 가능** (fx=fy=1109 정확 매칭) |
| Grasp 표현 | 6-DoF SE(3) + width + score (17-D) | 6-DoF SE(3) | 6-DoF SE(3) parallel-jaw + FleX success label |
| Frame | **camera frame** (정확 일치) | camera/world | object → renderer 가 cam transform 제공 |
| 마스크 | ✅ 0=bg, 1~88 obj id | ✅ + language | ✅ (renderer 출력) |
| Real/Sim | **Real** | Synthetic (Stable Diffusion + ZoeDepth) | Sim (NVIDIA FleX) |
| 단일 pixel target 네이티브? | ❌ (마스크 centroid 직접 추출 5줄) | ✅ **point-prompt 라벨 native** | ❌ (centroid 투영) |

**Effort 견적 (Top-3 통합)**:

| Dataset | Download | 포맷 변환 | 학습/평가 |
|---|---|---|---|
| GraspNet-1B | 6h (97GB, 100 scene zip) | 8~12h: 씬별 K 치환, 마스크→(u,v) centroid, parallel-jaw 폭 필터, train_seen/similar/novel split | eval-only 4~6h, retrain 24~36h |
| Grasp-Anything-6D | 2h Google Forms 대기 + 4h DL | 6~8h: point-prompt 이미 있음, depth+(u,v) pair 추출 | eval-only 3h, retrain 18~24h |
| ACRONYM | 30min + 4h ShapeNetSem 등록 | 10~14h: PyRender 로 fx=fy=1109 / 1280×720 렌더, mesh centroid 투영, success-grasp 필터 | retrain 12~18h |

→ **총 5~7 working day** (single engineer)

### 2-2. 인용만 하고 실행 skip ❌

| Dataset | 이유 |
|---|---|
| **YCB-Video** (RSS'18) | 6D pose only, **grasp 라벨 없음** |
| **Cornell** (Saxena'11) | **4-DoF rectangle**, 2011, 패러다임 mismatch |
| **Jacquard** (IROS'18) | 4-DoF + EULA 친화 안됨 (기관 메일 인증) |
| **OCID-grasp** (ICRA'21) | **4-DoF rectangle** |
| **EGAD!** (RA-L'20) | top-down antipodal Dex-Net format, mesh-only |
| **HOPE** (IROS'22) | 6D pose only, grasp 없음 |
| **REGRAD** (RA-L'22) | 6-DoF parallel-jaw 가능하지만 4순위 fallback (중국 미러 다운로드 느림) |
| **DexGraspNet 2.0** | 주로 dexterous, parallel-jaw split 작음 |
| **ClearGrasp** (ICRA'20) | transparent only, 니치 |

---

## 3. 베이스라인 — Generative family

### 🥇 SE(3)-DiffusionFields (Urain et al., **ICRA 2023**) — **omit = reject**
- **Repo**: [robotgradient/grasp_diffusion](https://github.com/robotgradient/grasp_diffusion)
- **License**: **MIT** ✅
- **활성도**: 350⭐, 마지막 push 2024-07, 공개 issue 7개
- **Stack**: PyTorch + `theseus-ai` (CUDA build 어려움 — Docker 권장) + mesh_to_sdf + trimesh + pyrender
- **GPU**: 24GB OK (batch 8~16)
- **사전학습 ckpt**: ✅ (Google Drive, issue #19 mirror 끊김 가능)
- **입출력**: 객체/씬 PC → SE(3) 행렬
- **(u,v) 적응**: depth + YOLO mask 로 segmented PC 추출 → 50줄 어댑터
- **Effort**: install 4h, eval-only 6h, retrain 2~3일
- **Pitfall**: `theseus` CUDA 12 빌드, headless `pyrender` EGL

### 🥈 EquiGraspFlow (Lim et al., **CoRL 2024**) — **method-family 직접 대비**
- **Repo**: [bdlim99/EquiGraspFlow](https://github.com/bdlim99/EquiGraspFlow)
- **License**: **MIT** ✅
- **활성도**: 34⭐, 마지막 push 2026-02-09, **issue 0**
- **Stack**: Vector Neurons (SO(3)-equivariant) + Flow Matching
- **GPU**: ACRONYM subset (Laptop/Mug/Bowl/Pencil) — 24GB 충분
- **사전학습 ckpt**: 있음 (project page Drive)
- **(u,v) 적응**: PC 기반 — same as SE(3)-DF
- **Effort**: install 3h, eval 4h, retrain 1일
- **주의**: 4 카테고리 한정 학습 — 우리 bottle/can/cube/marker/spam 은 **out-of-distribution** → (a) zero-shot eval (Mug/Bowl 모델로) + (b) 우리 grasp_v2.h5 로 retrain 두 트랙

### 🥉 GraspLDM (Barad et al., **IEEE Access 2024**) — **인스톨 가장 쉬움**
- **Repo**: [kuldeepbrd1/graspLDM](https://github.com/kuldeepbrd1/graspLDM)
- **License**: **Apache 2.0** ✅
- **활성도**: 마지막 push **2026-04-04**, 활발
- **Stack**: Python ≥3.8, torch 1.13+cu117, pytorch-lightning 1.8, diffusers, **Devcontainer + Docker 제공**
- **GPU**: 24GB 충분 (PVCNN small)
- **사전학습 ckpt**: ✅ HuggingFace `kuldeepbarad/GraspLDM` (full + partial PC 둘 다)
- **(u,v) 적응**: partial-PC ckpt 가 이미 있어 segmented depth crop → 1024 pts 로 직접 사용 가능
- **Effort**: install 3h (Docker), eval 3h, retrain 1~2일

### NVIDIA GraspGen (arXiv 2025) — **신 SOTA, 라이선스 주의**
- **Repo**: [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)
- **License**: ⚠️ "Other" (NVIDIA Source Code License) — 학술 비영리만, **상용 금지**. 논문 footnote 의무
- **활성도**: 436⭐, 마지막 push 2026-04-23 (가장 활발)
- **GPU**: 추론 24GB OK, retrain 1 GPU 위험 (batch 축소 필요)
- **Effort**: install 4h, eval 4h, retrain 위험

### Skip ❌
- **DexDiffuser** (RA-L'24): dexterous only, 우리 parallel-jaw 부적합
- **FFHFlow** (CoRL'25): Five-finger-hand, 코드 미공개
- **EFF-Grasp** (CVPR'26): dexterous, 코드 미공개 (cite-only)
- **CGDF** (IROS'24): constrained dual-arm, 우리 single-arm scope 밖

---

## 4. 베이스라인 — Direct regression / Sample-and-score

### 🥇 Direct MUST-have: **Contact-GraspNet** (Sundermeyer ICRA'21)
- **Repo**: [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet)
- **License**: ⚠️ NVIDIA Source Code License **Non-Commercial** (학술 OK, footnote 필수)
- **Stack**: TF 1.15 + CUDA 11.0 + Python 3.7 (legacy — Docker 권장)
- **GPU**: 6GB eval, 12GB train
- **사전학습 ckpt**: ✅ ACRONYM
- **입출력**: PC N×3 → 6-DoF contact-point grasps + score
- **(u,v) 적응**: YOLO mask 로 per-object PC segment → ~5h
- **Effort**: install + run total ~12h. Pitfall: TF1.15/protobuf 지옥

### 🥈 Sample-and-score MUST-have: **6-DoF GraspNet** (Mousavian ICCV'19)
- **Repo (PyTorch port 추천)**: [jsll/pytorch_6dof-graspnet](https://github.com/jsll/pytorch_6dof-graspnet) (TF1.12 dodge)
- **License**: NVIDIA NC (학술 OK)
- **사전학습 ckpt**: ✅ ACRONYM
- **입출력**: PC → 6-DoF (CVAE sample + grasp evaluator)
- **Narrative 가치**: *"CVAE sampler (2019) → Flow Matching sampler (ours)"* originality 섹션 핵심 hook
- **Effort**: ~10h

### 🥉 대안: **GraspNet baseline** (정정 2026-04-28: 라이선스 Non-Commercial)
- **Repo**: [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline) — 마지막 push **2026-04-26** (활발), 28 open issues
- **License**: ⚠️ **Non-Commercial Research Use Only** (LICENSE 파일 헤더 verbatim: *"ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY"*) — 학술 paper OK 단 footnote 의무. **이전 메모의 "Apache 2.0" 표기는 오류**
- **graspnetAPI** (별도 리포): [graspnet/graspnetAPI](https://github.com/graspnet/graspnetAPI) — **MIT** ✅ (eval 툴 + 데이터 로더)
- **Stack**: PyTorch 1.6, CUDA 10.1/11.0, **PointNet2 ops** (CUDA 12 빌드 가능 fork 존재)
- **사전학습 ckpt**: ✅ `checkpoint-rs.tar` (RealSense), `checkpoint-kn.tar` (Kinect) — Google Drive + Baidu Pan 미러
- **데모 즉시 실행**: `command_demo.sh` 가 임의 RGB-D 입력 → 6-DoF grasp 출력. 인스톨 후 우리 데이터에 즉시 추론 가능
- **공식 평가**: `command_test.sh` + `graspnetAPI.GraspNetEval` → AP_seen/similar/novel 자동 산출
- **Tolerance label**: 추가 생성 필요 (`dataset/generate_tolerance_label.py`) 또는 Drive 다운
- **(u,v) 적응**: 픽셀별 출력 → 마스크-then-rank ~4h
- **Effort**: ~8h
- **권고**: Contact-GraspNet 이 TF1.15 빌드 실패 시 swap

### Skip ❌
- **GG-CNN, GR-ConvNet v2**: **4-DoF only** — 6-DoF SE(3) 비교 부적합 (Related Work 만 인용)
- **AnyGrasp** (Fang T-RO'23): ⚠️ **Closed-source SDK + license key 필수, 상용 금지**. 학술 사용도 매년 갱신. **Table 1 포함 금지**, Discussion 만 mention
- **HGGD** (RA-L'23): MinkowskiEngine 빌드 1~2일 소요 — 보조 옵션
- **VGN** (CoRL'20): TSDF voxel 다중뷰 필요, 우리 single-view depth 부적합
- **GPD, PointNetGPD**: 휴리스틱 / dex-net heavy dep, 6-DoF GraspNet 이 더 강함

---

## 5. 베이스라인 — Pixel-prompted (가장 중요)

### 🎯 TOGNet (Lu et al., **ECCV-W 2024**) — **반드시 포함**
- **Repo**: [liuliu66/TOGNet](https://github.com/liuliu66/TOGNet) **(코드 공개 확인)**
- **License**: MIT ✅
- **사전학습 ckpt**: ✅ (GraspNet-1B 학습)
- **Stack**: graspnetAPI + PointNet2 CUDA ops
- **입출력**: 단일뷰 depth/PC + region prompt (mask/box/click) → **6-DoF**
- **(u,v) 적응**: K + uint16 depth → PC 변환 + (u,v)→region patch crop
- **Effort**: 10~16h (zero-shot 가능, retrain 옵션)
- **블로커**: GraspNet-1B 인트린식 ≠ 우리 K=1109 → zero-shot 또는 우리 데이터로 retrain
- **논문 메시지**: *"TOGNet 은 픽셀 prompt + anchor 회귀 (49 fixed templates) → mode collapse, 우리는 픽셀 prompt + Flow Matching → 분포 학습"*

### 보조: RegionNormalizedGrasp (Ma et al., 2024)
- **Repo**: [THU-VCLab/RegionNormalizedGrasp](https://github.com/THU-VCLab/RegionNormalizedGrasp)
- **License**: MIT ✅
- **사전학습 ckpt**: ✅, MED ~12h
- **포함 권고**: 시간 여유 시 secondary

### Skip ❌
- **GraspSAM** (2024): ✅ 코드 공개, **4-DoF planar only** — 6-DoF 비교 부적합
- **Click-to-Grasp / C2G** (IROS'24): ✅ 코드 공개 ([tsagkas/click2grasp](https://github.com/tsagkas/click2grasp)) but **AnyGrasp commercial license 의존** — 라이선스 위험
- **Murali Target-driven** (ICRA'20): 6-DoF GraspNet 에 흡수됨

### Pixel-prompted baseline 가 부족하면?
- 우리 자체 **Direct MLP baseline** ([src/direct_model.py](src/direct_model.py)) 가 *pixel-conditioned regression* 그 자체 → "Direct-MLP (ours, pixel-conditioned)" 표기로 controlled stand-in 충분

---

## 6. VLM future-work PoC stack

### 6-1. 권장 VLM (단일 모델, 한 호출로 text → (u,v))

#### 🥇 Primary: **Molmo-7B-D** (AI2)
- **HF**: [allenai/Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924)
- **License**: **Apache 2.0** ✅
- **크기**: 7B / fp16 ~16GB VRAM (24GB GPU 충분)
- **출력**: **`<point x="34.2" y="55.7"/>` XML 네이티브** — *PixMo-Points 학습 데이터로 픽셀 좌표 직접 출력*
- **사용**: prompt `Point to the {object_description}.` → x,y (% of W,H) → `u = x/100*W, v = y/100*H`
- **설치**:
  ```bash
  /home/robotics/anaconda3/bin/pip install "transformers>=4.45" accelerate einops Pillow huggingface_hub
  /home/robotics/anaconda3/bin/huggingface-cli download allenai/Molmo-7B-D-0924
  ```

#### 🥈 Fallback: **GroundingDINO**
- **Repo**: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- **License**: **Apache 2.0** ✅
- **크기**: 172~340M / 1.5GB / VRAM 6GB
- **출력**: text → bbox `[x1,y1,x2,y2]` → bbox center 가 (u,v)
- **장점**: 결정론적, 더 안정적
- **단점**: Molmo 처럼 점 출력 아님 (1줄 후처리)

### 6-2. Skip / 라이선스 위험

| 모델 | 이유 |
|---|---|
| **CLIP** | 점 출력 없음, Grad-CAM 후처리 필요 (effort 증가) |
| **LLaVA-1.5/1.6** | 코드 Apache-2.0 but **Llama-2/Vicuna weights research-only** — 상용 금지 |
| **CogVLM-Grounding** | research-only, 17B / 35GB (24GB 초과) |
| **PaLI-X / PaLI-3** | Google 내부, 공개 weights 없음 |
| **Qwen2.5-VL-72B** | 특수 라이선스 (≤100M MAU), 24GB 초과. **단 7B는 Apache-2.0** 사용 가능 |

### 6-3. 데이터셋 (language ↔ pixel grounding)

#### 🥇 Primary: **OCID-Ref** (NAACL 2021)
- **Repo**: [lluma/OCID-Ref](https://github.com/lluma/OCID-Ref)
- **License**: research-only (OCID base CC BY-NC-SA 4.0)
- **크기**: **305K expressions × 2.3K cluttered tabletop RGBD scenes**
- **포맷**: 언어 → 2D bbox + 마스크 + RGBD + 3D PC
- **(u,v) 추출**: bbox center 또는 mask centroid
- **적합도**: **우수** — RGBD + clutter + parallel-jaw scale, 우리 1280×720 K rescale 만 필요

#### 🥈 Secondary: **RoboRefIt** (Lu IROS 2023)
- **Repo**: [luyh20/VL-Grasp](https://github.com/luyh20/VL-Grasp)
- **License**: research-only
- **크기**: 50K expressions / 10K RGBD pairs
- **포맷**: 언어 → instance mask + bbox
- **적합도**: **우수** — VL-Grasp 가 사용한 데이터셋, parallel-jaw scale

#### Skip
- **RefCOCO/+/g**: depth 없음, grasp 없음 — VLM grounding pre-train sanity check 만
- **Grasp-Anything (Original)**: RGB only (depth 없음) — 우리 depth 컨디셔닝 호환 안 됨
- **TaskGrasp**: PC-only, (u,v) 없음
- **REVERIE / ScanRefer / HOI-Ref**: 내비/실내/에고센트릭 — out of scope

### 6-4. PoC 5-step 레시피 (1.5 일)

```
1. Molmo 다운 (~10분, 16GB)
   /home/robotics/anaconda3/bin/huggingface-cli download allenai/Molmo-7B-D-0924

2. Wrapper 작성 (scripts/vlm_to_uv.py, ~3h)
   - AutoModelForCausalLM (bfloat16, device_map="auto")
   - prompt: "Point to the {user_text}."
   - parse <point x= y=> XML
   - scale x/100*W, y/100*H

3. 기존 demo_inference.py 와 결합 (~2h)
   - YOLO uv → Molmo uv 로 source 교체
   - depth crop + encoder.onnx + velocity.onnx 그대로 재사용

4. Eval loop (~25min × 100 scenes × 3 prompts ≈ 2h actual)
   - img_dataset/random6/* 100 val scenes
   - 각 씬당 3 자연어 prompt ("the blue bottle on the left" 등)
   - Molmo→(u,v)→CFM(N=32)→collision filter→best-pick

5. Metric 보고 (~3h)
   - pixel error vs YOLO centroid
   - grasp success after collision filter
   - language accuracy (point 가 의도된 클래스 마스크 안에 있는지)
```

---

## 7. 통합 권고: Table 1 + 데이터셋 + 일정

### Table 1 최종 라인업 (6 baselines)

| 라인 | 베이스라인 | family | 라이선스 | metric | 우리 컨디셔닝 적응 |
|---|---|---|---|---|---|
| L1 | **Direct MLP (ours)** | Direct, pixel-cond | — | Pos MAE / Ang Err / COV / APD | done |
| L2 | **Contact-GraspNet** | Direct, scene 6-DoF | NVIDIA NC ★ | 동일 | YOLO mask → segmented PC |
| L3 | **6-DoF GraspNet (PyTorch port)** | Sample-and-score CVAE | NVIDIA NC ★ | 동일 | segmented PC |
| L4 | **SE(3)-DiffusionFields** | Generative diffusion | MIT | 동일 | segmented PC |
| L5 | **EquiGraspFlow** | Generative flow (CNF) | MIT | 동일 | segmented PC, retrain on grasp_v2 |
| L6 | **TOGNet** | **Pixel-prompted 6-DoF** | MIT | 동일 | (u,v) prompt 직접 |
| L7 (옵션) | **GraspLDM** | Generative latent diffusion | Apache 2.0 | 동일 | partial PC |
| L8 (옵션) | **GraspGen** | Generative DDPM-Transformer | NVIDIA NC ★ | 동일 | scene PC |

★ NVIDIA Source Code License — academic OK, 논문 footnote: *"Used under NVIDIA Source Code License for non-commercial research."*

### 데이터셋 라인업

| Dataset | 역할 | 비교 베이스라인 (코드) |
|---|---|---|
| **자체 grasp_v2.h5** (588 sim) | Primary 학습/평가 | All 6 |
| **GraspNet-1B** | Real-world 전이 검증 (test_seen/similar/novel) | Contact-GraspNet, 6-DoF GraspNet, GraspNet-baseline 사전학습 ckpt 그대로 |
| **ACRONYM** | 1109 px 매칭 + 물리 success label | SE(3)-DF, EquiGraspFlow, GraspLDM, GraspGen (모두 ACRONYM 학습됨) |
| **Grasp-Anything-6D** (옵션) | 단일 픽셀 prompt 차별화 강조 | LGD direct comparison |

### 일정 (single engineer)

| 주차 | 작업 | 예상 시간 |
|---|---|---|
| W1 | GraspNet-1B + ACRONYM 다운 + 포맷 변환 | 30~40h |
| W2 | SE(3)-DF + EquiGraspFlow + GraspLDM 인스톨 + zero-shot eval | 30h |
| W3 | Contact-GraspNet + 6-DoF GraspNet + TOGNet 인스톨 + zero-shot eval | 30h |
| W4 | 모든 baseline 우리 데이터로 (선택적) retrain + Table 1 채움 | 40~50h |
| W5 | Grasp-Anything-6D 추가 + LGD 비교 (옵션) | 25h |

→ **5주 = 1개월 정도** 풀타임이면 모든 베이스라인 + 데이터셋 정리 가능

### Future paper VLM PoC

| 시점 | 작업 | 시간 |
|---|---|---|
| 현 논문 출고 후 | Molmo PoC + OCID-Ref / RoboRefIt eval | 1.5일 |
| Future paper 작성 시 | OWG / LGD / FLASH 비교 + GraspVLA 차별화 | 2주 |

---

## 8. 라이선스 위험 / 실행 불가 항목 (블랙리스트)

### 절대 Table 1 에 포함하지 말 것 ⛔
| 항목 | 이유 |
|---|---|
| **AnyGrasp** | Closed-source binary SDK + 매년 academic key 갱신, **상용 금지**. Discussion 에서만 "commercial SOTA, not publicly reproducible" mention |
| **Click-to-Grasp** | AnyGrasp 의존 — 동일 라이선스 문제 |
| **ThinkGrasp** | GPT-4o API key + AnyGrasp 둘 다 필요 |
| **CogVLM-Grounding** | research-only license |
| **LLaVA on Llama-2 weights** | Llama-2 community license — 상용 금지 |

### 코드 미공개 (cite-only) ⚠️
| 항목 | 다음 액션 |
|---|---|
| **EFF-Grasp** (CVPR 2026) | "code not yet released" footnote |
| **FFHFlow** (CoRL 2025) | 동일 |
| **DexGraspVLA** (AAAI 2026) | 동일 |
| **FLASH** (CoRL 2025) | 동일 |
| **UniDiffGrasp** (arXiv 2505) | 동일 |
| **TOGNet** | ✅ **코드 공개됨** ([liuliu66/TOGNet](https://github.com/liuliu66/TOGNet)) — 실행 가능 |

### NVIDIA Source Code License (academic OK, footnote 필수)
- Contact-GraspNet
- 6-DoF GraspNet (NVlabs/6dof-graspnet 본가; PyTorch port 는 다른 라이선스)
- GraspGen

### CC BY-NC (상용 X, 학술 OK)
- GraspNet-1B
- ACRONYM 데이터
- Grasp-Anything (-6D)
- OCID-Ref / RoboRefIt

---

## 9. 실행 todo (단계별 effort)

### Phase 1 (1주차) — 데이터셋 확보
- [ ] GraspNet-1B 다운 + graspnetAPI 빌드 + 씬 1개 샘플 변환 시연 (8h)
- [ ] ACRONYM + ShapeNetSem 등록 + PyRender 렌더 1109 px (12h)
- [ ] Grasp-Anything-6D Google Forms (응답 대기, 후순위)

### Phase 2 (2주차) — Generative 베이스라인 zero-shot eval
- [ ] SE(3)-DiffusionFields: Docker + theseus + 사전학습 ckpt + 우리 val 100 obj eval (10h)
- [ ] GraspLDM: HF ckpt + 우리 partial PC 입력 (8h)
- [ ] EquiGraspFlow: VN-PointNet 빌드 + Mug/Bowl ckpt zero-shot (10h)
- [ ] (옵션) GraspGen: 라이선스 review + 사전학습 ckpt eval (8h)

### Phase 3 (3주차) — Direct + Sample-and-score + Pixel-prompted
- [ ] Contact-GraspNet: Docker (TF1.15) + ACRONYM ckpt + 우리 val (12h)
- [ ] 6-DoF GraspNet (PyTorch port): segmented PC + CVAE sample (10h)
- [ ] **TOGNet: GraspNet-1B ckpt + 우리 (u,v) 입력 + zero-shot (14h)**

### Phase 4 (4주차) — Table 1 통합 + retrain (선택)
- [ ] 메트릭 통합 스크립트: COV / APD / Pos MAE / Ang Err per-mode (8h)
- [ ] 베이스라인 중 1~2개 우리 grasp_v2.h5 로 retrain (20~40h)
- [ ] Fig 3 비교 갱신 + Table 1 final (8h)

### Phase 5 (5주차, 옵션) — Grasp-Anything-6D 추가
- [ ] LGD vs Ours direct comparison on GA-6D (25h)

### Phase 6 (논문 출고 후) — VLM PoC
- [ ] Molmo PoC + OCID-Ref eval (1.5일)
- [ ] Future paper outline 갱신

---

## 부록 A. 빠른 reference URL 모음

### Datasets
- GraspNet-1B: https://graspnet.net · API: https://github.com/graspnet/graspnetAPI
- ACRONYM: https://github.com/NVlabs/acronym
- Grasp-Anything: https://airvlab.github.io/grasp-anything/
- OCID-Ref: https://github.com/lluma/OCID-Ref
- RoboRefIt: https://github.com/luyh20/VL-Grasp
- DexGraspNet 2.0: https://huggingface.co/datasets/lhrlhr/DexGraspNet2.0

### Generative grasp baselines
- SE(3)-DiffusionFields: https://github.com/robotgradient/grasp_diffusion
- GraspLDM: https://github.com/kuldeepbrd1/graspLDM · HF ckpt: huggingface.co/kuldeepbarad/GraspLDM
- EquiGraspFlow: https://github.com/bdlim99/EquiGraspFlow · project: https://equigraspflow.github.io
- GraspGen: https://github.com/NVlabs/GraspGen · project: https://graspgen.github.io
- NGDF: https://github.com/facebookresearch/NGDF

### Direct / Sample-and-score baselines
- Contact-GraspNet: https://github.com/NVlabs/contact_graspnet
- 6-DoF GraspNet (orig TF): https://github.com/NVlabs/6dof-graspnet
- 6-DoF GraspNet (PyTorch port): https://github.com/jsll/pytorch_6dof-graspnet
- GraspNet baseline: https://github.com/graspnet/graspnet-baseline
- HGGD: https://github.com/THU-VCLab/HGGD
- AnyGrasp SDK ⚠️: https://github.com/graspnet/anygrasp_sdk

### Pixel-prompted
- **TOGNet: https://github.com/liuliu66/TOGNet ✅**
- RegionNormalizedGrasp: https://github.com/THU-VCLab/RegionNormalizedGrasp
- GraspSAM: https://github.com/gist-ailab/GraspSAM (4-DoF only)
- Click-to-Grasp: https://github.com/tsagkas/click2grasp ⚠️ AnyGrasp dep

### VLM (future)
- Molmo: https://huggingface.co/allenai/Molmo-7B-D-0924 · blog: https://molmo.allenai.org
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- SAM2: https://github.com/facebookresearch/sam2
- Grounded-SAM-2: https://github.com/IDEA-Research/Grounded-SAM-2
- Florence-2: https://huggingface.co/microsoft/Florence-2-large
- OWLv2: https://huggingface.co/google/owlv2-large-patch14-ensemble
- Qwen2.5-VL-7B: https://github.com/QwenLM/Qwen2.5-VL

### VLM-grasp baselines (future)
- LGD: https://github.com/Fsoft-AIC/LGD
- LLGD: https://github.com/Fsoft-AIC/LLGD
- OWG: https://github.com/gtziafas/OWG
- VL-Grasp: https://github.com/luyh20/VL-Grasp
- CROG: https://github.com/HilbertXu/CROG (4-DoF only)
- HiFi-CS: https://github.com/vineet2104/hifi-cs
- F3RM: https://github.com/f3rm/f3rm
- LERF-TOGO: https://github.com/lerftogo/lerftogo
- GraspGPT: https://github.com/mkt1412/GraspGPT_public
- GraspVLA: https://github.com/PKU-EPIC/GraspVLA
- ManipLLM: https://github.com/clorislili/ManipLLM
- CLIPort: https://github.com/cliport/cliport (4-DoF only)
- ThinkGrasp: https://github.com/H-Freax/ThinkGrasp ⚠️ GPT-4o + AnyGrasp dep
