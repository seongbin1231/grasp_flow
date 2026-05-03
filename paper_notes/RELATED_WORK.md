# Related Work — 정밀 선행연구 조사 및 차별화 전략

**작성일**: 2026-04-28
**작성 방식**: 5인 병렬 에이전트 (general-purpose × 5) + WebSearch + WebFetch 로 abstract / method 정독 팩트체크
**대상 논문**: *Pixel-Conditioned Multi-Modal 6-DoF Grasp Pose Generation via Conditional Flow Matching*

---

## 0. 목차

- [1. TL;DR — 핵심 결론](#1-tldr--핵심-결론)
- [2. 현재 framing의 문제 + 권고 재작업](#2-현재-framing의-문제--권고-재작업)
- [3. 픽셀-(u,v) conditioning 단독 검증](#3-픽셀-uv-conditioning-단독-검증-사용자-질문)
- [4. 가장 위협적인 prior — 상세 카드](#4-가장-위협적인-prior--상세-카드)
- [5. 6-DoF grasp 분류 (3-paradigm)](#5-6-dof-grasp-분류-3-paradigm)
- [6. VLM / 언어 컨디셔닝 grasp 풍경 (future work)](#6-vlm--언어-컨디셔닝-grasp-풍경-future-work)
- [7. 방법론 prior (FM / RF / SE(3) / 컨디셔닝)](#7-방법론-prior-fm--rf--se3--컨디셔닝)
- [8. 우리 차별화 5축](#8-우리-차별화-5축)
- [9. Table 1 비교 필수 베이스라인](#9-table-1-비교-필수-베이스라인)
- [10. 반드시 추가 인용할 논문 (must-cite)](#10-반드시-추가-인용할-논문-must-cite)
- [11. 다음 행동](#11-다음-행동)

---

## 1. TL;DR — 핵심 결론

| 질문 | 결과 |
|---|---|
| **"Flow Matching을 grasp에 처음 적용"** novelty 방어 가능? | ❌ **불가능**. EquiGraspFlow (CoRL 2024) + SNU MSc thesis (2024, 동일 저자) + EFF-Grasp (CVPR 2026) 이 이미 점령 |
| **"Rectified Flow specifically for 6-DoF grasp"** novelty? | ✅ **방어 가능**. 위 prior 모두 CNF / 일반 FM이지 Rectified Flow 명시 변형 아님 |
| **픽셀 (u,v) conditioning grasp 모델 존재?** | ⚠️ **부분 존재**. TOGNet (ECCV-W 2024) 가 가장 비슷. 단 discriminative anchor regression — 생성형 분포 아님 |
| **"depth + 단일 픽셀 (u,v) + Conditional Rectified Flow + 멀티모달 6-DoF + 1-step Euler ONNX"** 정확 조합 존재? | ✅ **0건**. 우리 unique pocket |
| **VLM future work** ("language → VLM → (u,v) → CFM grasp") 선점 여부? | ⚠️ **GraspVLA (2025) 가 8/10 일치**. 단 bbox 사용 (vs 우리 (u,v)), end-to-end (vs 모듈식). 빈틈 있음 |

**한 줄 결론**: 헤드라인 "first FM for grasp" 표현은 **반드시 폐기**. **"first pixel-(u,v) conditioned, depth-only, Rectified-Flow-based, deployment-friendly multi-modal 6-DoF grasp generator"** 5축 결합 novelty로 재작업 필요.

---

## 2. 현재 framing의 문제 + 권고 재작업

### 2-1. 현재 contribution (CLAUDE.md / research_paper_plan.md)

> **기여 1**. 다수 객체 scene 의 Depth 이미지에서 픽셀(u,v) 컨디셔닝으로 타겟 객체 파지 자세 생성
> **기여 2**. Conditional Flow Matching 디코더로 멀티모달 분포 직접 학습·샘플링

### 2-2. Reviewer 가 깰 수 있는 지점

| 깨질 지점 | Reviewer 가 가져올 prior |
|---|---|
| "FM for grasp 처음" 함축 | EquiGraspFlow (CoRL 2024), SNU MSc thesis (2024) |
| "픽셀 conditioning 처음" 함축 | TOGNet (ECCV-W 2024), GraspSAM, Click-to-Grasp |
| "멀티모달 분포 직접 학습 처음" | SE(3)-DiffusionFields (ICRA'23), GraspLDM (2024), CGDF (2024), GraspGen (2025) |
| "직접 회귀 vs 샘플-스코어 두 분류" 트리코토미 새것 함축 | 이미 GraspLDM/CGDF/GraspGen paper들이 동일 분류 사용 — 새것 아님 |

### 2-3. 권고 재작업 framing (방어 가능 5축 결합)

> **기여 1**. **단일 픽셀 (u,v) 를 직접 conditioning input** 으로 받아 그 객체에 대한 멀티모달 6-DoF grasp 분포를 생성하는 첫 **생성형** (생성 분포 학습) 시스템. 기존 픽셀-입력 grasp 모델 (TOGNet, GraspSAM) 은 모두 discriminative anchor regression — mode collapse 동일 한계.
>
> **기여 2**. 6-DoF SE(3) grasp 생성에 **Conditional Rectified Flow (Liu 2023)** 를 처음 적용. 기존 6-DoF flow grasp (EquiGraspFlow, CoRL 2024) 는 CNF / FFJORD 류 multi-step ODE — Rectified Flow 의 직선화된 path 덕에 **1-step Euler 추론으로 deployment-friendly**.
>
> **(부) 기여**. **Depth-only single-view + (u,v) bridge** 의 modular VLM-agnostic 설계. ONNX velocity MLP 단일 export + MATLAB 1-step Euler + 충돌 필터로 RoboCup ARM UR5e 통합 검증.

### 2-4. Originality 섹션 표현

직전 표현 ("3 paradigm 트리코토미 새것") 대신:

> Within the **generative-distribution family** (SE(3)-DiffusionFields, GraspLDM, CGDF, EquiGraspFlow, GraspGen) that resolves the mode-collapse pathology of direct regression (GG-CNN, Contact-GraspNet, AnyGrasp) without the two-stage cost of sample-and-score (6-DoF GraspNet, GPD), we are the first to (a) employ **Rectified Flow** for the velocity field rather than score-based diffusion or generic CNF, (b) condition on a **single pixel (u,v)** rather than a full segmented point cloud, and (c) ship the velocity field as a **single-step Euler ONNX** for industrial-grade real-time deployment.

---

## 3. 픽셀-(u,v) conditioning 단독 검증 (사용자 질문)

> 사용자 직접 질문: *"우리는 컨디션으로 픽셀을 넣잖아 그게 있냐는 말이야"*

### 3-1. 픽셀을 input/prompt로 받는 grasp 논문 — 4건 존재

| # | 논문 | 픽셀 사용 방식 | DoF | 생성형? | 우리와 정확 일치? |
|---|---|---|---|---|---|
| 1 | **TOGNet** (Lu et al., **ECCV-W 2024**) [arxiv](https://arxiv.org/abs/2408.11138) | 클릭 (u,v) → 즉시 3D 영역 중심 back-project → RGB-D patch crop 후 네트워크 입력 | 6-DoF | ❌ Discriminative regression (49 anchor) | ❌ anchor 회귀 — mode collapse 동일 |
| 2 | **GraspSAM** (Noh et al., 2024) [arxiv](https://arxiv.org/abs/2409.12521) | SAM-style 픽셀 prompt → SAM mask → grasp head gating | **4-DoF planar** | ❌ Discriminative | ❌ 4-DoF |
| 3 | **Click-to-Grasp / C2G** (Tsagkas et al., **IROS 2024**) [arxiv](https://arxiv.org/abs/2403.14526) | 다른 인스턴스 RGB의 클릭 → diffusion descriptor 매칭 | 6-DoF (단일) | ❌ 매칭 | ❌ 단일 grasp, 분포 아님 |
| 4 | **Point-and-Click teleop** (Kent et al., IROS 2017) | 클릭 → 3D point → 휴리스틱 그리퍼 | 6-DoF (단일) | ❌ Classical, 학습 모델 아님 | ❌ 학습모델 아님 |
| (참) | **Region-aware Grasp** (Ma et al., 2024) [arxiv](https://arxiv.org/abs/2406.01767) | 마스크 위에서 **네트워크 자체** (u,v) 샘플링 | 6-DoF | ❌ Discriminative | ❌ 사용자 입력 아님 |

### 3-2. 정확한 우리 조합 — *"픽셀 (u,v) → 생성형 (Flow) → 멀티모달 6-DoF 분포"*

이 5요소 결합으로 발표된 논문 **0건**. 검증된 모든 논문이 다음 중 1개 이상에서 어긋남:

- 픽셀을 받지만 → discriminative (TOGNet, GraspSAM, Region-aware)
- 픽셀을 받지만 → 4-DoF (GraspSAM)
- 픽셀을 받지만 → 단일 grasp (Click-to-Grasp, Point-and-Click)
- 6-DoF 생성형이지만 → **point cloud 입력** (EquiGraspFlow, SE(3)-DiffusionFields, GraspLDM, GraspGen, EFF-Grasp, LGD)
- 6-DoF 생성형 + 픽셀 비슷한 거 받지만 → **2D bounding box** (GraspVLA), **마스크** (Murali ICRA'20, HiFi-CS, OWG)

### 3-3. 결론

**"픽셀 conditioning 자체는 우리가 first 가 아님"** (TOGNet 존재). 그러나 **"픽셀 conditioning + 생성형 분포 (Rectified Flow) + 6-DoF SE(3)"** 이 한 줄 정확 조합은 비어있음.

논문 작성 시 의무사항:
- TOGNet 반드시 인용
- TOGNet vs 우리 차이 명시: *"TOGNet 은 픽셀로 영역 지정 후 anchor 회귀 → mode collapse, 우리는 픽셀 conditioning 을 생성형 flow 와 결합 → 분포 직접 학습"*
- Table 1 에 TOGNet 도 baseline 으로 (가능한 경우)

---

## 4. 가장 위협적인 prior — 상세 카드

### Card 1. 🔴 **EquiGraspFlow** (Lim et al., **CoRL 2024**)

- **링크**: [project](https://equigraspflow.github.io/) · [PMLR](https://proceedings.mlr.press/v270/lim25a.html)
- **저자**: Byeongdo Lim, Donghoon Kim, Junhyek Kim, Hokyun Lee, Frank C. Park (SNU)
- **방법**: SE(3)-equivariant **Continuous Normalizing Flow** for 6-DoF grasp generation. 입력은 **object point cloud**, 출력은 6-DoF SE(3) 분포.
- **위협도**: 🔴 **최고**. 같은 family (flow on SE(3)) + 같은 출력 (6-DoF grasp 분포). 우리가 모르고 발표하면 즉시 reject.
- **차이점 (= 우리 무기)**:
  1. EquiGraspFlow 는 **CNF / FFJORD 류** (multi-step adjoint ODE) — 우리는 **Rectified Flow** (1-step Euler 가능, deployment 친화)
  2. EquiGraspFlow 는 **point cloud 입력** — 우리는 **depth + 단일 픽셀 (u,v)**
  3. EquiGraspFlow 는 **SE(3)-equivariant 구조** — 우리는 **AdaLN-Zero + cross-attn** (DiT 계열, 더 단순)
  4. ONNX 배포 / MATLAB 통합 사례 미공개 — 우리는 검증 완료
- **must-cite 표현**: *"Most closely related, EquiGraspFlow [Lim et al., CoRL 2024] applies SE(3)-equivariant continuous normalizing flow to 6-DoF grasp generation, but conditions on a full segmented object point cloud and requires multi-step ODE integration. We instead employ Rectified Flow with a single-pixel (u,v) condition, enabling 1-step Euler inference deployable as a single ONNX MLP."*

### Card 2. 🔴 **SNU MSc thesis "6-DoF Grasp Pose Generation with Riemannian Flow Matching"** (Lim, 2024)

- **링크**: [s-space](https://s-space.snu.ac.kr/handle/10371/209476)
- **저자**: Byeongdo Lim (Park Lab, SNU) — EquiGraspFlow 의 1저자 = 동일 인물의 precursor
- **위협도**: 🔴 (제목 verbatim 거의 동일)
- **사실 점검**:
  - 학술지 / 학회 peer review **미통과** (석사논문 only)
  - 인용 의무 약하지만, 우리가 "first FM for grasp" 표현을 못 쓰는 결정적 근거
  - method 도 point cloud 입력 가정, Riemannian FM 사용
- **대응**: 본문 인용 불필요, 단 우리는 "first FM" 같은 표현을 쓰면 안 됨을 인지

### Card 3. 🟠 **EFF-Grasp** (Zhao et al., **CVPR 2026**)

- **링크**: [arxiv](https://arxiv.org/html/2603.16151)
- **방법**: Energy-Field **Flow Matching** with training-free physics guidance, **dexterous grasp generation** (ShadowHand 33-DoF)
- **위협도**: 🟠 높음. Flow Matching 명시 + 6-DoF 이상 + 멀티모달
- **차이점**:
  1. **Dexterous (33-DoF ShadowHand)** vs 우리 parallel-jaw 6-DoF (전혀 다른 application)
  2. Object **point cloud** 입력 vs depth+(u,v)
  3. Energy-field guidance (training-free) 강조 — 우리는 학습된 conditioning

### Card 4. 🟠 **TOGNet** (Lu et al., **ECCV-W 2024**) — *픽셀 conditioning 가장 가까움*

- **링크**: [arxiv](https://arxiv.org/abs/2408.11138) · [HTML](https://arxiv.org/html/2408.11138v1)
- **방법**: 클릭 / 포인팅 / 언어 → 통합된 3D 영역 중심 → RGB-D patch crop → **49 anchor 6-DoF grasp regression**
- **위협도**: 🟠 매우 높음 (사상이 가장 비슷)
- **차이점 (= 우리 무기)**:
  1. **Discriminative anchor regression** vs 우리 **생성형 분포 (Rectified Flow)** — anchor 49개에 closest 1개 회귀 = mode collapse 같은 한계
  2. **RGB-D 입력** vs depth-only
  3. (u,v) 즉시 3D point 변환 vs 우리는 **(u,v) 자체를 conditioning token + local crop 중심으로 동시 사용**
  4. CFM-기반 분포 sampling 없음
- **권고**: 우리 Table 1 의 baseline 후보 1순위. **반드시 본문 인용 + 차별화**.

### Card 5. 🟠 **GraspGen** (NVIDIA, **arXiv 2025**)

- **링크**: [arxiv](https://arxiv.org/abs/2507.13097) · [project](https://graspgen.github.io/)
- **방법**: Diffusion-Transformer + on-generator-trained discriminator, 53M-grasp 데이터셋. ACRONYM 에서 Contact-GraspNet 대비 +17%.
- **위협도**: 🟡 중. Diffusion (우리는 RF), point cloud 입력
- **차이점**: DDPM-Transformer (우리는 RF + AdaLN), **scene/object PC** 입력 (우리 픽셀+depth), discriminator 별도
- **권고**: 새 SOTA — 안 인용하면 reviewer flag. Table 1 baseline 후보.

### Card 6. 🟠 **LGD (Language-Driven 6-DoF with Negative Prompt)** (Nguyen et al., **ECCV 2024**)

- **링크**: [arxiv](https://arxiv.org/abs/2407.13842)
- **방법**: CLIP text embedding 을 **diffusion** 6-DoF grasp generator 에 cross-attn conditioning
- **위협도**: 🟡 중 (우리 future work 와 더 가까움)
- **차이점**: (u,v) bridge 안 거치고 **언어 임베딩 직접** cross-attn. point cloud 입력. Diffusion (RF 아님)

### Card 7. 🔴 **GraspVLA** (Deng et al., 2025) — *VLM future work 가장 가까움*

- **링크**: [project](https://pku-epic.github.io/GraspVLA-web/)
- **방법**: 내부 VLM backbone → **2D bounding box** 예측 → **Flow-Matching action expert** (Progressive Action Generation) → 6-DoF + chunked action
- **위협도**: 🔴 **future paper 의 8/10 일치**
- **차이점 (= 우리 future work 무기)**:
  1. **2D bounding box bridge** vs 우리 **단일 (u,v) 픽셀** — 더 minimal
  2. **End-to-end joint training** vs 우리 **VLM-agnostic 모듈식** (VLM 교체 자유)
  3. RGB 입력 + multi-step action chunking vs depth-only + static grasp
  4. 학습 데이터 대규모 vs RoboCup-scale 데이터로 검증
- **권고**: 우리 future paper 의 직접 경쟁자. **반드시 future work 섹션에 명시 인용 + 차별화 4축**.

---

## 5. 6-DoF grasp 분류 (3-paradigm)

| Family | 대표 논문 | 출력 표현 | 멀티모달 처리 | 한계 |
|---|---|---|---|---|
| **(A) Direct regression** | GG-CNN (RSS'18), GR-ConvNet v2 (Sensors'22), Contact-GraspNet (ICRA'21), HGGD (RA-L'23), AnyGrasp (T-RO'23) | 픽셀별 heatmap + residual rotation/width | 공간 mode 만 분해, **회전 mode collapse** | 같은 객체 다중 grasp 불가 |
| **(B) Sample-and-score** | GPD (IJRR'17), PointNetGPD (IROS'19), 6-DoF GraspNet VAE (ICCV'19), GraspNet-1B (CVPR'20) | 6-DoF SE(3) | 휴리스틱/VAE proposal → discriminator score → top-k | 2-stage 느림 (VAE + evaluator) |
| **(C) Generative-distribution** ← 우리 위치 | SE(3)-DiffusionFields (ICRA'23), CGDF (IROS'24), GraspLDM (Access'24), DexDiffuser (RA-L'24), **EquiGraspFlow (CoRL'24)**, GraspGen (NVIDIA 2025), **우리** | 6-DoF SE(3) (Lie / quaternion+pos) | p(grasp\|obs) 직접 학습 + 다중 sampling | multi-step (대부분), point-cloud 입력 가정 |

**우리 차별 위치**: (C) family 안에서 (a) Rectified Flow 채택, (b) 픽셀 conditioning, (c) 1-step Euler 배포.

---

## 6. VLM / 언어 컨디셔닝 grasp 풍경 (future work)

| 그룹 | 대표 논문 | 우리 future work 와 일치도 | 차별화 가능성 |
|---|---|---|---|
| **End-to-end VLA (joint trained)** | GraspVLA (2025), π₀ (2024), RT-2 (2023), OpenVLA (2024), DexGraspVLA (AAAI'26), ManipLLM (CVPR'24) | 7~8/10 | 모듈성, VLM-agnostic, depth-only |
| **Direct language-conditioned grasp** | LGD (ECCV'24), FLASH (CoRL'25), LLGD (IROS'24), CROG (CoRL'23), GraspGPT (RA-L'23) | 6/10 | (u,v) bridge 명시 vs language 직접 입력, Rectified Flow |
| **Modular: VLM grounding → 2D mask/bbox → off-the-shelf grasp** | OWG (CoRL'24), ThinkGrasp (CoRL'24), HiFi-CS (2024), VL-Grasp (2023), F3RM (CoRL'23), LERF-TOGO (CoRL'23) | 7/10 | **GraspNet 대신 우리 CFM 분포 generator** |
| **Pixel-prompted (non-language)** | TOGNet (ECCV-W'24), GraspSAM (2024), Click-to-Grasp (IROS'24) | 4/10 | language → VLM → (u,v) 자동화 |

### Future paper 정확한 unoccupied 조합:
> **"Open-vocabulary natural language → VLM attention map → minimal single-pixel (u,v) bridge → Conditional Rectified Flow generative 6-DoF grasp distribution → collision filter + best pick → industrial deployment"**

GraspVLA 가 가장 가깝지만 (a) bbox vs (u,v), (b) end-to-end vs 모듈, (c) depth-only 의 3개 빈틈 존재.

### Future paper 인용 의무
- **GraspVLA, LGD, OWG, F3RM, ThinkGrasp, ManipLLM, π₀, FLASH, HiFi-CS, LERF-TOGO** 10개를 future work 섹션에 grouped citation

---

## 7. 방법론 prior (FM / RF / SE(3) / 컨디셔닝)

| # | Paper | Venue / Year | 영역 | 우리에게 주는 prior |
|---|---|---|---|---|
| 1 | **Lipman, "Flow Matching for Generative Modeling"** [arXiv 2210.02747](https://arxiv.org/abs/2210.02747) | ICLR 2023 | 이미지 / R^d | FM 학습 framework foundational |
| 2 | **Liu, "Flow Straight and Fast: Rectified Flow"** [arXiv 2209.03003](https://arxiv.org/abs/2209.03003) | ICLR 2023 Spotlight | 이미지 / R^d | **우리 핵심 — Rectified Flow 직선 path** |
| 3 | **Bose & Tong, "FoldFlow: SE(3)-Stochastic FM"** [arXiv 2310.02391](https://arxiv.org/abs/2310.02391) | ICLR 2024 | SE(3)^N (단백질) | **SE(3) 위에서의 FM 시초** — 우리 SE(3) 적용의 prior |
| 4 | **Esser et al., "SD3: RF Transformers"** [arXiv 2403.03206](https://arxiv.org/abs/2403.03206) | 2024 | 이미지 latent | RF 가 large-scale 동작 증거 |
| 5 | **Peebles & Xie, DiT (AdaLN-Zero)** [arXiv 2212.09748](https://arxiv.org/abs/2212.09748) | ICCV 2023 | 이미지 latent | 우리 conditioning 메커니즘 정확 출처 |
| 6 | **Ho & Salimans, CFG** [arXiv 2207.12598](https://arxiv.org/abs/2207.12598) | NeurIPS-W 2021 | 이미지 | 우리 CFG=2.5 사용 근거 |
| 7 | **Yim, "FrameDiff: SE(3) Diffusion"** [arXiv 2302.02277](https://arxiv.org/abs/2302.02277) | ICML 2023 | SE(3)^N (단백질) | SE(3) 생성 모델 prior |
| 8 | **Urain, SE(3)-DiffusionFields** [arXiv 2209.03855](https://arxiv.org/abs/2209.03855) | ICRA 2023 | SE(3) grasp | **grasp domain prior — Table 1 baseline** |
| 9 | **Chi, Diffusion Policy** [arXiv 2303.04137](https://arxiv.org/abs/2303.04137) | RSS 2023 | action sequence | robotics-domain diffusion 시초 |
| 10 | **Black, π₀ Flow Model** [arXiv 2410.24164](https://arxiv.org/abs/2410.24164) | 2024 | VLA action chunks | **robotics-FM 동시기 — future work 인용** |
| 11 | **Chen, "PixArt-α"** [arXiv 2310.00426](https://arxiv.org/abs/2310.00426) | ICLR 2024 Spotlight | 이미지 | **Cross-attn block 우리 v8 prior** |
| 12 | **Qi, "PointNet++ MSG"** | NeurIPS 2017 | 점군 | **Multi-scale crop 우리 v8 prior** |
| 13 | **Loshchilov, AdamW** | ICLR 2019 | optim | weight_decay 0.05 prior |
| 14 | **EquiGraspFlow** [project](https://equigraspflow.github.io/) | **CoRL 2024** | SE(3) grasp flow | **가장 가까운 grasp prior — 의무 인용** |
| 15 | **FlowPolicy** [arXiv 2412.04987](https://arxiv.org/abs/2412.04987) | AAAI 2025 Oral | 3D manipulation | Consistency FM in robotics |

---

## 8. 우리 차별화 5축

논문 본문 / abstract / contribution 모두 이 5축으로 통일:

| # | 축 | 우리 | 가까운 prior 와 차이 |
|---|---|---|---|
| 1 | **Conditioning input** | **단일 픽셀 (u,v) + depth** | EquiGraspFlow / SE(3)-DF / GraspLDM / GraspGen / LGD: object point cloud. GraspVLA: 2D bbox. HiFi-CS: mask. TOGNet: 클릭 + RGB-D |
| 2 | **Generative model** | **Conditional Rectified Flow (Liu 2023)** | EquiGraspFlow: CNF (FFJORD 류). SE(3)-DF: score-based. GraspLDM: latent diffusion. GraspGen: DDPM Transformer. LGD: diffusion |
| 3 | **Inference cost** | **1-step Euler (deployment)** | EquiGraspFlow / SE(3)-DF / LGD: multi-step ODE / DDPM (10~50 steps) |
| 4 | **Sensor minimum** | **Depth only** (single view) | 거의 모든 prior: RGB or RGB-D (CLIP feature 등 RGB 의존) |
| 5 | **Deployment artifact** | **Single ONNX velocity MLP + MATLAB integration 검증** | 거의 모든 prior: deployment 미공개 |

5축 모두 결합한 prior **0건** ← 우리 protected pocket.

---

## 9. Table 1 비교 필수 베이스라인

| 패밀리 | 베이스라인 | 코드 | 우리 비교에서 보여줄 메시지 |
|---|---|---|---|
| Direct regression | **Contact-GraspNet** (Sundermeyer ICRA'21) | [NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet) | 3D point regression — orientation mode collapse |
| Direct regression (자체 ablation) | **Direct MLP (Ours)** [src/direct_model.py](src/direct_model.py) | 자체 학습 완료 | 표준 MLP 회귀 — standing COV 33.3% |
| Sample-and-score | **6-DoF GraspNet** (Mousavian ICCV'19) | [NVlabs/6dof-graspnet](https://github.com/NVlabs/6dof-graspnet) | VAE 2-stage 느림 |
| Generative (직접 대비) | **SE(3)-DiffusionFields** (Urain ICRA'23) | [robotgradient/grasp_diffusion](https://github.com/robotgradient/grasp_diffusion) | 다단계 score diffusion, point-cloud |
| Generative (FM 직접 대비) | **EquiGraspFlow** (Lim CoRL'24) | [project](https://equigraspflow.github.io/) | CNF, point-cloud, equivariant |
| Pixel-conditioning 직접 대비 | **TOGNet** (Xie ECCV-W'24) | 코드 미공개 — **RNGNet (Chen et al., 2024)** 으로 대체 | anchor-based, mode collapse 동일 |
| (옵션) 새 SOTA | **GraspGen** (NVIDIA 2025) | [graspgen.github.io](https://graspgen.github.io/) | 53M dataset, DDPM-Transformer |
| (옵션) | **GraspLDM** (Barad Access'24) | [code](https://github.com/kuldeepbrd1/graspLDM) | Latent diffusion |

**우리 metric** (이미 정립): Pos MAE, Ang Err, **COV** (Achlioptas ICML'18), **APD** (5cm/30° threshold, Sundermeyer ICRA'21). per-mode breakdown.

---

## 10. 반드시 추가 인용할 논문 (must-cite)

### 10-1. 안 인용하면 reject 위험 매우 큼

1. **EquiGraspFlow** (Lim et al., CoRL 2024) — *closest published prior*
2. **GraspGen** (NVIDIA, arXiv 2025) — new SOTA
3. **TOGNet** (Lu et al., ECCV-W 2024) — pixel-conditioning closest
4. **GraspVLA** (2025) — VLM+FM closest (future work 섹션)
5. **SE(3)-DiffusionFields** (Urain, ICRA 2023) — generative grasp prior
6. **Liu, Rectified Flow** (ICLR 2023) — method foundation
7. **Lipman, Flow Matching** (ICLR 2023) — method foundation
8. **FoldFlow (Bose & Tong, ICLR 2024)** — SE(3) FM 시초
9. **DiT / AdaLN-Zero** (Peebles & Xie, ICCV 2023) — conditioning 출처
10. **CFG** (Ho & Salimans, NeurIPS-W 2021) — CFG=2.5 출처

### 10-2. Family 별 보강 인용

| Family | 추가 인용 |
|---|---|
| Direct regression | GG-CNN (Morrison RSS'18), GR-ConvNet v2 (Sensors'22), Contact-GraspNet (ICRA'21), HGGD (RA-L'23), AnyGrasp (T-RO'23) |
| Sample-and-score | GPD (IJRR'17), 6-DoF GraspNet (ICCV'19), GraspNet-1B (CVPR'20) |
| Generative grasp | CGDF (IROS'24), GraspLDM (Access'24), DexDiffuser (RA-L'24), NGDF (CoRL'22), EFF-Grasp (CVPR'26) |
| Pixel-prompted | TOGNet, GraspSAM, Click-to-Grasp |
| Robotics-FM | π₀ (2024), FlowPolicy (AAAI'25), ManiFlow (2025), Affordance-FM (2024), PointFlowMatch (CoRL'24) |
| Future work (VLM) | LGD (ECCV'24), FLASH (CoRL'25), OWG (CoRL'24), ThinkGrasp (CoRL'24), HiFi-CS (2024), VL-Grasp, F3RM (CoRL'23), LERF-TOGO (CoRL'23), CROG (CoRL'23), CLIPort (CoRL'21), GraspGPT (RA-L'23), DexGraspVLA (AAAI'26), ManipLLM (CVPR'24) |

---

## 11. 다음 행동

### 11-1. 즉시 (1~2 일)

- [ ] **CLAUDE.md 변경 이력**에 "2026-04-28 선행연구 정밀 조사 — framing 재작업" 행 추가
- [ ] **research_paper_plan.md 메모리** 업데이트: 신규 framing 5축, must-cite 10편 반영
- [ ] **TOGNet PDF 직접 정독** (HTML mirror: arxiv.org/html/2408.11138v1) — method 디테일 확보
- [ ] **EquiGraspFlow PDF 정독** — CNF 학습 디테일, 우리 RF 와의 정확한 차이 정리

### 11-2. 단기 (1 주)

- [ ] **Abstract / Contribution / Originality 재작성** — 위 5축 framing 반영
- [ ] **Related Work 섹션 작성** — 본 파일을 베이스로 3 paradigm + pixel-conditioning + VLM-grasp + method-prior 4 sub-section 구성
- [ ] **Table 1 baseline 확장 검토** — TOGNet / EquiGraspFlow / SE(3)-DF 코드 fetch 가능 여부, 우리 데이터셋 (random6 val 400 obj) 포팅 비용 평가

### 11-3. 중기 (Future paper 준비)

- [ ] **GraspVLA / LGD / FLASH PDF 정독** — VLM-grasp 차별화 5축 정밀화
- [ ] **VLM PoC** — CLIP 또는 GroundingDINO attention → (u,v) → 우리 Flow → grasp 예시 1~2 케이스 구현
- [ ] **Future paper 제목 후보**: *"Open-Vocabulary Multi-Modal Grasping via Pixel-Bridged Conditional Flow Matching"*

---

## 부록 A. 검증되지 않은/추가 조사 필요 후보

| 논문 | 상태 | 다음 액션 |
|---|---|---|
| TOGNet method 섹션 | abstract 만 정독 | full method 정독 (anchor 49 디테일) |
| EquiGraspFlow CNF 학습 | abstract+project page | full method 정독 (FFJORD 디테일) |
| GraspVLA 학습 데이터 / inference 비용 | project page 만 | technical report 정독 |
| GraspGen ACRONYM 결과 | abstract | full benchmark 표 정독 |
| EFF-Grasp dexterous 한정 여부 | HTML 일부 | CVPR 2026 PDF 공개 시 정독 |

## 부록 B. 사용자 GitHub 리포지토리 + 본 파일 위치

- 원격: https://github.com/seongbin1231/grasp_flow
- 본 파일: `paper_notes/RELATED_WORK.md`
- 권장: 본 파일도 git push 해서 협업자 / 지도교수 공유 가능하게
