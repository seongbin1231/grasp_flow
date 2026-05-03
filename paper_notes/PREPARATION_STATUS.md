# IEEE급 논문 준비 상태 보고서

**작성일**: 2026-04-28 (2026-04-29 사실 검증 + 표현 정정 완료)
**목적**: 실험 시작 전, 자료 조사·가능성 검증·의사결정 항목을 한 곳에 정리해서 본격 실행 준비 완료 상태를 점검

---

## 📂 0. paper_notes/ 디렉터리 인덱스 (2026-04-28 갱신)

| 파일 | 용도 | 상태 |
|---|---|---|
| [RELATED_WORK.md](RELATED_WORK.md) | 선행연구 정밀 조사 + 차별화 5축 + must-cite 10편 | ✅ 완성 |
| [EXPERIMENTAL_SETUP.md](EXPERIMENTAL_SETUP.md) | 데이터셋 / 베이스라인 사용가능성 + 라이선스 / VLM future PoC | ✅ 완성 (graspnet-baseline 라이선스 정정 반영) |
| [DATASET_STRATEGY.md](DATASET_STRATEGY.md) | 데이터셋 변형 전략 + 활용 + risk | ✅ 완성 (8회 fact-check) |
| [EXPERIMENT_PLAYBOOK.md](EXPERIMENT_PLAYBOOK.md) | 실험 명령 카탈로그 (실행 시 사용) | ⚠️ ERRATA 23건 정정 완료. 실행 시 ERRATA 우선 |
| [PRIOR_READINGS_EQUIGRASPFLOW_TOGNET.md](PRIOR_READINGS_EQUIGRASPFLOW_TOGNET.md) | EquiGraspFlow + TOGNet 정밀 정독, 5축 차별화, Related Work paragraph 초안, bibitem | ✅ 신규 2026-04-28 |
| [PRIOR_READINGS_METHOD.md](PRIOR_READINGS_METHOD.md) | Lipman 2023 / Liu 2023 / DiT 정밀 정독, 인용 표현, 방정식 verbatim | ✅ 신규 2026-04-28 |
| **PREPARATION_STATUS.md** (이 파일) | 종합 준비 상태 + 의사결정 항목 | ✅ |

---

## 1. 논문 포지셔닝 (확정)

### 제목
*Pixel-Conditioned Multi-Modal 6-DoF Grasp Pose Generation via Conditional Flow Matching*

### 핵심 기여 (검증된 5축 차별화)
1. **단일 픽셀 (u,v) conditioning** — 생성형 6-DoF grasp 분포 모델 중 첫 사례
2. **Conditional Rectified Flow** 채택 — EquiGraspFlow 의 CNF, SE(3)-DiffusionFields 의 score-based diffusion 과 구분되는 방법론
3. **1-step Euler 추론** — multi-step ODE prior 대비 ONNX 배포 친화
4. **Depth-only single-view** 입력 — RGB·CLIP 의존성 제거
5. **다봉 분포 직접 학습** — Direct MLP 회귀의 mode collapse 와 정량 대조

이 5축 결합 사례 0건 ([RELATED_WORK.md](RELATED_WORK.md) §3 에서 검증)

### 차별화 위협 prior (반드시 인용 + 차별화)
- EquiGraspFlow (CoRL 2024) — 가장 가까움
- TOGNet (ECCV-W 2024) — 픽셀 prompt 가장 가까움 (단 anchor regression)
- GraspVLA (2025) — VLM future work 가장 가까움 (8/10 일치)
- SE(3)-DiffusionFields (ICRA 2023) — 같은 family
- GraspGen (NVIDIA 2025) — 새 SOTA

---

## 2. 데이터셋 — 사용가능성 검증 결과

| Dataset | (u,v) 추출 | 학습 가능 | 라이선스 | 작업 비용 | 결정 |
|---|---|---|---|---|---|
| 자체 grasp_v2.h5 (588 sim) | ✅ done | ✅ done | — | 0 | 🥇 Primary |
| **GraspNet-1B** (131 GB) | ✅ 100% (`label/*.png` instance mask) | ✅ ~30~50h 변환 | CC BY-NC-SA 4.0 (학술 OK) | 8~12h DL + 5d dev | 🥈 Real-world 검증용 |
| **ACRONYM** (1.6 + 12.2 GB) | ✅ 렌더링 1~2일 | ✅ 3~5일 변환 | code MIT, data CC BY-NC | 4h DL + 3~5d dev | 🥉 Generalization (옵션) |
| ~~Grasp-Anything-6D~~ | ❌ native (u,v) 없음 (검증 결과) | ❌ ZoeDepth 도메인 갭 | MIT | — | **drop, cite-only** |

### 핵심 함정 (반드시 알아두기)
1. **ACRONYM gripper 원점은 mount** (손가락 끝 z=+0.1122). 잘못 다루면 11.2cm 빗나감 → 데이터셋 무효
2. **GraspNet-1B 카메라 K** — Kinect (631) vs RealSense (927) 분기 처리 필수
3. **GraspNet-1B 단일 씬 다운 불가** — 최소 `train_1.zip 20 GB`
4. **ShapeNetSem 12.2 GB** (이전 30 GB 추정 오류 정정)

---

## 3. 베이스라인 — 코드 / 라이선스 / 호환성 검증 결과

| 베이스라인 | 코드 공개 | 라이선스 | (u,v) 입력 호환 | 우리 GPU 호환 | 결정 |
|---|---|---|---|---|---|
| **Direct MLP (자체)** | ✅ | — | native | ✅ | 🟢 포함 (ablation) |
| **GraspLDM** | ✅ | Apache 2.0 ✅ | adapter 필요 | ✅ | 🟢 포함 (인스톨 가장 쉬움, Docker) |
| **SE(3)-DiffusionFields** | ✅ | MIT ✅ | adapter 필요 | ✅ (theseus 빌드 어려움) | 🟢 포함 (필수 — generative grasp prior) |
| **6-DoF GraspNet** (PT port) | ✅ | code MIT, weights CC-BY-NC-SA 2.0 | adapter 필요 | ✅ | 🟢 포함 (sample-and-score 대표) |
| **Contact-GraspNet** | ✅ | **MIT** (NVIDIA NC 표기 정정) | adapter 필요 | ✅ (TF 2.2 + CUDA 10.1/11.1) | 🟢 포함 (direct regression 대표) |
| **GraspGen** | ✅ | NVIDIA NC ⚠️ | adapter 필요 | ✅ (CUDA 12.1 OK) | 🟡 옵션 (새 SOTA, footnote 의무) |
| **graspnet-baseline** | ✅ | Non-Commercial Research Only ⚠️ | adapter 필요 | ✅ | 🟡 옵션 (T2 보조) |
| **TOGNet** | ❌ **리포 가짜 (404)** | — | — | — | 🔴 **사용 불가, cite-only** |
| EquiGraspFlow | ✅ | MIT ✅ | adapter 필요 | ✅ | 🟡 옵션 (4-cat ckpt 한정 — OOD 위험) |
| ~~AnyGrasp~~ | binary only | Closed | — | — | **drop, Discussion mention only** |
| ~~LGD~~ | ✅ | MIT | language only | ✅ | **drop, related work 인용만** |
| ~~CROG, GraspSAM, CLIPort, GG-CNN~~ | ✅ | 다양 | 4-DoF only | — | **drop, paradigm 불일치** |

### 검증된 사전학습 ckpt 위치 (재학습 불필요)
- GraspLDM: HF [`kuldeepbarad/GraspLDM`](https://huggingface.co/kuldeepbarad/GraspLDM)
- SE(3)-DiF: HF [`camusean/grasp_diffusion`](https://huggingface.co/camusean/grasp_diffusion)
- GraspGen: HF [`adithyamurali/GraspGenModels`](https://huggingface.co/adithyamurali/GraspGenModels)
- 6-DoF GraspNet PT: Drive (repo `checkpoints/download_models.sh`)
- Contact-GraspNet: Drive (`scene_test_2048_bs3_hor_sigma_001` 등)
- graspnet-baseline: Drive (`checkpoint-rs.tar`, `checkpoint-kn.tar`)

### TOGNet 대안 (리포 부재 대응) — **D1 결정 완료 (2026-04-28)**

**조사 결과**: TOGNet 코드는 paper-only, GitHub 어디에도 release 없음. 첫저자도 정정 — **Xie et al.** (Lu 가 아님). THU-VCLab 의 9개 공개 리포 중 TOGNet 없음.

**최종 선택**: **RNGNet (THU-VCLab/RNGNet)** — TOGNet 같은 랩 + 같은 연구 라인의 후속작
- ✅ Native (u,v) prompt API: `infer_from_rgbd_centers(rgb, depth, centers_2d)` — 우리 paradigm 정확 일치
- ✅ 사전학습 ckpt (RealSense 40MB + Kinect 40MB)
- ✅ 1280×720 native 학습 — 우리 K=1109 와 호환
- ✅ 활발 (마지막 push 2026-04-22)
- ✅ Discriminative method — 우리 generative vs discriminative 차별화 narrative 강화

**이중 Pixel-prompted baseline 라인업**:
1. **RNGNet** = 외부 SOTA reference (THU-VCLab 라인 재현)
2. **자체 Direct MLP** = controlled ablation (동일 architecture / data / training)

**Related Work 표현**: *"Because the TOGNet authors have not released code, we use the publicly released RNGNet (Chen et al., 2024) from the same research group as our pixel-prompted discriminative baseline."*

---

## 4. 메트릭 — 정의 + 인용 검증 결과

| Metric | 정의 | 인용 출처 (정정됨) |
|---|---|---|
| **Pos MAE** | 가장 가까운 GT grasp 까지 위치 오차 (cm), nearest-neighbor matching | 표준 SE(3) pose 메트릭 |
| **Ang Err** | 가장 가까운 GT 와의 회전 측지선 각도 (deg) | 표준 |
| **COV (Coverage)** | GT pool 중 generated sample 이 5cm/30° 이내로 매칭한 비율 | "**inspired by Achlioptas 2018, with our 5cm/30° SE(3) matching threshold**" |
| **APD (Avg Pairwise Dist)** | 생성된 N samples 간 평균 거리 (cm) — 다양성 지표 | 우리 정의 — 자체 motivation 명시 |
| **AP (GraspNet 공식)** | force-closure precision @ 6 friction × top-50 | graspnetAPI `eval_seen/similar/novel` |
| **FleX-success (ACRONYM)** | physics-sim binary (gripper close + shake + contact 유지) | Eppner 2021 §III |

### ⚠️ 중요 정정사항 (paper-risk 제거) — **2026-04-28 완료**
- 기존 표현 *"COV (Achlioptas 2018), 5cm/30° threshold (Sundermeyer 2021)"* → **두 인용 모두 misattribution 확인**
- 정정 표현: *"We define a coverage metric inspired by Achlioptas 2018, with a 5cm/30° matching threshold chosen to suit SE(3) parallel-jaw grasps."*
- ✅ **로컬 수정 완료**:
  - [scripts/make_paper_table1.py:101, 261](../scripts/make_paper_table1.py) — 코멘트/표 footer 정정
  - 메모리 [paper_metrics_official.md](/home/robotics/.claude/projects/-home-robotics-Competition-YOLO-Grasp/memory/paper_metrics_official.md) — 출처 표 + 임계값 섹션 + 본문 표현 + bibitem 모두 정정
- ❗ paper 본문 작성 시 위 정정 표현 그대로 사용

---

## 5. 실행 전 사용자 결정 필요 항목 (Open Decisions)

### ~~D1. TOGNet 대체 베이스라인~~ → ✅ **결정 완료 (2026-04-28)**
- 결정: **RNGNet (THU-VCLab/RNGNet)** + 자체 Direct MLP 이중 라인업
- 근거: §3 의 "TOGNet 대안" 섹션 참조

### D2. mode_id 처리 방식 (multi-dataset 학습 위해)
- (a) 모델 입력에 `mode_emb` 추가 (4-bucket: lying/standing/cube/unknown), 30줄 코드 추가
- (b) GraspNet/ACRONYM 학습 시 mode 입력 비활성화 (Stage 1/2/3 별도 모델)
- 현재 모델은 mode_id 가 입력 아님 — sampler/aux loss 만 사용

### D3. 학습 regime 선택
- (a) 3-stage progressive (ACRONYM → GraspNet → 자체) — 35h compute
- (b) 자체만 학습 + zero-shot transfer — 6h compute (이미 v7 보유)
- (c) Joint training (dataset_id FiLM) — 26h, 단일 ckpt

### D4. ACRONYM 사용 여부 / 범위
- (a) 매칭 5-class subset (~600 obj, generalization 미약하지만 fair head-to-head)
- (b) Top-20 ACRONYM 카테고리 (~5,000 obj, 더 강한 일반화 주장)
- (c) ACRONYM 전혀 안 함 (Table 1 + Table 2 만, 3주 단축)

### D5. Real-world UR5e + RealSense 실험 포함 여부
- (a) 50 picks 정성 + 정량 (Table 4) — 1주 추가
- (b) qualitative 만 (Fig 5)
- (c) 없음 (sim only paper)

### D6. 투고 venue 와 마감
- IEEE T-RO: rolling 투고, 14 page, review 4~6 개월
- IEEE RA-L: rolling 투고, 8 page, review 2~3 개월 (RA-L + IROS 동시 발표 옵션)
- ICRA 2026: 마감 지남 (2025-09)
- IROS 2026: 마감 지남 (2026-03)
- ICRA 2027: 마감 ~2026-09 (시간 충분)

D6 가 가장 시급 — 마감 정해져야 W1~W4 일정 역산 가능

### D7. graspnet-baseline (Non-Commercial) 사용 여부
- 학술 paper OK 단 footnote 의무
- 상용 deploy 에 ckpt 사용 금지 → RoboCup ARM 2026 deploy 에 영향 없음 (별도 자체 모델)
- 결정: footnote 명시하고 사용 권장

### D8. ablation 범위
- 필수: Direct vs Flow, RF vs DDPM step
- 권장: w/o cross-attn, w/o multi-scale crop
- 옵션: 다른 noise schedule, CFG 강도 sweep

---

## 6. Risk Register (사전 식별)

| Risk | Severity | Mitigation |
|---|---|---|
| ACRONYM gripper 원점 잘못 다루기 | 🔴 데이터셋 무효 | 첫 단계 시각 검증 (D-day 시작 전) |
| GraspNet K 카메라별 분기 누락 | 🟠 marker-grasp 불일치 | `camera` flag detect, K assertion |
| theseus / TF1.15 등 legacy 빌드 실패 | 🟠 베이스라인 1-2개 swap | Docker 사용, fallback baseline 미리 결정 |
| TOGNet 대안 결정 지연 | 🟡 픽셀 prompt 비교 약화 | RegionNormalizedGrasp 즉시 swap 가능 |
| EquiGraspFlow 가 우리 paper 직전 update / 새 claim | 🟡 차별화 약화 | 매주 arxiv watch, 차별화 5축 어느 하나 더 강화 |
| GraspNet-1B 다운 시간 (131 GB, 8~12h) | 🟢 시간 손실 | W0 사전 다운 시작, 변환 작업과 병행 |
| 베이스라인 metric 정의 차이 | 🟢 비교 fair X | uv_adapter + 통일 metric 스크립트 모든 베이스라인 적용 |
| 메트릭 인용 misattribution | 🔴 reviewer reject | 정정 표현 적용 (이미 §4 결정) |
| Grasp-Anything-6D 잘못 가정 한 게 다른 곳에 영향 | 🟡 paper claim 약화 | RELATED_WORK.md ERRATA 정확히 반영 |
| Sim-only 비판 | 🟠 reviewer | GraspNet-1B real RealSense 결과 → 실세계 1 row |

---

## 7. 읽어둘 논문 (paper 작성 전 정독 필수)

### 우선순위 A (반드시 정독)
1. **EquiGraspFlow** (Lim CoRL 2024) — 가장 가까운 prior, 차별화 정밀화 위해
2. **TOGNet** (Xie et al., ECCV-W 2024) — 픽셀 prompt 가장 비슷, 차별화 정밀화 (코드 미공개)
3. **Lipman, Flow Matching** (ICLR 2023) — method foundation
4. **Liu, Rectified Flow** (ICLR 2023) — 우리 핵심
5. **DiT/AdaLN-Zero** (Peebles ICCV 2023) — 우리 conditioning

### 우선순위 B (권장 정독)
6. **SE(3)-DiffusionFields** (Urain ICRA 2023) — Table 1 기준 베이스라인
7. **GraspGen** (NVIDIA 2025) — 새 SOTA
8. **GraspLDM** (Barad Access 2024) — latent diffusion 비교
9. **6-DoF GraspNet** (Mousavian ICCV 2019) — sample-and-score 대표
10. **Contact-GraspNet** (Sundermeyer ICRA 2021) — direct regression 대표

### 우선순위 C (Future work / Section 6 인용)
11. **GraspVLA** (2025) — VLM future work
12. **LGD** (Nguyen ECCV 2024) — 언어 컨디셔닝 generative
13. **F3RM** (Shen CoRL 2023) — VLM 모듈식
14. **π₀** (Black 2024) — robotics-FM
15. **FoldFlow** (Bose ICLR 2024) — SE(3) FM 시초

→ 우선순위 A 정독 + 메모 작성 필수 (paper Related Work 섹션 작성 시 직접 인용 표현)

---

## 8. Target Venue 비교 (의사결정 D6 도움)

| Venue | 페이지 | 마감 | 특징 | 우리 적합도 |
|---|---|---|---|---|
| IEEE T-RO | 14 | rolling | 가장 prestigious, 긴 review (4-6개월), real-robot 비중 |  높음 (5축 모두 포함 가능) |
| IEEE RA-L | 8 | rolling | RA-L+ICRA/IROS 옵션, 빠른 review (2-3개월), 짧은 페이지 강점/약점 | 가장 적합 — 짧고 강한 contribution |
| ICRA 2027 | 8 | ~2026-09 | 학회 prestige, novelty 강조 | 적합 — 시간 충분 |
| IROS 2026 | 8 | (지남) / 2027 ~2027-03 | 학회 prestige, application 비중 | 적합 |

### 권고: **RA-L + IROS 동시 발표 옵션** (최단 경로)
- RA-L 8 page 로 5축 차별화 + Table 1·2 + Fig 1~3 + Table 4 (ablation) 수용 가능
- 채택 시 IROS 발표 자동 부여
- Rolling 투고 → 4~5월 준비, 6월 제출 현실적
- ACRONYM (Table 3) 과 real UR5e (Table 5) 까지 포함하면 T-RO 14 page 확장 가능 (후속 작업)

---

## 9. 자원 견적 (확인 완료)

| 항목 | 필요 | 보유 | 부족 시 |
|---|---|---|---|
| GPU | 24GB single | ✅ | — |
| 디스크 | 150 GB | 확인 필요 (`df -h`) | 외장 SSD |
| Docker | TF 2.2 / theseus / pyrender 빌드 | 설치만 | — |
| 시간 | dev 150h + compute 53h ≈ 5주 | — | de-scope (T1+T2+T4 만 = 3주) |
| 인력 | 1 engineer | 1 (사용자) | — |

---

## 10. 준비 완료 점검 체크리스트

### 자료 조사 단계 (현재 = ✅ 완료)
- [x] 선행연구 정밀 조사 (RELATED_WORK.md)
- [x] 차별화 5축 확정
- [x] 데이터셋 사용 가능성 검증 (DATASET_STRATEGY.md)
- [x] 베이스라인 코드/라이선스 검증 (EXPERIMENTAL_SETUP.md)
- [x] 메트릭 정의 + 인용 정정 (§4)
- [x] Risk register 작성 (§6)
- [x] Target venue 비교 (§8)
- [x] 자원 견적 (§9)

### 의사결정 단계
- [x] **D1** — TOGNet 대안 결정 → **RNGNet** ✅ (2026-04-28 완료)
- [ ] D2 — mode_id 처리 방식 결정
- [ ] D3 — 학습 regime 결정 (1-stage / 3-stage progressive / joint)
- [ ] D4 — ACRONYM 범위 결정
- [ ] D5 — real UR5e 포함 여부
- [ ] **D6 — Target venue + 마감일 결정** (가장 시급)
- [ ] D7 — graspnet-baseline 사용 여부 (footnote 검토)
- [ ] D8 — ablation 범위 확정

### 실행 직전 단계 (D-day 결정 후)
- [x] 메트릭 인용 정정 (paper-risk 제거) ✅ (2026-04-28 완료, scripts + memory 모두)
- [x] EquiGraspFlow / TOGNet 정독 → 5축 차별화 검증 ✅ (2026-04-28, [PRIOR_READINGS_EQUIGRASPFLOW_TOGNET.md](PRIOR_READINGS_EQUIGRASPFLOW_TOGNET.md))
- [x] Lipman / Liu / DiT 정독 → Method 인용 정리 ✅ (2026-04-28, [PRIOR_READINGS_METHOD.md](PRIOR_READINGS_METHOD.md))
- [x] **`scripts/train_flow.py` 에 `--pretrained` 추가** ✅ (2026-04-29, v7 ckpt 로드 missing=0/unexpected=0 검증, 1-epoch smoke val_flow=0.327 정상)
- [x] **`scripts/baselines/uv_adapter.py` 작성** ✅ (2026-04-29, smoke 합성 depth 검증 centroid 오차 <0.01m, 5 random val rows PC 20K pts z=0.35~0.71m 모두 통과)
- [x] **`scripts/baselines/eval_baseline_unified.py` 작성** ✅ (2026-04-29, Oracle GT 검증 Pos MAE=0/COV=100% 정확)
- [ ] GraspNet-1B 1 씬 sanity (train_1.zip 다운 + load 검증, 1일) — 다운로드 필요
- [ ] ACRONYM gripper 시각 검증 (1일) — 다운로드 + 시각화 필요
- [ ] mode_id 처리 방식 결정 + 적용 (D2)
- [ ] EquiGraspFlow / Liu (RF) / Peebles (DiT) / SE(3)-DiF / GraspGen / GraspLDM / 6-DoF GraspNet / Contact-GraspNet 정독 — 우선순위 B 6편 (3일)

---

## 11. 결론 — 준비 상태 요약

**자료 조사 + 가능성 검증**: ✅ **완료** (2026-04-28)
- 선행연구 검증: 5축 결합 unique pocket 확정 (EquiGraspFlow + TOGNet 5축 모두 0/5 충족)
- 데이터셋 3개 가능성: 자체 + GraspNet-1B + ACRONYM 모두 사용 가능 (GA-6D 제외)
- 베이스라인 6개 가능성: 코드 공개 + ckpt 보유 (TOGNet → **RNGNet 으로 swap 결정 완료**)
- 메트릭 정의 + 인용 정정: 식별 + 정정 표현 확정 + **로컬 수정 완료** (paper-risk 제거)
- 컴퓨팅 / 디스크 / 시간 견적: 현실적 (5주)
- 우선순위 A 정독 5편 중 5편 완료 (EquiGraspFlow + TOGNet + Lipman + Liu + DiT 모두 verbatim 인용 + 차별화 표 + Related Work paragraph 초안)

**사용자 결정 남은 것**: 7개 (D2~D8). 가장 시급: **D6 venue / 마감**

**실행 가능 상태**: D2~D8 결정 완료 시 즉시 실행 가능. 모든 함정 / 라이선스 / 명령 정정 항목이 EXPERIMENT_PLAYBOOK.md ERRATA 에 명시됨.

**IEEE 급 부합도**: 5축 차별화 + 다중 데이터셋 + 7 베이스라인 (RNGNet 추가) + 표준 metric + ablation + (옵션) real robot — RA-L 8 page 충분, T-RO 14 page extension 가능.

### 진행 작업 요약 (2026-04-28 ~ 04-29)
1. ✅ 메트릭 인용 정정 — `make_paper_table1.py` + 메모리 `paper_metrics_official.md` (Achlioptas/Sundermeyer misattribution 제거)
2. ✅ EquiGraspFlow + TOGNet 정밀 정독 — 5축 차별화 검증 + 인용 표 + Related Work paragraph 초안
3. ✅ Lipman 2023 + Liu 2023 + DiT 정독 — Method 섹션 sketch + 방정식 verbatim 인용
4. ✅ TOGNet 대체 베이스라인 결정 — RNGNet (THU-VCLab, 같은 연구 그룹, native `(u,v)` prompt API, 사전학습 ckpt 제공)
5. ✅ TOGNet 첫저자 정정 — 모든 파일에서 "Lu et al." → **"Xie et al."** 일괄 변경 (실제 첫저자: Pengwei Xie)
6. ✅ RNGNet 인용 정정 — *"Ma et al., CoRL 2024"* 추정 폐기 → **"Chen et al., 2024 (arXiv:2406.01767)"** 확정 (첫저자: Siang Chen)
7. ✅ GraspNet-1B 용량 정정 — 132 GB → **131 GB core** (라벨 6.6 GB → 실제 2.3 GB)
8. ✅ ACRONYM gripper 기하 verbatim 검증 — fingertip z = 0.112169998 m (반올림 0.1122 m 사용 무방)
9. ✅ **`scripts/train_flow.py --pretrained` 인자 추가** — v7 ckpt warm-start 검증 (missing=0/unexpected=0, smoke val_flow=0.327)
10. ✅ **`scripts/baselines/uv_adapter.py` 작성 + 검증** — depth + (u,v) → segmented PC. Smoke 합성 depth centroid 오차 < 1mm, 실제 val 5 rows 모두 통과 (PC 20K pts, z=0.35~0.71m table-top 범위)
11. ✅ **`scripts/baselines/eval_baseline_unified.py` 작성 + 검증** — Phase 1 베이스라인 결과 통합 평가 스크립트. Oracle GT 입력 시 Pos MAE=0 / COV=100% 정확 산출 검증
