---
name: paper-figure-generation
description: "IEEE 급 논문용 figure (Fig 1 mode-wise GT, Fig 2 architecture, Fig 3 Direct vs Flow 비교) + Table 1 (COV/APD/Pos MAE/Ang Err) 생성·갱신·다듬기. matplotlib 3D + RGB-색칠 PC + mathtext + zorder 강제 + standing top-down/side balance + per-mode dist filter + Achlioptas/Sundermeyer 인용 metric. 사용자가 'Fig 1/2/3 다시', 'Table 갱신', '색·마진·뷰각·N개·CFG·threshold 변경', '논문 그림', 'mode coverage / COV / APD' 요청 시 반드시 사용."
---

# Paper Figure Generation

논문 figure / table 4 종을 일관된 디자인 규칙으로 생성·갱신.

## Quick reference — 산출물 4 종

| ID | 산출 | 스크립트 | 핵심 파라미터 |
|---|---|---|---|
| Fig 1 | `paper_figs/fig1_gt_synthesis.{png,pdf}` | `scripts/make_paper_fig1_gt.py` | CASES (sid/obj_idx), elev/azim, gripper scale 0.65 |
| Fig 2 | `paper_figs/fig2_architecture.{png,pdf}` | `scripts/make_paper_fig2_arch.py` | 박스 색·폰트, 16:4 wide |
| Fig 3 | `paper_figs/fig3_compare.{png,pdf}` | `scripts/make_paper_fig3_compare.py` | N_SAMPLES, CFG_W, DIST_THRESH_BY_MODE, SEED |
| Table 1 | `paper_figs/table1.{md,json}` | `scripts/make_paper_table1.py` | POS_TH_M=0.05, ANG_TH_DEG=30, CFG_W=2.5, max_objs |

## 공통 환경 상수

- 카메라: `K_FX=K_FY=1109, K_CX=640, K_CY=360`, 1280×720
- Depth: uint16 PNG mm → `/1000.0` meter
- Frame: camera 전용 (base 변환 금지)
- Python: `/home/robotics/anaconda3/bin/python`

## Fig 1 디자인 규칙 (모드별 GT 합성, 3-panel)

```python
# CASES = (label, sid, obj_idx, ply_short, mode, elev, azim)
("Standing can\n(top-down 8 + side-45 8 + side-cap 8 = 24)",
 "sample_random6_32", 0, "can", "standing", -15, 205),
("Lying can\n(4 pos x 3 tilts x 180 sym = 24)",
 "sample_random6_11", 0, "can", "lying", -15, 205),
("Cube\n(edge-aligned 2 yaws)",
 "sample_random6_30", 7, "cube_red", "cube", -15, 205),
```

- RGB 실측 색칠: `depth_mask_to_pc_rgb()` + `rgb_path_for_depth()` (depth 폴더 → captured_images 매핑)
- 그리퍼 색: `#1976d2` (진청, RGB-can 녹색과 충돌 회피)
- 그리퍼 35% 작게: `GRIPPER_HALF=0.0425*0.65, FINGER_LEN=0.040*0.65, PALM_BACK=0.025*0.65`
- View: `view_init(elev=-15, azim=205, vertical_axis='y')` + `ax.invert_yaxis()`
- z-order 강제: `ax.computed_zorder=False`, scatter `zorder=1, alpha=0.30, s=2.5`, plot `zorder=10, lw=1.5`
- 마진 0: `subplots_adjust(left=0,right=1,top=0.92,bottom=0,wspace=0)` + `pad_inches=0.02`
- 제목 폰트 15

## Fig 2 디자인 규칙 (architecture, 단일 행)

- `FancyBboxPatch` 둥근 사각형 + `FancyArrowPatch` 화살표
- mathtext: `D \in \mathbb{R}^{H \times W}`, `E_{\text{global}}(D)`, `E_{\text{local}}(D|_{u,v})`, `\mathbf{c}`, `g_t`, `v_\theta(g_t,t,\mathbf{c})`, `\hat{v} \in \mathbb{R}^8`
- 색: 입력/출력 베이지 `#fff5e0`, global encoder 초록 `#e7f4ea`, local encoder 민트 `#dff0e0`, cond 라일락 `#f3e6f9`, velocity 살구 `#fde4e1`
- 폰트: 입력/인코더 16, velocity 20, cond 26, 출력 17
- 16:4 aspect, `set_position([0,0,1,1])` + `pad_inches=0.02`
- **신경망 깊이/dim/AdaLN 내부 노출 금지** (사용자 강조)

## Fig 3 디자인 규칙 (Direct vs Flow 비교, 2×3 grid)

```python
N_SAMPLES = 16
N_OVERSAMPLE = 48          # filter+balance 위해 우선 많이 뽑음
T_EULER = 32
NOISE_TEMP = 0.8
CFG_W = 2.5                # 3.5 는 outlier 다수
DIRECT_UV_JITTER_PX = 2.0  # Direct N 샘플용
DIST_THRESH_BY_MODE = {"standing": 0.10, "lying": 0.07, "cube": 0.06, "default": 0.10}
SEED = 7

# CASES — seen 씬 사용
("Standing can",  "sample_random4_56", 0, "greenCan.ply",   -15, 205),
("Lying can",     "sample_random1_39", 0, "greenCan.ply",   -15, 205),
("Cube",          "sample_random4_41", 4, "cube.ply",       -15, 205),
```

- CFG 구현: **cond zero** (depth zero 아님)
  ```python
  cond_on = model.encode(d_b, u_b)
  cond_off = torch.zeros_like(cond_on)
  v_on = model.velocity(g_t, cond_on, t_emb, uv_norm)
  v_off = model.velocity(g_t, cond_off, t_emb, uv_norm)
  v = v_off + w_cfg * (v_on - v_off)
  ```
- Direct N 샘플: uv ±2px Gaussian jitter (deterministic 모델의 입력 불확실성 모사)
- `filter_and_balance()`:
  1. dist filter (객체 중심 기준)
  2. standing 한정 top-down/side 12:4 sub-sampling (`|a_z|>0.6` top)
- 색: Direct `#c0392b` 빨강, Ours `#1976d2` 진청
- `ax.computed_zorder=False`, scatter `zorder=1, alpha=0.30, s=2.5`, plot `zorder=10, lw=1.4`
- 그리퍼 35% 작게, 제목 폰트 15

## Table 1 디자인 규칙

- 4 metric × per-mode breakdown (standing / lying / cube / all)
- COV: GT-side group coverage (각 GT 그룹에 (5cm, 30°) 안 pred 있나)
- APD: pred set 의 평균 쌍별 위치 거리 (cm)
- 인용: Achlioptas et al., ICML 2018 (COV) + Sundermeyer et al., ICRA 2021 (5cm/30°)
- 출력: `table1.md` (markdown 표) + `table1.json` (numeric)

## Mode ID 주의

`grasp_v2.h5` 메타 `mode_names: ['lying','standing','cube']` → **0=lying, 1=standing, 2=cube** (헷갈리기 쉬움)

## Group ID

`group_names: ['top-down','side-cap','lying','cube','side-45']` → 0/1/2/3/4

- standing GT: groups [0, 1, 4] (3 모드)
- lying GT: group [2] (1 모드)
- cube GT: group [3] (1 모드)

## 자주 쓰는 명령

```bash
cd /home/robotics/Competition/YOLO_Grasp

# 단일 figure
/home/robotics/anaconda3/bin/python scripts/make_paper_fig1_gt.py
/home/robotics/anaconda3/bin/python scripts/make_paper_fig2_arch.py
/home/robotics/anaconda3/bin/python scripts/make_paper_fig3_compare.py

# 전체 figure 일괄
for f in 1 2 3; do
  /home/robotics/anaconda3/bin/python scripts/make_paper_fig${f}_*.py
done

# Table 1 (val 400 obj)
/home/robotics/anaconda3/bin/python scripts/make_paper_table1.py --max_objs 400
```

## 후보 탐색 (다른 씬으로 교체 시)

`scripts/scan_can_candidates.py` — 12 후보를 하나의 그림으로 나열. 사용자가 보고 sid/obj_idx 골라서 알려주면 CASES 갱신.

## 자주 받는 수정 요청 → 빠른 답

| 요청 | 수정 위치 |
|---|---|
| "Fig X 마진 더 줄여" | `subplots_adjust(top=0.92→0.96, bottom=0→0)` + `pad_inches=0.02→0.005` |
| "그리퍼 색 바꿔" | Fig 1 `color_map`, Fig 3 `grasp_color` |
| "N 더 늘려/줄여" | `N_SAMPLES`, `N_OVERSAMPLE` 비례 |
| "CFG 변경" | `CFG_W` (table 도 같이 갱신 권장) |
| "viewing angle 회전" | `CASES` 의 elev/azim |
| "다른 씬으로" | `scan_can_candidates.py` 로 후보 → CASES 교체 |
| "threshold 변경" | `POS_TH_M`, `ANG_TH_DEG` |
| "standing top-down 더 노출" | `filter_and_balance` 의 `n_top_target` 비율 ↑ |
| "lying outlier 제거" | `DIST_THRESH_BY_MODE['lying']` ↓ 또는 CFG ↓ |

## 본문 표현 (인용 가능)

상세는 메모리 `paper_metrics_official.md` 참조. 핵심 한 줄:

> "위치 5 cm·접근 30° 임계값(Sundermeyer ICRA 2021)에서 기준 모델은 standing(GT 3 모드)에서 **33.3% COV** 에 그치며 mode collapse 가 정량 확인되었고, 제안 모델은 **99.2% COV** 와 **20× APD** 를 달성하였다."
