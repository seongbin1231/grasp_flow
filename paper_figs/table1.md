# Table 1. Quantitative comparison (val split, random6)

_Pos threshold 5 cm, Ang threshold 30° (Contact-GraspNet ICRA 2021 / ACRONYM)_
_N_samples=32, CFG=2.5, T_euler=32._

| Mode | Model | n_obj | GT groups | Pos. MAE [cm] ↓ | Ang. Err. [°] ↓ | **COV [%] ↑** | **APD [cm] ↑** |
|---|---|---|---|---|---|---|---|
| standing | Direct MLP | 40 | 3.00 | 0.44 | 47.97 | 33.3 | 0.24 |
| standing | **Ours (Flow)** | 40 | 3.00 | 0.50 | 50.93 | 99.2 | 0.50 |
| lying | Direct MLP | 268 | 1.00 | 1.29 | 2.94 | 100.0 | 0.19 |
| lying | **Ours (Flow)** | 268 | 1.00 | 1.84 | 5.29 | 100.0 | 6.26 |
| cube | Direct MLP | 92 | 1.00 | 0.65 | 0.15 | 100.0 | 0.31 |
| cube | **Ours (Flow)** | 92 | 1.00 | 0.58 | 0.90 | 100.0 | 0.35 |
| all | Direct MLP | 400 | 1.20 | 1.06 | 6.80 | 93.3 | 0.22 |
| all | **Ours (Flow)** | 400 | 1.20 | 1.41 | 8.84 | 99.9 | 4.32 |

_COV = Coverage [Achlioptas, ICML 2018]. APD = Average Pairwise Distance among predictions (diversity)._
