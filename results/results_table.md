| Policy | Seeds | Train episodes | Eval episodes / seed | Mean eval score (mean of seed means) | Std across seeds | Min..Max seed mean | Max single-game score |
|---|---|---|---|---|---|---|---|
| Random | 5 | n/a | 100 | 0.15 | 0.05 | 0.08..0.23 | 2 |
| DQN | 5 | 500 | 100 | 21.66 | 0.73 | 20.50..22.38 | 46 |
| Double DQN | 5 | 500 | 100 | 20.56 | 0.53 | 19.88..21.17 | 48 |

Per-seed mean scores (each over 100 eval games):

| Policy | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| Random | 0.14 | 0.08 | 0.23 | 0.16 | 0.13 |
| DQN | 22.38 | 21.55 | 22.16 | 21.69 | 20.50 |
| Double DQN | 20.78 | 19.88 | 20.13 | 21.17 | 20.82 |

Checkpoint comparison (mean ± std across seeds of per-checkpoint mean eval score):

| Checkpoint (episode) | DQN | Double DQN |
|---|---|---|
| 50 | 11.14 ± 1.80 | 9.02 ± 2.77 |
| 100 | 12.24 ± 5.74 | 12.78 ± 3.14 |
| 150 | 15.09 ± 1.99 | 14.26 ± 3.46 |
| 200 | 17.40 ± 2.36 | 16.66 ± 2.59 |
| 250 | 18.67 ± 1.62 | 18.97 ± 1.01 |
| 300 | 20.22 ± 1.61 | 19.87 ± 1.19 |
| 350 | 19.95 ± 0.89 | 19.96 ± 0.95 |
| 400 | 19.66 ± 0.68 | 20.95 ± 1.07 |
| 450 | 21.22 ± 1.00 | 20.62 ± 0.96 |
| 500 | 21.66 ± 0.73 | 20.56 ± 0.53 |
