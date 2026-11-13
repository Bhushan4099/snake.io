# Feature table (MUST / SHOULD / LATER)

Status values: DONE = implemented and covered by a test or run in this repo, PARTIAL = implemented with a stated limitation,
TODO = not implemented. Nothing marked LATER has been built.

## Project features

| Feature | Priority | Status | Where / note |
|---|---|---|---|
| Headless grid Snake environment (separate from rendering) | MUST | DONE | `snakelab/env.py`, `tests/test_env.py` |
| 19-feature state vector | MUST | DONE | `SnakeEnv.get_state`, table in README |
| 3 relative actions (straight / left / right) | MUST | DONE | `env.py`, tested incl. reversal guard for manual input |
| Replay buffer | MUST | DONE | `snakelab/replay.py`, `tests/test_replay.py` |
| DQN agent (online + target net, Huber loss, Adam) | MUST | DONE | `snakelab/agent.py` |
| Train 500 episodes, evaluate 100 episodes vs random | MUST | DONE | `run_all.py` |
| Random baseline | MUST | DONE | `evaluate.random_policy` |
| Manual Play mode (Pygame) | MUST | PARTIAL | headless smoke-tested only; not visually verified |
| AI Auto Play (speed slider, Q-values on screen) | MUST | PARTIAL | headless smoke-tested only; not visually verified |
| Train / Evaluate / Random from the Pygame menu | SHOULD | PARTIAL | runs headless tasks in a thread and prints text; smoke-tested only |
| Training curves (score + moving average, loss, epsilon) | MUST | DONE | `results/plots/` |
| Checkpoint compare (table + plot) | MUST | DONE | `results/checkpoint_compare*.csv`, plot |
| Double DQN ablation, same hyperparameters | MUST | DONE | `agent.compute_targets`, unit-tested by hand-computed targets |
| Seed-averaged evaluation (>= 3 seeds, 100 games each) | MUST | DONE | `results/summary.json`, `eval_per_run.csv` |
| README with Mermaid flow + every mode | MUST | DONE | `README.md` |
| One-command run + `--quick` | MUST | DONE | `run_all.py` |
| Methodology speech, viva questions | SHOULD | DONE | `docs/` |
| Statistical significance test between variants | SHOULD | TODO | only mean +/- std across seeds is reported; 3 seeds is too few for a meaningful test |
| More seeds (>= 5) / longer training (> 500 episodes) | SHOULD | TODO | limited by the compute budget of v1 |
| Hyperparameter sweep (lr, gamma, eps decay, target sync) | SHOULD | TODO | one fixed setting used for both variants, untuned |
| Prioritized replay, dueling, n-step | LATER | TODO | not implemented |
| Other RL algorithms (PPO, A2C, ...) | LATER | TODO | future work |
| CNN agent on raw pixels | LATER | TODO | future work |
| Optimal / perfect-game play (fills the board) | LATER | TODO | not achieved; no claim made |
| ESP32 deployment | LATER | TODO | future work; nothing built or tested |
| Windows-native Pygame verification by a human | MUST | TODO | the student must do this before the review |

## State features (19)

| # | Name | Priority | Status |
|---|---|---|---|
| 0-2 | danger straight / left / right | MUST | DONE |
| 3-6 | heading one-hot (up/right/down/left) | MUST | DONE |
| 7-10 | food direction flags (up/right/down/left) | MUST | DONE |
| 11-12 | signed normalised food dx, dy | SHOULD | DONE |
| 13-15 | normalised free distance (ray) straight / left / right | SHOULD | DONE |
| 16 | snake length / board area | SHOULD | DONE |
| 17 | free-space fraction reachable via flood fill from the cell ahead | SHOULD | DONE |
| 18 | hunger: steps since food / hunger limit | SHOULD | DONE |
| - | tail-reachability / Hamiltonian-cycle hint | LATER | TODO |
| - | feature ablation (which of the 19 matter) | LATER | TODO |
