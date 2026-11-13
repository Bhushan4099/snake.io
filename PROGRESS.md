# PROGRESS

Status: v1 COMPLETE (after one interruption; training finished, evaluation re-run with --skip-train).

## Provenance
This is a v1 REBUILD from a written spec. The student's earlier implementation was not available; no code, results or scores
from it were reused or quoted. Every number in README/docs comes from files written to `results/` by this code.

## Done
- Headless env, 19 features, relative actions, replay buffer, DQN + Double DQN agent, training loop, evaluation, plots, pipeline, CLI, Pygame UI.
- pytest suite (env, replay, agent, headless UI smoke).
- Full training: 5 seeds x {DQN, Double DQN} x 500 episodes completed (10 final.pt + checkpoints every 50 episodes).
- README, docs/methodology_speech.md, docs/viva_questions.md, docs/feature_table.md written.

## Pipeline completion (verified from files)
- Training: results/run_all.log shows all 10 runs reached ep 500/500; 10 final.pt present (~300 s per run on 2 CPU cores, 2 runs in parallel).
- The first run_all was interrupted during evaluation (no summary.json/plots). Evaluation, checkpoint compare, plots, results_table.md,
  summary.json and the README results block were regenerated with `python run_all.py --skip-train --seeds 5` (log: results/run_all_eval.log).
  summary.json 'timing' is therefore empty (training was not re-run); training times are in run_all.log.
- pytest: 35 passed (results/pytest_output.txt).

## Interpretation and caveats
- DQN 21.66 +/- 0.73 (std across 5 seed means), Double DQN 20.56 +/- 0.53, random 0.15 +/- 0.05. Double DQN did NOT beat DQN here; seed ranges overlap; no significance test; do not claim either is better.
- Deaths of trained agents are mostly self-collision (body), not starvation; no run ever filled the board (0 'won', 0 'starved' in eval).
- Max single-game scores (46 DQN, 48 Double) are single games, not typical performance.
- Checkpoint table uses the same eval games as the headline; best-checkpoint picking would be optimistic, headline uses final only.
- Scores still rising slowly at ep 500 (see plots); 500 episodes is the original protocol, not convergence.

## Open issues
- Pygame UI not visually verified (headless smoke only).
- Windows-native run not tested (sandbox is Linux).

## Assumptions
- 12x12 grid; reward +10/-10 with +/-0.1 distance shaping; hunger limit 100+5*length ends episode (-10).
- Hyperparameters chosen up front, untuned, identical for both variants.
- Used 5 seeds (spec minimum 3; README default is 3) since compute allowed it.
- torch installed from PyPI (CPU index blocked); version 2.15.0 (CUDA build, used on CPU).

## Windows run + UI update (2026-11-13)
Details: `docs/session_log_2026-11-13.md`.
- Native Windows works on Python 3.12 (`.venv312`): 35 tests passed. Python 3.14 fails (no pygame wheel).
- Manual Play got a speed slider (2-30 steps/s, drag or +/-); the slider is now a shared `Slider` class.
- UI restyled: black/blue/red palette, square blocks, 5x7 bitmap pixel font (`snakelab/pixelfont.py`). Screenshots regenerated.
- The "Pygame UI not visually verified" and "Windows-native run not tested" open issues above are now resolved.

## Review pass (2026-11-11)
A separate reviewer was started and stopped before changing anything; the lead did a shorter pass.
- Tests: 35 passed.
- Every screen rendered to an image and inspected (menu, Manual Play, AI Auto Play with Q-values and speed slider, Evaluate DQN, Random Baseline). All draw correctly. Fixed: the menu's hint line was cut off at the bottom of the window. Screenshots in `docs/screenshots/`.
- RL update checked by reading the code: TD target `r + gamma * (1 - done) * Q_target(s', a*)`; Double DQN differs from DQN only in choosing `a*` with the online network; Huber loss; gradient clipping. No bug found.
- Docs scanned for overclaims: no claim that Double DQN is better, no optimal-play claim, ESP32 listed only as future work, rebuild-from-spec note present.
- Clean copy on Python 3.12.3 (fresh venv from requirements.txt): 35 tests passed.
- NOT done in this pass (the full reviewer brief covered these): checking each of the 19 features against hand-built boards, line-by-line audit of viva answers, real-display interaction on Windows.
