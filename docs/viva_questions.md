# Viva questions (15) with answers grounded in this code

Numbers quoted here are copied from `results/results_table.md` / `results/summary.json`.

**1. Why DQN and not tabular Q-learning for Snake?**
The state is a vector of 19 mixed binary and continuous features (`SnakeEnv.get_state`), including normalised distances and a flood-fill ratio. That space is far too large and continuous-valued to tabulate, so a function approximator (an MLP, `agent.make_net`) generalises across states.

**2. What exactly is the MDP?**
State: the 19-feature vector. Actions: 3, relative to heading (straight, left, right). Reward: +10 food, -10 death (wall, body, starvation), +/-0.1 shaping for getting closer to / farther from food (Manhattan). Transitions: deterministic except where new food spawns (uniformly on free cells). Discount gamma = 0.95 (`config.HParams`).

**3. Why relative actions?**
It removes the illegal 180-degree reversal from the action space and makes "turn left" mean the same thing in every heading, so the policy is more rotation-invariant. Manual Play converts absolute key presses with `env.relative_action`, and a reversal is mapped to straight.

**4. Why is the state only approximately Markov?**
The food position is encoded only relative to the head, and the body is encoded through danger flags, rays and a flood-fill ratio, not the full grid. Two different boards can map to the same vector, so the agent is working with a partially observed, hand-engineered state. The hunger feature (feature 18) is included so the starvation cutoff is observable.

**5. Why a replay buffer?**
Consecutive Snake transitions are highly correlated and the data distribution drifts as the policy changes. Uniform sampling from a 50,000-transition ring buffer (`replay.ReplayBuffer`) decorrelates minibatches and reuses data. Learning starts after 1,000 transitions.

**6. Why a target network, and how often is it updated?**
Bootstrapping from the same network that is being updated makes the regression target move with every step. `DQNAgent.target` is a frozen copy refreshed by a hard copy every 500 environment steps (`remember`/`sync_target`).

**7. Write the DQN target and the Double DQN target.**
DQN: y = r + gamma (1 - done) max_a' Q_target(s', a'). Double DQN: y = r + gamma (1 - done) Q_target(s', argmax_a' Q_online(s', a')). Both are in `DQNAgent.compute_targets`; `tests/test_agent.py` checks them against hand-computed values with deliberately disagreeing networks.

**8. Why does Double DQN help in theory?**
The max over noisy estimates is biased upward (maximization bias). Selecting with one network and evaluating with the other decouples selection from evaluation (van Hasselt et al. 2016). Here there are only 3 actions and the rewards are bounded, so the expected effect is small.

**9. Did Double DQN beat DQN in your runs?**
No. In `results/results_table.md` (5 seeds, 500 episodes, 100 greedy games per seed) DQN's mean of seed means is 21.66 and Double DQN's is 20.56, with std across seeds of 0.73 and 0.53. Double DQN is therefore not better here; the seed ranges (20.50..22.38 vs 19.88..21.17) overlap, and with 5 seeds and no significance test I only say DQN scored somewhat higher in this setup, not that it is provably better. A plausible reason for no gain is that there are only 3 actions and bounded rewards, so overestimation is mild, but I did not test that explanation.

**10. Why Huber loss and gradient clipping?**
TD targets can have large errors early (rewards of +/-10 against shaping of +/-0.1). Smooth-L1 is quadratic near zero and linear beyond, which limits gradient magnitude; clip-grad-norm at 10 adds a second guard (`DQNAgent.learn`).

**11. Explain the exploration schedule.**
Epsilon-greedy, epsilon starts at 1.0, multiplied by 0.99 after every episode, floored at 0.01 (`decay_epsilon`). With 500 episodes it reaches the floor region near the end of training. Evaluation uses epsilon = 0 (`agent.act(s, epsilon=0.0)`).

**12. How is evaluation made fair and seed-averaged?**
Each policy plays 100 games where game i uses env seed 1,000,000 + i, so all policies (every checkpoint, both variants, the random baseline) see the same start directions and the same food-spawn RNG stream. We train several independent seeds per variant, compute each seed's 100-game mean, then report mean and sample std (ddof=1) across seeds (`pipeline.evaluate_all`). The evaluation seeds are disjoint from training env seeds (0..k).

**13. What do the training curves show and what is their limitation?**
`results/plots/` has score with moving average (window 50), mean TD loss and epsilon per episode. The training score includes exploration noise (epsilon > 0), so it understates greedy performance; loss in DQN is not a convergence measure because the targets themselves move.

**14. What is the checkpoint compare, and what is its pitfall?**
Every 50 episodes a checkpoint is saved; `pipeline.compare_checkpoints` evaluates each one greedily on the same 100 games. Pitfall: picking the best-scoring checkpoint using those same games is selection on the test set, so the headline result uses only the final checkpoint.

**15. What are the main weaknesses and what is future work?**
Untuned hyperparameters; few seeds and no significance test; 12x12 only; hand-crafted features (no ablation of which of the 19 matter); starvation treated as terminal rather than truncation; the agent does not fill the board (no optimal-play claim). Future work: other RL algorithms, longer training, tuning, feature ablation, ESP32 deployment (none of it built).
