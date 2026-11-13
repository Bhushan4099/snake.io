# Two-minute methodology speech (DQN, for an RL professor)

*Spoken text below the line only. Quote any results from `results/results_table.md` on the day, not from memory.*

---

We model Snake as a Markov decision process on a 12 by 12 grid. The state is a hand-designed vector of 19 features: danger flags, heading, food direction, normalised food offsets, obstacle ray distances, snake length, a flood-fill free-space estimate, and a hunger counter. Actions are relative to the heading: straight, left, right. Reward is plus ten for food, minus ten for dying, and plus or minus 0.1 shaping for moving toward or away from the food.

The agent is a Deep Q-Network, as in Mnih and colleagues: an MLP with two hidden layers of 128 units and three Q-value outputs. We act epsilon-greedily, with epsilon decaying by 0.99 per episode from 1.0 to a floor of 0.01. Transitions go into a 50,000-entry replay buffer; after 1,000 transitions we take one gradient step per environment step on a uniform minibatch of 64, which breaks temporal correlation.

The regression target is the reward plus gamma, 0.95, times the maximum next-state Q-value from a separate target network, copied from the online network every 500 environment steps to stabilise bootstrapping. We minimise the Huber loss with Adam and gradient clipping.

As an ablation we implemented Double DQN, following van Hasselt and colleagues. Only the target changes: the online network selects the next action and the target network evaluates it, reducing the overestimation bias of the max operator. All hyperparameters are identical across variants.

Evaluation is greedy, epsilon zero, for 100 games on fixed environment seeds, shared with a uniform-random baseline. We train several independent seeds per variant and report mean and spread across seeds, because single deep RL runs are noisy.

The limitations are real. Hyperparameters are untuned, the seed count is small so we make no significance claim between the variants, and starvation is treated as terminal rather than as a time-limit truncation. Optimal play and other algorithms are future work.
