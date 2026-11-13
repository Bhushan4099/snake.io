"""DQN / Double DQN agent (PyTorch, CPU)."""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HParams, N_FEATURES, N_ACTIONS
from .replay import ReplayBuffer

VARIANTS = ("dqn", "double")


def make_net(hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(N_FEATURES, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, N_ACTIONS),
    )


class DQNAgent:
    def __init__(self, variant="dqn", hp: HParams = None, seed=0):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant = variant
        self.hp = hp or HParams()
        self.double = variant == "double"
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = make_net(self.hp.hidden)
        self.target = copy.deepcopy(self.online)
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=self.hp.lr)
        self.buffer = ReplayBuffer(self.hp.buffer_size, N_FEATURES, seed=seed)
        self.epsilon = self.hp.eps_start
        self.env_steps = 0

    # ----------------------------------------------------------------- acting
    @torch.no_grad()
    def q_values(self, state) -> np.ndarray:
        x = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.online(x)[0].numpy()

    def act(self, state, epsilon=None) -> int:
        """Epsilon-greedy. epsilon=0 gives the greedy policy used in evaluation."""
        eps = self.epsilon if epsilon is None else epsilon
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(N_ACTIONS))
        return int(np.argmax(self.q_values(state)))

    def decay_epsilon(self):
        self.epsilon = max(self.hp.eps_end, self.epsilon * self.hp.eps_decay)

    # --------------------------------------------------------------- learning
    def remember(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)
        self.env_steps += 1
        if self.env_steps % self.hp.target_sync_steps == 0:
            self.sync_target()

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def compute_targets(self, r, s2, d):
        """y = r + gamma * (1-d) * Q_target(s', a*) ; a* = argmax Q_target (DQN)
        or argmax Q_online (Double DQN, van Hasselt et al. 2016)."""
        with torch.no_grad():
            q2_target = self.target(s2)
            if self.double:
                a_star = self.online(s2).argmax(dim=1, keepdim=True)
                q2 = q2_target.gather(1, a_star).squeeze(1)
            else:
                q2 = q2_target.max(dim=1).values
            return r + self.hp.gamma * (1.0 - d) * q2

    def learn(self):
        """One gradient step on a replay minibatch. Returns the loss or None."""
        if len(self.buffer) < max(self.hp.batch_size, self.hp.learn_start):
            return None
        s, a, r, s2, d = (torch.as_tensor(x) for x in self.buffer.sample(self.hp.batch_size))
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        y = self.compute_targets(r, s2, d)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.hp.grad_clip)
        self.opt.step()
        return float(loss.item())

    # ------------------------------------------------------------ persistence
    def save(self, path, episode=None):
        torch.save({"model": self.online.state_dict(), "variant": self.variant,
                    "hp": self.hp.to_dict(), "episode": episode,
                    "epsilon": self.epsilon}, path)

    @classmethod
    def load(cls, path, seed=0):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        hp = HParams(**ck["hp"])
        agent = cls(ck["variant"], hp, seed=seed)
        agent.online.load_state_dict(ck["model"])
        agent.sync_target()
        agent.epsilon = 0.0
        agent.loaded_episode = ck.get("episode")
        return agent
