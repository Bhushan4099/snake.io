"""DQN agent: epsilon-greedy action selection and Double-network targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .model import QNetwork
from .replay_buffer import ReplayBuffer


@dataclass(slots=True)
class DQNConfig:
    state_size: int = 19
    action_size: int = 3
    hidden_size: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.90
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    replay_capacity: int = 100_000
    target_update_interval: int = 1_000
    gradient_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.state_size != 19 or self.action_size != 3:
            raise ValueError("This Snake environment requires state_size=19 and action_size=3")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= min <= start <= 1")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size >= 1")
        if self.target_update_interval < 1 or self.gradient_clip <= 0:
            raise ValueError("target_update_interval and gradient_clip must be positive")


class DQNAgent:
    """DQN learner designed to run on CPU and keep inference lightweight."""

    def __init__(
        self,
        config: DQNConfig | None = None,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.online_network = QNetwork(self.config.state_size, self.config.action_size, self.config.hidden_size).to(self.device)
        self.target_network = QNetwork(self.config.state_size, self.config.action_size, self.config.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.config.learning_rate
        )
        self.loss_function = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(self.config.replay_capacity, seed=seed)
        self.epsilon = self.config.epsilon_start
        self.learn_steps = 0

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for the three actions without modifying the model."""
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.config.state_size,):
            raise ValueError(f"state must have shape ({self.config.state_size},)")
        tensor = torch.as_tensor(state_array, dtype=torch.float32, device=self.device)
        self.online_network.eval()
        values = self.online_network(tensor).squeeze(0).cpu().numpy().copy()
        self.online_network.train()
        return values

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Choose a relative action; set explore=False for greedy evaluation/play."""
        if explore and random.random() < self.epsilon:
            return random.randrange(self.config.action_size)
        return int(np.argmax(self.q_values(state)))

    def remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def learn(self) -> float | None:
        """Perform one optimizer update; return loss or None until buffer is warm."""
        if len(self.memory) < self.config.batch_size:
            return None

        batch = self.memory.sample(self.config.batch_size, self.device)
        current_q = self.online_network(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(batch.next_states).max(dim=1).values
            target_q = batch.rewards + self.config.gamma * next_q * (1.0 - batch.dones)

        loss = self.loss_function(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.config.target_update_interval == 0:
            self.sync_target_network()
        return float(loss.detach().cpu().item())

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_steps": self.learn_steps,
            "config": asdict(self.config),
            "extra": extra or {},
        }
        torch.save(checkpoint, destination)
        return destination

    def load(self, path: str | Path, load_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint.get("target_network", checkpoint["online_network"]))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.config.epsilon_min))
        self.learn_steps = int(checkpoint.get("learn_steps", 0))
        return dict(checkpoint.get("extra", {}))
