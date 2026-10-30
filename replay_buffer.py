"""Fixed-capacity experience replay buffer for DQN."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Store transitions and sample independent mini-batches."""

    def __init__(self, capacity: int = 100_000, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self._memory: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._memory)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        if state_array.ndim != 1 or next_state_array.shape != state_array.shape:
            raise ValueError("state and next_state must be equally sized 1D arrays")
        if not np.isfinite(state_array).all() or not np.isfinite(next_state_array).all():
            raise ValueError("states must contain only finite values")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if action not in (0, 1, 2):
            raise ValueError("action must be 0, 1, or 2")

        # Copy the arrays so later environment mutations cannot alter memory.
        self._memory.append(
            Transition(state_array.copy(), int(action), float(reward), next_state_array.copy(), bool(done))
        )

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> ReplayBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(self._memory) < batch_size:
            raise ValueError(
                f"not enough experiences: requested {batch_size}, have {len(self._memory)}"
            )

        transitions = self._rng.sample(list(self._memory), batch_size)
        return ReplayBatch(
            states=torch.as_tensor(np.stack([t.state for t in transitions]), dtype=torch.float32, device=device),
            actions=torch.as_tensor([t.action for t in transitions], dtype=torch.int64, device=device),
            rewards=torch.as_tensor([t.reward for t in transitions], dtype=torch.float32, device=device),
            next_states=torch.as_tensor(
                np.stack([t.next_state for t in transitions]), dtype=torch.float32, device=device
            ),
            dones=torch.as_tensor([t.done for t in transitions], dtype=torch.float32, device=device),
        )
