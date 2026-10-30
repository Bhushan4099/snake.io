"""PyTorch Q-network used by the Snake DQN agent."""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map a 19-feature Snake observation to 3 action-value estimates."""

    def __init__(self, state_size: int = 19, action_size: int = 3, hidden_size: int = 128) -> None:
        super().__init__()
        if state_size <= 0 or action_size <= 0 or hidden_size <= 0:
            raise ValueError("network dimensions must be positive")

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.network(state)
