"""Hyperparameters and shared constants (single source of truth)."""
from dataclasses import dataclass, asdict

GRID_W = 12
GRID_H = 12
N_FEATURES = 19
N_ACTIONS = 3

# Evaluation env seeds are disjoint from training seeds so that every policy
# (each checkpoint, each variant, and the random baseline) sees the SAME
# 100 games: episode i uses env seed EVAL_SEED_BASE + i.
EVAL_SEED_BASE = 1_000_000


@dataclass
class HParams:
    episodes: int = 500
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    learn_start: int = 1_000      # transitions in buffer before the first update
    target_sync_steps: int = 500  # hard target-network copy every N env steps
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.99       # multiplicative, applied once per episode
    hidden: int = 128
    grad_clip: float = 10.0
    ckpt_every: int = 50          # save a checkpoint every N episodes (+ final)

    def to_dict(self):
        return asdict(self)
