"""Fixed-size uniform replay buffer (ring buffer over preallocated NumPy arrays)."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed=None):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def push(self, s, a, r, s2, done):
        i = self.pos
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError("not enough samples in buffer")
        idx = self.rng.integers(0, self.size, size=batch_size)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]
