import numpy as np
import pytest
from snakelab.replay import ReplayBuffer


def test_push_len_and_sample_shapes():
    b = ReplayBuffer(10, 19, seed=0)
    for i in range(5):
        b.push(np.full(19, i), i % 3, 1.0, np.full(19, i + 1), i == 4)
    assert len(b) == 5
    s, a, r, s2, d = b.sample(4)
    assert s.shape == (4, 19) and a.shape == (4,) and s2.shape == (4, 19) and d.shape == (4,)


def test_ring_overwrites_oldest():
    b = ReplayBuffer(3, 2, seed=0)
    for i in range(5):
        b.push(np.array([i, i]), 0, 0.0, np.array([i, i]), False)
    assert len(b) == 3
    assert sorted(b.s[:, 0].tolist()) == [2.0, 3.0, 4.0]


def test_sample_consistency_and_errors():
    b = ReplayBuffer(8, 2, seed=1)
    for i in range(8):
        b.push(np.array([i, i]), i % 3, float(i), np.array([i + 1, i + 1]), False)
    s, a, r, s2, d = b.sample(8)
    assert np.allclose(s2, s + 1) and np.allclose(r, s[:, 0])
    with pytest.raises(ValueError):
        ReplayBuffer(8, 2).sample(1)


def test_seeded_sampling_reproducible():
    def mk():
        b = ReplayBuffer(20, 1, seed=7)
        for i in range(20):
            b.push(np.array([i]), 0, 0.0, np.array([i]), False)
        return b.sample(5)[0]
    assert np.array_equal(mk(), mk())
