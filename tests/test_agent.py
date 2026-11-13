import numpy as np
import pytest
import torch
from snakelab.agent import DQNAgent
from snakelab.config import HParams
from snakelab.env import SnakeEnv


def small_hp(**kw):
    base = dict(learn_start=64, batch_size=32, buffer_size=1000, episodes=3, ckpt_every=1)
    base.update(kw)
    return HParams(**base)


def test_q_shape_and_act_range():
    a = DQNAgent("dqn", small_hp(), seed=0)
    s = SnakeEnv(seed=0).get_state()
    assert a.q_values(s).shape == (3,)
    assert all(a.act(s, epsilon=1.0) in (0, 1, 2) for _ in range(50))
    assert a.act(s, epsilon=0.0) == int(np.argmax(a.q_values(s)))


def test_invalid_variant():
    with pytest.raises(ValueError):
        DQNAgent("rainbow")


def test_epsilon_decay_and_floor():
    a = DQNAgent("dqn", small_hp(eps_decay=0.5, eps_end=0.1), seed=0)
    for _ in range(10):
        a.decay_epsilon()
    assert a.epsilon == pytest.approx(0.1)


def test_learn_waits_for_buffer_then_updates():
    a = DQNAgent("dqn", small_hp(), seed=0)
    env = SnakeEnv(seed=0)
    s = env.get_state()
    assert a.learn() is None
    for _ in range(80):
        s2, r, d, _ = env.step(a.act(s, 1.0))
        a.remember(s, 0, r, s2, d)
        s = env.reset() if d else s2
    before = [p.clone() for p in a.online.parameters()]
    loss = a.learn()
    assert loss is not None and np.isfinite(loss)
    assert any(not torch.equal(b, p) for b, p in zip(before, a.online.parameters()))


def test_target_sync():
    a = DQNAgent("dqn", small_hp(target_sync_steps=5), seed=0)
    for p in a.online.parameters():
        p.data.add_(1.0)
    assert not all(torch.equal(x, y) for x, y in zip(a.online.parameters(), a.target.parameters()))
    z = np.zeros(19, np.float32)
    for _ in range(5):
        a.remember(z, 0, 0.0, z, False)
    assert all(torch.equal(x, y) for x, y in zip(a.online.parameters(), a.target.parameters()))


def test_targets_dqn_vs_double_hand_computed():
    """Make online and target disagree on argmax so the two rules give different y."""
    hp = small_hp(gamma=0.9)
    dq, dd = DQNAgent("dqn", hp, seed=0), DQNAgent("double", hp, seed=0)
    for ag in (dq, dd):
        for p in list(ag.online.parameters()) + list(ag.target.parameters()):
            torch.nn.init.zeros_(p)
        ag.online[-1].bias.data = torch.tensor([0.0, 5.0, 0.0])   # online picks action 1
        ag.target[-1].bias.data = torch.tensor([4.0, 1.0, 0.0])   # target max is action 0 (=4)
    r, s2, d = torch.tensor([1.0, 1.0]), torch.zeros(2, 19), torch.tensor([0.0, 1.0])
    y_dqn = dq.compute_targets(r, s2, d)
    y_dd = dd.compute_targets(r, s2, d)
    assert y_dqn[0].item() == pytest.approx(1 + 0.9 * 4.0)   # max_a Q_target
    assert y_dd[0].item() == pytest.approx(1 + 0.9 * 1.0)    # Q_target(argmax Q_online)
    assert y_dqn[1].item() == pytest.approx(1.0) and y_dd[1].item() == pytest.approx(1.0)  # terminal


def test_save_load_roundtrip(tmp_path):
    a = DQNAgent("double", small_hp(), seed=0)
    p = tmp_path / "c.pt"
    a.save(str(p), episode=7)
    b = DQNAgent.load(str(p))
    s = SnakeEnv(seed=2).get_state()
    assert b.variant == "double" and b.loaded_episode == 7 and b.epsilon == 0.0
    assert np.allclose(a.q_values(s), b.q_values(s))


def test_short_training_run_writes_outputs(tmp_path):
    from snakelab.train import train
    out = tmp_path / "run"
    agent, rows = train("double", 0, small_hp(episodes=3, ckpt_every=1), str(out))
    assert len(rows) == 3
    assert (out / "training_log.csv").exists() and (out / "final.pt").exists()
    assert (out / "checkpoints" / "ep0003.pt").exists()


def test_training_is_seed_reproducible():
    from snakelab.train import train
    hp = small_hp(episodes=4)
    _, r1 = train("dqn", 3, hp)
    _, r2 = train("dqn", 3, hp)
    assert [x["score"] for x in r1] == [x["score"] for x in r2]
    assert [x["steps"] for x in r1] == [x["steps"] for x in r2]
