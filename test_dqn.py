from __future__ import annotations

import numpy as np
import pytest
import torch

from snake_ai_lab.ai import DQNAgent, DQNConfig, QNetwork, ReplayBuffer


def test_q_network_returns_three_q_values() -> None:
    model = QNetwork()
    output = model(torch.zeros(19))
    assert output.shape == (1, 3)


def test_replay_buffer_samples_expected_shapes() -> None:
    buffer = ReplayBuffer(capacity=10, seed=1)
    for index in range(4):
        state = np.full(19, index, dtype=np.float32)
        buffer.add(state, index % 3, float(index), state + 1, index == 3)
    batch = buffer.sample(3)
    assert batch.states.shape == (3, 19)
    assert batch.actions.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.next_states.shape == (3, 19)
    assert batch.dones.shape == (3,)


def test_replay_buffer_rejects_insufficient_batch() -> None:
    buffer = ReplayBuffer(capacity=10)
    with pytest.raises(ValueError, match="not enough experiences"):
        buffer.sample(1)


def test_agent_selects_valid_action_and_greedy_mode_is_stable() -> None:
    agent = DQNAgent(seed=7)
    state = np.zeros(19, dtype=np.float32)
    action = agent.select_action(state, explore=False)
    assert action in (0, 1, 2)
    assert agent.select_action(state, explore=False) == action


def test_agent_learns_after_buffer_reaches_batch_size() -> None:
    config = DQNConfig(batch_size=4, replay_capacity=20, target_update_interval=1)
    agent = DQNAgent(config=config, seed=2)
    state = np.zeros(19, dtype=np.float32)
    next_state = np.ones(19, dtype=np.float32)
    assert agent.learn() is None
    for index in range(4):
        agent.remember(state, index % 3, 1.0, next_state, index == 3)
    loss = agent.learn()
    assert loss is not None
    assert np.isfinite(loss)
    assert agent.learn_steps == 1


def test_checkpoint_save_and_load(tmp_path) -> None:
    agent = DQNAgent(seed=3)
    agent.epsilon = 0.42
    path = tmp_path / "nested" / "checkpoint.pt"
    agent.save(path, extra={"episode": 12})

    restored = DQNAgent(seed=9)
    extra = restored.load(path)
    assert restored.epsilon == pytest.approx(0.42)
    assert extra["episode"] == 12
    for key, value in agent.online_network.state_dict().items():
        assert torch.equal(value, restored.online_network.state_dict()[key])


def test_train_dqn_reports_progress_and_saves_checkpoint(tmp_path) -> None:
    from train import train_dqn

    updates: list[dict] = []
    checkpoint = tmp_path / "model.pt"
    log = tmp_path / "training.csv"
    result = train_dqn(
        episodes=1,
        seed=11,
        checkpoint=checkpoint,
        log=log,
        save_every=1,
        progress_callback=updates.append,
        verbose=False,
    )

    assert result["completed"] == 1
    assert result["stopped"] is False
    assert checkpoint.exists()
    assert log.exists()
    assert len(updates) == 1
    assert updates[0]["completed"] == 1
    assert updates[0]["total"] == 1


def test_train_dqn_resumes_from_saved_checkpoint(tmp_path) -> None:
    from train import train_dqn

    checkpoint = tmp_path / "model.pt"
    log = tmp_path / "training.csv"
    first = train_dqn(
        episodes=1, seed=12, checkpoint=checkpoint, log=log,
        save_every=1, verbose=False,
    )
    second = train_dqn(
        episodes=1, seed=12, checkpoint=checkpoint, log=log,
        save_every=1, resume=True, verbose=False,
    )

    assert first["last_episode"] == 1
    assert second["last_episode"] == 2
    assert second["completed"] == 1
