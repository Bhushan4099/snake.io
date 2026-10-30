from __future__ import annotations

import numpy as np
import pytest

from snake_ai_lab.core.environment import Action, Direction, SnakeEnvironment


def test_reset_returns_valid_19_feature_state() -> None:
    env = SnakeEnvironment()
    state = env.reset(seed=123)

    assert isinstance(state, np.ndarray)
    assert state.shape == (19,)
    assert state.dtype == np.float32
    assert set(np.unique(state)).issubset({0.0, 1.0})
    assert len(env.snake) == 3
    assert env.food not in env.snake
    assert env.score == 0


def test_reset_is_reproducible_with_same_seed() -> None:
    env = SnakeEnvironment()
    first_state = env.reset(seed=99)
    first_food = env.food

    second_state = env.reset(seed=99)
    second_food = env.food

    np.testing.assert_array_equal(first_state, second_state)
    assert first_food == second_food


def test_straight_action_moves_one_cell_and_applies_step_penalty() -> None:
    env = SnakeEnvironment()
    env.reset(seed=1)
    old_head = env.head

    state, reward, done, info = env.step(Action.STRAIGHT)

    assert env.head == (old_head[0] + 1, old_head[1])
    assert reward == pytest.approx(-0.01)
    assert done is False
    assert info["steps"] == 1
    assert state.shape == (19,)


def test_relative_left_and_right_actions_update_heading() -> None:
    env = SnakeEnvironment()
    env.reset(seed=2)

    env.step(Action.LEFT)
    assert env.direction == Direction.UP
    assert env.head[1] == env.height // 2 - 1

    env.reset(seed=2)
    env.step(Action.RIGHT)
    assert env.direction == Direction.DOWN
    assert env.head[1] == env.height // 2 + 1


def test_food_collection_rewards_and_grows_snake() -> None:
    env = SnakeEnvironment()
    env.reset(seed=3)
    old_length = len(env.snake)
    env.food = (env.head[0] + 1, env.head[1])

    _, reward, done, info = env.step(Action.STRAIGHT)

    assert reward == pytest.approx(10.0)
    assert done is False
    assert len(env.snake) == old_length + 1
    assert env.score == 1
    assert info["ate_food"] is True
    assert env.food not in env.snake


def test_wall_collision_ends_episode_with_penalty() -> None:
    env = SnakeEnvironment(width=7, height=7)
    env.reset(seed=4)
    env.snake = [(6, 3), (5, 3), (4, 3)]
    env.direction = Direction.RIGHT
    env.food = (0, 0)

    _, reward, done, info = env.step(Action.STRAIGHT)

    assert done is True
    assert reward == pytest.approx(-10.0)
    assert info["reason"] == "collision"
    assert env.game_over is True


def test_body_collision_ends_episode() -> None:
    env = SnakeEnvironment(width=8, height=8)
    env.reset(seed=5)
    # Head moves upward into the body segment at (3, 2).
    env.snake = [(3, 3), (3, 2), (2, 2), (2, 3)]
    env.direction = Direction.RIGHT
    env.food = (7, 7)

    _, reward, done, info = env.step(Action.LEFT)

    assert done is True
    assert reward == pytest.approx(-10.0)
    assert info["reason"] == "collision"


def test_moving_into_vacating_tail_cell_is_legal() -> None:
    env = SnakeEnvironment(width=8, height=8)
    env.reset(seed=6)
    # Head at (2, 3), facing left. Turning left means down; arrange a
    # U-shape where the next cell is the current tail.
    env.snake = [(2, 3), (2, 2), (3, 2), (3, 3)]
    env.direction = Direction.UP
    env.food = (7, 7)

    # Facing up, turn right -> right; this enters (3, 3), the tail cell.
    _, reward, done, info = env.step(Action.RIGHT)

    assert done is False
    assert reward == pytest.approx(-0.01)
    assert env.head == (3, 3)
    assert len(env.snake) == 4
    assert info["reason"] is None


def test_step_after_episode_end_requires_reset() -> None:
    env = SnakeEnvironment(width=5, height=5)
    env.reset(seed=7)
    env.snake = [(4, 2)]
    env.direction = Direction.RIGHT
    env.food = (0, 0)
    env.step(Action.STRAIGHT)

    with pytest.raises(RuntimeError, match="call reset"):
        env.step(Action.STRAIGHT)


def test_invalid_action_raises_value_error() -> None:
    env = SnakeEnvironment()
    with pytest.raises(ValueError, match="action must be"):
        env.step(3)


def test_starvation_limit_ends_episode() -> None:
    env = SnakeEnvironment(width=8, height=8, max_steps_without_food=2)
    env.reset(seed=8)
    env.food = (7, 7)

    _, _, done1, _ = env.step(Action.STRAIGHT)
    _, reward2, done2, info2 = env.step(Action.STRAIGHT)

    assert done1 is False
    assert done2 is True
    assert reward2 == pytest.approx(-10.0)
    assert info2["reason"] == "starvation"



def test_state_includes_body_probes_at_relative_distances() -> None:
    env = SnakeEnvironment(width=10, height=10)
    env.reset(seed=12)
    # Facing right. Put body cells at left-adjacent and straight two cells ahead.
    env.snake = [(4, 4), (4, 3), (6, 4), (3, 4)]
    env.direction = Direction.RIGHT
    env.food = (8, 8)

    state = env.get_state()

    assert state.shape == (19,)
    # Index 11 is left-adjacent body, 12 right-adjacent body.
    assert state[11] == 1.0
    assert state[12] == 0.0
    # Probe order: distance 2 straight, left, right; distance 3 same.
    assert state[13] == 1.0  # straight two cells ahead: (6, 4)

def test_constructor_validates_dimensions() -> None:
    with pytest.raises(ValueError):
        SnakeEnvironment(width=4, height=20)

    with pytest.raises(ValueError):
        SnakeEnvironment(width=20, height=20, initial_length=20)
