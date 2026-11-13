import numpy as np
import pytest
from snakelab.env import (SnakeEnv, UP, RIGHT, DOWN, LEFT, STRAIGHT, TURN_LEFT, TURN_RIGHT,
                          FEATURE_NAMES, relative_action, R_FOOD, R_DEATH, hunger_limit)
from snakelab.config import N_FEATURES


def env_with(snake, d, food, w=12, h=12):
    e = SnakeEnv(w, h, seed=0)
    e.set_state(snake, d, food)
    return e


def test_reset_initial_state():
    e = SnakeEnv(seed=1)
    assert len(e.snake) == 3 and e.score == 0 and not e.done
    assert e.food not in e.snake
    assert e.head == (6, 6)


def test_straight_movement_all_headings():
    for d, (dx, dy) in enumerate([(0, -1), (1, 0), (0, 1), (-1, 0)]):
        e = env_with([(6, 6), (6 - dx, 6 - dy), (6 - 2 * dx, 6 - 2 * dy)], d, (0, 0))
        e.step(STRAIGHT)
        assert e.head == (6 + dx, 6 + dy) and len(e.snake) == 3


def test_relative_turns():
    # heading UP: left -> LEFT(3), right -> RIGHT(1)
    e = env_with([(6, 6), (6, 7), (6, 8)], UP, (0, 0)); e.step(TURN_LEFT)
    assert e.direction == LEFT and e.head == (5, 6)
    e = env_with([(6, 6), (6, 7), (6, 8)], UP, (0, 0)); e.step(TURN_RIGHT)
    assert e.direction == RIGHT and e.head == (7, 6)
    # heading RIGHT: left -> UP, right -> DOWN
    e = env_with([(6, 6), (5, 6), (4, 6)], RIGHT, (0, 0)); e.step(TURN_LEFT)
    assert e.direction == UP
    e = env_with([(6, 6), (5, 6), (4, 6)], RIGHT, (0, 0)); e.step(TURN_RIGHT)
    assert e.direction == DOWN


def test_relative_action_helper_blocks_reversal():
    assert relative_action(UP, UP) == STRAIGHT
    assert relative_action(UP, DOWN) == STRAIGHT
    assert relative_action(UP, LEFT) == TURN_LEFT
    assert relative_action(UP, RIGHT) == TURN_RIGHT
    assert relative_action(LEFT, UP) == TURN_RIGHT


def test_wall_collision_each_side():
    cases = [((0, 5), [(0, 5), (1, 5), (2, 5)], LEFT), ((11, 5), [(11, 5), (10, 5), (9, 5)], RIGHT),
             ((5, 0), [(5, 0), (5, 1), (5, 2)], UP), ((5, 11), [(5, 11), (5, 10), (5, 9)], DOWN)]
    for _, snake, d in cases:
        e = env_with(snake, d, (6, 6))
        _, r, done, info = e.step(STRAIGHT)
        assert done and r == R_DEATH and info["cause"] == "wall"


def test_self_collision():
    # head (5,5) moving UP into own body at (5,4) after a tight loop
    snake = [(5, 5), (6, 5), (6, 4), (5, 4), (4, 4)]
    e = env_with(snake, LEFT, (0, 0))
    _, r, done, info = e.step(TURN_RIGHT)  # LEFT -> turn right = UP -> (5,4)
    assert done and info["cause"] == "body" and r == R_DEATH


def test_moving_into_vacating_tail_is_legal():
    snake = [(5, 5), (6, 5), (6, 4), (5, 4)]  # 2x2 loop; tail at (5,4)
    e = env_with(snake, LEFT, (0, 0))
    _, _, done, _ = e.step(TURN_RIGHT)  # head -> (5,4) == tail cell, which is vacated
    assert not done and e.head == (5, 4)


def test_eating_grows_scores_and_respawns_food():
    e = env_with([(6, 6), (5, 6), (4, 6)], RIGHT, (7, 6))
    _, r, done, _ = e.step(STRAIGHT)
    assert r == R_FOOD and not done and e.score == 1 and len(e.snake) == 4
    assert e.food != (7, 6) and e.food not in e.snake
    assert e.steps_since_food == 0


def test_food_never_spawns_on_snake():
    e = SnakeEnv(seed=3)
    for _ in range(200):
        e._place_food()
        assert e.food not in e.snake


def test_step_after_done_raises():
    e = env_with([(0, 5), (1, 5), (2, 5)], LEFT, (6, 6))
    e.step(STRAIGHT)
    with pytest.raises(RuntimeError):
        e.step(STRAIGHT)


def test_starvation_terminates():
    e = SnakeEnv(seed=0)
    e.set_state([(6, 6), (5, 6), (4, 6)], RIGHT, (11, 0))
    e.steps_since_food = hunger_limit(3) - 1
    _, r, done, info = e.step(STRAIGHT)
    assert done and info["cause"] == "starved" and r == R_DEATH


def test_determinism_with_seed():
    a, b = SnakeEnv(seed=5), SnakeEnv(seed=5)
    assert a.food == b.food and a.direction == b.direction
    sa = [a.step(i % 3)[1] for i in range(10) if not a.done]
    sb = [b.step(i % 3)[1] for i in range(10) if not b.done]
    assert sa == sb


# ---------------------------------------------------------------- 19 features
def test_state_shape_dtype_and_names():
    e = SnakeEnv(seed=0)
    s = e.get_state()
    assert s.shape == (19,) and s.dtype == np.float32 and len(FEATURE_NAMES) == 19


def test_state_values_open_field():
    e = env_with([(6, 6), (5, 6), (4, 6)], RIGHT, (9, 3))
    s = e.get_state()
    assert list(s[0:3]) == [0, 0, 0]                       # no danger
    assert list(s[3:7]) == [0, 1, 0, 0]                    # heading right
    assert list(s[7:11]) == [1, 1, 0, 0]                   # food is up and right
    assert s[11] == pytest.approx(3 / 11) and s[12] == pytest.approx(-3 / 11)
    assert s[13] == pytest.approx(6 / 12)                  # ray straight: x 7..11 free, wall at 12
    assert s[14] == pytest.approx(7 / 12)                  # ray left (up): y 5..0 free, wall at -1
    assert s[15] == pytest.approx(6 / 12)                  # ray right (down): y7..11, wall at 12
    assert s[16] == pytest.approx(3 / 144)
    assert s[17] == pytest.approx(141 / 141)               # all free cells reachable from ahead
    assert s[18] == 0.0


def test_state_danger_flags_and_rays():
    # head at top-left corner heading UP: wall straight and left, free right
    e = env_with([(0, 0), (0, 1), (0, 2)], UP, (5, 5))
    s = e.get_state()
    assert list(s[0:3]) == [1, 1, 0]
    assert s[13] == pytest.approx(1 / 12) and s[14] == pytest.approx(1 / 12)
    assert s[17] == 0.0                                    # ahead is blocked -> no free space


def test_danger_from_body():
    e = env_with([(5, 5), (6, 5), (6, 4), (5, 4), (4, 4), (4, 5)], LEFT, (0, 0))
    # heading LEFT at (5,5): ahead (4,5) is body, right (up) (5,4) is body, left (down) free
    s = e.get_state()
    assert list(s[0:3]) == [1, 0, 1]


def test_hunger_and_length_features_update():
    e = env_with([(6, 6), (5, 6), (4, 6)], RIGHT, (0, 11))
    e.step(STRAIGHT)
    s = e.get_state()
    assert s[18] == pytest.approx(1 / hunger_limit(3))


def test_state_all_finite_and_bounded_during_random_play():
    rng = np.random.default_rng(0)
    e = SnakeEnv(seed=0)
    for _ in range(20):
        e.reset()
        while not e.done:
            s, *_ = e.step(int(rng.integers(3)))
            assert np.all(np.isfinite(s)) and s.min() >= -1.0 and s.max() <= 1.0
