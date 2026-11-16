"""Check all 19 state features against hand-built boards.

Every expected value below was worked out by hand from the board drawing in the
comment, NOT by calling the code. Coordinates are (x, y), y grows downward.
Feature order: see FEATURE_NAMES in snakelab/env.py.
"""
import numpy as np
import pytest
from snakelab.env import SnakeEnv, UP, RIGHT, DOWN, LEFT, FEATURE_NAMES, hunger_limit
from snakelab.config import N_FEATURES


def board(snake, d, food, w=12, h=12):
    e = SnakeEnv(w, h, seed=0)
    e.set_state(snake, d, food)
    return e


def check(state, expected: dict):
    """expected: {feature_index: value}; every index not listed must be 0 or checked elsewhere."""
    for i, v in expected.items():
        assert state[i] == pytest.approx(v, abs=1e-6), f"feature {i} ({FEATURE_NAMES[i]}): {state[i]} != {v}"


def test_board_A_open_field_heading_right():
    # 12x12, head (5,5) heading RIGHT, body behind it at x=4,3,2, food at (8,2). Nothing adjacent is blocked.
    s = board([(5, 5), (4, 5), (3, 5), (2, 5)], RIGHT, (8, 2)).get_state()
    assert s.shape == (N_FEATURES,) and s.dtype == np.float32
    check(s, {
        0: 0, 1: 0, 2: 0,                      # no danger
        3: 0, 4: 1, 5: 0, 6: 0,                # heading right
        7: 1, 8: 1, 9: 0, 10: 0,               # food is up and to the right
        11: 3 / 11, 12: -3 / 11,               # dx=+3, dy=-3, divided by (12-1)
        13: 7 / 12,                            # right: x=6..11 free (6 cells) then wall -> n=7
        14: 6 / 12,                            # left of RIGHT is UP: y=4..0 free (5) then wall -> n=6
        15: 7 / 12,                            # right of RIGHT is DOWN: y=6..11 free (6) then wall -> n=7
        16: 4 / 144,
        17: 1.0,                               # all 140 free cells reachable from the cell ahead
        18: 0.0,
    })


def test_board_B_corner_walls():
    # head (0,0) heading UP in the top-left corner, body below it, food at (3,3)
    s = board([(0, 0), (0, 1), (0, 2), (0, 3)], UP, (3, 3)).get_state()
    check(s, {
        0: 1,                                  # ahead is outside the board
        1: 1,                                  # left of UP is LEFT: x=-1 is outside
        2: 0,                                  # right of UP is RIGHT: (1,0) is free
        3: 1, 4: 0, 5: 0, 6: 0,
        7: 0, 8: 1, 9: 1, 10: 0,               # food is down and to the right
        11: 3 / 11, 12: 3 / 11,
        13: 1 / 12, 14: 1 / 12,                # touching a wall: ray length 1
        15: 12 / 12,                           # x=1..11 free (11) then wall -> n=12
        16: 4 / 144,
        17: 0.0,                               # the cell ahead is blocked -> no free space
        18: 0.0,
    })


def test_board_C_flood_fill_counts_only_the_reachable_pocket():
    # 5x5 board. The snake is a horizontal wall across row 2: head (4,2), body (3,2),(2,2),(1,2),(0,2).
    # Heading UP. Cell ahead is (4,1); rows 0-1 form a closed 10-cell pocket above the wall.
    #   row0: . . . . .
    #   row1: . . . . .
    #   row2: B B B B H
    #   row3: . . . . .   (unreachable from the pocket)
    #   row4: . . . . .
    s = board([(4, 2), (3, 2), (2, 2), (1, 2), (0, 2)], UP, (0, 0), w=5, h=5).get_state()
    check(s, {
        0: 0,                                  # (4,1) free
        1: 1,                                  # left of UP is (3,2) = own body
        2: 1,                                  # right of UP is (5,2) = outside
        13: 3 / 5,                             # up: y=1,0 free then wall -> n=3, max(w,h)=5
        14: 1 / 5, 15: 1 / 5,
        16: 5 / 25,
        17: 10 / 20,                           # 10 reachable cells / (25 - 5 snake cells)
    })


def test_board_D_hunger_uses_length_dependent_limit():
    e = board([(5, 5), (5, 6), (5, 7)], UP, (5, 0))
    e.steps_since_food = 50
    assert hunger_limit(3) == 115
    check(e.get_state(), {18: 50 / 115})


def test_board_E_food_straight_ahead_heading_up():
    s = board([(5, 5), (5, 6), (5, 7), (5, 8)], UP, (5, 2)).get_state()
    check(s, {
        3: 1, 4: 0, 5: 0, 6: 0,
        7: 1, 8: 0, 9: 0, 10: 0,               # food is straight up, no horizontal offset
        11: 0.0, 12: -3 / 11,
        13: 6 / 12,                            # y=4..0 free (5) then wall -> n=6
        14: 6 / 12,                            # left: x=4..0 free (5) then wall -> n=6
        15: 7 / 12,                            # right: x=6..11 free (6) then wall -> n=7
    })


def test_board_F_left_and_right_are_relative_to_heading():
    # Heading DOWN, so "left" is screen-east and "right" is screen-west.
    # Snake: (5,5)(5,4)(6,4)(6,5)(6,6) is contiguous; (6,5) is east of the head.
    s = board([(5, 5), (5, 4), (6, 4), (6, 5), (6, 6)], DOWN, (0, 11)).get_state()
    check(s, {
        0: 0,                                  # (5,6) free
        1: 1,                                  # left of DOWN = RIGHT -> (6,5) is body
        2: 0,                                  # right of DOWN = LEFT -> (4,5) free
        3: 0, 4: 0, 5: 1, 6: 0,
        7: 0, 8: 0, 9: 1, 10: 1,               # food is down and to the left
        14: 1 / 12,                            # ray to the body next to us
        15: 6 / 12,                            # x=4..0 free (5 cells) then wall -> n=6
    })


def test_heading_one_hot_for_all_four_directions():
    for d in (UP, RIGHT, DOWN, LEFT):
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][d]
        s = board([(6, 6), (6 - dx, 6 - dy), (6 - 2 * dx, 6 - 2 * dy)], d, (0, 0)).get_state()
        assert list(s[3:7]) == [1.0 if i == d else 0.0 for i in range(4)]


def test_all_features_stay_in_range_over_random_play():
    rng = np.random.default_rng(7)
    e = SnakeEnv(seed=3)
    for _ in range(3000):
        s = e.get_state()
        assert s.shape == (N_FEATURES,) and np.isfinite(s).all()
        assert (s[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] >= 0).all() and (s[:11] <= 1).all()
        assert (np.abs(s[11:13]) <= 1.0 + 1e-6).all()
        assert (s[13:] >= 0).all() and (s[13:] <= 1.0 + 1e-6).all()
        _, _, done, _ = e.step(int(rng.integers(3)))
        if done:
            e.reset()
