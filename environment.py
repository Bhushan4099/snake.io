"""Pygame-independent Snake environment for reinforcement learning.

Observation (19 float32 values):
    0..2   immediate danger straight, left, right
    3..6   current direction up, down, left, right
    7..10  food direction left, right, up, down
    11..18 body occupancy probes: near left/right, then straight/left/right
            at distances 2 and 3

Actions are relative to the snake's current heading:
    0 = straight, 1 = turn left, 2 = turn right

The public API intentionally stays small and dependency-light so this
module can be used by headless training, Pygame, tests, and future firmware.
"""

from __future__ import annotations

from enum import IntEnum
import random
from typing import TypeAlias

import numpy as np


Point: TypeAlias = tuple[int, int]


class Action(IntEnum):
    """Actions available to the agent."""

    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


class Direction(IntEnum):
    """Cardinal directions in clockwise order."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


_DIRECTION_VECTORS: dict[Direction, Point] = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}


class SnakeEnvironment:
    """A compact, Gym-like Snake environment.

    Args:
        width: Number of grid cells horizontally.
        height: Number of grid cells vertically.
        initial_length: Starting snake length.
        max_steps_without_food: End an episode after this many moves without
            collecting food. This prevents a looping policy from running
            forever. Set to ``None`` to disable the limit.

    ``step(action)`` returns ``(observation, reward, done, info)``.
    ``reset(seed=...)`` returns the initial observation.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        initial_length: int = 3,
        max_steps_without_food: int | None = 100,
    ) -> None:
        if width < 5 or height < 5:
            raise ValueError("width and height must both be at least 5")
        if initial_length < 1 or initial_length >= width:
            raise ValueError("initial_length must be >= 1 and smaller than width")
        if max_steps_without_food is not None and max_steps_without_food < 1:
            raise ValueError("max_steps_without_food must be positive or None")

        self.width = int(width)
        self.height = int(height)
        self.initial_length = int(initial_length)
        self.max_steps_without_food = max_steps_without_food
        self._rng = random.Random()

        # Public state fields are useful to renderers and debugging tools.
        self.snake: list[Point] = []  # Head is always index 0.
        self.direction: Direction = Direction.RIGHT
        self.food: Point | None = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        self.done_reason: str | None = None

        self.reset()

    @property
    def head(self) -> Point:
        """Current head coordinate (x, y)."""
        return self.snake[0]

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new episode and return its 19-feature observation.

        Passing the same seed to reset produces the same initial food
        placement, which is useful for reproducible evaluation.
        """
        if seed is not None:
            self._rng.seed(seed)

        head_x = max(self.initial_length - 1, self.width // 2)
        head_y = self.height // 2
        self.snake = [(head_x - offset, head_y) for offset in range(self.initial_length)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.game_over = False
        self.done_reason = None
        self.food = self._spawn_food()
        return self.get_state()

    def step(self, action: int | Action) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        """Advance one grid step.

        Rewards:
            * +10.0 for collecting food
            * -0.01 for a normal movement step
            * -10.0 for collision or starvation timeout

        Entering the cell currently occupied by the tail is legal when the
        snake is not growing on that step, because the tail moves away.
        """
        if self.game_over:
            raise RuntimeError("step() called after episode ended; call reset() first")

        try:
            selected_action = Action(action)
        except (TypeError, ValueError) as exc:
            raise ValueError("action must be 0 (straight), 1 (left), or 2 (right)") from exc

        next_direction = self._turn(self.direction, selected_action)
        dx, dy = _DIRECTION_VECTORS[next_direction]
        next_head = (self.head[0] + dx, self.head[1] + dy)
        will_eat = next_head == self.food
        self.steps += 1

        if self._would_collide(next_head, will_grow=will_eat):
            self.direction = next_direction
            self.game_over = True
            self.done_reason = "collision"
            return self.get_state(), -10.0, True, self._info(ate_food=False, won=False)

        self.direction = next_direction
        self.snake.insert(0, next_head)

        if will_eat:
            self.score += 1
            self.steps_since_food = 0
            reward = 10.0
            self.food = self._spawn_food()
            if self.food is None:
                self.game_over = True
                self.done_reason = "won"
                return self.get_state(), reward, True, self._info(ate_food=True, won=True)
        else:
            self.snake.pop()
            self.steps_since_food += 1
            reward = -0.01
            if (
                self.max_steps_without_food is not None
                and self.steps_since_food >= self.max_steps_without_food
            ):
                self.game_over = True
                self.done_reason = "starvation"
                reward = -10.0

        return self.get_state(), reward, self.game_over, self._info(
            ate_food=will_eat, won=self.done_reason == "won"
        )

    def get_state(self) -> np.ndarray:
        """Encode the current game into the documented 19-feature vector."""
        left_direction = Direction((int(self.direction) - 1) % 4)
        right_direction = Direction((int(self.direction) + 1) % 4)

        danger_straight = self._danger_in_direction(self.direction)
        danger_left = self._danger_in_direction(left_direction)
        danger_right = self._danger_in_direction(right_direction)

        direction_one_hot = [
            float(self.direction == Direction.UP),
            float(self.direction == Direction.DOWN),
            float(self.direction == Direction.LEFT),
            float(self.direction == Direction.RIGHT),
        ]

        food_left = float(self.food is not None and self.food[0] < self.head[0])
        food_right = float(self.food is not None and self.food[0] > self.head[0])
        food_up = float(self.food is not None and self.food[1] < self.head[1])
        food_down = float(self.food is not None and self.food[1] > self.head[1])

        # Body awareness probes are relative to the current heading. The first
        # three immediate-danger features already encode distance 1; these eight
        # extra features reveal nearby body segments at distances 1..3.
        relative_directions = (self.direction, left_direction, right_direction)
        body_cells = set(self.snake[1:])
        body_probes: list[float] = []

        # Distance 1: side probes (straight is already represented by danger).
        for direction in (left_direction, right_direction):
            dx, dy = _DIRECTION_VECTORS[direction]
            body_probes.append(float((self.head[0] + dx, self.head[1] + dy) in body_cells))

        # Distances 2 and 3: straight, left, right.
        for distance in (2, 3):
            for direction in relative_directions:
                dx, dy = _DIRECTION_VECTORS[direction]
                point = (self.head[0] + dx * distance, self.head[1] + dy * distance)
                body_probes.append(float(point in body_cells))

        state = np.asarray(
            [
                float(danger_straight),
                float(danger_left),
                float(danger_right),
                *direction_one_hot,
                food_left,
                food_right,
                food_up,
                food_down,
                *body_probes,
            ],
            dtype=np.float32,
        )
        assert state.shape == (19,)
        return state

    def _turn(self, direction: Direction, action: Action) -> Direction:
        if action == Action.LEFT:
            return Direction((int(direction) - 1) % 4)
        if action == Action.RIGHT:
            return Direction((int(direction) + 1) % 4)
        return direction

    def _danger_in_direction(self, direction: Direction) -> bool:
        dx, dy = _DIRECTION_VECTORS[direction]
        target = (self.head[0] + dx, self.head[1] + dy)
        return self._would_collide(target, will_grow=(target == self.food))

    def _would_collide(self, point: Point, will_grow: bool = False) -> bool:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # If the snake grows, its tail stays in place; otherwise it vacates.
        occupied_body = self.snake if will_grow else self.snake[:-1]
        return point in occupied_body

    def _spawn_food(self) -> Point | None:
        occupied = set(self.snake)
        empty_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        return self._rng.choice(empty_cells) if empty_cells else None

    def _info(self, ate_food: bool, won: bool) -> dict[str, object]:
        return {
            "score": self.score,
            "steps": self.steps,
            "ate_food": ate_food,
            "won": won,
            "reason": self.done_reason,
        }
