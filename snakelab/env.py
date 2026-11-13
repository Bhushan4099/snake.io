"""Headless grid Snake environment (no Pygame import anywhere in this file).

Coordinates are (x, y) with y growing DOWNWARDS (screen convention).
Directions: 0=up, 1=right, 2=down, 3=left (clockwise order).
Actions are RELATIVE to the current heading: 0=straight, 1=turn left, 2=turn right.
"""
from collections import deque
import numpy as np

from .config import GRID_W, GRID_H, N_FEATURES

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES = ["up", "right", "down", "left"]
STRAIGHT, TURN_LEFT, TURN_RIGHT = 0, 1, 2
ACTION_NAMES = ["straight", "left", "right"]
_ACTION_DELTA = {STRAIGHT: 0, TURN_LEFT: -1, TURN_RIGHT: +1}

R_FOOD = 10.0
R_DEATH = -10.0
R_SHAPING = 0.1   # +/- per step closer to / farther from food (Manhattan)

FEATURE_NAMES = [
    "danger_straight", "danger_left", "danger_right",            # 0-2
    "dir_up", "dir_right", "dir_down", "dir_left",               # 3-6
    "food_up", "food_right", "food_down", "food_left",           # 7-10
    "food_dx_norm", "food_dy_norm",                              # 11-12
    "ray_straight", "ray_left", "ray_right",                     # 13-15
    "length_norm", "free_space_ahead", "hunger",                 # 16-18
]
assert len(FEATURE_NAMES) == N_FEATURES


def relative_action(current_dir: int, desired_dir: int) -> int:
    """Convert an absolute desired direction to a relative action.
    A 180-degree reversal is illegal in Snake, so it maps to STRAIGHT."""
    diff = (desired_dir - current_dir) % 4
    if diff == 1:
        return TURN_RIGHT
    if diff == 3:
        return TURN_LEFT
    return STRAIGHT  # diff 0 (same) or 2 (reverse, ignored)


def hunger_limit(length: int) -> int:
    """Max steps allowed without eating before the episode is ended."""
    return 100 + 5 * length


class SnakeEnv:
    def __init__(self, width=GRID_W, height=GRID_H, seed=None, shaping=R_SHAPING):
        self.w, self.h = width, height
        self.shaping = shaping
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        d = int(self.rng.integers(4))
        cx, cy = self.w // 2, self.h // 2
        dx, dy = DIRS[d]
        # head at centre, two body segments directly behind it
        self.snake = deque([(cx, cy), (cx - dx, cy - dy), (cx - 2 * dx, cy - 2 * dy)])
        self._occ = set(self.snake)
        self.direction = d
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.cause = None
        self.food = None
        self._place_food()
        return self.get_state()

    def set_state(self, snake, direction, food):
        """Test/debug helper: force an exact configuration (head first)."""
        self.snake = deque(snake)
        self._occ = set(self.snake)
        self.direction = direction
        self.food = tuple(food)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.cause = None

    def _place_food(self):
        free = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in self._occ]
        if not free:
            self.food = None
            return
        self.food = free[int(self.rng.integers(len(free)))]

    # ------------------------------------------------------------------- step
    @property
    def head(self):
        return self.snake[0]

    def _blocked(self, cell):
        x, y = cell
        return not (0 <= x < self.w and 0 <= y < self.h) or cell in self._occ

    def _dist_food(self, cell):
        return abs(cell[0] - self.food[0]) + abs(cell[1] - self.food[1])

    def step(self, action: int):
        """Apply a relative action. Returns (state, reward, done, info)."""
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        self.direction = (self.direction + _ACTION_DELTA[action]) % 4
        dx, dy = DIRS[self.direction]
        hx, hy = self.head
        new_head = (hx + dx, hy + dy)
        self.steps += 1
        self.steps_since_food += 1

        ate = new_head == self.food
        # The tail cell is vacated this step unless the snake grows, so moving
        # into the current tail cell is legal when not eating.
        tail = self.snake[-1]
        out = not (0 <= new_head[0] < self.w and 0 <= new_head[1] < self.h)
        hit_body = new_head in self._occ and (ate or new_head != tail)
        if out or hit_body:
            self.done, self.cause = True, "wall" if out else "body"
            return self.get_state(), R_DEATH, True, self._info()

        prev_dist = self._dist_food(self.head)
        self.snake.appendleft(new_head)
        self._occ.add(new_head)
        reward = 0.0
        if ate:
            self.score += 1
            self.steps_since_food = 0
            reward = R_FOOD
            self._place_food()
            if self.food is None:  # board full: perfect game
                self.done, self.cause = True, "won"
                return self.get_state(), reward, True, self._info()
        else:
            self.snake.pop()
            if tail != new_head:
                self._occ.discard(tail)
            reward = self.shaping * (1.0 if self._dist_food(new_head) < prev_dist else -1.0)

        if self.steps_since_food >= hunger_limit(len(self.snake)):
            self.done, self.cause = True, "starved"
            return self.get_state(), R_DEATH, True, self._info()
        return self.get_state(), reward, False, self._info()

    def _info(self):
        return {"score": self.score, "steps": self.steps, "cause": self.cause}

    # ---------------------------------------------------------------- features
    def _ray(self, start, d):
        """Steps from `start` along direction d until a wall/body cell (>=1)."""
        dx, dy = DIRS[d]
        n, x, y = 0, start[0], start[1]
        while True:
            x, y, n = x + dx, y + dy, n + 1
            if self._blocked((x, y)):
                return n

    def _flood(self, start):
        if self._blocked(start):
            return 0
        seen, q = {start}, deque([start])
        while q:
            x, y = q.popleft()
            for dx, dy in DIRS:
                c = (x + dx, y + dy)
                if c not in seen and not self._blocked(c):
                    seen.add(c)
                    q.append(c)
        return len(seen)

    def get_state(self) -> np.ndarray:
        """19-feature float32 vector; see FEATURE_NAMES / README for the table."""
        d, head = self.direction, self.head
        dl, dr = (d - 1) % 4, (d + 1) % 4
        s = np.zeros(N_FEATURES, dtype=np.float32)
        ahead = (head[0] + DIRS[d][0], head[1] + DIRS[d][1])
        left = (head[0] + DIRS[dl][0], head[1] + DIRS[dl][1])
        right = (head[0] + DIRS[dr][0], head[1] + DIRS[dr][1])
        s[0], s[1], s[2] = self._blocked(ahead), self._blocked(left), self._blocked(right)
        s[3 + d] = 1.0
        if self.food is not None:
            fx, fy = self.food
            s[7] = fy < head[1]
            s[8] = fx > head[0]
            s[9] = fy > head[1]
            s[10] = fx < head[0]
            s[11] = (fx - head[0]) / (self.w - 1)
            s[12] = (fy - head[1]) / (self.h - 1)
        m = max(self.w, self.h)
        s[13] = self._ray(head, d) / m
        s[14] = self._ray(head, dl) / m
        s[15] = self._ray(head, dr) / m
        s[16] = len(self.snake) / (self.w * self.h)
        free = self.w * self.h - len(self.snake)
        s[17] = self._flood(ahead) / free if free > 0 else 0.0
        s[18] = self.steps_since_food / hunger_limit(len(self.snake))
        return s
