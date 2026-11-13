"""Pygame front end. All game logic lives in env.py; this file only draws and reads input.
Import of this module requires pygame; training/evaluation never import it.
For headless smoke tests set SDL_VIDEODRIVER=dummy before importing.

Look: minimal black / blue / red palette, square blocks, 5x7 bitmap pixel font (pixelfont.py)."""
import os
import threading

import numpy as np
import pygame

from .config import GRID_W, GRID_H, HParams
from .env import (SnakeEnv, DIRS, relative_action, ACTION_NAMES, UP, RIGHT, DOWN, LEFT)
from .pixelfont import PixelFont

CELL = 32
PANEL = 300
PIXEL = 2                                   # text pixel size (scale factor)

# Palette: black, blue, red only.
BLACK = (0, 0, 0)
GRID_C = (10, 16, 42)                       # very dark blue grid lines
BLUE = (60, 130, 255)                       # main text, snake body
BLUE_HI = (140, 190, 255)                   # snake head
DIM = (35, 65, 150)                         # hints, inactive elements
RED = (235, 45, 55)                         # titles, food, highlights
BG = BLACK
HEAD_C, BODY_C, FOOD_C = BLUE_HI, BLUE, RED
TEXT_C, DIM_C, ACC_C = BLUE, DIM, RED

KEYS = {pygame.K_UP: UP, pygame.K_w: UP, pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
        pygame.K_DOWN: DOWN, pygame.K_s: DOWN, pygame.K_LEFT: LEFT, pygame.K_a: LEFT}
PLUS_KEYS = (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS)
MINUS_KEYS = (pygame.K_MINUS, pygame.K_KP_MINUS)
DEFAULT_CKPT = os.path.join("results", "runs", "dqn_seed0", "final.pt")


class Slider:
    """Horizontal speed slider (steps per second). Drag with the mouse; +/- keys nudge it."""

    def __init__(self, x, y, w, lo, hi, value, step=1.0):
        self.rect = pygame.Rect(x, y, w, 8)
        self.lo, self.hi, self.value, self.step = lo, hi, value, step
        self.dragging = False

    def handle(self, e):
        if e.type == pygame.KEYDOWN:
            if e.key in PLUS_KEYS:
                self.value = min(self.hi, self.value + self.step)
            elif e.key in MINUS_KEYS:
                self.value = max(self.lo, self.value - self.step)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and self.rect.inflate(0, 24).collidepoint(e.pos):
            self.dragging = True
        if e.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if e.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN) and self.dragging:
            frac = min(1.0, max(0.0, (e.pos[0] - self.rect.x) / self.rect.w))
            self.value = self.lo + frac * (self.hi - self.lo)

    def draw(self, sc, label_y):
        sc.text(f"SPEED {self.value:.0f} STEPS/S", self.rect.x, label_y)
        pygame.draw.rect(sc.surf, DIM, self.rect)
        frac = (self.value - self.lo) / (self.hi - self.lo)
        fill = pygame.Rect(self.rect.x, self.rect.y, int(frac * self.rect.w), self.rect.h)
        pygame.draw.rect(sc.surf, BLUE, fill)
        kx = int(self.rect.x + frac * self.rect.w)
        pygame.draw.rect(sc.surf, RED, (kx - 6, self.rect.centery - 8, 12, 16))


class Screen:
    def __init__(self, w=GRID_W, h=GRID_H):
        pygame.init()
        pygame.display.set_caption("Snake AI Lab")
        self.board_w, self.board_h = w * CELL, h * CELL
        self.surf = pygame.display.set_mode((self.board_w + PANEL, max(self.board_h, 420)))
        self.font = PixelFont(PIXEL)
        self.big = PixelFont(PIXEL + 1)
        self.clock = pygame.time.Clock()

    def text(self, s, x, y, color=TEXT_C, font=None):
        self.surf.blit((font or self.font).render(s, False, color), (x, y))

    def draw_board(self, env):
        self.surf.fill(BG)
        for x in range(env.w):
            for y in range(env.h):
                pygame.draw.rect(self.surf, GRID_C, (x * CELL, y * CELL, CELL, CELL), 1)
        pygame.draw.line(self.surf, DIM, (self.board_w, 0), (self.board_w, self.surf.get_height()), 2)
        if env.food is not None:
            pygame.draw.rect(self.surf, FOOD_C, (env.food[0] * CELL + 6, env.food[1] * CELL + 6, CELL - 12, CELL - 12))
        for i, (x, y) in enumerate(env.snake):
            pygame.draw.rect(self.surf, HEAD_C if i == 0 else BODY_C,
                             (x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4))

    def overlay(self, msg, sub):
        s = pygame.Surface((self.board_w, self.board_h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 190))
        self.surf.blit(s, (0, 0))
        self.text(msg, 20, self.board_h // 2 - 30, RED, self.big)
        self.text(sub, 20, self.board_h // 2 + 10, BLUE)


def _quit_event(e):
    return e.type == pygame.QUIT


def run_manual(max_frames=None, screen=None):
    """Arrow keys / WASD steer. Speed slider (drag) or +/- keys. R restarts, ESC returns."""
    sc = screen or Screen()
    env = SnakeEnv()
    desired, acc, frames = env.direction, 0, 0
    x = self_x(sc)
    slider = Slider(x, 200, PANEL - 50, 2.0, 30.0, 8.0)
    while True:
        dt = sc.clock.tick(60)
        frames += 1
        for e in pygame.event.get():
            if _quit_event(e):
                return "quit"
            slider.handle(e)
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return "menu"
                if e.key == pygame.K_r:
                    env.reset(); desired = env.direction
                if e.key in KEYS:
                    desired = KEYS[e.key]
        acc += dt
        if acc >= 1000.0 / slider.value and not env.done:
            acc = 0
            env.step(relative_action(env.direction, desired))
        sc.draw_board(env)
        sc.text("MANUAL PLAY", x, 20, ACC_C, sc.big)
        sc.text(f"SCORE {env.score}", x, 70)
        sc.text(f"STEPS {env.steps}", x, 100)
        slider.draw(sc, 170)
        sc.text("ARROWS/WASD STEER", x, 250, DIM_C)
        sc.text("+/- OR DRAG: SPEED", x, 275, DIM_C)
        sc.text("R RESTART  ESC MENU", x, 300, DIM_C)
        if env.done:
            sc.overlay("GAME OVER", f"{env.cause.upper()}. PRESS R")
        pygame.display.flip()
        if max_frames is not None and frames >= max_frames:
            return "done"


def self_x(sc):
    return sc.board_w + 20


def run_autoplay(checkpoint=DEFAULT_CKPT, max_frames=None, screen=None):
    """Trained agent plays greedily; Q-values for the 3 actions are drawn live.
    Speed slider (drag) or +/- keys; SPACE pauses; R resets; ESC returns."""
    from .agent import DQNAgent
    sc = screen or Screen()
    if not os.path.exists(checkpoint):
        return _message(sc, f"No checkpoint at {checkpoint}",
                        "Train first: python main.py train", max_frames)
    agent = DQNAgent.load(checkpoint)
    env = SnakeEnv()
    state = env.get_state()
    paused, acc, frames = False, 0.0, 0
    x = self_x(sc)
    slider = Slider(x, 330, PANEL - 50, 1.0, 60.0, 10.0, step=2.0)
    while True:
        dt = sc.clock.tick(60)
        frames += 1
        for e in pygame.event.get():
            if _quit_event(e):
                return "quit"
            slider.handle(e)
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return "menu"
                if e.key == pygame.K_SPACE:
                    paused = not paused
                if e.key == pygame.K_r:
                    state = env.reset()
        q = agent.q_values(state)
        best = int(np.argmax(q))
        acc += dt
        if not paused and not env.done and acc >= 1000.0 / slider.value:
            acc = 0.0
            state, _, _, _ = env.step(best)
        if env.done and not paused and acc >= 1000.0:   # auto-restart after 1 s
            state, acc = env.reset(), 0.0
        sc.draw_board(env)
        sc.text("AI AUTO PLAY", x, 20, ACC_C, sc.big)
        sc.text(f"{agent.variant.upper()}  CKPT EP {getattr(agent, 'loaded_episode', '?')}", x, 62, DIM_C)
        sc.text(f"SCORE {env.score}  STEPS {env.steps}", x, 90)
        sc.text("Q-VALUES  RED=CHOSEN", x, 130, DIM_C)
        lo, hi = float(q.min()), float(q.max())
        span = max(hi - lo, 1e-6)
        for i, name in enumerate(ACTION_NAMES):
            y = 160 + i * 42
            w = int(8 + (q[i] - lo) / span * (PANEL - 150))
            pygame.draw.rect(sc.surf, RED if i == best else DIM, (x, y + 20, w, 10))
            sc.text(f"{name.upper():8s} {q[i]:+.2f}", x, y, RED if i == best else BLUE)
        slider.draw(sc, 300)
        sc.text("SPACE PAUSE  R RESET", x, 360, DIM_C)
        sc.text("+/- SPEED  ESC MENU", x, 385, DIM_C)
        if paused:
            sc.text("PAUSED", x, 410, RED)
        if env.done:
            sc.overlay("GAME OVER", f"{env.cause.upper()}; RESTARTING...")
        pygame.display.flip()
        if max_frames is not None and frames >= max_frames:
            return "done"


def _message(sc, line1, line2, max_frames=None):
    frames = 0
    while True:
        sc.clock.tick(30)
        frames += 1
        for e in pygame.event.get():
            if _quit_event(e):
                return "quit"
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                return "menu"
        sc.surf.fill(BG)
        sc.text(line1, 20, 40, RED)
        sc.text(line2, 20, 70, BLUE)
        sc.text("ESC: BACK", 20, 110, DIM_C)
        pygame.display.flip()
        if max_frames is not None and frames >= max_frames:
            return "done"


def run_task_screen(title, worker, max_frames=None, screen=None):
    """Run a blocking headless task (train/eval/baseline) in a thread and show its
    status lines. worker(state) updates state['lines'] and polls state['stop']."""
    sc = screen or Screen()
    state = {"lines": ["starting..."], "stop": False, "done": False}
    th = threading.Thread(target=lambda: _safe(worker, state), daemon=True)
    th.start()
    frames = 0
    while True:
        sc.clock.tick(15)
        frames += 1
        for e in pygame.event.get():
            if _quit_event(e):
                state["stop"] = True
                return "quit"
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                state["stop"] = True
                return "menu"
        sc.surf.fill(BG)
        sc.text(title.upper(), 20, 20, ACC_C, sc.big)
        for i, ln in enumerate(list(state["lines"])[-14:]):
            sc.text(ln, 20, 75 + i * 24, RED if ln.startswith("ERROR") else BLUE)
        sc.text("ESC: BACK (STOPS A RUNNING TASK)", 20, sc.surf.get_height() - 30, DIM_C)
        pygame.display.flip()
        if max_frames is not None and frames >= max_frames:
            state["stop"] = True
            return "done"


def _safe(worker, state):
    try:
        worker(state)
    except Exception as ex:  # show errors in the window instead of dying silently
        state["lines"].append(f"ERROR: {ex}")
    state["done"] = True


def task_train(state):
    from .train import train
    hp = HParams()
    state["lines"] = [f"Training DQN seed 0, {hp.episodes} episodes", "(headless, CPU)"]

    def prog(ep, total, row):
        if ep % 10 == 0 or ep == total:
            state["lines"].append(f"ep {ep}/{total} score {row['score']} eps {row['epsilon']}")
    train("dqn", 0, hp, os.path.join("results", "runs", "dqn_seed0"),
          progress=prog, should_stop=lambda: state["stop"])
    state["lines"].append("done. checkpoint: " + DEFAULT_CKPT)


def task_eval(state):
    from .agent import DQNAgent
    from .evaluate import run_episodes, agent_policy, random_policy, summarize
    if not os.path.exists(DEFAULT_CKPT):
        state["lines"] = [f"No checkpoint at {DEFAULT_CKPT}", "Run Train DQN first."]
        return
    state["lines"] = ["Evaluating 100 greedy episodes..."]
    a = summarize(run_episodes(agent_policy(DQNAgent.load(DEFAULT_CKPT)), 100))
    b = summarize(run_episodes(random_policy(0), 100))
    state["lines"] += [f"DQN    mean {a['mean_score']:.2f}  std {a['std_score']:.2f}  max {a['max_score']}",
                       f"Random mean {b['mean_score']:.2f}  std {b['std_score']:.2f}  max {b['max_score']}"]


def task_random(state):
    from .evaluate import run_episodes, random_policy, summarize
    state["lines"] = ["Random baseline, 100 episodes..."]
    b = summarize(run_episodes(random_policy(0), 100))
    state["lines"] += [f"mean {b['mean_score']:.2f}  std {b['std_score']:.2f}  max {b['max_score']}",
                       f"causes: {b['causes']}"]


MENU = [("Manual Play", "manual"), ("Train DQN", "train"), ("AI Auto Play", "auto"),
        ("Evaluate DQN", "eval"), ("Random Baseline", "random"), ("Quit", "quit")]


def run_menu(max_frames=None, checkpoint=DEFAULT_CKPT, screen=None):
    sc = screen or Screen()
    sel, frames = 0, 0
    while True:
        sc.clock.tick(30)
        frames += 1
        choice = None
        for e in pygame.event.get():
            if _quit_event(e):
                return
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_UP, pygame.K_w):
                    sel = (sel - 1) % len(MENU)
                elif e.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % len(MENU)
                elif e.key in (pygame.K_RETURN, pygame.K_SPACE):
                    choice = MENU[sel][1]
                elif e.key == pygame.K_ESCAPE:
                    return
            if e.type == pygame.MOUSEBUTTONDOWN:
                for i in range(len(MENU)):
                    if pygame.Rect(40, 90 + i * 46, 320, 40).collidepoint(e.pos):
                        sel, choice = i, MENU[i][1]
        if choice == "quit":
            return
        if choice:
            r = {"manual": lambda: run_manual(screen=sc),
                 "auto": lambda: run_autoplay(checkpoint, screen=sc),
                 "train": lambda: run_task_screen("Train DQN", task_train, screen=sc),
                 "eval": lambda: run_task_screen("Evaluate DQN", task_eval, screen=sc),
                 "random": lambda: run_task_screen("Random Baseline", task_random, screen=sc)}[choice]()
            if r == "quit":
                return
        sc.surf.fill(BG)
        sc.text("SNAKE AI LAB", 40, 30, ACC_C, sc.big)
        for i, (label, _) in enumerate(MENU):
            rect = pygame.Rect(40, 90 + i * 46, 320, 40)
            on = i == sel
            pygame.draw.rect(sc.surf, RED if on else DIM, rect, 2)
            sc.text(("> " if on else "  ") + label.upper(), 56, 102 + i * 46, RED if on else BLUE)
        sc.text("UP/DOWN + ENTER, OR CLICK", 40, 90 + len(MENU) * 46 + 6, DIM_C)
        pygame.display.flip()
        if max_frames is not None and frames >= max_frames:
            return
