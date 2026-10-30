"""Pygame application with in-app, non-blocking DQN training and evaluation."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
from typing import Any

import pygame

from snake_ai_lab.ai import DQNAgent, DQNConfig
from snake_ai_lab.core import Action, Direction, SnakeEnvironment


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "models" / "checkpoints" / "dqn_snake.pt"
TRAINING_LOG = ROOT / "logs" / "training" / "training.csv"
CELL = 25
BOARD_W = 20
BOARD_H = 20
BOARD_PIXELS_W = BOARD_W * CELL
BOARD_PIXELS_H = BOARD_H * CELL
SIDEBAR_W = 270
WINDOW_W = BOARD_PIXELS_W + SIDEBAR_W
WINDOW_H = BOARD_PIXELS_H + 70

BG = (16, 20, 28)
PANEL = (26, 33, 45)
GRID = (38, 47, 61)
TEXT = (235, 240, 247)
MUTED = (155, 168, 185)
GREEN = (62, 205, 128)
GREEN_DARK = (34, 145, 91)
RED = (245, 91, 91)
GOLD = (246, 193, 75)
BLUE = (89, 156, 245)


class SnakeAIApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Snake AI Lab")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 20)
        self.small = pygame.font.SysFont("segoeui", 15)
        self.title_font = pygame.font.SysFont("segoeui", 34, bold=True)
        self.env = SnakeEnvironment(width=BOARD_W, height=BOARD_H)
        self.mode = "menu"
        self.play_mode = "manual"
        self.agent: DQNAgent | None = None
        self.state = self.env.get_state()
        self.pending_action = Action.STRAIGHT
        self.speed = 8.0
        self.accumulator = 0.0
        self.message = "Choose a mode to begin."
        self.message_until = 0
        self.menu_buttons: list[tuple[str, pygame.Rect]] = []

        # Background workers keep the Pygame event loop responsive.
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.worker_stop: threading.Event | None = None
        self.worker_kind = "training"
        self.worker_running = False
        self.worker_stop_requested = False
        self.worker_progress: dict[str, Any] = {}
        self.worker_result: dict[str, Any] | None = None
        self.worker_error: str | None = None
        self.worker_buttons: list[tuple[str, pygame.Rect]] = []

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            self._drain_worker_queue()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    if self.worker_running and self.worker_stop is not None:
                        self.worker_stop.set()
                elif self.mode == "menu":
                    self._handle_menu_event(event)
                elif self.mode in ("game", "paused"):
                    self._handle_game_event(event)
                elif self.mode in ("worker", "worker_done"):
                    self._handle_worker_event(event)

            if self.mode in ("game", "paused"):
                self._update_game(dt)
                self._draw_game()
            elif self.mode in ("worker", "worker_done"):
                self._draw_worker_screen()
            else:
                self._draw_menu()
            pygame.display.flip()

        if self.worker_running and self.worker_stop is not None:
            self.worker_stop.set()
            if self.worker_thread is not None:
                self.worker_thread.join(timeout=10.0)
        pygame.quit()

    def _set_message(self, text: str, seconds: float = 4.0) -> None:
        self.message = text
        self.message_until = pygame.time.get_ticks() + int(seconds * 1000)

    def _draw_text(self, text: str, x: int, y: int, font=None, color=TEXT) -> None:
        surface = (font or self.font).render(text, True, color)
        self.screen.blit(surface, (x, y))

    def _draw_menu(self) -> None:
        self.screen.fill(BG)
        self._draw_text("SNAKE AI LAB", 52, 42, self.title_font, GREEN)
        self._draw_text("Deep Q-Network reinforcement learning", 55, 91, self.font, MUTED)
        self._draw_text("Select an action", 58, 150, self.font)

        options = [
            ("Manual Play", "Play with arrow keys / WASD"),
            ("AI Auto Play", "Load the trained DQN checkpoint"),
            ("Train DQN (500 episodes)", "Train inside this window — live progress and stop control"),
            ("Evaluate DQN (100 episodes)", "Evaluate inside this window — no terminal progress"),
        ]
        self.menu_buttons = []
        y = 175
        for label, description in options:
            rect = pygame.Rect(58, y, WINDOW_W - 116, 68)
            pygame.draw.rect(self.screen, PANEL, rect, border_radius=10)
            pygame.draw.rect(self.screen, GRID, rect, width=1, border_radius=10)
            highlight = GREEN if label.startswith("AI") or label.startswith("Train") else TEXT
            self._draw_text(label, rect.x + 18, rect.y + 9, self.font, highlight)
            self._draw_text(description, rect.x + 18, rect.y + 39, self.small, MUTED)
            self.menu_buttons.append((label, rect))
            y += 76

        model_status = "DQN checkpoint found" if CHECKPOINT.exists() else "No checkpoint yet — train the DQN first"
        status_color = GREEN if CHECKPOINT.exists() else GOLD
        self._draw_text(model_status, 60, WINDOW_H - 82, self.small, status_color)
        if pygame.time.get_ticks() < self.message_until:
            self._draw_text(self.message, 60, WINDOW_H - 58, self.small, GOLD)
        self._draw_text("Esc / window close: quit", 60, WINDOW_H - 34, self.small, MUTED)

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for label, rect in self.menu_buttons:
                if not rect.collidepoint(event.pos):
                    continue
                if label == "Manual Play":
                    self._start_game("manual")
                elif label == "AI Auto Play":
                    if not CHECKPOINT.exists():
                        self._set_message("Train the DQN first: choose Train DQN.")
                    else:
                        self._start_game("ai")
                elif label.startswith("Train"):
                    self._start_training(episodes=500)
                elif label.startswith("Evaluate"):
                    if not CHECKPOINT.exists():
                        self._set_message("No checkpoint found. Train the DQN first.")
                    else:
                        self._start_evaluation(episodes=100)
                break

    def _start_training(self, episodes: int) -> None:
        if self.worker_running:
            self._set_message("A background task is already running.")
            return
        self.worker_kind = "training"
        self.worker_progress = {"completed": 0, "total": episodes, "episode": 0}
        self.worker_result = None
        self.worker_error = None
        self.worker_stop_requested = False
        self.worker_stop = threading.Event()
        self.worker_running = True
        resume = CHECKPOINT.exists()
        self.mode = "worker"

        def work() -> None:
            try:
                from train import train_dqn

                result = train_dqn(
                    episodes=episodes,
                    seed=42,
                    checkpoint=CHECKPOINT,
                    log=TRAINING_LOG,
                    save_every=100,
                    resume=resume,
                    progress_callback=lambda item: self.worker_queue.put(("progress", item)),
                    stop_event=self.worker_stop,
                    verbose=False,
                )
                self.worker_queue.put(("finished", result))
            except Exception as exc:  # Forward errors to the UI thread.
                self.worker_queue.put(("error", f"{type(exc).__name__}: {exc}"))

        self.worker_thread = threading.Thread(target=work, name="snake-dqn-training", daemon=True)
        self.worker_thread.start()

    def _start_evaluation(self, episodes: int) -> None:
        if self.worker_running:
            self._set_message("A background task is already running.")
            return
        self.worker_kind = "evaluation"
        self.worker_progress = {"completed": 0, "total": episodes}
        self.worker_result = None
        self.worker_error = None
        self.worker_stop_requested = False
        self.worker_stop = threading.Event()
        self.worker_running = True
        self.mode = "worker"

        def work() -> None:
            try:
                from evaluate import evaluate_checkpoint

                result = evaluate_checkpoint(
                    checkpoint=CHECKPOINT,
                    episodes=episodes,
                    progress_callback=lambda item: self.worker_queue.put(("progress", item)),
                )
                self.worker_queue.put(("finished", result))
            except Exception as exc:
                self.worker_queue.put(("error", f"{type(exc).__name__}: {exc}"))

        self.worker_thread = threading.Thread(target=work, name="snake-dqn-evaluation", daemon=True)
        self.worker_thread.start()

    def _drain_worker_queue(self) -> None:
        while True:
            try:
                event_name, payload = self.worker_queue.get_nowait()
            except queue.Empty:
                break
            if event_name == "progress":
                self.worker_progress = dict(payload)
            elif event_name == "finished":
                self.worker_result = dict(payload)
                self.worker_running = False
                self.mode = "worker_done"
            elif event_name == "error":
                self.worker_error = str(payload)
                self.worker_running = False
                self.mode = "worker_done"

    def _handle_worker_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.worker_running:
                self._set_message("Stop the task before leaving this screen.")
            else:
                self.mode = "menu"
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for label, rect in self.worker_buttons:
                if not rect.collidepoint(event.pos):
                    continue
                if label == "Stop & Save" and self.worker_running and self.worker_stop is not None:
                    self.worker_stop_requested = True
                    self.worker_stop.set()
                elif label == "Back to Menu" and not self.worker_running:
                    self.mode = "menu"
                break

    def _draw_worker_screen(self) -> None:
        self.screen.fill(BG)
        is_training = self.worker_kind == "training"
        title = "DQN TRAINING" if is_training else "DQN EVALUATION"
        subtitle = (
            "Learning on CPU — Pygame stays responsive while training runs"
            if is_training else "Testing the saved policy — no exploration or weight updates"
        )
        self._draw_text("SNAKE AI LAB", 48, 34, self.small, GREEN)
        self._draw_text(title, 48, 68, self.title_font, GREEN if is_training else BLUE)
        self._draw_text(subtitle, 50, 116, self.small, MUTED)

        completed = int(self.worker_progress.get("completed", 0))
        total = max(1, int(self.worker_progress.get("total", 1)))
        fraction = max(0.0, min(1.0, completed / total))
        progress_rect = pygame.Rect(55, 174, WINDOW_W - 110, 18)
        pygame.draw.rect(self.screen, GRID, progress_rect, border_radius=9)
        fill_rect = pygame.Rect(progress_rect.x, progress_rect.y, int(progress_rect.width * fraction), progress_rect.height)
        if fill_rect.width > 0:
            pygame.draw.rect(self.screen, GREEN if is_training else BLUE, fill_rect, border_radius=9)
        self._draw_text(f"{completed} / {total} episodes  ({fraction * 100:.1f}%)", 55, 202, self.font)

        if is_training:
            episode = int(self.worker_progress.get("episode", 0))
            score = int(self.worker_progress.get("score", 0))
            avg_score = float(self.worker_progress.get("avg_score", 0.0))
            epsilon = float(self.worker_progress.get("epsilon", 1.0))
            loss = float(self.worker_progress.get("loss", 0.0))
            best_score = int(self.worker_progress.get("best_score", 0))
            elapsed = float(self.worker_progress.get("elapsed", 0.0))
            self._metric_card("Current episode", str(episode), 55, 258, 195)
            self._metric_card("Latest score", str(score), 270, 258, 195)
            self._metric_card("Average score (last 50)", f"{avg_score:.2f}", 485, 258, 230)
            self._metric_card("Exploration (epsilon)", f"{epsilon:.3f}", 55, 342, 195)
            self._metric_card("Training loss", f"{loss:.4f}", 270, 342, 195)
            self._metric_card("Best score", str(best_score), 485, 342, 230)
            self._draw_text(f"Elapsed: {elapsed:.1f}s", 58, 440, self.small, MUTED)
            self._draw_text(
                "Saves every 100 episodes; Stop & Save finishes this episode first.",
                58, 465, self.small, MUTED,
            )
        else:
            score = int(self.worker_progress.get("score", 0))
            avg_score = float(self.worker_progress.get("avg_score", 0.0))
            best_score = int(self.worker_progress.get("best_score", 0))
            avg_steps = float(self.worker_progress.get("avg_steps", 0.0))
            self._metric_card("Latest score", str(score), 55, 270, 195)
            self._metric_card("Average score", f"{avg_score:.2f}", 270, 270, 195)
            self._metric_card("Best score", str(best_score), 485, 270, 230)
            self._metric_card("Average survival steps", f"{avg_steps:.1f}", 55, 354, 195)

        if self.worker_running:
            if self.worker_stop_requested:
                status = "Stop requested — finishing current episode and saving checkpoint..."
                status_color = GOLD
            else:
                status = "RUNNING — live progress appears here; no terminal training output."
                status_color = GREEN
            self._draw_text(status, 58, 500, self.small, status_color)
            button_label = "Stop & Save" if is_training else "Back to Menu"
            self.worker_buttons = []
            if is_training:
                rect = pygame.Rect(WINDOW_W - 220, 488, 165, 48)
                pygame.draw.rect(self.screen, RED, rect, border_radius=8)
                self._draw_text("Stop & Save", rect.x + 25, rect.y + 13, self.small, (255, 255, 255))
                self.worker_buttons.append((button_label, rect))
        else:
            if self.worker_error:
                self._draw_text(f"Task failed: {self.worker_error[:85]}", 58, 500, self.small, RED)
            elif self.worker_result is not None:
                if is_training:
                    if self.worker_result.get("stopped"):
                        status = "Training stopped safely. Latest checkpoint saved."
                    else:
                        status = "Training complete. Latest checkpoint saved."
                    self._draw_text(status, 58, 500, self.small, GREEN)
                else:
                    self._draw_text("Evaluation complete.", 58, 500, self.small, GREEN)
            self.worker_buttons = []
            rect = pygame.Rect(WINDOW_W - 220, 488, 165, 48)
            pygame.draw.rect(self.screen, BLUE, rect, border_radius=8)
            self._draw_text("Back to Menu", rect.x + 20, rect.y + 13, self.small, (255, 255, 255))
            self.worker_buttons.append(("Back to Menu", rect))

    def _metric_card(self, label: str, value: str, x: int, y: int, width: int) -> None:
        rect = pygame.Rect(x, y, width, 66)
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=9)
        self._draw_text(label, x + 12, y + 8, self.small, MUTED)
        self._draw_text(value, x + 12, y + 32, self.font, TEXT)

    def _start_game(self, mode: str) -> None:
        self.play_mode = mode
        self.agent = None
        if mode == "ai":
            self.agent = DQNAgent(DQNConfig(), device="cpu")
            try:
                self.agent.load(CHECKPOINT, load_optimizer=False)
            except Exception as exc:
                self.agent = None
                self.mode = "menu"
                self._set_message(f"Could not load checkpoint: {exc}", 8)
                return
            self.agent.epsilon = 0.0
        self.state = self.env.reset()
        self.pending_action = Action.STRAIGHT
        self.accumulator = 0.0
        self.mode = "game"

    def _handle_game_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.mode = "menu"
            elif event.key in (pygame.K_p, pygame.K_SPACE):
                self.mode = "paused" if self.mode == "game" else "game"
            elif event.key in (pygame.K_r, pygame.K_RETURN) and self.env.game_over:
                self._start_game(self.play_mode)
            elif self.play_mode == "manual" and not self.env.game_over:
                self._set_manual_direction(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            slider = self._speed_slider_rect()
            if slider.inflate(12, 20).collidepoint(event.pos):
                self._set_speed_from_x(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
            slider = self._speed_slider_rect()
            if slider.inflate(12, 20).collidepoint(event.pos):
                self._set_speed_from_x(event.pos[0])

    def _set_manual_direction(self, key: int) -> None:
        key_to_direction = {
            pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP,
            pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT,
            pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT,
        }
        desired = key_to_direction.get(key)
        if desired is None:
            return
        current = self.env.direction
        delta = (int(desired) - int(current)) % 4
        if delta == 0:
            self.pending_action = Action.STRAIGHT
        elif delta == 1:
            self.pending_action = Action.RIGHT
        elif delta == 3:
            self.pending_action = Action.LEFT
        # Ignore 180-degree reversals.

    def _speed_slider_rect(self) -> pygame.Rect:
        return pygame.Rect(BOARD_PIXELS_W + 28, 235, SIDEBAR_W - 56, 8)

    def _set_speed_from_x(self, x: int) -> None:
        rect = self._speed_slider_rect()
        fraction = max(0.0, min(1.0, (x - rect.x) / rect.width))
        self.speed = 4.0 + fraction * 16.0

    def _update_game(self, dt: float) -> None:
        if self.mode != "game" or self.env.game_over:
            return
        self.accumulator += dt
        interval = 1.0 / self.speed
        while self.accumulator >= interval and not self.env.game_over:
            if self.play_mode == "ai" and self.agent is not None:
                action = self.agent.select_action(self.state, explore=False)
            else:
                action = self.pending_action
            self.state, _, _, _ = self.env.step(action)
            self.pending_action = Action.STRAIGHT
            self.accumulator -= interval

    def _draw_game(self) -> None:
        self.screen.fill(BG)
        board = pygame.Rect(0, 0, BOARD_PIXELS_W, BOARD_PIXELS_H)
        pygame.draw.rect(self.screen, (12, 16, 23), board)
        for x in range(0, BOARD_PIXELS_W + 1, CELL):
            pygame.draw.line(self.screen, GRID, (x, 0), (x, BOARD_PIXELS_H), 1)
        for y in range(0, BOARD_PIXELS_H + 1, CELL):
            pygame.draw.line(self.screen, GRID, (0, y), (BOARD_PIXELS_W, y), 1)

        if self.env.food is not None:
            fx, fy = self.env.food
            pygame.draw.circle(self.screen, RED, (fx * CELL + CELL // 2, fy * CELL + CELL // 2), CELL // 2 - 3)
        for index, (sx, sy) in enumerate(self.env.snake):
            rect = pygame.Rect(sx * CELL + 2, sy * CELL + 2, CELL - 4, CELL - 4)
            pygame.draw.rect(self.screen, GREEN if index == 0 else GREEN_DARK, rect, border_radius=5)

        panel_x = BOARD_PIXELS_W
        pygame.draw.rect(self.screen, PANEL, (panel_x, 0, SIDEBAR_W, WINDOW_H))
        label = "AI AUTO MODE" if self.play_mode == "ai" else "MANUAL MODE"
        self._draw_text(label, panel_x + 24, 30, self.font, GREEN if self.play_mode == "ai" else BLUE)
        self._draw_text(f"Score: {self.env.score}", panel_x + 24, 78, self.font)
        self._draw_text(f"Length: {len(self.env.snake)}", panel_x + 24, 110, self.small, MUTED)
        self._draw_text(f"Speed: {self.speed:.1f} steps/sec", panel_x + 24, 180, self.small, TEXT)
        slider = self._speed_slider_rect()
        pygame.draw.line(self.screen, MUTED, slider.midleft, slider.midright, 5)
        knob_x = int(slider.x + (self.speed - 4.0) / 16.0 * slider.width)
        pygame.draw.circle(self.screen, GOLD, (knob_x, slider.centery), 9)
        self._draw_text("Drag slider to change speed", panel_x + 24, 255, self.small, MUTED)

        if self.play_mode == "ai" and self.agent is not None:
            q = self.agent.q_values(self.state)
            names = ["Straight", "Left", "Right"]
            self._draw_text("DQN Q-values", panel_x + 24, 315, self.font)
            for i, (name, value) in enumerate(zip(names, q)):
                y = 355 + i * 48
                self._draw_text(f"{name}: {value:+.3f}", panel_x + 24, y, self.small, GREEN if i == int(q.argmax()) else TEXT)
        else:
            self._draw_text("Controls", panel_x + 24, 325, self.font)
            self._draw_text("Arrow keys / WASD", panel_x + 24, 360, self.small, TEXT)
        self._draw_text("P / Space: pause", panel_x + 24, WINDOW_H - 112, self.small, MUTED)
        self._draw_text("R / Enter: restart after game over", panel_x + 24, WINDOW_H - 86, self.small, MUTED)
        self._draw_text("Esc: back to menu", panel_x + 24, WINDOW_H - 60, self.small, MUTED)

        if self.env.game_over:
            overlay = pygame.Surface((BOARD_PIXELS_W, BOARD_PIXELS_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 155))
            self.screen.blit(overlay, (0, 0))
            title = "YOU WIN" if self.env.done_reason == "won" else "GAME OVER"
            self._draw_text(title, 205, 250, self.title_font, GOLD)
            self._draw_text(f"Score: {self.env.score}   |   Press R to restart", 145, 300, self.font)
        elif self.mode == "paused":
            overlay = pygame.Surface((BOARD_PIXELS_W, BOARD_PIXELS_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 135))
            self.screen.blit(overlay, (0, 0))
            self._draw_text("PAUSED", 240, 275, self.title_font, GOLD)


def main() -> None:
    app = SnakeAIApp()
    app.run()
