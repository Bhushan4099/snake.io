"""Headless Pygame smoke tests (SDL dummy driver). These do NOT verify the visuals."""
import pygame
import pytest
from snakelab import ui


def test_manual_and_menu_open_step_close():
    sc = ui.Screen()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
    assert ui.run_manual(max_frames=10, screen=sc) == "done"
    ui.run_menu(max_frames=5, screen=sc)
    pygame.quit()


def test_autoplay_with_checkpoint(tmp_path):
    from snakelab.agent import DQNAgent
    from snakelab.config import HParams
    p = tmp_path / "c.pt"
    DQNAgent("dqn", HParams(), seed=0).save(str(p), 0)
    sc = ui.Screen()
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
    assert ui.run_autoplay(str(p), max_frames=30, screen=sc) == "done"
    pygame.quit()


def test_autoplay_missing_checkpoint_shows_message(tmp_path):
    sc = ui.Screen()
    assert ui.run_autoplay(str(tmp_path / "nope.pt"), max_frames=3, screen=sc) == "done"
    pygame.quit()


def test_random_baseline_task_screen():
    sc = ui.Screen()
    assert ui.run_task_screen("Random Baseline", ui.task_random, max_frames=3, screen=sc) == "done"
    pygame.quit()
