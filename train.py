"""Train a DQN agent on the headless Snake environment.

The train_dqn() function is also used by the Pygame UI, so training progress
can be shown inside the application without blocking its event loop.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable

import numpy as np
import torch

from snake_ai_lab.ai import DQNAgent, DQNConfig
from snake_ai_lab.core import SnakeEnvironment

ProgressCallback = Callable[[dict[str, Any]], None]


def train_dqn(
    *,
    episodes: int = 500,
    seed: int = 42,
    width: int = 20,
    height: int = 20,
    max_steps_without_food: int = 100,
    checkpoint: Path | str = Path("models/checkpoints/dqn_snake.pt"),
    log: Path | str = Path("logs/training/training.csv"),
    save_every: int = 100,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the agent and report per-episode progress.

    Stop requests are handled safely at episode boundaries. The latest model
    is always saved before this function returns, including a user stop.
    """
    if episodes < 1 or save_every < 1:
        raise ValueError("episodes and save_every must be positive")

    checkpoint = Path(checkpoint)
    log = Path(log)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    env = SnakeEnvironment(
        width=width,
        height=height,
        max_steps_without_food=max_steps_without_food,
    )
    agent = DQNAgent(config=DQNConfig(), device="cpu", seed=seed)

    start_episode = 1
    resume_compatible = False
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume: checkpoint not found: {checkpoint}")
        try:
            extra = agent.load(checkpoint)
            start_episode = int(extra.get("episode", 0)) + 1
            resume_compatible = True
        except (RuntimeError, KeyError, ValueError) as exc:
            # A checkpoint trained with the old 11-feature observation cannot
            # load into the new 19-feature network. Start a fresh v2 run.
            if verbose:
                print(
                    "Existing checkpoint is incompatible with the 19-feature DQN; "
                    "starting fresh training. The old checkpoint file will be "
                    "replaced when the new model is saved."
                )
            start_episode = 1
            resume_compatible = False

    log.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    append_log = log.exists() and resume_compatible
    file_mode = "a" if append_log else "w"

    recent_scores: list[int] = []
    recent_rewards: list[float] = []
    recent_losses: list[float] = []
    completed = 0
    last_episode = start_episode - 1
    start_time = time.time()
    stopped = False

    with log.open(file_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if file_mode == "w":
            writer.writerow([
                "episode", "score", "steps", "total_reward", "mean_loss",
                "epsilon", "duration_seconds",
            ])

        for run_episode in range(1, episodes + 1):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break

            episode = start_episode + run_episode - 1
            state = env.reset(seed=seed + episode)
            total_reward = 0.0
            losses: list[float] = []
            done = False
            info: dict[str, Any] = {"score": 0, "steps": 0}

            while not done:
                action = agent.select_action(state, explore=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss_value = agent.learn()
                if loss_value is not None:
                    losses.append(loss_value)
                state = next_state
                total_reward += reward

            agent.decay_epsilon()
            score = int(info["score"])
            mean_loss = float(np.mean(losses)) if losses else 0.0
            recent_scores.append(score)
            recent_rewards.append(total_reward)
            if losses:
                recent_losses.extend(losses)

            elapsed = time.time() - start_time
            writer.writerow([
                episode,
                score,
                info["steps"],
                round(total_reward, 4),
                round(mean_loss, 6),
                round(agent.epsilon, 5),
                round(elapsed, 2),
            ])
            csv_file.flush()

            completed += 1
            last_episode = episode
            progress = {
                "completed": completed,
                "total": episodes,
                "episode": episode,
                "score": score,
                "steps": int(info["steps"]),
                "avg_score": float(np.mean(recent_scores[-50:])),
                "avg_reward": float(np.mean(recent_rewards[-50:])),
                "epsilon": float(agent.epsilon),
                "loss": mean_loss,
                "best_score": max(recent_scores) if recent_scores else 0,
                "elapsed": elapsed,
            }
            if progress_callback is not None:
                progress_callback(progress)

            if verbose and (run_episode == 1 or run_episode % 10 == 0):
                print(
                    f"Episode {episode:4d} | score={score:2d} | "
                    f"avg50={progress['avg_score']:.2f} | "
                    f"reward50={progress['avg_reward']:.2f} | "
                    f"epsilon={agent.epsilon:.3f} | loss={mean_loss:.4f}"
                )

            if episode % save_every == 0:
                agent.save(checkpoint, extra={"episode": episode, "seed": seed})

            # The stop request is deliberately checked after an episode so a
            # complete transition sequence is retained and the model can save.
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break

    if last_episode >= start_episode:
        agent.save(checkpoint, extra={"episode": last_episode, "seed": seed})

    result = {
        "completed": completed,
        "requested": episodes,
        "last_episode": last_episode,
        "stopped": stopped,
        "checkpoint": str(checkpoint),
        "log": str(log),
        "avg_score": float(np.mean(recent_scores[-50:])) if recent_scores else 0.0,
        "best_score": max(recent_scores) if recent_scores else 0,
        "elapsed": time.time() - start_time,
        "epsilon": float(agent.epsilon),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent to play Snake")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--max-steps-without-food", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/checkpoints/dqn_snake.pt"))
    parser.add_argument("--log", type=Path, default=Path("logs/training/training.csv"))
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--resume", nargs="?", const="", default=None, metavar="CHECKPOINT",
        help="Resume from the default checkpoint or from CHECKPOINT if supplied",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_dqn(
        episodes=args.episodes,
        seed=args.seed,
        width=args.width,
        height=args.height,
        max_steps_without_food=args.max_steps_without_food,
        checkpoint=Path(args.resume) if args.resume else args.checkpoint,
        log=args.log,
        save_every=args.save_every,
        resume=args.resume is not None,
        verbose=True,
    )
    print("\nTraining stopped after user request." if result["stopped"] else "\nTraining complete.")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"CSV log:    {result['log']}")
    print(f"Episodes:   {result['completed']}/{result['requested']}")
    print(f"Mean score (last up to 50): {result['avg_score']:.2f}")
    print(f"Elapsed:    {result['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
