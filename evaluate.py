"""Evaluate a trained Snake DQN, a random baseline, or compare both fairly."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import torch

from snake_ai_lab.ai import DQNAgent, DQNConfig
from snake_ai_lab.core import SnakeEnvironment

ProgressCallback = Callable[[dict[str, Any]], None]
DEFAULT_CHECKPOINT = Path("models/checkpoints/dqn_snake.pt")


def _summarize(scores: list[int], steps: list[int], episodes: int) -> dict[str, Any]:
    return {
        "episodes": episodes,
        "avg_score": float(np.mean(scores)),
        "best_score": max(scores),
        "avg_steps": float(np.mean(steps)),
        "scores": scores,
    }


def evaluate_checkpoint(
    *,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    episodes: int = 100,
    seed: int = 10_000,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Evaluate a saved DQN checkpoint without exploration."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Train the DQN first.")


    torch.set_num_threads(1)
    agent = DQNAgent(DQNConfig(), device="cpu", seed=seed)
    agent.load(checkpoint, load_optimizer=False)
    agent.epsilon = 0.0
    env = SnakeEnvironment()
    scores: list[int] = []
    steps: list[int] = []

    for episode_index in range(episodes):
        state = env.reset(seed=seed + episode_index)
        done = False
        info: dict[str, Any] = {"score": 0, "steps": 0}
        while not done:
            action = agent.select_action(state, explore=False)
            state, _, done, info = env.step(action)
        scores.append(int(info["score"]))
        steps.append(int(info["steps"]))
        if progress_callback is not None:
            progress_callback({
                "completed": episode_index + 1,
                "total": episodes,
                "score": scores[-1],
                "avg_score": float(np.mean(scores)),
                "best_score": max(scores),
                "avg_steps": float(np.mean(steps)),
            })

    return _summarize(scores, steps, episodes)



def evaluate_random(
    *,
    episodes: int = 100,
    seed: int = 10_000,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Evaluate a uniformly random relative-action policy.



    Uses the same environment and deterministic episode seeds as DQN evaluation,
    making this a simple baseline for measuring whether the trained policy helps.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = SnakeEnvironment()
    action_rng = random.Random(seed)
    scores: list[int] = []
    steps: list[int] = []

    for episode_index in range(episodes):
        env.reset(seed=seed + episode_index)
        done = False
        info: dict[str, Any] = {"score": 0, "steps": 0}
        while not done:
            # Action IDs match the environment: 0=straight, 1=left, 2=right.
            action = action_rng.randrange(3)
            _, _, done, info = env.step(action)
        scores.append(int(info["score"]))
        steps.append(int(info["steps"]))
        if progress_callback is not None:
            progress_callback({
                "completed": episode_index + 1,
                "total": episodes,
                "score": scores[-1],
                "avg_score": float(np.mean(scores)),
                "best_score": max(scores),
                "avg_steps": float(np.mean(steps)),
            })

    return _summarize(scores, steps, episodes)



def _print_result(title: str, result: dict[str, Any]) -> None:
    print(title)
    print(f"Evaluation episodes: {result['episodes']}")
    print(f"Average score:      {result['avg_score']:.3f}")
    print(f"Best score:         {result['best_score']}")
    print(f"Average steps:      {result['avg_steps']:.1f}")
    print(f"Scores:             {result['scores']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Snake DQN or random baseline")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--random", action="store_true", help="Evaluate random-action baseline")
    policy.add_argument("--compare", action="store_true", help="Evaluate DQN and random baseline")
    args = parser.parse_args()

    if args.random:
        result = evaluate_random(episodes=args.episodes, seed=args.seed)
        _print_result("Random baseline", result)
        return

    dqn_result = evaluate_checkpoint(
        checkpoint=args.checkpoint, episodes=args.episodes, seed=args.seed
    )
    _print_result("DQN evaluation", dqn_result)

    if args.compare:
        random_result = evaluate_random(episodes=args.episodes, seed=args.seed)
        _print_result("\nRandom baseline (same episode seeds)", random_result)
        difference = dqn_result["avg_score"] - random_result["avg_score"]
        relative = (
            100.0 * difference / random_result["avg_score"]
            if random_result["avg_score"] > 0
            else float("inf")
        )
        print("\nComparison")
        print(f"DQN minus random average score: {difference:+.3f}")
        if np.isfinite(relative):
            print(f"DQN relative improvement:       {relative:+.1f}%")
        else:
            print("DQN relative improvement:       undefined (random average is zero)")
    

if __name__ == "__main__":
    main()
