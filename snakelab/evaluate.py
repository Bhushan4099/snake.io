"""Evaluation: greedy DQN policy vs a uniform-random baseline on identical games."""
import numpy as np

from .config import EVAL_SEED_BASE, N_ACTIONS
from .env import SnakeEnv


def run_episodes(policy, n_episodes=100, base_seed=EVAL_SEED_BASE):
    """policy(state) -> action. Episode i uses env seed base_seed+i, so every policy
    plays exactly the same sequence of food placements/start directions (as far as
    its own trajectory allows)."""
    env = SnakeEnv()
    out = []
    for i in range(n_episodes):
        s = env.reset(seed=base_seed + i)
        done = False
        while not done:
            s, _, done, info = env.step(policy(s))
        out.append({"episode": i, "score": info["score"], "steps": info["steps"],
                    "cause": info["cause"]})
    return out


def agent_policy(agent):
    return lambda s: agent.act(s, epsilon=0.0)


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda s: int(rng.integers(N_ACTIONS))


def summarize(episodes):
    sc = np.array([e["score"] for e in episodes], dtype=float)
    causes = {}
    for e in episodes:
        causes[e["cause"]] = causes.get(e["cause"], 0) + 1
    return {"n": len(sc), "mean_score": float(sc.mean()), "std_score": float(sc.std(ddof=1)) if len(sc) > 1 else 0.0,
            "median_score": float(np.median(sc)), "max_score": int(sc.max()),
            "mean_steps": float(np.mean([e["steps"] for e in episodes])), "causes": causes}
