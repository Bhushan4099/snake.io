"""Training loop (headless)."""
import csv
import os
import time
import numpy as np
import torch

from .agent import DQNAgent
from .config import HParams
from .env import SnakeEnv

LOG_FIELDS = ["episode", "score", "reward", "steps", "epsilon", "loss", "cause"]


def train(variant="dqn", seed=0, hp: HParams = None, out_dir=None,
          progress=None, should_stop=None, verbose=False):
    """Train one agent. Writes <out_dir>/training_log.csv and checkpoints
    <out_dir>/checkpoints/ep{N:04d}.pt (every hp.ckpt_every episodes) and final.pt.
    `progress(ep, total, row)` is called after each episode; `should_stop()` may abort.
    Returns (agent, rows)."""
    hp = hp or HParams()
    torch.set_num_threads(1)
    env = SnakeEnv(seed=seed)
    agent = DQNAgent(variant, hp, seed=seed)
    rows = []
    ck_dir = None
    if out_dir:
        ck_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ck_dir, exist_ok=True)
    t0 = time.time()
    for ep in range(1, hp.episodes + 1):
        if should_stop and should_stop():
            break
        s = env.reset()
        done, total_r, losses = False, 0.0, []
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.remember(s, a, r, s2, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            s, total_r = s2, total_r + r
        eps_used = agent.epsilon
        agent.decay_epsilon()
        row = {"episode": ep, "score": info["score"], "reward": round(total_r, 3),
               "steps": info["steps"], "epsilon": round(eps_used, 5),
               "loss": round(float(np.mean(losses)), 5) if losses else "",
               "cause": info["cause"]}
        rows.append(row)
        if ck_dir and (ep % hp.ckpt_every == 0):
            agent.save(os.path.join(ck_dir, f"ep{ep:04d}.pt"), ep)
        if progress:
            progress(ep, hp.episodes, row)
        if verbose and ep % 50 == 0:
            recent = np.mean([r_["score"] for r_ in rows[-50:]])
            print(f"[{variant} s{seed}] ep {ep}/{hp.episodes} avg50 score {recent:.2f} "
                  f"eps {eps_used:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if out_dir:
        agent.save(os.path.join(out_dir, "final.pt"), len(rows))
        with open(os.path.join(out_dir, "training_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            w.writeheader()
            w.writerows(rows)
    return agent, rows
