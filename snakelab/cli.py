"""Single entry point: `python main.py <command>`. No command opens the Pygame menu."""
import argparse
import os
import sys

from .config import HParams


def _default_ckpt():
    return os.path.join("results", "runs", "dqn_seed0", "final.pt")


def build_parser():
    p = argparse.ArgumentParser(prog="main.py", description="Snake AI Lab")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("menu", help="Pygame menu (default)")
    sub.add_parser("manual", help="Manual Play (Pygame)")
    a = sub.add_parser("autoplay", help="AI Auto Play (Pygame, speed slider, Q-values)")
    a.add_argument("--checkpoint", default=_default_ckpt())
    t = sub.add_parser("train", help="Train DQN (headless)")
    t.add_argument("--variant", choices=["dqn", "double"], default="dqn")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--episodes", type=int, default=500)
    t.add_argument("--out", default=None, help="default results/runs/<variant>_seed<seed>")
    e = sub.add_parser("evaluate", help="Evaluate a checkpoint vs the random baseline")
    e.add_argument("--checkpoint", default=_default_ckpt())
    e.add_argument("--episodes", type=int, default=100)
    r = sub.add_parser("random", help="Random baseline")
    r.add_argument("--episodes", type=int, default=100)
    c = sub.add_parser("compare", help="Tabulate every checkpoint in a checkpoints/ directory")
    c.add_argument("--dir", default=os.path.join("results", "runs", "dqn_seed0", "checkpoints"))
    c.add_argument("--episodes", type=int, default=100)
    ra = sub.add_parser("run-all", help="Full pipeline (same as run_all.py)")
    ra.add_argument("--quick", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cmd = args.cmd or "menu"
    if cmd in ("menu", "manual", "autoplay"):
        from . import ui
        if cmd == "menu":
            ui.run_menu()
        elif cmd == "manual":
            ui.run_manual()
        else:
            ui.run_autoplay(args.checkpoint)
        return 0
    if cmd == "train":
        from .train import train
        out = args.out or os.path.join("results", "runs", f"{args.variant}_seed{args.seed}")
        train(args.variant, args.seed, HParams(episodes=args.episodes), out, verbose=True)
        print("saved to", out)
    elif cmd == "evaluate":
        from .agent import DQNAgent
        from .evaluate import run_episodes, agent_policy, random_policy, summarize
        a = summarize(run_episodes(agent_policy(DQNAgent.load(args.checkpoint)), args.episodes))
        b = summarize(run_episodes(random_policy(0), args.episodes))
        print(f"{'policy':8s} {'mean':>7s} {'std':>7s} {'max':>5s}  causes")
        for n, s in (("agent", a), ("random", b)):
            print(f"{n:8s} {s['mean_score']:7.2f} {s['std_score']:7.2f} {s['max_score']:5d}  {s['causes']}")
    elif cmd == "random":
        from .evaluate import run_episodes, random_policy, summarize
        s = summarize(run_episodes(random_policy(0), args.episodes))
        print(f"random baseline over {s['n']} games: mean {s['mean_score']:.2f} std {s['std_score']:.2f} max {s['max_score']} causes {s['causes']}")
    elif cmd == "compare":
        import glob
        from .agent import DQNAgent
        from .evaluate import run_episodes, agent_policy, summarize
        print(f"{'checkpoint':14s} {'mean':>7s} {'std':>7s} {'max':>5s}")
        for pth in sorted(glob.glob(os.path.join(args.dir, "*.pt"))):
            s = summarize(run_episodes(agent_policy(DQNAgent.load(pth)), args.episodes))
            print(f"{os.path.basename(pth):14s} {s['mean_score']:7.2f} {s['std_score']:7.2f} {s['max_score']:5d}")
    elif cmd == "run-all":
        import run_all
        return run_all.main(["--quick"] if args.quick else [])
    return 0
