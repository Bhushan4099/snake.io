"""One command: train (DQN + Double DQN x seeds), evaluate vs random, compare checkpoints,
regenerate plots and tables.

  python run_all.py            # full protocol: 3 seeds x 2 variants x 500 episodes, 100 eval games
  python run_all.py --quick    # smoke run (tiny; writes to results_quick/, README untouched)
"""
import argparse
import os

from snakelab.pipeline import run_all


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=None, help="number of seeds (default 3; quick 2)")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default=None)
    p.add_argument("--skip-train", action="store_true", help="reuse existing runs/ and only re-evaluate/plot")
    a = p.parse_args(argv)
    if a.quick:
        n_seeds, episodes, n_eval, ck, out, readme = a.seeds or 2, a.episodes or 60, a.eval_episodes or 20, 20, a.out or "results_quick", None
        # Never let a smoke run overwrite the reported full-run results.
        if os.path.abspath(out) == os.path.abspath("results"):
            p.error("--quick must not write to results/ (it holds the reported full-run results); use another --out")
    else:
        n_seeds, episodes, n_eval, ck, out = a.seeds or 3, a.episodes or 500, a.eval_episodes or 100, 50, a.out or "results"
        readme = "README.md"
    agg, meta = run_all(out, tuple(range(n_seeds)), episodes, n_eval, a.workers, ck, readme, a.skip_train)
    for k, v in agg.items():
        print(f"{k:7s} mean {v['mean_of_seed_means']:.2f}  std-across-seeds {v['std_across_seeds']:.2f}  seed means {v['seed_means']}")
    print("outputs in", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
