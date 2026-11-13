"""One-command pipeline: train all (variant, seed) runs, evaluate, compare checkpoints,
write CSV/JSON/Markdown tables and plots. Every number in the docs comes from here."""
import csv
import glob
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .agent import DQNAgent
from .config import HParams, EVAL_SEED_BASE
from .evaluate import run_episodes, agent_policy, random_policy, summarize
from .train import train
from . import plots

VARIANTS = ["dqn", "double"]


def _train_job(args):
    variant, seed, hp_dict, out_dir = args
    t = time.time()
    train(variant, seed, HParams(**hp_dict), out_dir, verbose=True)
    return variant, seed, time.time() - t


def train_all(out, seeds, hp, workers=2):
    jobs = [(v, s, hp.to_dict(), os.path.join(out, "runs", f"{v}_seed{s}"))
            for s in seeds for v in VARIANTS]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_train_job, jobs))
    else:
        res = [_train_job(j) for j in jobs]
    wall = time.time() - t0
    return {"wall_seconds": round(wall, 1),
            "per_run_seconds": {f"{v}_seed{s}": round(t, 1) for v, s, t in res}}


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _ms(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def evaluate_all(out, seeds, n_eval):
    ep_rows, run_rows = [], []
    per_seed = {"dqn": [], "double": [], "random": []}
    for v in VARIANTS:
        for s in seeds:
            agent = DQNAgent.load(os.path.join(out, "runs", f"{v}_seed{s}", "final.pt"), seed=s)
            eps = run_episodes(agent_policy(agent), n_eval)
            ep_rows += [dict(variant=v, seed=s, **e) for e in eps]
            sm = summarize(eps)
            run_rows.append(dict(variant=v, seed=s, **{k: sm[k] for k in
                              ("n", "mean_score", "std_score", "median_score", "max_score", "mean_steps")},
                              **{f"n_{c}": sm["causes"].get(c, 0) for c in ("wall", "body", "starved", "won")}))
            per_seed[v].append(sm["mean_score"])
    # Random baseline: one independent RNG stream per seed, same 100 games.
    for s in seeds:
        eps = run_episodes(random_policy(seed=s), n_eval)
        ep_rows += [dict(variant="random", seed=s, **e) for e in eps]
        sm = summarize(eps)
        run_rows.append(dict(variant="random", seed=s, **{k: sm[k] for k in
                          ("n", "mean_score", "std_score", "median_score", "max_score", "mean_steps")},
                          **{f"n_{c}": sm["causes"].get(c, 0) for c in ("wall", "body", "starved", "won")}))
        per_seed["random"].append(sm["mean_score"])
    _write_csv(os.path.join(out, "eval_episodes.csv"), ep_rows,
               ["variant", "seed", "episode", "score", "steps", "cause"])
    fields = ["variant", "seed", "n", "mean_score", "std_score", "median_score", "max_score",
              "mean_steps", "n_wall", "n_body", "n_starved", "n_won"]
    _write_csv(os.path.join(out, "eval_per_run.csv"), run_rows, fields)
    agg = {}
    for name, vals in per_seed.items():
        m, sd = _ms(vals)
        pooled = [r["score"] for r in ep_rows if r["variant"] == name]
        agg[name] = {"seeds": list(seeds), "seed_means": vals, "mean_of_seed_means": m,
                     "std_across_seeds": sd, "min_seed_mean": min(vals), "max_seed_mean": max(vals),
                     "pooled_max_score": int(max(pooled)), "n_eval_episodes_per_seed": n_eval}
    return agg, per_seed


def compare_checkpoints(out, seeds, n_eval):
    """Evaluate every saved intermediate checkpoint of every run (greedy, same games)."""
    rows, table = [], {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for s in seeds:
            d = os.path.join(out, "runs", f"{v}_seed{s}", "checkpoints")
            for p in sorted(glob.glob(os.path.join(d, "ep*.pt"))):
                ep = int(re.search(r"ep(\d+)\.pt", p).group(1))
                agent = DQNAgent.load(p, seed=s)
                sm = summarize(run_episodes(agent_policy(agent), n_eval))
                rows.append(dict(variant=v, seed=s, checkpoint_episode=ep,
                                 mean_score=sm["mean_score"], std_score=sm["std_score"],
                                 max_score=sm["max_score"], mean_steps=sm["mean_steps"]))
                table[v].setdefault(ep, []).append(sm["mean_score"])
    _write_csv(os.path.join(out, "checkpoint_compare_per_run.csv"), rows,
               ["variant", "seed", "checkpoint_episode", "mean_score", "std_score", "max_score", "mean_steps"])
    agg_rows = []
    for v in VARIANTS:
        for ep in sorted(table[v]):
            m, sd = _ms(table[v][ep])
            agg_rows.append(dict(variant=v, checkpoint_episode=ep, n_seeds=len(table[v][ep]),
                                 mean_of_seed_means=round(m, 3), std_across_seeds=round(sd, 3)))
    _write_csv(os.path.join(out, "checkpoint_compare.csv"), agg_rows,
               ["variant", "checkpoint_episode", "n_seeds", "mean_of_seed_means", "std_across_seeds"])
    return table, agg_rows


def checkpoint_markdown(agg_rows):
    eps = sorted({r["checkpoint_episode"] for r in agg_rows})
    lines = ["| Checkpoint (episode) | " + " | ".join(plots.LABELS[v] for v in VARIANTS) + " |",
             "|---|" + "---|" * len(VARIANTS)]
    for e in eps:
        cells = []
        for v in VARIANTS:
            r = next((x for x in agg_rows if x["variant"] == v and x["checkpoint_episode"] == e), None)
            cells.append(f"{r['mean_of_seed_means']:.2f} ± {r['std_across_seeds']:.2f}" if r else "-")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_markdown(agg, meta, ckpt_md):
    n = meta["n_seeds"]
    L = ["| Policy | Seeds | Train episodes | Eval episodes / seed | Mean eval score (mean of seed means) | Std across seeds | Min..Max seed mean | Max single-game score |",
         "|---|---|---|---|---|---|---|---|"]
    for name in ("random", "dqn", "double"):
        a = agg[name]
        te = "n/a" if name == "random" else str(meta["episodes"])
        L.append(f"| {plots.LABELS[name]} | {n} | {te} | {a['n_eval_episodes_per_seed']} | "
                 f"{a['mean_of_seed_means']:.2f} | {a['std_across_seeds']:.2f} | "
                 f"{a['min_seed_mean']:.2f}..{a['max_seed_mean']:.2f} | {a['pooled_max_score']} |")
    per = ["", "Per-seed mean scores (each over %d eval games):" % meta["n_eval"], "",
           "| Policy | " + " | ".join(f"seed {s}" for s in meta["seeds"]) + " |",
           "|---|" + "---|" * len(meta["seeds"])]
    for name in ("random", "dqn", "double"):
        per.append(f"| {plots.LABELS[name]} | " + " | ".join(f"{x:.2f}" for x in agg[name]["seed_means"]) + " |")
    return "\n".join(L + per + ["", "Checkpoint comparison (mean ± std across seeds of per-checkpoint mean eval score):", "", ckpt_md])


def make_plots(out, seeds, agg_per_seed, ckpt_table):
    pd = os.path.join(out, "plots")
    os.makedirs(pd, exist_ok=True)
    by_v = {}
    for v in VARIANTS:
        paths = [os.path.join(out, "runs", f"{v}_seed{s}", "training_log.csv") for s in seeds]
        by_v[v] = paths
        plots.plot_training_curves(paths[0], f"{plots.LABELS[v]} training curves (seed {seeds[0]})",
                                   os.path.join(pd, f"training_curves_{v}_seed{seeds[0]}.png"))
    plots.plot_variant_comparison(by_v, os.path.join(pd, "training_dqn_vs_double.png"))
    plots.plot_eval_bars(agg_per_seed, os.path.join(pd, "eval_scores.png"))
    plots.plot_checkpoint_compare(ckpt_table, float(np.mean(agg_per_seed["random"])),
                                  os.path.join(pd, "checkpoint_compare.png"))


def update_readme(readme, results_md_text, meta):
    with open(readme, encoding="utf-8") as f:
        txt = f.read()
    block = (f"<!-- RESULTS_START -->\n{results_md_text}\n\n"
             f"_Generated by `python run_all.py` from `results/summary.json`; "
             f"{meta['n_seeds']} seeds, {meta['episodes']} training episodes, "
             f"{meta['n_eval']} eval episodes per seed._\n<!-- RESULTS_END -->")
    new = re.sub(r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->", lambda m: block, txt, flags=re.S)
    with open(readme, "w", encoding="utf-8") as f:
        f.write(new)


def run_all(out="results", seeds=(0, 1, 2), episodes=500, n_eval=100, workers=2,
            ckpt_every=50, readme=None, skip_train=False):
    os.makedirs(out, exist_ok=True)
    hp = HParams(episodes=episodes, ckpt_every=ckpt_every)
    timing = {}
    if not skip_train:
        timing = train_all(out, seeds, hp, workers)
    agg, per_seed = evaluate_all(out, seeds, n_eval)
    ck_table, ck_rows = compare_checkpoints(out, seeds, n_eval)
    meta = {"seeds": list(seeds), "n_seeds": len(seeds), "episodes": episodes, "n_eval": n_eval,
            "eval_seed_base": EVAL_SEED_BASE, "hyperparameters": hp.to_dict(), "timing": timing,
            "grid": [12, 12]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"meta": meta, "aggregate": agg}, f, indent=2)
    md = results_markdown(agg, meta, checkpoint_markdown(ck_rows))
    with open(os.path.join(out, "results_table.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    make_plots(out, list(seeds), per_seed, ck_table)
    if readme:
        update_readme(readme, md, meta)
    return agg, meta
