"""Plots: training curves, variant comparison, evaluation bars, checkpoint comparison."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"dqn": "#1f77b4", "double": "#d95f02", "random": "#7f7f7f"}
LABELS = {"dqn": "DQN", "double": "Double DQN", "random": "Random"}


def moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - w + 1): i + 1].mean()
    return out


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ep = np.array([int(r["episode"]) for r in rows])
    score = np.array([float(r["score"]) for r in rows])
    eps = np.array([float(r["epsilon"]) for r in rows])
    loss = np.array([float(r["loss"]) if r["loss"] != "" else np.nan for r in rows])
    return ep, score, loss, eps


def plot_training_curves(log_path, title, out_path, window=50):
    """Score + moving average, loss, epsilon for ONE run."""
    ep, score, loss, eps = read_log(log_path)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, score, alpha=0.3, label="score")
    ax[0].plot(ep, moving_average(score, window), lw=2, label=f"moving avg ({window})")
    ax[0].set(title="Score per episode", xlabel="episode", ylabel="score (food eaten)")
    ax[0].legend()
    ax[1].plot(ep, loss)
    ax[1].set(title="Mean TD loss per episode (Huber)", xlabel="episode", ylabel="loss")
    ax[2].plot(ep, eps)
    ax[2].set(title="Epsilon (exploration rate)", xlabel="episode", ylabel="epsilon")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_variant_comparison(logs_by_variant, out_path, window=50):
    """logs_by_variant: {variant: [log_path, ...]} -> mean moving-average score +/- seed std."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for v, paths in logs_by_variant.items():
        scs, losses = [], []
        for p in paths:
            ep, score, loss, _ = read_log(p)
            scs.append(moving_average(score, window))
            losses.append(moving_average(np.nan_to_num(loss, nan=0.0), window))
        for arrs, a in ((scs, ax[0]), (losses, ax[1])):
            n = min(len(x) for x in arrs)
            m = np.stack([x[:n] for x in arrs])
            a.plot(ep[:n], m.mean(0), color=COLORS[v], label=f"{LABELS[v]} (n={len(arrs)} seeds)")
            a.fill_between(ep[:n], m.min(0), m.max(0), color=COLORS[v], alpha=0.2)
    ax[0].set(title="Training score, moving avg (band = seed min..max)", xlabel="episode", ylabel="score")
    ax[1].set(title="Training loss, moving avg", xlabel="episode", ylabel="Huber loss")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_eval_bars(per_seed, out_path):
    """per_seed: {name: [seed mean scores]} (name in dqn/double/random)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(per_seed)
    for i, n in enumerate(names):
        vals = np.array(per_seed[n])
        ax.bar(i, vals.mean(), color=COLORS[n], alpha=0.7,
               yerr=vals.std(ddof=1) if len(vals) > 1 else 0, capsize=5)
        ax.scatter([i] * len(vals), vals, color="k", zorder=3, s=18)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([LABELS[n] for n in names])
    ax.set(ylabel="mean eval score", title="Eval score (bar = mean of seed means, whiskers = std across seeds)")
    ax.title.set_fontsize(9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_checkpoint_compare(table, random_mean, out_path):
    """table: {variant: {episode: [seed means]}}."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, d in table.items():
        eps = sorted(d)
        m = np.array([np.mean(d[e]) for e in eps])
        s = np.array([np.std(d[e], ddof=1) if len(d[e]) > 1 else 0 for e in eps])
        ax.errorbar(eps, m, yerr=s, color=COLORS[v], marker="o", capsize=3, label=LABELS[v])
    ax.axhline(random_mean, color=COLORS["random"], ls="--", label="Random baseline")
    ax.set(xlabel="checkpoint (training episode)", ylabel="mean eval score",
           title="Checkpoint comparison (mean +/- std across seeds)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
