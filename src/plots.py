"""
Figures. Reproduces the three reference heatmap styles from the brief, plus a
before/after self-improvement figure. All styling kept close to the references:
sequential RdYlBu for pass rates (blue = high), Reds for co-failure, diverging
RdBu_r centred at 0 for uplift deltas.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIG = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="white")
# DejaVu Sans ships with matplotlib and carries the logical-symbol glyphs
# (and, or, up-tack, negation) used by the natural-deduction skill labels;
# Arial lacks them and renders boxes.
plt.rcParams["font.family"] = "DejaVu Sans"


def plot_model_task_grid(df, title, fname, out_dir=None):
    h = max(6, 0.16 * len(df))
    fig, ax = plt.subplots(figsize=(7.5, h))
    sns.heatmap(df, cmap="RdYlBu", vmin=0, vmax=1, cbar_kws={"label": "Pass Rate"},
                linewidths=0.3, linecolor="white", ax=ax)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Model (Agent)"); ax.set_ylabel("Task")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    target = (out_dir or FIG) / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig); return target


def plot_skill_pair_failure(fail_df, cnt_df, model, k, fname):
    annot = fail_df.copy().astype(object)
    for x in fail_df.index:
        for y in fail_df.columns:
            v, n = fail_df.loc[x, y], cnt_df.loc[x, y]
            annot.loc[x, y] = ("\u2014" if (v != v) else f"{v*100:.0f}%\n(n={int(n)})")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(fail_df.astype(float) * 100, cmap="Reds", vmin=0, vmax=100,
                annot=annot.values, fmt="", annot_kws={"fontsize": 7},
                cbar_kws={"label": "Failure rate"}, linewidths=0.4,
                linecolor="white", ax=ax)
    ax.set_title(f"{model}   k={k}   skill-pair failure (all_fail)", fontsize=12, pad=12)
    ax.set_xlabel("Skill Y (required together)")
    ax.set_ylabel("Skill X (required together)")
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=150, bbox_inches="tight")
    plt.close(fig); return FIG / fname


def plot_baseline_uplift(base_df, uplift_df, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(base_df) + 2))
    sns.heatmap(base_df, cmap="YlGnBu", vmin=0, vmax=1, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7}, cbar_kws={"label": "Pass Rate"},
                linewidths=0.4, linecolor="white", ax=axes[0])
    axes[0].set_title("Baseline Pass Rate"); axes[0].set_xlabel("Model"); axes[0].set_ylabel("Task")
    lim = max(0.3, float(np.nanmax(np.abs(uplift_df.values))))
    sns.heatmap(uplift_df, cmap="RdBu", center=0, vmin=-lim, vmax=lim,
                annot=uplift_df.map(lambda v: f"{v:+.2f}"), fmt="",
                annot_kws={"fontsize": 7}, cbar_kws={"label": "Score Delta"},
                linewidths=0.4, linecolor="white", ax=axes[1])
    axes[1].set_title("Skill Uplift (curated \u2212 baseline)")
    axes[1].set_xlabel("Model"); axes[1].set_ylabel("")
    for a in axes:
        a.set_xticklabels(a.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=150, bbox_inches="tight")
    plt.close(fig); return FIG / fname


def plot_self_improvement(df, model, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["before"], width=0.4, label="before (CoT)", color="#c0c5ce")
    ax.bar(x + 0.2, df["after"], width=0.4, label="after (best-of-16, keep-4)", color="#3b7dd8")
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=80, ha="right", fontsize=5)
    ax.set_ylabel("Pass rate"); ax.set_ylim(0, 1)
    mean_gain = df["delta"].mean()
    ax.set_title(f"Self-improvement on {model}: mean +{mean_gain*100:.1f} pp", fontsize=12)
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    return FIG / fname


def plot_metacog(summary_df, model, fname):
    """Grouped bars: the four scaffolding metrics, pre- vs post-training."""
    metrics = ["enumeration_recall", "selection_accuracy", "name_before_use",
               "answer_correct"]
    labels = ["enumeration\nrecall", "selection\naccuracy", "name-before\n-use",
              "answer\ncorrect"]
    pre = [summary_df.loc["pre_training", m] for m in metrics]
    post = [summary_df.loc["post_training", m] for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x - 0.2, pre, 0.4, label="pre-training (no catalogue)", color="#c0c5ce")
    ax.bar(x + 0.2, post, 0.4, label="post-training (catalogue in context)",
           color="#3b7dd8")
    for xi, (p, q) in enumerate(zip(pre, post)):
        ax.text(xi - 0.2, p + 0.02, f"{p:.2f}", ha="center", fontsize=7)
        ax.text(xi + 0.2, q + 0.02, f"{q:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"Metacognitive scaffolding on {model}: enumerate → choose "
                 f"→ name → justify", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=1.0); fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    return FIG / fname


def plot_real_vs_sim(df, real_model, sim_model, fname, out_dir=None):
    """Paired bars: real (open model) vs simulated pass rate per task."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["real"], 0.4, label=f"real ({real_model})", color="#2e8b57")
    ax.bar(x + 0.2, df["sim"], 0.4, label=f"simulated ({sim_model})", color="#c0c5ce")
    ax.set_xticks(x); ax.set_xticklabels(df.index, rotation=80, ha="right", fontsize=6)
    ax.set_ylabel("Pass rate"); ax.set_ylim(0, 1)
    mae = float((df["real"] - df["sim"]).abs().mean())
    ax.set_title(f"Real open model vs simulation on a shared task slice "
                 f"(mean |Δ| = {mae*100:.1f} pp)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=1.0)
    fig.tight_layout()
    target = (out_dir or FIG) / fname
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, bbox_inches="tight"); plt.close(fig)
    return target
