"""
Evaluation engine.

Builds the analyses behind the three figures plus the self-improvement result:

  1. model x task pass-rate matrix, rows sorted easy -> hard  (SkillsBench-style)
  2. skill-pair co-failure matrix for a fixed model            (natural deduction)
  3. baseline pass rate vs skill uplift (curated - baseline)   (small models)
  4. STaR-style best-of-16 / keep-4 self-improvement, before vs after

All randomness is seeded, so results reproduce exactly with the simulation
backend; with a real backend the same code paths run live models.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from .models import MODEL_ROSTER, UPLIFT_MODELS, get_backend
from .skills import build_skill_library


# --------------------------------------------------------------------------- #
# Core: pass rate of a model on a task over n trials
# --------------------------------------------------------------------------- #
def pass_rate(backend, model, task, n_trials=5, uplift=0.0):
    passes = sum(backend.solve(model, task, t, uplift).passed for t in range(n_trials))
    return passes / n_trials


# --------------------------------------------------------------------------- #
# 1. Model x task grid, sorted easy -> hard
# --------------------------------------------------------------------------- #
def model_task_matrix(backend, tasks, models, n_trials=5) -> pd.DataFrame:
    rows = {}
    for task in tasks:
        rows[task.label] = {m: pass_rate(backend, m, task, n_trials) for m in models}
    df = pd.DataFrame(rows).T[models]
    # Easy (high mean pass) at the top, hard (low) at the bottom.
    df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]
    return df


# --------------------------------------------------------------------------- #
# 2. Skill-pair co-failure for a fixed model
# --------------------------------------------------------------------------- #
def skill_pair_failure(backend, nd_tasks, model, n_trials=5):
    """For every ordered pair of ND rules, failure rate over tasks needing both.

    Returns (failure_df, count_df) indexed by the rule symbols, mirroring the
    reference 'skill-pair failure (all_fail)' heatmap.
    """
    symbols = [s.symbol for s in build_skill_library()["natural_deduction"]]
    fails = defaultdict(list)
    for task in nd_tasks:
        syms = task.meta["symbols"]
        # binary "did the model fail this task" = majority of trials failed
        pr = pass_rate(backend, model, task, n_trials, uplift=0.0)
        failed = pr < 0.5
        for x, y in itertools.product(syms, syms):
            fails[(x, y)].append(failed)
    fail_mat = pd.DataFrame(index=symbols, columns=symbols, dtype=float)
    cnt_mat = pd.DataFrame(index=symbols, columns=symbols, dtype=float)
    for x in symbols:
        for y in symbols:
            vals = fails.get((x, y), [])
            cnt_mat.loc[x, y] = len(vals)
            fail_mat.loc[x, y] = (np.mean(vals) if vals else np.nan)
    return fail_mat, cnt_mat


# --------------------------------------------------------------------------- #
# 3. Baseline vs uplift on language tasks for the 4 small models
# --------------------------------------------------------------------------- #
def baseline_uplift(backend, lang_tasks, models=UPLIFT_MODELS, n_trials=5):
    base = {}
    curated = {}
    for task in lang_tasks:
        base[task.label] = {m: pass_rate(backend, m, task, n_trials, 0.0) for m in models}
        curated[task.label] = {m: pass_rate(backend, m, task, n_trials, task.base_uplift)
                               for m in models}
    base_df = pd.DataFrame(base).T[models]
    cur_df = pd.DataFrame(curated).T[models]
    uplift_df = cur_df - base_df
    return base_df, uplift_df


# --------------------------------------------------------------------------- #
# 4. STaR-style self-improvement: best-of-16, keep 4, "train", re-evaluate
# --------------------------------------------------------------------------- #
def self_improvement(backend, tasks, model, n_sample=16, keep=4,
                     gain_per_kept=0.18, n_trials=8):
    """Rejection-sampling self-improvement (STaR / best-of-n).

    For each task we draw n_sample trajectories; the correct ones are the
    "kept" rationales (capped at `keep`). A task that yields >=1 correct sample
    contributes training signal, modelled as an ability gain on tasks sharing its
    skills. We then re-evaluate to report before/after pass rate.

    Returns a tidy DataFrame with before/after pass rates per task.
    """
    # before
    before = {t.label: pass_rate(backend, model, t, n_trials) for t in tasks}

    # collect kept rationales and per-skill training mass
    skill_mass = defaultdict(float)
    for t in tasks:
        correct = [backend.solve(model, t, s).passed for s in range(n_sample)]
        n_correct = sum(correct)
        kept = min(keep, n_correct)
        if kept > 0:
            for uid in t.skill_uids:
                skill_mass[uid] += kept / keep
    # "train": ability uplift on a task proportional to training mass on its skills
    after = {}
    for t in tasks:
        mass = np.mean([skill_mass.get(u, 0.0) for u in t.skill_uids]) if t.skill_uids else 0.0
        uplift = gain_per_kept * mass
        after[t.label] = pass_rate(backend, model, t, n_trials, uplift=uplift)

    df = pd.DataFrame({"before": before, "after": after})
    df["delta"] = df["after"] - df["before"]
    return df.sort_values("before")


# --------------------------------------------------------------------------- #
# 5. Metacognitive scaffolding: pre- vs post-training enumerate/choose/justify
# --------------------------------------------------------------------------- #
def metacog_eval(backend, tasks, model, lib,
                 conditions=("pre_training", "post_training")):
    """Run the named-skill protocol (src/metacog.py) on each task under each
    condition and aggregate the four protocol metrics.

    Returns (summary_df indexed by condition, list[ScaffoldTrace]). The traces
    carry the full enumerate/choose/justify text for the report and dashboard.
    """
    from . import metacog as MC

    metrics = ["enumeration_recall", "selection_accuracy", "name_before_use",
               "answer_correct"]
    rows, traces = [], []
    for cond in conditions:
        per = {m: [] for m in metrics}
        for t in tasks:
            tr = MC.scaffold_solve(backend, model, t, lib, cond)
            sc = MC.score_trace(tr, t)
            for m in metrics:
                per[m].append(sc[m])
            traces.append(tr)
        row = {m: float(np.mean(per[m])) for m in metrics}
        row["condition"] = cond
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("condition")[metrics]
    return summary, traces


# --------------------------------------------------------------------------- #
# 6. Real vs simulated pass rate on the same task slice
# --------------------------------------------------------------------------- #
def real_vs_sim(real_backend, sim_backend, tasks, real_model,
                sim_model="qwen2.5-7b", n_trials=3):
    """Run the same tasks on a real open model and on the simulation backend.

    Answers, honestly, whether the IRT calibration tracks real small-open-model
    behaviour on the overlapping slice. Returns a DataFrame indexed by task with
    columns real, sim, family.
    """
    rows = {}
    for t in tasks:
        rows[t.label] = {
            "real": pass_rate(real_backend, real_model, t, n_trials),
            "sim": pass_rate(sim_backend, sim_model, t, n_trials),
            "family": t.family,
        }
    df = pd.DataFrame(rows).T
    df["real"] = df["real"].astype(float)
    df["sim"] = df["sim"].astype(float)
    return df
