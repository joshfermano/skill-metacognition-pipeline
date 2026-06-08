"""Real open-model run with on-disk caching, writing to outputs/real/.

Evaluates a small task slice on real open-weight models via an OpenAI-compatible
gateway (OpenRouter by default, Groq also supported) and produces a real model x
task grid, a real-vs-simulation comparison, and metacognition traces. Every call
is cached so a committed cache replays with no key:

    python -m src.run_real                 # real run, records the cache
    python -m src.run_real --use-cache     # replay the committed cache, no key

The API key is read only from the environment or .env, never cached.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import pandas as pd

from .skills import build_skill_library
from .tasks import natural_deduction_tasks, broad_tasks
from .code_tasks import code_tasks
from .models import SimulationBackend, CachedBackend, get_backend
from . import pipeline as P
from . import plots
from . import metacog as MC

ROOT = Path(__file__).resolve().parent.parent
OUTR = ROOT / "outputs" / "real"
TABR = OUTR / "tables"
FIGR = OUTR / "figures"
CACHE = OUTR / "raw_cache.json"

# Open-weight models on OpenRouter, ordered roughly weak -> strong. Fast,
# non-queuing models so the run finishes in a few minutes; the ':free' variants
# queue heavily and can stall, so add them via --models with a higher TIMEOUT.
DEFAULT_MODELS = [
    "google/gemma-3-12b-it",
    "deepseek/deepseek-v4-flash",
]
DEFAULT_PRIMARY = "deepseek/deepseek-v4-flash"   # fast + capable, for traces/comparison


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE lines from a git-ignored .env, without overriding the shell.

    A real shell export always wins, since variables already set are left alone.
    """
    import os
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def short_name(slug: str) -> str:
    """Compact display/column name for a provider slug."""
    return slug.split(":")[0].split("/")[-1]


def build_slice():
    """Small mixed slice: deterministic-gold code and logic, plus a couple broad."""
    code = code_tasks()
    nd = natural_deduction_tasks(k=2, per_pair=1)[:2]
    broad = sorted(broad_tasks(18), key=lambda t: t.difficulty)
    broad = [broad[4], broad[12]]                    # easy-ish / hard-ish
    return code, nd, broad


def preflight(backend, models, log):
    """Drop models that do not respond, so one bad slug does not stall the grid."""
    working = []
    for m in models:
        try:
            backend.inner._chat(m, "Reply with the single word: ok")
            working.append(m)
            log(f"[real] model OK: {m}")
        except Exception as e:                       # noqa: BLE001
            log(f"[real] model unavailable, skipped: {m} ({str(e)[:90]})")
    return working


def build_grid(backend, tasks, model_slugs, n_trials, log, save_cb=None):
    """Model x task pass-rate grid; a failing call becomes NaN, not a crash.

    Persists the partial CSV via save_cb after each model, so an interrupted run
    still leaves usable data on disk.
    """
    cols = {}
    for i, slug in enumerate(model_slugs, 1):
        col = {}
        for t in tasks:
            try:
                col[t.label] = P.pass_rate(backend, slug, t, n_trials)
            except Exception as e:                   # noqa: BLE001
                col[t.label] = math.nan
                log(f"[real]   {short_name(slug)} / {t.label}: failed ({str(e)[:60]})")
        cols[short_name(slug)] = col
        if save_cb:
            save_cb(pd.DataFrame(cols))               # incremental persist
        log(f"[real] grid: {short_name(slug)} done ({i}/{len(model_slugs)} models)")
    grid = pd.DataFrame(cols)
    # Highest mean pass rate (easiest) at the top.
    grid = grid.loc[grid.mean(axis=1, skipna=True).sort_values(ascending=False).index]
    return grid


def main():
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "groq"])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated provider slugs.")
    ap.add_argument("--primary", default=DEFAULT_PRIMARY,
                    help="model used for real-vs-sim and the metacognition traces.")
    ap.add_argument("--sim-model", default="qwen2.5-7b")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--use-cache", action="store_true",
                    help="replay the committed cache only; make no API calls.")
    args = ap.parse_args()

    for d in (TABR, FIGR):
        d.mkdir(parents=True, exist_ok=True)
    log = print
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.use_cache:
        backend = CachedBackend.load(CACHE, inner=None)
        log(f"[real] replay-only from {CACHE.name} ({len(backend.cache)} entries)")
    else:
        inner = get_backend(args.backend, max_tokens=args.max_tokens)
        existing = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        backend = CachedBackend(inner=inner, cache=existing)
        log(f"[real] {args.backend} backend, {len(models)} models "
            f"(resuming {len(existing)} cached calls)")

    sim = SimulationBackend()
    lib = build_skill_library()
    code, nd, broad = build_slice()
    # Only self-contained tasks: code embeds its source, ND carries its premises.
    # The broad tasks are vague difficulty probes, so they are left out here.
    slice_tasks = code + nd

    summary = {"backend": args.backend, "sim_model": args.sim_model,
               "trials": args.trials, "n_slice_tasks": len(slice_tasks)}
    try:
        working = models if args.use_cache else preflight(backend, models, log)
        if not working:
            raise RuntimeError("no working models; check keys/slugs in .env")
        summary["models"] = [short_name(m) for m in working]

        # --- real model x task grid (persists incrementally) ---------------
        def _save_partial(df):
            df.to_csv(TABR / "model_task_real.csv")
            if not args.use_cache:
                backend.save(CACHE)
        grid = build_grid(backend, slice_tasks, working, args.trials, log,
                          save_cb=_save_partial)
        grid.to_csv(TABR / "model_task_real.csv")
        f_grid = plots.plot_model_task_grid(
            grid, "Real open models on a task slice (easy → hard)",
            "fig_real_grid.png", out_dir=FIGR)
        log(f"[real] grid -> {f_grid}")

        # --- pick the primary model (for comparison + traces) --------------
        primary = args.primary if short_name(args.primary) in grid.columns else working[0]
        primary_col = short_name(primary)
        summary["primary"] = primary_col

        # --- real vs simulation (reuses the grid's cached solves) ----------
        sim_col = {t.label: P.pass_rate(sim, args.sim_model, t, args.trials)
                   for t in slice_tasks}
        fam = {t.label: t.family for t in slice_tasks}
        rvs = pd.DataFrame({"real": grid[primary_col],
                            "sim": pd.Series(sim_col),
                            "family": pd.Series(fam)}).dropna(subset=["real"])
        rvs.to_csv(TABR / "real_vs_sim.csv")
        mae = float((rvs["real"].astype(float) - rvs["sim"].astype(float)).abs().mean())
        plots.plot_real_vs_sim(rvs, primary_col, args.sim_model,
                               "fig7_real_vs_sim.png", out_dir=FIGR)
        summary["real_vs_sim_mae_pp"] = round(mae * 100, 2)
        log(f"[real] real-vs-sim ({primary_col}) mean|delta| = {mae*100:.1f} pp")

        # --- real metacognition traces (best effort) -----------------------
        try:
            mc_tasks = code[:3] + nd[:1]
            mc_summary, mc_traces = P.metacog_eval(backend, mc_tasks, primary, lib)
            mc_summary.to_csv(TABR / "metacog.csv")
            by_uid = {t.uid: t for t in mc_tasks}
            shown = {t.label for t in mc_tasks[:3]}
            examples = []
            for tr in mc_traces:
                if tr.label not in shown:
                    continue
                if tr.condition == "post_training":
                    MC.interrogate(backend, primary, by_uid[tr.task_uid], tr, 0)
                examples.append(dataclasses.asdict(tr))
            (OUTR / "metacog_traces.json").write_text(json.dumps(examples, indent=2))
            summary["metacog_post"] = mc_summary.loc["post_training"].round(3).to_dict()
            summary["metacog_pre"] = mc_summary.loc["pre_training"].round(3).to_dict()
            log(f"[real] metacog post selection_accuracy "
                f"{mc_summary.loc['post_training', 'selection_accuracy']:.2f}")
        except Exception as e:                       # noqa: BLE001
            log(f"[real] metacog step skipped: {str(e)[:120]}")
    finally:
        if not args.use_cache:
            backend.save(CACHE)
        summary["cached_calls"] = len(backend.cache)

    (OUTR / "summary.json").write_text(json.dumps(summary, indent=2))
    log("[real summary] " + json.dumps(summary))


if __name__ == "__main__":
    main()
