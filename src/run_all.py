"""End-to-end runner: builds every table and figure under outputs/.

    python -m src.run_all                 # simulation backend (default)
    python -m src.run_all --backend ollama
    python -m src.run_all --backend hf
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from .skills import build_skill_library
from .tasks import natural_deduction_tasks, language_tasks, broad_tasks
from .code_tasks import code_tasks
from .models import MODEL_ROSTER, UPLIFT_MODELS, get_backend
from . import pipeline as P
from . import plots

OUT = Path(__file__).resolve().parent.parent / "outputs"
TAB = OUT / "tables"; TAB.mkdir(parents=True, exist_ok=True)


def export_skills(lib):
    rows = []
    for grp, skills in lib.items():
        for s in skills:
            rows.append({"group": grp, "skill_uid": s.uid, "name": s.name,
                         "slug": s.name.replace(" ", "_"),  # underscore slug
                         "family": s.family, "cognitive_rank": s.cognitive_rank,
                         "symbol": s.symbol or "", "n_members": len(s.members)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "skills.csv", index=False)
    return df


def export_tasks(all_tasks):
    rows = [{"task_uid": t.uid, "label": t.label, "family": t.family, "k": t.k,
             "difficulty": round(t.difficulty, 3), "base_uplift": round(t.base_uplift, 3),
             "skill_uids": "|".join(t.skill_uids), "prompt": t.prompt}
            for t in all_tasks]
    df = pd.DataFrame(rows); df.to_csv(TAB / "tasks.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="simulation",
                    choices=["simulation", "ollama", "hf", "openrouter", "groq"])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--nd-k", type=int, default=2)
    ap.add_argument("--fixed-model", default="qwen2.5-7b")
    ap.add_argument("--scale", default="full", choices=["full", "small"],
                    help="'small' shrinks tasks/trials to fit OpenRouter free limits.")
    ap.add_argument("--models", default="", help="comma-separated model override.")
    ap.add_argument("--metacog", action=argparse.BooleanOptionalAction, default=True,
                    help="run the metacognitive scaffolding eval (fig5). --no-metacog to skip.")
    args = ap.parse_args()

    # 'small' scale keeps a real run inside free rate limits.
    if args.scale == "small":
        args.trials = min(args.trials, 2)
        n_nd_per_pair, n_lang, n_broad = 1, 8, 18
        default_models = ["qwen3-0.6b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
    else:
        n_nd_per_pair, n_lang, n_broad = 6, 15, 84
        default_models = list(MODEL_ROSTER)

    # Groq has no sub-8B models; use its open-model lineup unless overridden.
    if args.backend == "groq" and not args.models:
        default_models = ["llama3.1-8b", "qwen3-32b", "llama3.3-70b", "gpt-oss-120b"]
        if args.fixed_model == "qwen2.5-7b":
            args.fixed_model = "llama3.1-8b"

    models_all = args.models.split(",") if args.models else default_models

    backend = get_backend(args.backend)
    print(f"[backend] {args.backend}  scale={args.scale}  trials={args.trials}")

    # --- skills -------------------------------------------------------------
    lib = build_skill_library()
    sk_df = export_skills(lib)
    print(f"[skills] extracted={len(lib['extracted'])} "
          f"natural_deduction={len(lib['natural_deduction'])} -> tables/skills.csv")

    # --- tasks --------------------------------------------------------------
    nd = natural_deduction_tasks(k=args.nd_k, per_pair=n_nd_per_pair)
    lang = language_tasks(n=n_lang)
    broad = broad_tasks(n=n_broad)
    code = code_tasks()
    export_tasks(nd + lang + broad + code)
    print(f"[tasks] nd={len(nd)} language={len(lang)} broad={len(broad)} "
          f"code={len(code)} -> tables/tasks.csv")

    # --- Fig 1: model x task grid, easy -> hard -----------------------------
    mt = P.model_task_matrix(backend, broad, models_all, n_trials=args.trials)
    mt.to_csv(TAB / "model_task_pass.csv")
    f1 = plots.plot_model_task_grid(
        mt, "SkillsBench-style: task pass rate per open model (easy \u2192 hard)",
        "fig1_model_task_grid.png")
    print(f"[fig1] {f1}")

    # --- Fig 2: skill-pair co-failure for a fixed model ---------------------
    fail, cnt = P.skill_pair_failure(backend, nd, args.fixed_model, n_trials=args.trials)
    fail.to_csv(TAB / "skill_pair_failure.csv")
    f2 = plots.plot_skill_pair_failure(fail, cnt, args.fixed_model, args.nd_k,
                                       "fig2_skill_pair_failure.png")
    print(f"[fig2] {f2}")

    # --- Fig 3: baseline vs uplift, small models ----------------------------
    uplift_models = [m for m in UPLIFT_MODELS if m in models_all] or models_all[:4]
    base, uplift = P.baseline_uplift(backend, lang, uplift_models, n_trials=args.trials)
    base.to_csv(TAB / "baseline.csv"); uplift.to_csv(TAB / "uplift.csv")
    f3 = plots.plot_baseline_uplift(base, uplift, "fig3_baseline_uplift.png")
    print(f"[fig3] {f3}")

    # --- Fig 4: STaR-style self-improvement (mid-difficulty band) -----------
    ordered = sorted(broad, key=lambda t: t.difficulty)
    lo = len(ordered) // 2 - len(ordered) // 4
    mid = ordered[lo: lo + min(30, len(ordered))]
    si = P.self_improvement(backend, mid, args.fixed_model)
    si.to_csv(TAB / "self_improvement.csv")
    f4 = plots.plot_self_improvement(si, args.fixed_model, "fig4_self_improvement.png")
    print(f"[fig4] {f4}  (mean delta {si['delta'].mean()*100:+.1f} pp)")

    # --- Fig 6: code-derived task grid (model x code-task) ------------------
    code_mt = P.model_task_matrix(backend, code, models_all, n_trials=args.trials)
    code_mt.to_csv(TAB / "code_tasks.csv")
    f6 = plots.plot_model_task_grid(
        code_mt, "Code-derived tasks (PICO-8 Lua): pass rate per open model",
        "fig6_code_grid.png")
    print(f"[fig6] {f6}")

    # --- Fig 5: metacognitive scaffolding (pre- vs post-training) -----------
    metacog_summary = None
    if args.metacog:
        from . import metacog as MC
        n_mc = 6 if args.scale == "small" else 12
        mc_tasks = sorted(broad, key=lambda t: t.difficulty)
        mc_tasks = mc_tasks[:: max(1, len(mc_tasks) // n_mc)][:n_mc]
        metacog_summary, mc_traces = P.metacog_eval(backend, mc_tasks, args.fixed_model, lib)
        metacog_summary.to_csv(TAB / "metacog.csv")
        f5 = plots.plot_metacog(metacog_summary, args.fixed_model, "fig5_metacog.png")
        # Serialise a few full traces for the report and dashboard.
        by_uid = {t.uid: t for t in mc_tasks}
        shown_labels = {t.label for t in mc_tasks[:3]}
        examples = []
        for tr in mc_traces:
            if tr.label not in shown_labels:
                continue
            if tr.condition == "post_training":
                MC.interrogate(backend, args.fixed_model, by_uid[tr.task_uid], tr, 0)
            examples.append(dataclasses.asdict(tr))
        (OUT / "metacog_traces.json").write_text(json.dumps(examples, indent=2))
        print(f"[fig5] {f5}  (post selection_accuracy "
              f"{metacog_summary.loc['post_training', 'selection_accuracy']:.2f})")

    # --- summary ------------------------------------------------------------
    summary = {
        "backend": args.backend,
        "n_extracted_skills": len(lib["extracted"]),
        "n_nd_skills": len(lib["natural_deduction"]),
        "n_tasks": len(nd) + len(lang) + len(broad) + len(code),
        "fixed_model": args.fixed_model,
        "skillmix_uplift_mean_pp": round(uplift.values.mean() * 100, 2),
        "self_improvement_mean_pp": round(si["delta"].mean() * 100, 2),
    }
    if metacog_summary is not None:
        summary["metacog_pre"] = metacog_summary.loc["pre_training"].round(3).to_dict()
        summary["metacog_post"] = metacog_summary.loc["post_training"].round(3).to_dict()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[summary]", json.dumps(summary))


if __name__ == "__main__":
    main()
