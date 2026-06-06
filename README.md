# Skill-Metacognition Pipeline — LLM skill extraction, evaluation, and self-improvement

A small, reproducible pipeline that **extracts skills, generates skill-targeted
tasks, runs chain-of-thought, detects which skills a solution used, sorts tasks
easy → hard by pass rate, and self-improves a model via best-of-_n_ rejection
sampling** — then visualizes the results as the three heatmap styles requested in
the brief, plus a before/after self-improvement chart.

It is grounded in the Princeton (Arora lab) line of work on LLM *skills* and
*metacognition* — **Skill-Mix**, **LLM Metacognition**, and **Instruct-SkillMix** —
and uses **only open / open-weight models**. See `TECHNICAL_REPORT.md` for the
methodology, citations, and the rationale for open models.

## Quickstart (no GPU, no API key)

```bash
pip install -r requirements.txt
python -m src.run_all                 # default: deterministic simulation backend
python -m tests.test_pipeline         # smoke tests
```

Outputs land in `outputs/`:

```
outputs/
├── summary.json
├── metacog_traces.json                    (enumerate → choose → name → justify traces)
├── tables/   skills.csv  tasks.csv  model_task_pass.csv  code_tasks.csv
│             skill_pair_failure.csv  baseline.csv  uplift.csv  self_improvement.csv  metacog.csv
├── figures/  fig1_model_task_grid.png      (SkillsBench-style model × task, easy→hard)
│             fig2_skill_pair_failure.png   (natural-deduction skill-pair co-failure)
│             fig3_baseline_uplift.png      (baseline vs curated-skill uplift, 4 small models)
│             fig4_self_improvement.png     (best-of-16 / keep-4 before vs after)
│             fig5_metacog.png              (metacognition scaffolding, pre vs post training)
│             fig6_code_grid.png            (code-derived PICO-8 Lua tasks)
└── real/     (written by `python -m src.run_real`; real Groq numbers + fig7_real_vs_sim.png)
```

`python -m src.run_all --no-metacog` skips figure 5 if you only want the core four-plus-code figures.

## Run against real open models

All backends are open-weight only. Choose by what you have available:

```bash
# Groq (OpenAI-compatible LPU gateway, open models, fast, free tier ~14.4k req/day on 8B)
export GROQ_API_KEY=...               # free key, no card: https://console.groq.com
python -m src.run_all --backend groq --scale small

# OpenRouter (OpenAI-compatible gateway, open models, free tier ~50-1000 req/day)
export OPENROUTER_API_KEY=...         # free key, no card: https://openrouter.ai/keys
python -m src.run_all --backend openrouter --scale small

# Ollama (local, open weights, no key, no rate limits) — best for the small-model figure
ollama serve & ollama pull qwen2.5:3b-instruct
python -m src.run_all --backend ollama

# Hugging Face transformers (local, open weights)
pip install torch transformers accelerate
python -m src.run_all --backend hf
```

`--scale small` shrinks task counts and trials to fit free rate limits. Use
`--models a,b,c` to override the roster (raw provider slugs allowed, e.g.
`--models llama-3.1-8b-instant,qwen/qwen3-32b`). Gateway backends pace requests
to their RPM cap and back off on HTTP 429 automatically.

**Backend guidance:** Groq is the best free gateway for call volume (the 8B
workhorse allows ~14,400 req/day); use it for the model×task grid, skill-pair
grid, and self-improvement. Groq has no sub-8B models, so the small-model uplift
figure (qwen3-0.6b … qwen2.5-3b) is best produced on Ollama locally or left on
the simulation backend. The simulation backend remains the always-reproducible
default that needs no key and no GPU.

The backend is the only thing that changes; the extraction, generation,
detection, sorting, grading, self-improvement and plotting code is identical. The
simulation backend exists so the methodology and figures reproduce on any
machine; the real backends run the same pipeline on live open models.

## Real run with caching, and the web dashboard

For a self-contained real measurement that is committed to the repo, use the
dedicated runner. It evaluates a small mixed slice (code + logic + broad) on an
open-weight model via Groq, builds a real-versus-simulation comparison, and
caches every call so the result replays with no key:

```bash
export GROQ_API_KEY=...                # free key, no card: https://console.groq.com
python -m src.run_real                 # real numbers + traces -> outputs/real/
python -m src.run_real --use-cache     # replay the committed cache, no key needed
```

`outputs/real/` sits beside the simulation outputs (it does not overwrite them)
and holds `fig7_real_vs_sim.png`, `tables/real_vs_sim.csv`, `metacog_traces.json`
(genuine enumerate→choose→justify text), and `raw_cache.json`. The API key is
read only from the environment and never written to the cache.

A static **Next.js** dashboard over these tables and traces lives in a separate
sibling repository, `skill-metacognition-dashboard`:

```bash
cd ../skill-metacognition-dashboard && npm install && npm run build   # static export to out/
```

See that repo's `README.md`. Its prebuild step reads the CSVs and JSON from this
pipeline's `outputs/` (path overridable via `PIPELINE_OUTPUTS`) and renders the
heatmaps, the metacognition traces, and the real-versus-simulation panel.

## How it maps to the literature

| Pipeline stage (`src/`)              | Method / source |
|--------------------------------------|-----------------|
| `skills.py` extract + cluster        | Metacognition skill labelling + clustering (Didolkar et al. 2024); Instruct-SkillMix extraction (Kaur et al. 2024) |
| `tasks.py` k-skill task generation   | Skill-Mix generation (Yu et al. 2023); Instruct-SkillMix data generation |
| `grading.py` k+3 rubric & metrics    | Skill-Mix grading (Ratio of Full Marks / All Skills / Skill Fraction) |
| `grading.detect_skills`              | Metacognition skill labelling applied to a CoT trace |
| `pipeline.model_task_matrix`         | SkillsBench-style task-pass grid, sorted easy→hard |
| `pipeline.skill_pair_failure`        | Skill co-failure analysis over natural-deduction rules |
| `pipeline.baseline_uplift`           | Curated-skill uplift Δ (curated − baseline) |
| `pipeline.self_improvement`          | STaR / rejection-sampling (best-of-16, keep-4) self-improvement |

## Repository layout

```
src/uids.py        Uint64 hex IDs  (xxxx-xxxx-xxxx-xxxx)
src/skills.py      skill extraction, clustering (pure-numpy), 4-word names + slug, cognitive sort
src/tasks.py       natural-deduction / language / broad task generators
src/code_tasks.py  code-derived tasks parsed from data/code (locate/grep/trace/explain/decompose)
src/metacog.py     named-skill scaffolding: enumerate → choose → name → justify + interrogation
src/models.py      backends: simulation (IRT) | ollama | hf | openrouter | groq + caching wrapper
src/grading.py     Skill-Mix rubric + metrics, deterministic verifiers, skill detection
src/pipeline.py    matrices, easy→hard sort, co-failure, uplift, self-improvement, metacog, real-vs-sim
src/plots.py       the seven figures
src/run_all.py     simulation/real orchestrator (CLI)
src/run_real.py    Groq open-model run with on-disk cache (real numbers + traces)
data/seed_skills.json   fine-grained skill corpus + natural-deduction + code rule sets
data/code/dodge.p8.lua  small self-contained PICO-8-style Lua sample (MIT)
tests/test_pipeline.py  smoke tests (UIDs, metrics, code gold, metacog parser, cache)
```

The web dashboard lives in the separate sibling repo `../skill-metacognition-dashboard`.

## Notes on scope and honesty
- The IRT simulation backend produces **synthetic** pass/fail outcomes calibrated
  to the *magnitudes* reported in the papers; it is clearly labelled as such and
  is not a measurement of any specific model. Swap in `--backend ollama`/`hf` to
  measure real models.
- The Uint64 hex-UID and 4-word-name conventions are this project's design
  choices (consistent with the metacognition naming idea), not a verbatim
  reproduction of any paper.
- See `TECHNICAL_REPORT.md` §7 for provenance caveats (e.g. SkillsBench is a
  separate multi-institution benchmark, and the natural-deduction skill set is a
  standard logic rule set used here illustratively).
