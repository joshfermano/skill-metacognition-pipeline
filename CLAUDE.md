# CLAUDE.md — skill-metacognition-pipeline

Persistent context for Claude Code. Read this fully before acting.

## What this project is

This project (`skill-metacognition-pipeline`) is a reproducible pipeline for **LLM skill extraction, metacognition-based
task generation, evaluation, and self-improvement**, built for a technical assessment
(ML/LLM Data Specialist). It extracts named skills, generates skill-targeted tasks,
runs chain-of-thought, detects skills used, sorts tasks easy→hard by pass rate, and
self-improves a model via best-of-n rejection sampling. It outputs four figures and
seven tables. `TECHNICAL_REPORT.md` is the written deliverable; read it for full
methodology and citations.

## Grounding (do not mis-attribute)

- **Skill-Mix** (Yu et al., ICLR 2024, arXiv:2310.17567) — k-skill generation + k+3 grading.
- **LLM Metacognition** (Didolkar et al., NeurIPS 2024, arXiv:2405.12205) — skill labelling + clustering.
- **Instruct-SkillMix** (Kaur et al., ICLR 2025, arXiv:2408.14774) — skill→synthetic-data pipeline.
- **STaR** (Zelikman et al., 2022) — the best-of-n / rejection-sampling self-improvement loop.
- SkillsBench is a SEPARATE multi-institution benchmark (not Arora lab); the natural-deduction
  rule set is standard logic used illustratively. Keep these distinctions in any writing.

## Architecture (`src/`)

- `uids.py` — Uint64 hex IDs `xxxx-xxxx-xxxx-xxxx`, deterministic (BLAKE2b). Do not change the format.
- `skills.py` — extraction + clustering, 4-word names + underscore slug, cognitive-load sort. Clustering is
  a small pure-numpy average-linkage routine (`_agglomerative_average`); `sklearn.cluster` is intentionally
  NOT imported (its compiled DLL is blocked by Windows Application Control on this machine). TF-IDF +
  cosine_distances from sklearn still used. Three families: `extracted`, `natural_deduction`, `code`.
- `tasks.py` — three task sets: natural-deduction proofs, `language-XXXX`, broad mixed.
- `code_tasks.py` — code-derived tasks parsed from `data/code/dodge.p8.lua` (locate/grep/trace/explain/
  decompose), with deterministic gold checked against the source; references the fixed `code` skill family.
- `metacog.py` — named-skill scaffolding protocol: enumerate candidates → choose → name-before-use →
  justify-on-interrogation. pre_training (no catalogue) vs post_training (catalogue in context). Real
  backends (those with `_chat`) emit genuine text; simulation emits a structured scaffold.
- `models.py` — backends behind one interface: `simulation` (IRT, default), `ollama`, `hf`,
  `openrouter`, `groq`, plus `CachedBackend` (record/replay to disk; inner=None ⇒ replay-only).
  Open models only. OpenRouter/Groq share `_OpenAICompatBackend` (RPM pacing, 429 backoff, low default
  temperature); need `OPENROUTER_API_KEY` / `GROQ_API_KEY`. Groq has no sub-8B models.
- `grading.py` — Skill-Mix k+3 rubric + three metrics, deterministic verifiers (logic/math/code), skill detection.
- `pipeline.py` — model×task matrix, easy→hard sort, skill-pair co-failure, baseline-vs-uplift,
  self-improvement, `metacog_eval`, `real_vs_sim`.
- `plots.py` — figs 1-4 plus fig5 metacog (pre/post bars), fig6 code grid, fig7 real-vs-sim. Font forced to
  DejaVu Sans so logical symbols render.
- `run_all.py` — simulation/real orchestrator/CLI (`--metacog`/`--no-metacog`). `run_real.py` — Groq run
  with caching to `outputs/real/` (`--use-cache` replays, no key). `tests/test_pipeline.py` — smoke tests.

## Conventions (keep consistent)

- Open / open-weight models only; never add a paid-API dependency to the core path.
- The simulation backend produces SYNTHETIC outcomes calibrated to published magnitudes; always label
  simulated numbers as simulated. The same code paths must run real models via `--backend ollama|hf`.
- Determinism: everything is seeded; identical runs must reproduce identical figures/tables.
- Writing style (author preference): plain academic prose, no em-dashes, avoid "robust"/"rigorous",
  minimal bullet/header decoration in prose deliverables.

## Run / verify

```bash
pip install -r requirements.txt
python -m src.run_all            # simulation backend; writes outputs/{figures,tables,summary.json}
python -m tests.test_pipeline    # 9 smoke tests, all must pass
python -m src.run_real           # real Groq open-model slice -> outputs/real/ (needs GROQ_API_KEY)
python -m src.run_real --use-cache   # replays the committed cache, no key
cd ../skill-metacognition-dashboard && npm install && npm run build   # dashboard (separate repo)
```

## Current state

Complete and passing: pipeline + 7 figures (figs 1-6 simulation, fig7 real-vs-sim) + report (13 sections)
+ README + 9 tests. Three families of skills (extracted/ND/code), the metacognition scaffolding protocol,
code-derived PICO-8 tasks, a cached real Groq run, and a static Next.js dashboard in the separate
sibling repo `../skill-metacognition-dashboard`.

## Watch-outs

- `sklearn.cluster` is blocked by Windows Application Control here; clustering is pure-numpy. Do not
  re-import `sklearn.cluster`.
- Never write `GROQ_API_KEY` to any file; it is read from the environment only. `outputs/real/raw_cache.json`
  stores model outputs, never the key.
- Output schema is consumed by the sibling dashboard repo; new tables/columns are additive. Keep determinism on the
  simulation backend.
