"""
Model backends.

Every model in this project is open-source / open-weight. We never call a paid
API. Three interchangeable backends implement one interface:

  * SimulationBackend  - deterministic, dependency-light, runs anywhere (CI, a
                         laptop, a reviewer's machine with no GPU). Pass/fail is
                         drawn from an Item-Response-Theory (IRT) model whose
                         parameters are calibrated to the magnitudes reported in
                         the literature (see TECHNICAL_REPORT.md S5). This is the
                         default so the whole pipeline + figures reproduce with
                         `pip install -r requirements.txt && python -m src.run_all`.

  * OllamaBackend      - talks to a local Ollama server (http://localhost:11434).
                         Pull e.g. `ollama pull qwen2.5:3b-instruct`. No API key.

  * HFBackend          - loads an open model with transformers locally.

The interface is intentionally tiny: `solve(task) -> Trajectory` (a chain of
thought + final answer) and `label_skill(text, skills) -> name`. Swapping the
backend changes nothing else in the pipeline.

WHY OPEN MODELS (transparency note, also in the report): (1) no paid API key is
required to grade or reproduce the submission; (2) open weights make the
evaluation fully auditable and deterministic given a seed; (3) the small open
models used here (0.6B-7B) are exactly the regime the metacognition/skill papers
target for cheap, scalable skill labelling and self-improvement; (4) the IRT
simulation backend lets the methodology be demonstrated and the figures
regenerated even on hardware without a GPU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# Open-weight model roster. `ability` is the IRT latent ability theta used by the
# simulation backend (logit scale). Values are an ordering calibrated so that the
# capability ranking and the easy->hard gradient match the published figures;
# they are NOT measured scores and are clearly labelled as simulation parameters.
MODEL_ROSTER: dict[str, dict] = {
    "qwen3-0.6b":      {"ability": -0.9, "params_b": 0.6, "hf": "Qwen/Qwen3-0.6B"},
    "llama3.2-1b":     {"ability": -0.6, "params_b": 1.0, "hf": "meta-llama/Llama-3.2-1B-Instruct"},
    "qwen2.5-1.5b":    {"ability": -0.2, "params_b": 1.5, "hf": "Qwen/Qwen2.5-1.5B-Instruct"},
    "gemma2-2b":       {"ability":  0.1, "params_b": 2.0, "hf": "google/gemma-2-2b-it"},
    "qwen2.5-3b":      {"ability":  0.5, "params_b": 3.0, "hf": "Qwen/Qwen2.5-3B-Instruct"},
    "phi3.5-mini":     {"ability":  0.8, "params_b": 3.8, "hf": "microsoft/Phi-3.5-mini-instruct"},
    "qwen2.5-7b":      {"ability":  1.3, "params_b": 7.0, "hf": "Qwen/Qwen2.5-7B-Instruct"},
}

# The four small models used for the baseline-vs-uplift figure (matches the brief's
# reference image exactly).
UPLIFT_MODELS = ["llama3.2-1b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen3-0.6b"]


@dataclass
class Trajectory:
    task_uid: str
    model: str
    cot: str            # chain-of-thought text
    answer: str         # final answer
    passed: bool        # did it satisfy the verifier / rubric?
    score: float        # graded score in [0, 1]
    used_skills: list[str]   # skill UIDs detected in the CoT


def _hash01(*parts: str) -> float:
    """Deterministic pseudo-random float in [0,1) from string parts."""
    h = hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class SimulationBackend:
    """Deterministic IRT pass-probability model.

    P(pass) = sigmoid( a * (theta_model - b_task + uplift) )

      theta_model : latent model ability (MODEL_ROSTER)
      b_task      : task difficulty (set by tasks.py)
      uplift      : skill effect when the curated skill is supplied (>=0 helps,
                    <0 hurts: reproduces the SkillsBench finding that poor or
                    self-generated skills can reduce performance)
      a           : discrimination (slope)

    Trials are independent Bernoulli draws seeded by (model, task, trial) so the
    same configuration always yields the same outcomes.
    """

    def __init__(self, discrimination: float = 1.3, seed: int = 0):
        self.a = discrimination
        self.seed = seed

    def pass_prob(self, model: str, difficulty: float, uplift: float = 0.0) -> float:
        theta = MODEL_ROSTER[model]["ability"]
        return float(sigmoid(self.a * (theta - difficulty + uplift)))

    def trial(self, model: str, task_uid: str, difficulty: float, trial_idx: int,
              uplift: float = 0.0) -> bool:
        p = self.pass_prob(model, difficulty, uplift)
        u = _hash01(str(self.seed), model, task_uid, str(trial_idx))
        return u < p

    def solve(self, model: str, task, trial_idx: int = 0, uplift: float = 0.0) -> Trajectory:
        passed = self.trial(model, task.uid, task.difficulty, trial_idx, uplift)
        # A schematic CoT that names the required skills (so skill-detection has
        # signal). Real backends produce genuine reasoning text here.
        steps = [f"Step {i+1}: apply '{name}'." for i, name in enumerate(task.skill_names)]
        cot = " ".join(steps) + (" Therefore the answer follows." if passed
                                 else " ... the derivation stalls here.")
        answer = task.gold if passed else "(incorrect / incomplete)"
        score = 1.0 if passed else round(_hash01(model, task.uid, "partial") * 0.5, 3)
        used = task.skill_uids if passed else task.skill_uids[: max(0, len(task.skill_uids) - 1)]
        return Trajectory(task.uid, model, cot, answer, passed, score, used)


class OllamaBackend:
    """Local Ollama server. Requires `ollama serve` and a pulled model."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host

    def _chat(self, model: str, prompt: str) -> str:
        import json
        import urllib.request
        tag = {"qwen2.5-3b": "qwen2.5:3b-instruct",
               "qwen2.5-1.5b": "qwen2.5:1.5b-instruct",
               "llama3.2-1b": "llama3.2:1b",
               "qwen3-0.6b": "qwen3:0.6b"}.get(model, model)
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps({"model": tag, "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["response"]

    def solve(self, model: str, task, trial_idx: int = 0, uplift: float = 0.0) -> Trajectory:
        from .grading import grade
        skill_hint = ""
        if uplift > 0:  # curated-skill condition: inject the SKILL.md-style guidance
            skill_hint = "\n\nRelevant skills:\n" + "\n".join(f"- {n}" for n in task.skill_names)
        prompt = (f"Solve the problem. Think step by step, then give the final "
                  f"answer after 'ANSWER:'.{skill_hint}\n\nProblem: {task.prompt}")
        text = self._chat(model, prompt)
        cot, sep, ans = text.partition("ANSWER:")
        # Reasoning models do not always emit the literal ANSWER: marker; when the
        # marker is absent, grade against the whole response so a correct answer in
        # the body is not scored as empty.
        answer = ans.strip() if sep else text.strip()
        result = grade(task, answer, cot if sep else text)
        return Trajectory(task.uid, model, cot.strip(), answer,
                          result.passed, result.score, result.used_skills)


class HFBackend:
    """Local transformers backend (open weights). Lazy-imports torch/transformers."""

    def __init__(self):
        self._pipes: dict = {}

    def _pipe(self, model: str):
        if model not in self._pipes:
            from transformers import pipeline
            self._pipes[model] = pipeline(
                "text-generation", model=MODEL_ROSTER[model]["hf"],
                device_map="auto", torch_dtype="auto",
            )
        return self._pipes[model]

    def solve(self, model: str, task, trial_idx: int = 0, uplift: float = 0.0) -> Trajectory:
        from .grading import grade
        pipe = self._pipe(model)
        skill_hint = ("\nRelevant skills:\n" + "\n".join(f"- {n}" for n in task.skill_names)
                      if uplift > 0 else "")
        msg = [{"role": "user", "content":
                f"Think step by step, then answer after 'ANSWER:'.{skill_hint}\n\n{task.prompt}"}]
        out = pipe(msg, max_new_tokens=512)[0]["generated_text"][-1]["content"]
        cot, _, answer = out.partition("ANSWER:")
        result = grade(task, answer.strip(), cot)
        return Trajectory(task.uid, model, cot.strip(), answer.strip(),
                          result.passed, result.score, result.used_skills)


# ---- OpenAI-compatible gateways (open-weight models) ----------------------- #
# OpenRouter and Groq both expose an OpenAI-compatible /chat/completions endpoint
# and serve open-WEIGHT models. They share one base class. Slugs/limits change;
# verify at https://openrouter.ai/models and https://console.groq.com/docs/models.

OPENROUTER_SLUGS: dict[str, str] = {
    "qwen3-0.6b":   "qwen/qwen3-0.6b",
    "llama3.2-1b":  "meta-llama/llama-3.2-1b-instruct",
    "qwen2.5-1.5b": "qwen/qwen-2.5-1.5b-instruct",
    "qwen2.5-3b":   "qwen/qwen-2.5-3b-instruct",
    "qwen2.5-7b":   "qwen/qwen-2.5-7b-instruct",
    "gemma2-2b":    "google/gemma-2-2b-it",
    "llama3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek-r1":  "deepseek/deepseek-r1-distill:free",
    "gpt-oss-20b":  "openai/gpt-oss-20b:free",
}

# Groq has no sub-8B models; map the small roster names onto the smallest Groq
# option and provide capability-ordered open models for the grid. Free-tier RPD:
# llama-3.1-8b-instant ~14,400; qwen3-32b / gpt-oss ~1,000; plan runs accordingly.
GROQ_SLUGS: dict[str, str] = {
    "llama3.1-8b":  "llama-3.1-8b-instant",      # workhorse, highest RPD
    "qwen3-32b":    "qwen/qwen3-32b",
    "llama3.3-70b": "llama-3.3-70b-versatile",
    "gpt-oss-20b":  "openai/gpt-oss-20b",        # open-weight MoE, fast, high RPD
    "gpt-oss-120b": "openai/gpt-oss-120b",       # open-weight flagship MoE
    # roster aliases (small models fall back to the 8B workhorse on Groq):
    "qwen3-0.6b": "llama-3.1-8b-instant", "llama3.2-1b": "llama-3.1-8b-instant",
    "qwen2.5-1.5b": "llama-3.1-8b-instant", "qwen2.5-3b": "llama-3.1-8b-instant",
    "qwen2.5-7b": "llama-3.1-8b-instant",
}


class _OpenAICompatBackend:
    """Shared base for OpenAI-compatible gateways (OpenRouter, Groq).

    Subclasses set HOST, ENV_KEY, SLUGS, and RPM. Handles slug mapping, optional
    request pacing (to respect requests-per-minute caps) and 429 backoff.
    """

    HOST = ""
    ENV_KEY = ""
    SLUGS: dict[str, str] = {}
    RPM = 0  # 0 = no client-side pacing
    TIMEOUT = 90  # per-request seconds; keep low so a queued free model fails fast

    def __init__(self, max_tokens: int = 700, temperature: float = 0.2):
        import os
        self.max_tokens = max_tokens
        # Low default temperature so graded pass rates on the deterministic-gold
        # tasks are stable across a real run; still leaves room in the scaffolding
        # text. Override per instance if more diversity is wanted.
        self.temperature = temperature
        self.key = os.environ.get(self.ENV_KEY, "")
        self._last_call = 0.0
        if not self.key:
            raise RuntimeError(f"Set {self.ENV_KEY} (see provider docs).")

    def _slug(self, model: str) -> str:
        return model if "/" in model else self.SLUGS.get(model, model)

    def _pace(self):
        import time
        if self.RPM:
            min_gap = 60.0 / self.RPM
            wait = min_gap - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _chat(self, model: str, prompt: str) -> str:
        import json
        import time
        import urllib.error
        import urllib.request
        body = json.dumps({
            "model": self._slug(model),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens, "temperature": self.temperature,
        }).encode()
        headers = {"Authorization": f"Bearer {self.key}",
                   "Content-Type": "application/json", "X-Title": "skill-metacognition-pipeline",
                   # Both gateways sit behind Cloudflare, which 1010-blocks the default
                   # Python-urllib User-Agent; a normal UA is required to be served.
                   "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                                  "Chrome/124.0.0.0 Safari/537.36"),
                   "HTTP-Referer": "https://github.com/skill-metacognition-pipeline"}
        for attempt in range(5):
            self._pace()
            try:
                req = urllib.request.Request(f"{self.HOST}/chat/completions",
                                             data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                    data = json.loads(r.read())
                choices = data.get("choices")
                if not choices:
                    raise RuntimeError(f"no choices: {str(data.get('error', data))[:160]}")
                msg = choices[0].get("message") or {}
                # Reasoning models may return content=null with the text in
                # a separate reasoning field; fall back to it.
                return msg.get("content") or msg.get("reasoning") or ""
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    time.sleep(2 ** attempt * 3)
                    continue
                raise
        return ""

    def solve(self, model: str, task, trial_idx: int = 0, uplift: float = 0.0) -> Trajectory:
        from .grading import grade
        skill_hint = ("\n\nRelevant skills:\n" + "\n".join(f"- {n}" for n in task.skill_names)
                      if uplift > 0 else "")
        prompt = (f"Solve the problem. Think step by step, then give the final "
                  f"answer after 'ANSWER:'.{skill_hint}\n\nProblem: {task.prompt}")
        text = self._chat(model, prompt)
        cot, sep, ans = text.partition("ANSWER:")
        # Reasoning models do not always emit the literal ANSWER: marker; when the
        # marker is absent, grade against the whole response so a correct answer in
        # the body is not scored as empty.
        answer = ans.strip() if sep else text.strip()
        result = grade(task, answer, cot if sep else text)
        return Trajectory(task.uid, model, cot.strip(), answer,
                          result.passed, result.score, result.used_skills)


class OpenRouterBackend(_OpenAICompatBackend):
    """OpenRouter gateway. Free key (no card): https://openrouter.ai/keys.
    Free tier ~20 req/min, 50-1000 req/day -> prefer --scale small."""
    HOST = "https://openrouter.ai/api/v1"
    ENV_KEY = "OPENROUTER_API_KEY"
    SLUGS = OPENROUTER_SLUGS
    RPM = 20


class GroqBackend(_OpenAICompatBackend):
    """Groq LPU gateway, open models only. Free key (no card): console.groq.com.
    Free tier 30 req/min, up to ~14,400 req/day on llama-3.1-8b-instant."""
    HOST = "https://api.groq.com/openai/v1"
    ENV_KEY = "GROQ_API_KEY"
    SLUGS = GROQ_SLUGS
    RPM = 28


def get_backend(name: str = "simulation", **kw):
    return {"simulation": SimulationBackend,
            "ollama": OllamaBackend,
            "hf": HFBackend,
            "openrouter": OpenRouterBackend,
            "groq": GroqBackend}[name](**kw)


class CachedBackend:
    """Record-and-replay wrapper around a real backend.

    Every solve()/_chat() result is memoised to an on-disk JSON cache keyed by
    its inputs. With `inner` set it calls the real model on a cache miss and
    writes the result through; with `inner=None` it serves only from the cache
    and raises on a miss. This lets a single real Groq run be committed to the
    repo so the report and dashboard reproduce the real numbers and traces with
    no API key (`python -m src.run_real --use-cache`). The API key is never part
    of a cache key or value; only model outputs are stored.
    """

    def __init__(self, inner=None, cache: dict | None = None):
        self.inner = inner
        self.cache = cache or {}
        self.misses = 0

    @classmethod
    def load(cls, path, inner=None):
        import json
        from pathlib import Path
        p = Path(path)
        cache = json.loads(p.read_text()) if p.exists() else {}
        return cls(inner=inner, cache=cache)

    def save(self, path):
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.cache, indent=1))

    def _hash(self, text: str) -> str:
        return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()

    def solve(self, model: str, task, trial_idx: int = 0, uplift: float = 0.0):
        import dataclasses
        key = f"solve|{model}|{task.uid}|{trial_idx}|{round(uplift, 3)}"
        if key in self.cache:
            return Trajectory(**self.cache[key])
        if self.inner is None:
            self.misses += 1
            raise KeyError(f"cache miss in replay mode: {key}")
        traj = self.inner.solve(model, task, trial_idx, uplift)
        self.cache[key] = dataclasses.asdict(traj)
        return traj

    def _chat(self, model: str, prompt: str) -> str:
        key = f"chat|{model}|{self._hash(prompt)}"
        if key in self.cache:
            return self.cache[key]
        if self.inner is None:
            self.misses += 1
            raise KeyError(f"cache miss in replay mode: {key}")
        out = self.inner._chat(model, prompt)
        self.cache[key] = out
        return out
