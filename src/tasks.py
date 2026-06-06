"""
Task generation.

Following Skill-Mix (Yu et al., 2023) and Instruct-SkillMix (Kaur et al., 2024),
tasks are built by sampling k skills and asking for an artefact that exercises
all of them. Each task carries:

  uid           : Uint64 hex (uids.task_uid)
  prompt        : the instruction shown to the model
  family        : domain tag (logic | math | code | language | safety | mixed)
  skill_uids    : required skills (the "required together" set)
  skill_names   : their canonical names
  difficulty    : IRT difficulty b (logit scale); rises with k and skill load
  gold          : reference answer (for deterministic verification where possible)
  base_uplift   : how much a *curated* skill helps this task (logit units). Most
                  are positive; a minority are <=0 to reproduce the SkillsBench
                  observation that some skills do not help (or hurt).

Three task sets are produced:
  * natural-deduction proof tasks  -> skill-pair co-failure heatmap
  * language-XXXX tasks            -> baseline vs uplift heatmap (4 small models)
  * broad mixed tasks              -> SkillsBench-style model x task grid
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .skills import Skill, build_skill_library
from .uids import stable_uint64, task_uid


@dataclass
class Task:
    uid: str
    label: str
    prompt: str
    family: str
    skill_uids: list[str]
    skill_names: list[str]
    difficulty: float
    gold: str = ""
    base_uplift: float = 0.0
    k: int = 1
    meta: dict = field(default_factory=dict)


def _rng(*parts: str) -> np.random.Generator:
    return np.random.default_rng(stable_uint64("|".join(parts)) % (2 ** 32))


def _difficulty_from(skills: list[Skill], k: int, rng: np.random.Generator) -> float:
    """Higher cognitive load and more required skills -> harder task."""
    load = np.mean([s.cognitive_rank for s in skills]) if skills else 3.0
    base = 0.45 * (load - 3.0) + 0.55 * (k - 1)
    return float(base + rng.normal(0, 0.35))


# ----------------------------------------------------------------------------- #
# Natural-deduction proof tasks (skill-pair co-failure heatmap)
# ----------------------------------------------------------------------------- #
def natural_deduction_tasks(k: int = 2, per_pair: int = 6) -> list[Task]:
    """Generate propositional-logic proof tasks, each requiring k ND rules.

    `per_pair` controls how many tasks are generated per unordered skill pair, so
    the co-failure matrix has a sensible support count (n=...) per cell.
    """
    nd = build_skill_library()["natural_deduction"]
    by_symbol = {s.symbol: s for s in nd}
    symbols = list(by_symbol)
    tasks: list[Task] = []
    for combo in itertools.combinations(symbols, k):
        skills = [by_symbol[s] for s in combo]
        for j in range(per_pair):
            rng = _rng("nd", *combo, str(j))
            premises = rng.choice(
                ["P, P->Q", "P&Q", "P, Q", "P|Q, ~P", "P->Q, Q->R, P", "~~P"],
                size=1)[0]
            goal = rng.choice(["Q", "R", "P&Q", "Q|R", "P", "~P"], size=1)[0]
            prompt = (f"Using natural deduction, prove {goal} from premises "
                      f"{premises}. Cite the inference rule at each line "
                      f"(rules to use: {', '.join(combo)}).")
            uid = task_uid(prompt + f"#{j}")
            diff = _difficulty_from(skills, k, rng) + 0.4  # logic runs harder
            tasks.append(Task(
                uid=uid, label=f"nd-{'-'.join(combo)}-{j}", prompt=prompt,
                family="logic", skill_uids=[s.uid for s in skills],
                skill_names=[s.symbol for s in skills], difficulty=diff,
                gold=f"QED ({goal})", base_uplift=0.6, k=k,
                meta={"symbols": list(combo)},
            ))
    return tasks


# ----------------------------------------------------------------------------- #
# language-XXXX tasks (baseline vs uplift heatmap)
# ----------------------------------------------------------------------------- #
def language_tasks(n: int = 15) -> list[Task]:
    """Short language/instruction-following tasks, mirroring the brief's image.

    Some tasks get a near-zero or negative base_uplift so the uplift panel shows
    the realistic mix of helped / unaffected / hurt seen in SkillsBench.
    """
    lib = build_skill_library()["extracted"]
    lang = [s for s in lib if s.family in ("language", "safety")]
    templates = [
        "Rewrite the passage in {sk}, keeping it under three sentences.",
        "Produce a short paragraph that demonstrates {sk} on the topic of gardening.",
        "Edit the text so it satisfies {sk} without changing the meaning.",
        "Write two sentences about dueling that exhibit {sk}.",
    ]
    tasks: list[Task] = []
    for i in range(n):
        rng = _rng("language", str(i))
        skill = lang[int(rng.integers(len(lang)))]
        tmpl = templates[int(rng.integers(len(templates)))]
        prompt = tmpl.format(sk=skill.name)
        uid = task_uid(prompt + f"::{i}")
        label = f"language-{stable_uint64(uid, namespace='lang') & 0xffffffff:08x}"
        diff = _difficulty_from([skill], 1, rng) + 1.4  # tuned so small models mostly fail baseline
        # Mixed uplift: ~65% helpful, ~20% neutral, ~15% harmful.
        roll = rng.random()
        uplift = (rng.uniform(0.4, 1.2) if roll < 0.65
                  else (rng.uniform(-0.1, 0.1) if roll < 0.85 else rng.uniform(-1.0, -0.3)))
        tasks.append(Task(
            uid=uid, label=label, prompt=prompt, family="language",
            skill_uids=[skill.uid], skill_names=[skill.name],
            difficulty=diff, gold="", base_uplift=float(uplift), k=1,
            meta={"skill_family": skill.family},
        ))
    return tasks


# ----------------------------------------------------------------------------- #
# Broad mixed tasks (SkillsBench-style model x task grid)
# ----------------------------------------------------------------------------- #
_DOMAINS = ["SW Eng", "Science", "Office", "Finance", "Media", "Cyber",
            "Health", "Robotics", "Energy", "Math", "Mfg"]


def broad_tasks(n: int = 84) -> list[Task]:
    """A spread of tasks across domains and difficulty (easy -> very hard)."""
    lib = build_skill_library()["extracted"]
    tasks: list[Task] = []
    for i in range(n):
        rng = _rng("broad", str(i))
        k = int(rng.integers(1, 4))
        skills = list(rng.choice(lib, size=min(k, len(lib)), replace=False))
        domain = _DOMAINS[int(rng.integers(len(_DOMAINS)))]
        slug = "-".join(s.name.split()[0] for s in skills)[:28]
        prompt = (f"[{domain}] Complete a task requiring: "
                  + ", ".join(s.name for s in skills) + ".")
        uid = task_uid(prompt + f"@{i}")
        # Spread difficulty widely so the grid spans uniformly-easy to uniformly-hard.
        spread = np.linspace(-1.5, 3.2, n)[i]
        diff = float(spread + _difficulty_from(skills, k, rng) * 0.3)
        tasks.append(Task(
            uid=uid, label=f"{slug} ({domain})", prompt=prompt, family="mixed",
            skill_uids=[s.uid for s in skills], skill_names=[s.name for s in skills],
            difficulty=diff, base_uplift=float(rng.uniform(0.0, 0.9)), k=k,
            meta={"domain": domain},
        ))
    return tasks


if __name__ == "__main__":
    nd = natural_deduction_tasks(k=2, per_pair=6)
    lang = language_tasks()
    broad = broad_tasks()
    print(f"natural-deduction tasks: {len(nd)}  (k=2)")
    print(f"language tasks:          {len(lang)}")
    print(f"broad mixed tasks:       {len(broad)}")
    print("\nexample ND task:\n ", nd[0].label, "->", nd[0].prompt)
    print("example language task:\n ", lang[0].label, "->", lang[0].prompt,
          f"(uplift {lang[0].base_uplift:+.2f})")
    print("example broad task:\n ", broad[40].label, "->", broad[40].prompt)
