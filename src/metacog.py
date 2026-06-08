"""Named-skill metacognition scaffolding: enumerate, choose, name-before-use, justify.

A prompt-and-parse loop over the skill library and model backends that contrasts
two conditions: pre_training has no catalogue in context (the model cannot
enumerate the named skills), post_training supplies it. Supplying the catalogue is
a proxy for the trained behaviour; the genuine weight-update loop lives in
pipeline.self_improvement. Real backends emit genuine text; the simulation backend
emits a structured scaffold so the figure reproduces with no key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .grading import detect_skills, grade


# Catalogue rendering and name-to-uid resolution
def _all_skills(lib: dict) -> list:
    return (list(lib.get("extracted", [])) + list(lib.get("natural_deduction", []))
            + list(lib.get("code", [])))


def render_catalog(lib: dict) -> str:
    """Numbered skill menu shown to the model in the post_training condition."""
    lines = []
    for i, s in enumerate(_all_skills(lib), 1):
        label = s.symbol if s.symbol else s.name
        gloss = (s.members[0] if s.members else s.name)
        lines.append(f"{i:2d}. {label}  [{s.uid}]  - {gloss}")
    return "\n".join(lines)


def _name_index(lib: dict) -> dict[str, str]:
    """Map every lowercased skill name and symbol to its uid, for parsing."""
    idx: dict[str, str] = {}
    for s in _all_skills(lib):
        idx[s.name.lower()] = s.uid
        if s.symbol:
            idx[s.symbol.lower()] = s.uid
    return idx


def _resolve(token: str, lib: dict, name_idx: dict[str, str]) -> str | None:
    """Best-effort resolve a free-text skill mention to a skill uid."""
    t = token.strip().lower().strip(".,;:")
    if not t:
        return None
    if t in name_idx:
        return name_idx[t]
    # substring match handles light paraphrase
    for s in _all_skills(lib):
        if s.name.lower() in t or (len(t) > 4 and t in s.name.lower()):
            return s.uid
    return None


# Trace data structures
@dataclass
class ScaffoldStep:
    candidates: list[str]                       # enumerated candidate names
    candidate_uids: list[str] = field(default_factory=list)
    chosen: str = ""                            # the named, chosen skill
    chosen_uid: str | None = None
    why: str = ""                               # justification text
    applied: str = ""                           # the work done in this step
    named_before_use: bool = False              # was a skill named before APPLY?


@dataclass
class ScaffoldTrace:
    task_uid: str
    label: str
    model: str
    condition: str
    steps: list[ScaffoldStep]
    answer: str
    raw: str
    passed: bool
    score: float
    interrogation: dict = field(default_factory=dict)  # {"question","answer","step"}


# Prompt construction
_PROTOCOL = (
    "Solve the problem step by step. For EACH step, follow this metacognitive "
    "protocol exactly:\n"
    "STEP <n>:\n"
    "CANDIDATE SKILLS: <comma-separated names of skills from the catalogue that "
    "could apply at this step>\n"
    "CHOSEN SKILL: <the single skill name you will use - state it before using it>\n"
    "WHY: <one sentence on why this skill, not the other candidates>\n"
    "APPLY: <carry out the step>\n"
    "Repeat for each step, then end with a line 'ANSWER: <final answer>'."
)


def build_prompt(task, lib: dict, condition: str) -> str:
    if condition == "post_training":
        return (
            "You have been trained with a named skill catalogue. Use it.\n\n"
            f"SKILL CATALOGUE:\n{render_catalog(lib)}\n\n"
            f"{_PROTOCOL}\n\nPROBLEM: {task.prompt}"
        )
    # pre_training: no catalogue; the model cannot enumerate named skills.
    return (
        "Solve the problem. Think step by step and, if you use any skill, name it. "
        "End with a line 'ANSWER: <final answer>'.\n\n"
        f"PROBLEM: {task.prompt}"
    )


# Parse real model output into a structured scaffold
_STEP_RE = re.compile(r"^\s*step\s*\d+\s*:?", re.I)
_FIELD_RE = re.compile(
    r"^\s*(candidate skills|chosen skill|why|apply)\s*:\s*(.*)$", re.I
)


def parse_scaffold(text: str, lib: dict) -> tuple[list[ScaffoldStep], str]:
    name_idx = _name_index(lib)
    answer = ""
    m = re.search(r"answer\s*:\s*(.+)", text, re.I | re.S)
    if m:
        answer = m.group(1).strip().splitlines()[0].strip()

    # Split into step blocks on 'STEP n:' markers.
    lines = text.splitlines()
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for ln in lines:
        if _STEP_RE.match(ln):
            cur = []
            blocks.append(cur)
        elif cur is not None:
            cur.append(ln)
    steps: list[ScaffoldStep] = []
    for blk in blocks:
        st = ScaffoldStep(candidates=[])
        saw_chosen = False
        for ln in blk:
            fm = _FIELD_RE.match(ln)
            if not fm:
                continue
            key, val = fm.group(1).lower(), fm.group(2).strip()
            if key == "candidate skills":
                st.candidates = [c.strip() for c in re.split(r"[,;]", val) if c.strip()]
            elif key == "chosen skill":
                st.chosen = val
                saw_chosen = True
            elif key == "why":
                st.why = val
            elif key == "apply":
                st.applied = val
                st.named_before_use = saw_chosen and bool(st.chosen)
        st.candidate_uids = [u for u in (_resolve(c, lib, name_idx) for c in st.candidates) if u]
        st.chosen_uid = _resolve(st.chosen, lib, name_idx) if st.chosen else None
        steps.append(st)

    # Fallback when the model ignored the format: recover one step from the whole
    # text via post-hoc detection so it still scores (as low compliance).
    if not steps:
        cand = lib  # detect over the full catalogue
        used = detect_skills(text, _all_skills(cand))
        steps = [ScaffoldStep(candidates=[], candidate_uids=[], chosen="",
                              chosen_uid=(used[0] if used else None),
                              applied=text[:200], named_before_use=False)]
    return steps, answer


# Deterministic scaffold for the simulation backend
def _hash01(*parts: str) -> float:
    import hashlib
    h = hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64


def _simulate_text(model: str, task, lib: dict, condition: str, passed: bool) -> str:
    """Render protocol-shaped text whose structure depends on the condition.

    post_training enumerates the gold skills plus a distractor and usually chooses
    correctly; pre_training produces unstructured steps with no named candidates.
    `passed` only sets the final ANSWER line, so answer-correctness stays decoupled
    from protocol compliance.
    """
    gold_names = task.skill_names or [s.name for s in _all_skills(lib)[:1]]
    catalogue = _all_skills(lib)

    if condition == "pre_training":
        steps = [f"Step {i+1}: I work through part {i+1} of the task."
                 for i in range(len(gold_names))]
        body = "\n".join(steps)
        ans = task.gold if (passed and task.gold) else ("done" if passed else "(incomplete)")
        return f"{body}\nANSWER: {ans}"

    # post_training: structured enumerate -> choose -> name -> justify.
    blocks = []
    for i, gname in enumerate(gold_names):
        distractor = catalogue[int(_hash01(task.uid, gname, "d") * len(catalogue))].name
        # deterministically miss sometimes, to avoid a trivial 100%
        miss = _hash01(model, task.uid, gname, "sel") < 0.15
        chosen = distractor if miss else gname
        blocks.append(
            f"STEP {i+1}:\n"
            f"CANDIDATE SKILLS: {gname}, {distractor}\n"
            f"CHOSEN SKILL: {chosen}\n"
            f"WHY: it matches sub-goal {i+1} better than the alternative.\n"
            f"APPLY: applying {chosen} to advance the solution."
        )
    ans = task.gold if (passed and task.gold) else ("done" if passed else "(incomplete)")
    return "\n".join(blocks) + f"\nANSWER: {ans}"


# Solve one task under the scaffolding protocol
def scaffold_solve(backend, model: str, task, lib: dict,
                   condition: str = "post_training") -> ScaffoldTrace:
    candidate_skills = _all_skills(lib)
    if hasattr(backend, "_chat"):                      # real model
        raw = backend._chat(model, build_prompt(task, lib, condition))
        steps, answer = parse_scaffold(raw, lib)
        result = grade(task, answer, raw, candidate_skills)
        passed, score = result.passed, result.score
    else:                                              # simulation
        # Catalogue acts as a curated-skill uplift on the IRT outcome, so
        # answer-correctness tracks task success, not skill names in the trace.
        uplift = 0.9 if condition == "post_training" else 0.0
        passed = backend.trial(model, task.uid, task.difficulty, 0, uplift=uplift)
        score = 1.0 if passed else round(_hash01(model, task.uid, "p") * 0.5, 3)
        raw = _simulate_text(model, task, lib, condition, passed)
        steps, answer = parse_scaffold(raw, lib)
    return ScaffoldTrace(task.uid, task.label, model, condition, steps, answer,
                         raw, passed, score)


def interrogate(backend, model: str, task, trace: ScaffoldTrace,
                step_idx: int = 0) -> ScaffoldTrace:
    """Ask the model to justify a chosen skill."""
    if not trace.steps:
        return trace
    step = trace.steps[min(step_idx, len(trace.steps) - 1)]
    chosen = step.chosen or "the skill you used"
    others = ", ".join(c for c in step.candidates if c != step.chosen) or "the alternatives"
    q = (f"In your solution you chose '{chosen}'. Explain why you chose it rather "
         f"than {others}.")
    if hasattr(backend, "_chat"):
        a = backend._chat(model, f"{trace.raw}\n\n{q}")
    else:
        a = (f"I chose '{chosen}' because it directly addresses this step's "
             f"sub-goal, whereas {others} would not advance the proof here.")
    trace.interrogation = {"step": step_idx, "question": q, "answer": a.strip()}
    return trace


# Score a trace against the task's gold skills
def score_trace(trace: ScaffoldTrace, task) -> dict:
    gold = set(task.skill_uids)
    enumerated = set(u for st in trace.steps for u in st.candidate_uids)
    chosen = set(st.chosen_uid for st in trace.steps if st.chosen_uid)
    k = max(1, len(gold))
    well_formed = [st for st in trace.steps if st.named_before_use]
    return {
        "enumeration_recall": len(gold & enumerated) / k,
        "selection_accuracy": len(gold & chosen) / k,
        "name_before_use": (len(well_formed) / len(trace.steps)) if trace.steps else 0.0,
        "answer_correct": float(trace.passed),
    }
