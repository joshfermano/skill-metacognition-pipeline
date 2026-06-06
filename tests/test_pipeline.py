"""Lightweight smoke tests: `python -m tests.test_pipeline` (no pytest needed)."""

from __future__ import annotations

import re

from src.uids import skill_uid, task_uid, format_uid, stable_uint64
from src.grading import ratio_full_marks, ratio_all_skills, skill_fraction
from src.models import SimulationBackend, MODEL_ROSTER, CachedBackend
from src.skills import build_skill_library
from src.tasks import natural_deduction_tasks
from src.code_tasks import code_tasks, count_calls, _load
from src.metacog import parse_scaffold
from src.pipeline import metacog_eval

UID_RE = re.compile(r"^[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}$")


def test_uid_format_and_determinism():
    a = skill_uid("apply modus ponens")
    assert UID_RE.match(a), a
    assert a == skill_uid("apply modus ponens")              # deterministic
    assert skill_uid("x") != skill_uid("y")                  # distinct
    assert format_uid(0) == "0000-0000-0000-0000"
    assert format_uid(2 ** 64 - 1) == "ffff-ffff-ffff-ffff"


def test_skillmix_metrics():
    # k=3, all skills + all 3 criteria -> full marks
    assert ratio_full_marks(3, 3, 3) == 1
    assert ratio_all_skills(3, 3, 3) == 1
    assert skill_fraction(3, 3, 3) == 1.0
    # all skills but one criterion slip -> All-Skills yes, Full-Marks no
    assert ratio_full_marks(3, 2, 3) == 0
    assert ratio_all_skills(3, 2, 3) == 1
    # missing a skill, all criteria -> Skill Fraction = 2/3
    assert abs(skill_fraction(2, 3, 3) - 2 / 3) < 1e-9
    assert ratio_all_skills(2, 3, 3) == 0


def test_irt_monotonic_in_ability():
    b = SimulationBackend()
    diff = 0.5
    probs = [b.pass_prob(m, diff) for m in MODEL_ROSTER]
    abilities = [MODEL_ROSTER[m]["ability"] for m in MODEL_ROSTER]
    # higher ability -> higher pass prob
    order = sorted(range(len(abilities)), key=lambda i: abilities[i])
    sorted_probs = [probs[i] for i in order]
    assert sorted_probs == sorted(sorted_probs)


def test_uplift_helps():
    b = SimulationBackend()
    assert b.pass_prob("qwen2.5-3b", 1.0, uplift=0.8) > b.pass_prob("qwen2.5-3b", 1.0, 0.0)


def test_skill_library_and_tasks():
    lib = build_skill_library()
    assert len(lib["natural_deduction"]) == 9
    assert all(UID_RE.match(s.uid) for s in lib["extracted"])
    nd = natural_deduction_tasks(k=2, per_pair=3)
    assert all(len(t.skill_uids) == 2 for t in nd)


def test_code_skills_and_verifiable_gold():
    # Fixed code-skill family is well-formed.
    code = build_skill_library()["code"]
    assert len(code) == 7
    assert all(UID_RE.match(s.uid) for s in code)
    assert all(s.family == "code" for s in code)
    # Code-task gold answers match a fresh parse of the bundled Lua source.
    by = {t.label: t for t in code_tasks()}
    text, funcs = _load()
    assert by["code-count-funcs"].gold == str(len(funcs))
    assert by["code-grep-spr"].gold == str(count_calls(text, "spr"))
    assert by["code-count-callers"].gold == str(count_calls(text, "reset_game"))
    assert by["code-locate-spawn"].gold == "spawn_enemy"


def test_metacog_parse_roundtrip():
    lib = build_skill_library()
    name = lib["code"][0].name                       # a real catalogue skill
    text = (f"STEP 1:\nCANDIDATE SKILLS: {name}, read file role\n"
            f"CHOSEN SKILL: {name}\nWHY: because it fits.\nAPPLY: do the step.\n"
            f"ANSWER: 2")
    steps, answer = parse_scaffold(text, lib)
    assert answer == "2"
    assert len(steps) == 1
    assert steps[0].chosen == name
    assert steps[0].named_before_use
    assert steps[0].chosen_uid is not None


def test_metacog_post_beats_pre():
    b = SimulationBackend()
    lib = build_skill_library()
    summary, _ = metacog_eval(b, code_tasks()[:4], "qwen2.5-7b", lib)
    # Before training the model cannot enumerate the named catalogue at all.
    assert summary.loc["pre_training", "enumeration_recall"] == 0.0
    # The catalogue lets it enumerate and select the gold skills.
    assert (summary.loc["post_training", "selection_accuracy"]
            > summary.loc["pre_training", "selection_accuracy"])


def test_cache_backend_records_and_replays():
    inner = SimulationBackend()
    t0, t1 = code_tasks()[0], code_tasks()[1]
    rec = CachedBackend(inner=inner)
    tr1 = rec.solve("qwen2.5-7b", t0, 0, 0.0)
    # Replay-only from the recorded cache, with no inner backend.
    rep = CachedBackend(inner=None, cache=dict(rec.cache))
    tr2 = rep.solve("qwen2.5-7b", t0, 0, 0.0)
    assert tr1.task_uid == tr2.task_uid and tr1.passed == tr2.passed
    # A miss in replay mode raises rather than silently calling a model.
    try:
        rep.solve("qwen2.5-7b", t1, 0, 0.0)
        raised = False
    except KeyError:
        raised = True
    assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
