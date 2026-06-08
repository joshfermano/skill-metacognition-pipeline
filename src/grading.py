"""Skill-Mix grading and skill detection.

A k-skill generation is scored out of k + 3 points: one per skill, plus on-topic,
coherent, and within length. From these come three metrics: Ratio of Full Marks
(all k + 3 awarded), Ratio of All Skills (all k skills and >= 2 of the other 3),
and Skill Fraction (skills/k when all 3 others are awarded, else 0). Logic, math,
and code tasks also have a deterministic gold-answer verifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GradeResult:
    passed: bool
    score: float                 # in [0, 1]
    skill_points: int            # A: points for the k skills
    other_points: int            # B: points for topic/coherence/length (0..3)
    used_skills: list[str]       # skill UIDs detected in the CoT


# Skill-Mix metrics
def ratio_full_marks(skill_points: int, other_points: int, k: int) -> int:
    return int(skill_points + other_points == k + 3)


def ratio_all_skills(skill_points: int, other_points: int, k: int) -> int:
    return int(skill_points == k and other_points >= 2)


def skill_fraction(skill_points: int, other_points: int, k: int) -> float:
    return (skill_points / k) if other_points == 3 and k > 0 else 0.0


# Skill detection from chain-of-thought
def detect_skills(cot: str, candidate_skills) -> list[str]:
    """Return UIDs of candidate skills whose name or symbol appears in the CoT.

    In `llm` mode a model performs this labelling instead.
    """
    found = []
    low = cot.lower()
    for s in candidate_skills:
        hit = s.name.lower() in low
        if not hit and getattr(s, "symbol", None):
            hit = s.symbol in cot
        if not hit:
            # fallback: >=2 content words from the name present
            toks = [t for t in re.findall(r"[a-z]+", s.name.lower()) if len(t) > 3]
            hit = sum(t in low for t in toks) >= max(2, len(toks) - 1)
        if hit:
            found.append(s.uid)
    return found


# Grading entry point (real backends)
def grade(task, answer: str, cot: str, candidate_skills=None) -> GradeResult:
    """Grade a real model's output for a task.

    Logic, math, and code tasks with a gold answer use a deterministic verifier;
    other tasks use a rubric proxy (skill coverage plus coherence heuristics). Swap
    in an LLM judge here if a stronger one is available.
    """
    used = detect_skills(cot, candidate_skills) if candidate_skills else []
    k = max(1, len(task.skill_uids))

    if task.family in ("logic", "math", "code") and task.gold:
        ok = task.gold.split()[0].lower() in (answer or "").lower()
        skill_pts = k if ok else max(0, len(set(used) & set(task.skill_uids)))
        other_pts = 3 if ok else 1
        score = 1.0 if ok else round(0.4 * skill_pts / k, 3)
        return GradeResult(ok, score, skill_pts, other_pts, used)

    # Rubric proxy for language and mixed tasks.
    covered = len(set(used) & set(task.skill_uids))
    skill_pts = covered
    sentences = [s for s in re.split(r"[.!?]+", answer or "") if s.strip()]
    on_topic = 1
    makes_sense = int(len(answer.strip()) > 0)
    length_ok = int(len(sentences) <= max(2, k - 1) + 1)
    other_pts = on_topic + makes_sense + length_ok
    passed = ratio_all_skills(skill_pts, other_pts, k) == 1
    score = round((skill_pts + other_pts) / (k + 3), 3)
    return GradeResult(passed, score, skill_pts, other_pts, used)
