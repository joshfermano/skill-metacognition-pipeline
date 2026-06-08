"""Tasks derived from a real code file.

Parses a self-contained PICO-8-style Lua file (data/code/dodge.p8.lua) into
navigation, grep, trace, explanation, and ticket-decomposition tasks. Answers are
checkable against the source, so a real run is graded by a deterministic verifier
rather than a rubric proxy. Tasks reference the fixed `code` skill family and span
one-step locates to multi-step ticket breakdowns.
"""

from __future__ import annotations

import re
from pathlib import Path

from .skills import build_skill_library
from .tasks import Task
from .uids import task_uid

CODE_DIR = Path(__file__).resolve().parent.parent / "data" / "code"

_FUNC_RE = re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)


def _load(fname: str = "dodge.p8.lua") -> tuple[str, list[str]]:
    text = (CODE_DIR / fname).read_text()
    return text, _FUNC_RE.findall(text)


def count_calls(text: str, name: str) -> int:
    """Call sites of `name`: all `name(` occurrences minus its definition."""
    total = len(re.findall(rf"\b{re.escape(name)}\s*\(", text))
    defs = len(re.findall(rf"function\s+{re.escape(name)}\s*\(", text))
    return total - defs


def _pick(skills, kw: str):
    return next(s for s in skills if kw in s.name)


def code_tasks(fname: str = "dodge.p8.lua") -> list[Task]:
    text, funcs = _load(fname)
    code = build_skill_library()["code"]
    locate = _pick(code, "locate")
    grep = _pick(code, "grep")
    readf = _pick(code, "read")
    count = _pick(code, "count")
    explain = _pick(code, "explain")
    trace = _pick(code, "trace")
    decomp = _pick(code, "decompose")

    tasks: list[Task] = []

    # Embed the source in every prompt; the model has no file-system tools here.
    src_header = f"Source file `{fname}`:\n```lua\n{text}\n```\n\n"

    def mk(label, prompt, gold, skills, difficulty, uplift=0.6, meta=None):
        full = f"{src_header}Question ({fname}): {prompt}"
        tasks.append(Task(
            uid=task_uid(full), label=label, prompt=full, family="code",
            skill_uids=[s.uid for s in skills], skill_names=[s.name for s in skills],
            difficulty=difficulty, gold=gold, base_uplift=uplift, k=len(skills),
            meta=meta or {},
        ))

    # --- locate / navigation (small, one step) ------------------------------
    mk("code-locate-spawn",
       "Which function adds a new enemy to the game? Answer with the function name only.",
       "spawn_enemy", [locate], -0.6)
    mk("code-locate-loselife",
       "Which function decreases the player's lives and sets the state to 'over' "
       "when lives reach zero? Function name only.",
       "lose_life", [locate], 0.1)

    # --- grep / counting (small) --------------------------------------------
    mk("code-grep-spr",
       "How many times is the spr() built-in called in this file? Answer with a number.",
       str(count_calls(text, "spr")), [grep], -0.3)
    mk("code-count-funcs",
       "How many functions are defined in this file? Answer with a single number.",
       str(len(funcs)), [grep, readf], 0.2)
    mk("code-count-callers",
       "How many call sites invoke reset_game()? Answer with a single number "
       "(do not count its definition).",
       str(count_calls(text, "reset_game")), [count], 0.6)

    # --- trace dependencies (medium) ----------------------------------------
    mk("code-trace-collision",
       "Inside check_collision(), which function is invoked when the player is "
       "hit by an enemy? Function name only.",
       "lose_life", [trace], 0.5)

    # --- explanation (medium, rubric-graded) --------------------------------
    mk("code-explain-update-enemies",
       "Explain step by step what update_enemies() does, including when an enemy "
       "is removed and when the score changes.",
       "", [explain], 0.4,
       meta={"expects": ["move", "score", "del", "off-screen"]})

    # --- ticket decomposition (large -> small steps, rubric-graded) ---------
    mk("code-ticket-multiplier",
       "Ticket: add a score multiplier that doubles the score gained per dodge "
       "after the player has survived 10 enemies. Break this ticket into ordered "
       "small implementation steps, and name the function(s) each step touches.",
       "", [decomp, explain], 1.4,
       meta={"expects_funcs": ["reset_game", "update_enemies", "draw_hud"],
             "scale": "large"})

    return tasks


if __name__ == "__main__":
    ts = code_tasks()
    text, funcs = _load()
    print(f"file functions ({len(funcs)}): {', '.join(funcs)}")
    print(f"spr() calls: {count_calls(text, 'spr')}  reset_game() callers: "
          f"{count_calls(text, 'reset_game')}")
    for t in ts:
        print(f"  {t.label:24s} k={t.k} diff={t.difficulty:+.1f} gold={t.gold!r}  "
              f"skills={t.skill_names}")
