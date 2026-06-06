"""
Deterministic Uint64 identifiers rendered as lowercase hex `xxxx-xxxx-xxxx-xxxx`.

The assessment brief asks for skill UIDs and task UIDs as Uint64 in lowercase
hex, grouped as four 16-bit fields. We derive the integer deterministically from
the object's defining text (skill name, or task prompt) using BLAKE2b with an
8-byte digest. Determinism matters: the same skill/task always maps to the same
UID across runs and machines, so figures, tables and the report stay consistent.
"""

from __future__ import annotations

import hashlib


def stable_uint64(text: str, *, namespace: str = "") -> int:
    """Map an arbitrary string to a stable 64-bit unsigned integer."""
    payload = f"{namespace}\x1f{text}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def format_uid(value: int) -> str:
    """Render a Uint64 as four hyphen-separated lowercase 16-bit hex groups."""
    if not 0 <= value < 2 ** 64:
        raise ValueError("value must fit in a Uint64")
    groups = [(value >> (16 * i)) & 0xFFFF for i in reversed(range(4))]
    return "-".join(f"{g:04x}" for g in groups)


def skill_uid(name: str) -> str:
    return format_uid(stable_uint64(name, namespace="skill"))


def task_uid(prompt: str) -> str:
    return format_uid(stable_uint64(prompt, namespace="task"))


if __name__ == "__main__":
    for s in ["apply modus ponens inference", "introduce misleading red herring"]:
        print(f"{skill_uid(s)}  {s}")
