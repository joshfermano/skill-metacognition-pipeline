"""Skill extraction and clustering.

Two stages: fine-grained phrases (read from data/seed_skills.json, or labelled
live in `llm` mode) are merged into coarse skills by TF-IDF + agglomerative
clustering, and the cluster medoid becomes the canonical skill. Each skill gets a
4-word name, a Uint64 hex UID, and a cognitive-load rank for sorting. The
natural-deduction rules load as a fixed, pre-named family.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from .uids import skill_uid

DATA = Path(__file__).resolve().parent.parent / "data"

# Signal words that map a skill to a cognitive-load rank, used only for display order.
_LOAD_HINTS = {
    1: ["assume", "read", "grep", "locate", "cite", "extract", "mask", "redact"],
    2: ["paraphrase", "summarize", "introduce", "eliminate", "convert", "follow", "obey", "rank", "validate"],
    3: ["apply", "compute", "isolate", "track", "set up", "parse", "explain"],
    4: ["chain", "decompose", "refactor", "factor", "ground", "model"],
    5: ["split", "infer", "detect", "navigate", "trace"],
    6: ["contradiction", "reductio", "negate", "absurdity", "syllogism"],
}


@dataclass
class Skill:
    uid: str
    name: str                       # 4-word canonical name
    family: str                     # "language" | "math" | "code" | "logic" | "safety"
    members: list[str] = field(default_factory=list)   # fine-grained phrases in cluster
    cognitive_rank: int = 3
    symbol: str | None = None       # for natural-deduction skills

    def to_dict(self) -> dict:
        return asdict(self)


def _family_of(phrase: str) -> str:
    p = phrase.lower()
    if any(w in p for w in ["conjunction", "disjunction", "modus", "conditional", "contradiction", "negat", "proof", "hypothesis", "absurd"]):
        return "logic"
    if any(w in p for w in ["equation", "arithmetic", "average", "fraction", "quadratic", "chain rule", "variable", "units", "numeric"]):
        return "math"
    if any(w in p for w in ["code", "function", "repository", "grep", "file", "unit test", "refactor", "csv", "variable across"]):
        return "code"
    if any(w in p for w in ["pii", "identifiable", "redact", "mask", "sensitive", "refuse", "out of scope", "ground"]):
        return "safety"
    return "language"


def _cognitive_rank(phrase: str) -> int:
    p = phrase.lower()
    best = 3
    for rank, hints in _LOAD_HINTS.items():
        if any(h in p for h in hints):
            best = max(best, rank) if rank >= 4 else rank
    return best


def _to_four_words(phrase: str) -> str:
    """Reduce a phrase to its first up-to-four meaningful, de-duplicated tokens."""
    stop = {"a", "an", "the", "to", "of", "for", "into", "from", "with", "via",
            "and", "another", "without", "s", "its", "in", "two", "one", "both"}
    tokens = [t for t in re.findall(r"[a-zA-Z]+", phrase.lower()) if t not in stop]
    seen, name = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            name.append(t)
        if len(name) == 4:
            break
    return " ".join(name) if name else phrase.lower()


def load_natural_deduction_skills() -> list[Skill]:
    raw = json.loads((DATA / "seed_skills.json").read_text())["natural_deduction_skills"]
    skills = []
    for r in raw:
        name = _to_four_words(r["name"])
        skills.append(Skill(
            uid=skill_uid("nd:" + r["symbol"]),
            name=name,
            family="logic",
            members=[r["full"]],
            cognitive_rank=r["cognitive_rank"],
            symbol=r["symbol"],
        ))
    return skills


def load_code_skills() -> list[Skill]:
    """Fixed code-navigation skill family, so each code task references a stable uid
    rather than depending on where a phrase lands in the clustering."""
    raw = json.loads((DATA / "seed_skills.json").read_text()).get("code_skills", [])
    skills = []
    for r in raw:
        name = _to_four_words(r["name"])
        skills.append(Skill(
            uid=skill_uid("code:" + r["name"]),
            name=name,
            family="code",
            members=[r["full"]],
            cognitive_rank=r["cognitive_rank"],
        ))
    skills.sort(key=lambda s: (s.cognitive_rank, s.name))
    return skills


def _agglomerative_average(D: np.ndarray, n_clusters: int) -> np.ndarray:
    """Average-linkage agglomerative clustering from a precomputed distance matrix.

    Hand-rolled to avoid sklearn.cluster, whose compiled DLL is blocked on this
    machine; the corpus is tiny so the naive O(n^3) merge loop is fine. Merges the
    pair with the smallest mean cross-pair distance, breaking ties on the smallest
    cluster-id pair so the result is deterministic.
    """
    clusters: dict[int, list[int]] = {i: [i] for i in range(D.shape[0])}
    while len(clusters) > n_clusters:
        ids = sorted(clusters)
        best, best_d = None, np.inf
        for ai in range(len(ids)):
            for bi in range(ai + 1, len(ids)):
                a, b = ids[ai], ids[bi]
                d = float(D[np.ix_(clusters[a], clusters[b])].mean())
                if d < best_d:          # first minimum wins, so smallest (a, b)
                    best_d, best = d, (a, b)
        a, b = best
        clusters[a].extend(clusters[b])
        del clusters[b]
    labels = np.empty(D.shape[0], dtype=int)
    for c_idx, cid in enumerate(sorted(clusters)):
        for m in clusters[cid]:
            labels[m] = c_idx
    return labels


def extract_skills(n_clusters: int = 18, seed: int = 0) -> list[Skill]:
    """Cluster the fine-grained seed corpus into coarse skills."""
    phrases = json.loads((DATA / "seed_skills.json").read_text())["fine_grained_skills"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(phrases)
    D = cosine_distances(X)
    n_clusters = min(n_clusters, len(phrases))
    labels = _agglomerative_average(D, n_clusters)

    skills: list[Skill] = []
    Xd = np.asarray(X.todense())
    for c in range(n_clusters):
        idx = [i for i, l in enumerate(labels) if l == c]
        if not idx:
            continue
        # Canonical member = phrase closest to the cluster centroid.
        centroid = Xd[idx].mean(axis=0, keepdims=True)
        dists = cosine_distances(Xd[idx], centroid).ravel()
        medoid = phrases[idx[int(np.argmin(dists))]]
        members = [phrases[i] for i in idx]
        name = _to_four_words(medoid)
        rank = max(_cognitive_rank(m) for m in members)
        skills.append(Skill(
            uid=skill_uid(name),
            name=name,
            family=_family_of(medoid),
            members=members,
            cognitive_rank=rank,
        ))
    # Sort low cognitive load to high.
    skills.sort(key=lambda s: (s.cognitive_rank, s.family, s.name))
    return skills


def build_skill_library(n_clusters: int = 18) -> dict[str, list[Skill]]:
    """Full skill library: extracted, natural-deduction, and code families."""
    return {
        "extracted": extract_skills(n_clusters=n_clusters),
        "natural_deduction": load_natural_deduction_skills(),
        "code": load_code_skills(),
    }


if __name__ == "__main__":
    lib = build_skill_library()
    for grp, sk in lib.items():
        print(f"\n== {grp} ({len(sk)}) ==")
        for s in sk:
            sym = f" [{s.symbol}]" if s.symbol else ""
            print(f"  {s.uid}  r{s.cognitive_rank}  {s.family:8s}  {s.name}{sym}")
