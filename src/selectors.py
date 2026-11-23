"""Selectors: choose k of n datasets with the OT score from src/pipeline.py.

Every selector returns {"chosen": [...], "score": mean OT distance of the
chosen combination, "calls": number of combinations scored, "seconds": ...}.

    one_shot       score each dataset alone; take the k best. LAVA applied one
                   dataset at a time: blind to overlap between picks.
    greedy         add, k times, the dataset whose addition gives the lowest
                   pooled score. Overlap-aware: a near-copy of something
                   already chosen adds (almost) nothing.
    mmr            one-shot scores plus a penalty for similarity to already-
                   chosen datasets: shared images (overlap) or distribution
                   similarity (OT distance between datasets).
    dedup_oneshot  baseline: drop datasets that mostly duplicate a better one
                   (detected overlap > theta), then one-shot.
    exhaustive     score every k-subset; the best is the reference "OT optimum".
    random         k datasets at random (averaged over draws in the analysis).

Scores are averaged over `seeds` OT subsamples to reduce scoring noise.
"""

from __future__ import annotations

import itertools
import time

import numpy as np


class Scorer:
    """Memoized mean OT score of a combination over several seeds."""

    def __init__(self, ctx, seeds=(0, 1, 2), dedupe=True):
        self.ctx, self.seeds, self.dedupe = ctx, tuple(seeds), dedupe
        self.cache: dict[tuple, list[float]] = {}
        self.calls = 0

    def per_seed(self, combo) -> list[float]:
        key = tuple(sorted(combo))
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = [self.ctx.score(key, s, self.dedupe) for s in self.seeds]
        return self.cache[key]

    def __call__(self, combo) -> float:
        return float(np.mean(self.per_seed(combo)))


def _result(chosen, scorer, t0, calls0, **extra):
    return {"chosen": list(chosen), "score": scorer(chosen),
            "calls": scorer.calls - calls0, "seconds": round(time.perf_counter() - t0, 2),
            **extra}


def one_shot(ids, k, scorer):
    t0, c0 = time.perf_counter(), scorer.calls
    ranked = sorted(ids, key=lambda d: scorer([d]))
    return _result(ranked[:k], scorer, t0, c0,
                   ranking=[(d, scorer([d])) for d in ranked])


def greedy(ids, k, scorer):
    t0, c0 = time.perf_counter(), scorer.calls
    chosen, trace = [], []
    for _ in range(k):
        cand = [d for d in ids if d not in chosen]
        scores = {d: scorer(chosen + [d]) for d in cand}
        best = min(scores, key=scores.get)
        chosen.append(best)
        trace.append({"added": best, "score": scores[best]})
    return _result(chosen, scorer, t0, c0, trace=trace)


def mmr(ids, k, scorer, similarity: np.ndarray, lam: float | None = None,
        lam_mult: float | None = None):
    """Maximal marginal relevance: pick, k times, the dataset minimizing
        single OT score + lam * (max similarity to an already-chosen dataset).

    similarity : n x n matrix in [0, 1] (overlap_similarity or ot_similarity)
    lam        : penalty in distance units for similarity 1. Default: lam_mult
                 (default 1) x the spread of the single-dataset scores, so a
                 fully similar dataset costs as much as the gap between the best
                 and the worst dataset. lam = 0 reduces to one-shot.
    """
    t0, c0 = time.perf_counter(), scorer.calls
    single = {d: scorer([d]) for d in ids}
    spread = max(single.values()) - min(single.values())
    if lam is None:
        lam = (1.0 if lam_mult is None else lam_mult) * spread
    pos = {d: i for i, d in enumerate(ids)}
    chosen, trace = [], []
    for _ in range(k):
        def val(d):
            pen = max((similarity[pos[d], pos[c]] for c in chosen), default=0.0)
            return single[d] + lam * pen
        best = min((d for d in ids if d not in chosen), key=val)
        trace.append({"added": best, "value": val(best)})
        chosen.append(best)
    return _result(chosen, scorer, t0, c0, lam=lam, lam_mult=lam / spread if spread else None,
                   trace=trace)


def overlap_similarity(overlap: np.ndarray) -> np.ndarray:
    """Symmetric similarity from the detected duplicate-overlap matrix:
    the larger of a-in-b and b-in-a. 1 = one is a full copy of the other."""
    return np.maximum(overlap, overlap.T)


def ot_similarity(ctx, ids, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Distribution similarity between datasets: class-wise OT distance between
    every pair (embeddings, near-duplicates merged), rescaled so the closest
    pair is 1 and the farthest 0. Catches datasets that are alike without
    sharing images. Returns (similarity, raw distances)."""
    from .ot import class_wise_ot
    n = len(ids)
    D = np.zeros((n, n))
    for a in range(n):
        ga = ctx.index([ids[a]])
        for b in range(a + 1, n):
            gb = ctx.index([ids[b]])
            D[a, b] = D[b, a] = class_wise_ot(ctx.F[ga], ctx.Y[ga], ctx.F[gb], ctx.Y[gb],
                                              seed=seed, n_max=ctx.n_max,
                                              label_weight=ctx.label_weight)["distance"]
    off = D[~np.eye(n, dtype=bool)]
    S = (off.max() - D) / max(off.max() - off.min(), 1e-12)
    np.fill_diagonal(S, 1.0)
    return S, D


def dedup_oneshot(ids, k, scorer, overlap: np.ndarray, theta: float = 0.5):
    t0, c0 = time.perf_counter(), scorer.calls
    pos = {d: i for i, d in enumerate(ids)}
    ranked = sorted(ids, key=lambda d: scorer([d]))
    kept = []
    for d in ranked:
        if all(max(overlap[pos[d], pos[c]], overlap[pos[c], pos[d]]) <= theta for c in kept):
            kept.append(d)
    chosen = (kept + [d for d in ranked if d not in kept])[:k]   # pad if too few survive
    return _result(chosen, scorer, t0, c0, theta=theta, dropped=[d for d in ranked if d not in kept])


def exhaustive(ids, k, scorer, workers: int = 1, log=print):
    """Score every k-subset; returns the best plus all scores."""
    t0, c0 = time.perf_counter(), scorer.calls
    combos = [tuple(sorted(c)) for c in itertools.combinations(ids, k)]
    todo = [c for c in combos if c not in scorer.cache]
    if workers > 1 and todo:
        _parallel_fill(scorer, todo, workers, log)
    for n, c in enumerate(combos, 1):
        scorer(c)
        if n % 25 == 0 or n == len(combos):
            log(f"  exhaustive: {n}/{len(combos)} combinations scored")
    best = min(combos, key=scorer)
    return _result(best, scorer, t0, c0,
                   all=[{"combo": list(c), "score": scorer(c), "per_seed": scorer.per_seed(c)}
                        for c in combos])


def random_draws(ids, k, n_draws=20, seed=0):
    rng = np.random.default_rng(seed)
    return [sorted(rng.choice(ids, k, replace=False).tolist()) for _ in range(n_draws)]


# ---------------------------------------------------------------- parallel
_CTX = None


def _work(args):
    combo, seeds, dedupe = args
    return combo, [_CTX.score(combo, s, dedupe) for s in seeds]


def _parallel_fill(scorer, combos, workers, log):
    """Score many combinations in worker processes (Linux fork shares the
    loaded embeddings without copying)."""
    import multiprocessing as mp
    global _CTX
    _CTX = scorer.ctx
    scorer.ctx.pairs()                                  # build caches before forking
    with mp.get_context("fork").Pool(workers) as pool:
        for n, (combo, vals) in enumerate(pool.imap_unordered(
                _work, [(c, scorer.seeds, scorer.dedupe) for c in combos], chunksize=2), 1):
            scorer.cache[tuple(combo)] = vals
            scorer.calls += 1
            if n % 25 == 0 or n == len(combos):
                log(f"  scored {n}/{len(combos)} combinations ({workers} workers)")
