"""Steps 6-7: select k of n datasets with every selector, clean the picks, and
save everything the analysis needs.

LAVA scores points within one dataset; this extends it to choosing a
combination of datasets, with near-duplicate merging and cleaning.

Writes repos/<name>/results/<encoder>_k<k>/:
    combo_scores.csv   OT score of every k-subset (mean and per seed), with
                       and without duplicate merging
    selections.json    each selector's pick, score, calls, runtime, cleaning
    dedupe_check.json  duplicate detection graded against the truth at
                       several thresholds (diagnostic only)

Run after building and embedding the repository:
    python scripts/build_repo.py configs/starter.json --leakage-encoder dinov2_vits14
    python -m src.embed --repo repos/starter
    python scripts/run_selection.py --repo repos/starter --k 3 --workers 8
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import RepoContext  # noqa: E402
from src.repo import load_truth  # noqa: E402
from src.selectors import (Scorer, dedup_oneshot, exhaustive, greedy, mmr,  # noqa: E402
                           one_shot, ot_similarity, overlap_similarity, random_draws)


def dedupe_check(ctx, thresholds=(0.90, 0.93, 0.95, 0.97, 0.98, 0.99)):
    """GRADING ONLY: precision/recall of detected cross-dataset duplicates
    against the true shared images."""
    truth = load_truth(ctx.repo)
    keys = np.concatenate([truth[d] for d in ctx.ids])
    i, j, s = ctx.pairs()
    cross = ctx.owner[i] != ctx.owner[j]
    # true cross-dataset duplicate pairs
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    true_pairs = set()
    start = 0
    for e in range(1, len(ks) + 1):
        if e == len(ks) or ks[e] != ks[start]:
            members = order[start:e]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    p, q = sorted((members[a], members[b]))
                    if ctx.owner[p] != ctx.owner[q]:
                        true_pairs.add((int(p), int(q)))
            start = e
    out = {"true_pairs": len(true_pairs), "by_threshold": []}
    for t in thresholds:
        m = cross & (s >= t)
        det = {(int(min(a, b)), int(max(a, b))) for a, b in zip(i[m], j[m])}
        tp = len(det & true_pairs)
        out["by_threshold"].append({
            "threshold": t, "detected": len(det),
            "precision": tp / len(det) if det else float("nan"),
            "recall": tp / len(true_pairs) if true_pairs else float("nan")})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--encoder", default="dinov2_vits14")
    p.add_argument("--emb-root", default="emb")
    p.add_argument("--seeds", type=int, default=3, help="OT subsample seeds per combination")
    p.add_argument("--dup-threshold", type=float, default=0.95,
                   help="cosine similarity at which two images count as the same")
    p.add_argument("--dup-features", default="pixels", choices=["pixels", "emb"],
                   help="duplicate detection on pixel fingerprints (default) or embeddings")
    p.add_argument("--dup-min-emb", type=float, default=0.0,
                   help="also require this embedding cosine for a duplicate pair")
    p.add_argument("--clean-q", type=float, default=0.95,
                   help="drop points costlier than this quantile of clean points")
    p.add_argument("--n-max", type=int, default=3000)
    p.add_argument("--mmr-similarity", nargs="+", default=["overlap"], choices=["overlap", "ot"],
                   help="MMR similarity: shared images (overlap) and/or distribution (ot)")
    p.add_argument("--mmr-lambdas", nargs="+", type=float, default=[1.0],
                   help="MMR penalty as multiples of the single-score spread; one selector "
                        "per value, e.g. 0 0.25 0.5 1 2")
    p.add_argument("--dedup-theta", type=float, default=0.5)
    p.add_argument("--no-exhaustive", action="store_true")
    p.add_argument("--no-dedupe-ablation", action="store_true",
                   help="skip scoring every combination without duplicate merging")
    p.add_argument("--workers", type=int, default=1, help="processes for exhaustive scoring")
    a = p.parse_args()

    t0 = time.perf_counter()
    ctx = RepoContext(a.repo, a.encoder, a.emb_root, a.dup_threshold, a.clean_q, a.n_max,
                      dup_features=a.dup_features, dup_min_emb=a.dup_min_emb)
    ids = ctx.ids
    out = Path(a.repo) / "results" / f"{a.encoder}_k{a.k}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"repository {a.repo}: {len(ids)} datasets, k = {a.k}, "
          f"{math.comb(len(ids), a.k)} combinations")

    ctx.pairs()
    ctx.point_costs()
    overlap = ctx.detected_overlap()
    scorer = Scorer(ctx, seeds=range(a.seeds), dedupe=True)

    sel = {}
    print("one-shot ...");      sel["one_shot"] = one_shot(ids, a.k, scorer)
    print("greedy ...");        sel["greedy"] = greedy(ids, a.k, scorer)
    sims = {}
    if "overlap" in a.mmr_similarity:
        sims["overlap"] = overlap_similarity(overlap)
    if "ot" in a.mmr_similarity:
        print("dataset-to-dataset OT distances for MMR ...")
        sims["ot"], ot_dist = ot_similarity(ctx, ids)
    for sname, S in sims.items():
        for lm in a.mmr_lambdas:
            name = "mmr" if (sname == "overlap" and lm == 1.0 and len(sims) == 1
                             and len(a.mmr_lambdas) == 1) else f"mmr_{sname}_x{lm:g}"
            print(f"{name} ...")
            sel[name] = mmr(ids, a.k, scorer, S, lam_mult=lm)
            sel[name]["similarity"] = sname
    print("dedup+one-shot ..."); sel["dedup_oneshot"] = dedup_oneshot(ids, a.k, scorer, overlap,
                                                                      a.dedup_theta)
    rows = []
    if not a.no_exhaustive:
        print("exhaustive ...")
        ex = exhaustive(ids, a.k, scorer, a.workers)
        rows = ex.pop("all")
        sel["exhaustive"] = ex
        if not a.no_dedupe_ablation:
            print("exhaustive without duplicate merging (ablation) ...")
            plain = Scorer(ctx, seeds=range(a.seeds), dedupe=False)
            ex2 = exhaustive(ids, a.k, plain, a.workers)
            plain_scores = {tuple(r["combo"]): r["score"] for r in ex2.pop("all")}
            sel["exhaustive_no_dedupe"] = ex2
            for r in rows:
                r["score_no_dedupe"] = plain_scores[tuple(r["combo"])]

    draws = random_draws(ids, a.k)
    sel["random"] = {"draws": draws}

    for name, r in sel.items():
        if "chosen" in r:
            _, stats = ctx.clean(r["chosen"])
            r["cleaning"] = stats

    for r in rows:                     # unique images after merging near-duplicates
        r["n_unique"] = int(ctx.groups(ctx.index(r["combo"])).max() + 1)
    if rows:
        with open(out / "combo_scores.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["combo", "score", "sd", "n_unique"] + [f"seed{s}" for s in range(a.seeds)]
                       + (["score_no_dedupe"] if "score_no_dedupe" in rows[0] else []))
            for r in sorted(rows, key=lambda r: r["score"]):
                w.writerow(["+".join(r["combo"]), f"{r['score']:.5f}",
                            f"{np.std(r['per_seed']):.5f}", r["n_unique"]]
                           + [f"{v:.5f}" for v in r["per_seed"]]
                           + ([f"{r['score_no_dedupe']:.5f}"] if "score_no_dedupe" in r else []))

    meta = {"repo": str(a.repo), "encoder": a.encoder, "k": a.k, "seeds": a.seeds,
            "dup_features": a.dup_features, "dup_min_emb": a.dup_min_emb,
            "dup_threshold": a.dup_threshold, "clean_q": a.clean_q, "n_max": a.n_max,
            "clean_threshold": ctx.reference()["threshold"],
            "detected_overlap": {d: {e: round(float(overlap[i, j]), 4)
                                     for j, e in enumerate(ids) if overlap[i, j] > 0}
                                 for i, d in enumerate(ids)},
            "single_scores": {d: scorer([d]) for d in ids},
            "mmr_similarity": {k: [[round(float(v), 4) for v in row] for row in S]
                               for k, S in sims.items()},
            "seconds_total": round(time.perf_counter() - t0, 1)}
    (out / "selections.json").write_text(json.dumps({"meta": meta, "selectors": sel}, indent=2,
                                                    default=float))
    dc = dedupe_check(ctx)
    (out / "dedupe_check.json").write_text(json.dumps(dc, indent=2))

    print(f"\n{'selector':21s} {'chosen':36s} {'score':>9s} {'calls':>6s} {'secs':>7s}  cleaning")
    for name, r in sel.items():
        if "chosen" not in r:
            continue
        c = r["cleaning"]
        print(f"{name:21s} {'+'.join(r['chosen']):36s} {r['score']:9.4f} {r['calls']:6d} "
              f"{r['seconds']:7.1f}  -{c['removed_duplicates']} dup, -{c['removed_costly']} costly "
              f"({c['n_in']} -> {c['n_out']})")
    print(f"\nduplicate detection vs truth ({dc['true_pairs']} true cross-dataset pairs; "
          f"features: {a.dup_features}; current threshold {a.dup_threshold}):")
    for r in dc["by_threshold"]:
        print(f"  cosine >= {r['threshold']:.2f}: {r['detected']:7d} detected, "
              f"precision {r['precision']:.3f}, recall {r['recall']:.3f}")
    print(f"\nsaved to {out}  ({meta['seconds_total']} s)")


if __name__ == "__main__":
    main()
