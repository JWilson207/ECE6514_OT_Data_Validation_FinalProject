"""Analysis: does the OT score pick the combination that trains best, and does
accounting for overlap (and cleaning) matter?

Reads repos/<name>/results/<encoder>_k<k>/{combo_scores.csv, selections.json,
ground_truth.jsonl} plus the repository's manifest and truth (grading only),
and reports:

    rank correlation  Spearman / Kendall between OT score and accuracy over all
                      combinations (negative is good: lower distance, higher
                      accuracy), with a bootstrap 95% CI; also without
                      duplicate merging
    proxy error       accuracy of the truly best combination minus that of the
                      best-scoring one
    per selector      accuracy of its pick (raw and cleaned), search error,
                      regret, rank among all combinations, true redundancy
                      (share of the pick's images that are copies) and harmful
                      share (noisy / out-of-domain datasets picked)

Usage:
    python scripts/analyze.py --repo repos/starter --k 3
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repo import load_manifest, load_truth  # noqa: E402


def spearman_ci(x, y, n_boot=2000, seed=0):
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(i)) > 2:
            vals.append(spearmanr(x[i], y[i]).statistic)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--encoder", default="dinov2_vits14")
    p.add_argument("--metric", default="acc_cifar10", choices=["acc_cifar10", "acc_cifar101"])
    p.add_argument("--ground-truth", default=None,
                   help="ground_truth.jsonl from another run (e.g. the DINOv2 results) to grade "
                        "this encoder's selections; only its raw arm is used, because the "
                        "clean arm depends on the encoder that did the cleaning")
    a = p.parse_args()

    from scipy.stats import kendalltau, spearmanr

    d = Path(a.repo) / "results" / f"{a.encoder}_k{a.k}"
    sel = json.loads((d / "selections.json").read_text())["selectors"]
    scores, scores_plain, n_unique = {}, {}, {}
    if (d / "combo_scores.csv").exists():
        with open(d / "combo_scores.csv") as f:
            for r in csv.DictReader(f):
                scores[r["combo"]] = float(r["score"])
                n_unique[r["combo"]] = int(r["n_unique"])
                if r.get("score_no_dedupe"):
                    scores_plain[r["combo"]] = float(r["score_no_dedupe"])

    runs = defaultdict(list)
    gt = Path(a.ground_truth) if a.ground_truth else d / "ground_truth.jsonl"
    if a.ground_truth:
        print(f"ground truth from {gt} (raw arm only)")
    if gt.exists():
        for line in gt.read_text().splitlines():
            r = json.loads(line)
            if a.ground_truth and r["arm"] != "raw":
                continue
            if r.get(a.metric) is not None:
                runs[("+".join(r["combo"]), r["arm"])].append(r[a.metric])
    acc = {k: float(np.mean(v)) for k, v in runs.items()}
    sd = {k: float(np.std(v)) for k, v in runs.items()}
    raw = {c: v for (c, arm), v in acc.items() if arm == "raw"}

    manifest = {r["id"]: r for r in load_manifest(a.repo)}
    truth = load_truth(a.repo)

    def redundancy(combo):
        keys = np.concatenate([truth[x] for x in combo])
        return 1 - len(np.unique(keys)) / len(keys)

    def harmful(combo):
        return np.mean([manifest[x]["type"] in ("noisy", "ood") for x in combo])

    summary = {"metric": a.metric, "n_combos_scored": len(scores),
               "n_combos_trained_raw": len(raw)}
    print(f"metric: {a.metric}   combinations scored: {len(scores)}   trained (raw): {len(raw)}")

    seed_sd = np.mean([v for (c, arm), v in sd.items() if arm == "raw"]) if raw else float("nan")
    summary["training_seed_sd"] = float(seed_sd)

    common = [c for c in scores if c in raw]
    if len(common) >= 5:
        x = [scores[c] for c in common]
        y = [raw[c] for c in common]
        rho = spearmanr(x, y).statistic
        tau = kendalltau(x, y).statistic
        lo, hi = spearman_ci(x, y)
        summary.update(spearman=rho, spearman_ci=[lo, hi], kendall=tau)
        print(f"\nOT score vs accuracy over {len(common)} combinations (negative = good):")
        print(f"  Spearman {rho:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})   Kendall {tau:+.3f}")
        if scores_plain:
            xp = [scores_plain[c] for c in common]
            rho_p = spearmanr(xp, y).statistic
            summary["spearman_no_dedupe"] = rho_p
            print(f"  without duplicate merging: Spearman {rho_p:+.3f}")
        rho_n = spearmanr([n_unique[c] for c in common], y).statistic
        summary["spearman_unique_images"] = rho_n
        print(f"  unique images (after merging duplicates) vs accuracy: Spearman {rho_n:+.3f}"
              "  (positive = more distinct data helps)")
        print(f"  training noise (mean sd across seeds): {seed_sd:.4f}")

    best_true = max(raw, key=raw.get) if raw else None
    ot_best = "+".join(sorted(sel["exhaustive"]["chosen"])) if "exhaustive" in sel else None
    if best_true and ot_best in raw:
        proxy = raw[best_true] - raw[ot_best]
        summary.update(best_true=best_true, best_true_acc=raw[best_true], ot_best=ot_best,
                       ot_best_acc=raw[ot_best], proxy_error=proxy)
        print(f"\ntruly best combination : {best_true}  acc {raw[best_true]:.4f}")
        print(f"best OT score          : {ot_best}  acc {raw[ot_best]:.4f}")
        print(f"proxy error            : {proxy:+.4f}")

    ranked = sorted(raw, key=raw.get, reverse=True)
    rows = []
    for name, r in sel.items():
        picks = [r["chosen"]] if "chosen" in r else r.get("draws", [])
        if not picks:
            continue
        keys = ["+".join(sorted(c)) for c in picks]
        def mean_of(dct):
            v = [dct[k] for k in keys if k in dct]
            return float(np.mean(v)) if v else float("nan")
        a_raw = mean_of(raw)
        a_clean = mean_of({c: v for (c, arm), v in acc.items() if arm == "clean"})
        row = {
            "selector": name, "pick": keys[0] if len(keys) == 1 else f"{len(keys)} random draws",
            "score": r.get("score", float("nan")), "acc_raw": a_raw, "acc_clean": a_clean,
            "clean_gain": a_clean - a_raw,
            "search_error": (raw[ot_best] - a_raw) if ot_best in raw else float("nan"),
            "regret": (raw[best_true] - a_raw) if best_true else float("nan"),
            "rank": (float(np.mean([ranked.index(k) + 1 for k in keys if k in ranked]))
                     if any(k in ranked for k in keys) else float("nan")),
            "redundancy": float(np.mean([redundancy(c) for c in picks])),
            "harmful": float(np.mean([harmful(c) for c in picks])),
            "calls": r.get("calls", ""), "seconds": r.get("seconds", ""),
        }
        rows.append(row)
    summary["selectors"] = rows

    print(f"\n{'selector':20s} {'pick':34s} {'acc raw':>8s} {'clean':>7s} {'gain':>6s} "
          f"{'regret':>7s} {'search':>7s} {'rank':>5s} {'redund':>7s} {'harm':>5s}")
    for r in rows:
        print(f"{r['selector']:20s} {r['pick']:34s} {r['acc_raw']:8.4f} {r['acc_clean']:7.4f} "
              f"{r['clean_gain']:+6.3f} {r['regret']:7.4f} {r['search_error']:+7.4f} "
              f"{r['rank']:5.1f} {r['redundancy']:7.1%} {r['harmful']:5.0%}")
    print("\nregret = truly best accuracy - pick's accuracy (raw); search = best-OT combo's "
          "accuracy - pick's; rank 1 = truly best;\nredund = share of the pick's images that "
          "are copies (truth); harm = share of picked datasets that are noisy or out-of-domain")

    (d / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nsaved {d / 'summary.json'}")


if __name__ == "__main__":
    main()
