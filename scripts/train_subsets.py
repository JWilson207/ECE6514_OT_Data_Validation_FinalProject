"""Step 8: ground truth. Train ResNet-18 on every k-subset of a repository
(or only the selectors' picks), as downloaded ("raw") and after cleaning
("clean"), for several seeds.

Resumable: each finished run is appended to
    repos/<name>/results/<encoder>_k<k>/ground_truth.jsonl
and skipped on restart. Split the work across GPUs or jobs with --part i/N.

Usage (from repo root, after scripts/run_selection.py):
    python scripts/train_subsets.py --repo repos/starter --k 3
    python scripts/train_subsets.py --repo repos/full --k 3 --part 1/4   # job 1 of 4
    python scripts/train_subsets.py --repo repos/full --k 3 --only-selected --arms clean
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--encoder", default="dinov2_vits14", help="used for the clean arm")
    p.add_argument("--emb-root", default="emb")
    p.add_argument("--data-root", default="data")
    p.add_argument("--arms", nargs="+", default=["raw", "clean"], choices=["raw", "clean"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--only-selected", action="store_true",
                   help="train only the combinations some selector picked")
    p.add_argument("--part", default="1/1", help="i/N: run every N-th job starting at i")
    p.add_argument("--dup-threshold", type=float, default=0.95)
    p.add_argument("--clean-q", type=float, default=0.95)
    p.add_argument("--dup-features", default="pixels", choices=["pixels", "emb"])
    p.add_argument("--dup-min-emb", type=float, default=0.0)
    a = p.parse_args()

    from src.data import load_dataset
    from src.pipeline import RepoContext
    from src.repo import dataset_ids
    from src.train import train_eval

    out_dir = Path(a.repo) / "results" / f"{a.encoder}_k{a.k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ground_truth.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add(("+".join(r["combo"]), r["arm"], r["seed"]))

    ids = dataset_ids(a.repo)
    if a.only_selected:
        sel = json.loads((out_dir / "selections.json").read_text())["selectors"]
        combos = {tuple(sorted(r["chosen"])) for r in sel.values() if "chosen" in r}
        combos |= {tuple(sorted(d)) for d in sel.get("random", {}).get("draws", [])}
        combos = sorted(combos)
    else:
        combos = [tuple(sorted(c)) for c in itertools.combinations(ids, a.k)]

    jobs = [(c, arm, s) for c in combos for arm in a.arms for s in a.seeds]
    i, n = (int(x) for x in a.part.split("/"))
    jobs = jobs[i - 1::n]
    todo = [j for j in jobs if ("+".join(j[0]), j[1], j[2]) not in done]
    print(f"{len(combos)} combinations x {len(a.arms)} arms x {len(a.seeds)} seeds; "
          f"this part: {len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to go")
    if not todo:
        return

    ctx = (RepoContext(a.repo, a.encoder, a.emb_root, a.dup_threshold, a.clean_q, verbose=True,
                       dup_features=a.dup_features, dup_min_emb=a.dup_min_emb)
           if "clean" in a.arms else None)
    tests = {"cifar10_test": load_dataset("cifar10_test", a.data_root)[:2]}
    try:
        tests["cifar101"] = load_dataset("cifar101", a.data_root)[:2]
    except FileNotFoundError:
        print("(CIFAR-10.1 not found; skipping it)")

    for n_run, (combo, arm, seed) in enumerate(todo, 1):
        t = time.perf_counter()
        if arm == "raw":
            from src.repo import load_repo_dataset
            import numpy as np
            parts = [load_repo_dataset(a.repo, d, "images") for d in combo]
            X = np.concatenate([q[0] for q in parts])
            y = np.concatenate([q[1] for q in parts])
            stats = None
        else:
            X, y, stats = ctx.training_set(list(combo), "clean")
        r = train_eval(X, y, tests, epochs=a.epochs, seed=seed, verbose=False)
        rec = {"combo": list(combo), "arm": arm, "seed": seed, "n_train": int(len(y)),
               "acc_cifar10": r["acc"]["cifar10_test"],
               "acc_cifar101": r["acc"].get("cifar101"),
               "seconds": round(time.perf_counter() - t, 1), "cleaning": stats}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{n_run}/{len(todo)}] {'+'.join(combo):40s} {arm:5s} seed {seed}  "
              f"n={len(y):5d}  acc {rec['acc_cifar10']:.4f}  ({rec['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
