"""Step 4: sanity checks for the class-wise OT score and the LAVA point signal.

Builds a few candidate datasets and checks the scores behave as the project
assumes:

  clean_A, clean_B : two different clean CIFAR shards (their gap = noise floor)
  duplicate_A      : shard A pooled with itself (should equal clean_A)
  ood_mnist        : an MNIST shard (should be far worse)
  noisy40_A        : A's images, CIFAR-10N worst labels (~40% wrong)
  noisy9_A         : A's images, CIFAR-10N aggregate labels (~9% wrong)

Two dataset-level signals are reported for each candidate:

  distance          class-wise OT distance to the validation set (lower = better)
  suspicious share  share of points whose own transport cost (how expensive
                    the point is to match to the validation set) exceeds a
                    level only 5% of clean points reach. The level is set by
                    matching half the clean validation set to the other half;
                    candidates are matched to the same half at the same size.
                    Clean data ~5%; rises with label noise. Lower = better.

For the noisy candidates we know exactly which labels are wrong, so the script
also reports AUROC (0.5 = chance, 1.0 = perfect) for finding them:
  AUROC grad : LAVA's calibrated gradients (ranks points within one dataset)
  AUROC cost : per-point transport cost (the signal behind suspicious share)

--label-ref chooses how class-to-class label distances are measured:
  pair : candidate class vs validation class (LAVA's original definition)
  val  : validation class vs validation class (clean, fixed reference)
  both : run both and print them side by side (default)

Usage (from repo root, after src.embed):
    python scripts/ot_sanity.py --fake                    # synthetic data, no GPU
    python scripts/ot_sanity.py                           # DINOv2 (default)
    python scripts/ot_sanity.py --encoder resnet50        # ablation
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ot import (auroc, class_wise_ot, clean_reference,  # noqa: E402
                    point_signal, val_label_distance_matrix)

CANDIDATES = ["clean_A", "clean_B", "duplicate_A", "ood_mnist", "noisy40_A", "noisy9_A"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def fake_world(seed=0, d=64, n_pool=20000, n_val=3000, n_ood=5000):
    """Gaussian class clusters. Returns the same structure as real data."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 3, (10, d))
    def draw(n, mu):
        y = rng.integers(0, 10, n)
        return (mu[y] + rng.normal(0, 1, (n, d))).astype(np.float32), y
    Xp, yp = draw(n_pool, means)
    Xv, yv = draw(n_val, means)
    Xo, yo = draw(n_ood, rng.normal(0, 3, (10, d)) + 6)       # shifted domain
    def flip(y, frac):
        y = y.copy()
        m = rng.random(len(y)) < frac
        y[m] = (y[m] + rng.integers(1, 10, m.sum())) % 10
        return y
    return {"pool": (Xp, yp), "val": (Xv, yv), "ood": (Xo, yo),
            "noisy40": flip(yp, 0.40), "noisy9": flip(yp, 0.09)}


def real_world(encoder, emb_root, data_root):
    from src.data import cifar10_train_val_split
    from src.embed import emb_path, load_embeddings
    X, y, _ = load_embeddings(encoder, "cifar10_train", emb_root)
    pool_idx, val_idx = cifar10_train_val_split(len(y))
    w = {"pool": (X[pool_idx], y[pool_idx]), "val": (X[val_idx], y[val_idx])}
    Xo, yo, _ = load_embeddings(encoder, "mnist_train", emb_root)
    w["ood"] = (Xo, yo)
    for key, ds in [("noisy40", "cifar10n_worse"), ("noisy9", "cifar10n_aggre")]:
        if emb_path(encoder, ds, emb_root).exists():
            w[key] = load_embeddings(encoder, ds, emb_root)[1][pool_idx]
        else:
            print(f"({ds} embeddings missing; skipping {key}. "
                  f"Run: python -m src.embed --encoder {encoder} --datasets {ds})")
    return w


# --------------------------------------------------------------------------
# one variant
# --------------------------------------------------------------------------

def run_variant(w, label_ref, a):
    Xp, yp = w["pool"]
    Xv, yv = w["val"]
    res = {k: {"dist": [], "share": [], "auroc": [], "auroc_cost": [], "label": []}
           for k in CANDIDATES}

    for s in range(a.seeds):
        rng = np.random.default_rng(100 + s)
        perm = rng.permutation(len(yp))
        A, B = perm[:a.shard], perm[a.shard:2 * a.shard]
        o = rng.choice(len(w["ood"][1]), a.shard, replace=False)
        Wval = (val_label_distance_matrix(Xv, yv, seed=s) if label_ref == "val" else None)
        ref = clean_reference(Xv, yv, q=a.clean_q, seed=s, n_max=a.n_max,
                              label_weight=a.label_weight, label_ref=label_ref)

        def score(key, X, y, ids=None, wrong=None):
            r = class_wise_ot(X, y, Xv, yv, ids_t=ids, seed=s, n_max=a.n_max,
                              label_weight=a.label_weight, label_ref=label_ref,
                              label_matrix=Wval, return_gradients=wrong is not None)
            res[key]["dist"].append(r["distance"])
            res[key]["label"].append(r["label_part"] / r["distance"])
            ps = point_signal(X, y, Xv, yv, ref, ids_t=ids)
            res[key]["share"].append(ps["suspicious_share"])
            if wrong is not None:
                res[key]["auroc"].append(auroc(r["gradients"], wrong[r["train_indices"]]))
                res[key]["auroc_cost"].append(auroc(ps["point_costs"],
                                                    wrong[ps["train_indices"]]))

        # ids = pool indices, so the duplicated shard's repeats are recognized
        score("clean_A", Xp[A], yp[A], A)
        score("clean_B", Xp[B], yp[B], B)
        AA = np.concatenate([A, A])
        score("duplicate_A", Xp[AA], yp[AA], AA)
        score("ood_mnist", w["ood"][0][o], w["ood"][1][o])
        for key, nk in [("noisy40_A", "noisy40"), ("noisy9_A", "noisy9")]:
            if nk in w:
                score(key, Xp[A], w[nk][A], A, wrong=(w[nk][A] != yp[A]))
        print(f"  [{label_ref}] seed {s} done")
    return {k: v for k, v in res.items() if v["dist"]}


def summarize(res, label_ref, a):
    mean = {k: {m: (np.mean(v[m]) if v[m] else np.nan) for m in v} for k, v in res.items()}
    sd = {k: {m: (np.std(v[m]) if v[m] else np.nan) for m in v} for k, v in res.items()}
    floor_d = max(float(np.mean(np.abs(np.subtract(res["clean_A"]["dist"],
                                                   res["clean_B"]["dist"])))), 1e-9)
    floor_s = max(float(np.mean(np.abs(np.subtract(res["clean_A"]["share"],
                                                   res["clean_B"]["share"])))), 1e-4)

    print(f"\n=== label distances: {label_ref} "
          f"({'candidate vs validation classes' if label_ref == 'pair' else 'validation classes only'}) ===")
    print(f"{'candidate':13s} {'distance':>10s} {'+/- sd':>7s}  {'suspicious':>10s} {'+/- sd':>7s}"
          f"  {'AUROC grad':>10s} {'AUROC cost':>10s}")
    for k in res:
        ag = f"{mean[k]['auroc']:.3f}" if res[k]["auroc"] else "-"
        ac = f"{mean[k]['auroc_cost']:.3f}" if res[k]["auroc_cost"] else "-"
        print(f"{k:13s} {mean[k]['dist']:10.4f} {sd[k]['dist']:7.4f}  "
              f"{mean[k]['share']:10.2%} {sd[k]['share']:7.2%}  {ag:>10s} {ac:>10s}")
    print(f"noise floor: distance {floor_d:.4f}, suspicious share {floor_s:.2%}   "
          f"label term share (clean) {mean['clean_A']['label']:.1%}")

    cA = mean["clean_A"]
    checks = [("duplicate ~ clean (distance)", abs(mean["duplicate_A"]["dist"] - cA["dist"]) < floor_d + 1e-6,
               f"diff {mean['duplicate_A']['dist'] - cA['dist']:+.4f}"),
              ("ood worse (distance)", mean["ood_mnist"]["dist"] - cA["dist"] > 5 * floor_d,
               f"{(mean['ood_mnist']['dist'] - cA['dist']) / floor_d:.0f}x floor")]
    go = mean["ood_mnist"]["share"] - cA["share"]
    checks.append(("ood worse (suspicious)", go > 5 * floor_s, f"{go:+.2%} = {go / floor_s:.0f}x floor"))
    for k, lab in [("noisy40_A", "40%"), ("noisy9_A", "9%")]:
        if k in mean:
            gd = mean[k]["dist"] - cA["dist"]
            gs = mean[k]["share"] - cA["share"]
            checks.append((f"noisy {lab} worse (distance)", gd > 2 * floor_d, f"{gd / floor_d:.1f}x floor"))
            checks.append((f"noisy {lab} worse (suspicious)", gs > 2 * floor_s,
                           f"{gs:+.2%} = {gs / floor_s:.1f}x floor"))
    for name, ok, info in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} {info}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--encoder", default="dinov2_vits14")
    p.add_argument("--emb-root", default="emb")
    p.add_argument("--data-root", default="data")
    p.add_argument("--shard", type=int, default=4000, help="images per candidate shard")
    p.add_argument("--n-max", type=int, default=3000, help="OT subsample size per side")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--label-weight", type=float, default=1.0)
    p.add_argument("--label-ref", choices=["pair", "val", "both"], default="both")
    p.add_argument("--clean-q", type=float, default=0.95,
                   help="suspicious = point cost above the q-quantile of clean "
                        "validation points (so clean data scores about 1 - q)")
    a = p.parse_args()

    w = fake_world() if a.fake else real_world(a.encoder, a.emb_root, a.data_root)
    refs = ["pair", "val"] if a.label_ref == "both" else [a.label_ref]
    results = {r: run_variant(w, r, a) for r in refs}
    for r in refs:
        summarize(results[r], r, a)


if __name__ == "__main__":
    main()
