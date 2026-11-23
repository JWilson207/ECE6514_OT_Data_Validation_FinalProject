"""Class-wise optimal transport distance (LAVA-style), numpy + POT only.

Following LAVA (Just et al., ICLR 2023), the cost between a training point
(x, y) and a validation point (x', y') is

    C((x,y), (x',y')) = ||x - x'|| + label_weight * W(y, y')

where W(y, y') is the OT (Wasserstein-1) distance between the class-conditional
feature distributions {x : label y} in the training set and {x' : label y'} in
the validation set. The dataset distance is the OT cost under C. Lower distance
= training data closer to the validation task = expected better accuracy.

Two variants of the label distance W (label_ref):
    "pair" : W(y, y') between candidate class y and validation class y'
             (LAVA's original definition). Mislabeled points contaminate the
             candidate's own classes, which partly hides mild label noise.
    "val"  : W(y, y') between validation classes y and y' only. The reference
             is always clean and identical for every candidate.

Point-level signal (LAVA's actual valuation): calibrated gradients of the
distance w.r.t. each training point's mass. Positive = the point pushes the
data away from the task. suspicious_share() turns them into a dataset score.

Runs on CPU with POT (no PyTorch needed), so it can be tested anywhere.
Inputs are precomputed embeddings (see src/embed.py).
"""

from __future__ import annotations

import numpy as np
import ot


def stratified_subsample(y: np.ndarray, n_max: int | None, rng: np.random.Generator) -> np.ndarray:
    """Indices of a class-stratified subsample of size <= n_max (all if None)."""
    n = len(y)
    if n_max is None or n <= n_max:
        return np.arange(n)
    idx = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        m = max(1, int(round(n_max * len(cls) / n)))
        idx.append(rng.choice(cls, size=min(m, len(cls)), replace=False))
    return np.sort(np.concatenate(idx))


def label_distance_matrix(Xt, yt, Xv, yv, n_per_class: int = 300,
                          seed: int = 0, n_classes: int = 10) -> np.ndarray:
    """W[i, j] = exact OT distance between train class i and validation class j
    (Euclidean ground cost, uniform weights, <= n_per_class points per class).
    Classes absent from a set get +inf (they are never used in the cost)."""
    rng = np.random.default_rng(seed)

    def per_class(X, y):
        out = {}
        for c in range(n_classes):
            idx = np.flatnonzero(y == c)
            if len(idx) == 0:
                continue
            if len(idx) > n_per_class:
                idx = rng.choice(idx, n_per_class, replace=False)
            out[c] = X[idx]
        return out

    T, V = per_class(Xt, yt), per_class(Xv, yv)
    W = np.full((n_classes, n_classes), np.inf)
    for i, A in T.items():
        for j, B in V.items():
            M = ot.dist(A, B, metric="euclidean")
            W[i, j] = ot.emd2(ot.unif(len(A)), ot.unif(len(B)), M)
    return W


def val_label_distance_matrix(Xv, yv, n_per_class: int = 300, seed: int = 0,
                              n_classes: int = 10) -> np.ndarray:
    """W[i, j] = exact OT distance between validation class i and validation
    class j (one subsample per class, so W[i, i] = 0). Depends only on the
    clean validation set, so it is the same for every candidate."""
    rng = np.random.default_rng(seed)
    V = {}
    for c in range(n_classes):
        idx = np.flatnonzero(yv == c)
        if len(idx) == 0:
            continue
        if len(idx) > n_per_class:
            idx = rng.choice(idx, n_per_class, replace=False)
        V[c] = Xv[idx]
    W = np.full((n_classes, n_classes), np.inf)
    for i, A in V.items():
        W[i, i] = 0.0
        for j, B in V.items():
            if j > i:
                M = ot.dist(A, B, metric="euclidean")
                W[i, j] = W[j, i] = ot.emd2(ot.unif(len(A)), ot.unif(len(B)), M)
    return W


def suspicious_share(values: np.ndarray, threshold: float,
                     weights: np.ndarray | None = None) -> float:
    """Mass-weighted share of points whose value exceeds `threshold`."""
    v = np.asarray(values, dtype=float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=float)
    return float(w[v > threshold].sum() / w.sum())


def clean_reference(Xv, yv, q: float = 0.95, seed: int = 0, **kw) -> dict:
    """Fixed target and threshold for the point-level signal.

    Splits the clean validation set into class-stratified halves R and T,
    matches R to T, and records the q-quantile of R's per-point transport
    costs. Candidates are later matched to the same T at the same subsample
    size, so their per-point costs are on the same scale: a clean candidate has
    about (1 - q) of its points above the threshold, a noisy one more.
    kw are passed to class_wise_ot (label_weight, label_ref, n_max, ...).
    """
    rng = np.random.default_rng(seed + 12345)
    half = []
    for c in np.unique(yv):
        idx = rng.permutation(np.flatnonzero(yv == c))
        half.append(idx[: len(idx) // 2])
    R = np.sort(np.concatenate(half))
    T = np.setdiff1d(np.arange(len(yv)), R)
    n_ref = min(len(R), len(T), kw.pop("n_max", None) or len(R))
    kw = {k: v for k, v in kw.items() if k not in ("return_gradients", "return_point_costs")}
    r = class_wise_ot(Xv[R], yv[R], Xv[T], yv[T], seed=seed, n_max=n_ref,
                      return_point_costs=True, **kw)
    return {"T": T, "n": n_ref, "threshold": float(np.quantile(r["point_costs"], q)),
            "q": q, "kw": kw, "seed": seed}


def point_signal(Xt, yt, Xv, yv, ref: dict, ids_t=None) -> dict:
    """Score a candidate's points against the clean reference.

    Returns suspicious_share (mass share of points costlier than the clean
    threshold; about 1 - q for clean data), plus per-point costs and the
    indices (into Xt) they refer to.
    """
    T = ref["T"]
    r = class_wise_ot(Xt, yt, Xv[T], yv[T], ids_t=ids_t, seed=ref["seed"],
                      n_max=ref["n"], return_point_costs=True, **ref["kw"])
    return {"suspicious_share": suspicious_share(r["point_costs"], ref["threshold"],
                                                 r["point_mass"]),
            "point_costs": r["point_costs"], "train_indices": r["train_indices"]}


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Area under the ROC curve: probability a random positive scores higher
    than a random negative (0.5 = chance, 1.0 = perfect). Ties count half."""
    from scipy.stats import rankdata
    pos = np.asarray(positives, dtype=bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def collapse_duplicates(ids: np.ndarray, weights: np.ndarray | None = None):
    """Merge repeated image ids into one point each, summing their mass.

    Returns (keep, w): indices of the first occurrence of each id and the total
    mass of that id. A dataset pooled with a copy of itself collapses to the
    original points with the same relative weights, so its OT distance is
    unchanged, as theory says. Without this, subsampling a pool that contains
    repeats yields fewer distinct points and inflates the distance slightly.
    """
    w = np.ones(len(ids)) if weights is None else np.asarray(weights, dtype=float)
    uniq, keep, inv = np.unique(ids, return_index=True, return_inverse=True)
    mass = np.bincount(inv, weights=w, minlength=len(uniq))
    order = np.argsort(keep)                     # keep original order
    return keep[order], mass[order]


def class_wise_ot(Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray, yv: np.ndarray,
                  weights_t: np.ndarray | None = None,
                  ids_t: np.ndarray | None = None,
                  label_weight: float = 1.0, n_max: int | None = 3000,
                  n_per_class_label: int = 300, method: str = "emd",
                  reg: float = 0.05, seed: int = 0,
                  label_ref: str = "pair", label_matrix: np.ndarray | None = None,
                  return_gradients: bool = False, grad_reg: float = 0.05,
                  return_point_costs: bool = False) -> dict:
    """Class-wise OT distance between a training set and the validation set.

    Xt, Xv     : float features (N, D), (M, D)
    yt, yv     : int labels
    weights_t  : optional per-point mass for the training set (default uniform);
                 used later for per-dataset vs per-image weighting
    ids_t      : optional image ids for the training set; repeated ids are
                 merged into one point carrying their summed mass (recommended
                 whenever a pool can contain the same image twice)
    n_max      : subsample each side to at most this many points (None = all)
    method     : "emd" (exact; default, about 0.5 s at 2,000 x 2,000 points) or
                 "sinkhorn" (entropic; only worth it for much larger subsamples)
    reg        : entropic regularization, relative to the largest cost (sinkhorn only)
    label_ref  : "pair" (LAVA: candidate class vs validation class) or
                 "val" (validation class vs validation class; see module doc)
    label_matrix : optional precomputed 10x10 W (overrides label_ref); pass
                 val_label_distance_matrix(...) to avoid recomputing it
    return_gradients : also return LAVA's calibrated gradients for the
                 (subsampled) training points (positive = harmful) and the
                 dataset's suspicious_share
    grad_reg   : Sinkhorn regularization (relative) used for the gradients.
                 Gradients rank points WITHIN one dataset (LAVA); they are
                 relative to that dataset's mean, so not comparable across
                 datasets. Use point_signal() for that.
    return_point_costs : also return each training point's transport cost
                 (its matched cost per unit mass, absolute units) and mass

    Returns dict with distance, its feature/label parts, sizes and, optionally,
    gradients plus the subsample indices they refer to.
    """
    rng = np.random.default_rng(seed)
    base = np.arange(len(yt))
    if ids_t is not None:
        base, weights_t = collapse_duplicates(ids_t, weights_t)
        Xt, yt = Xt[base], yt[base]
    it = stratified_subsample(yt, n_max, rng)
    iv = stratified_subsample(yv, n_max, rng)
    Xt_, yt_, Xv_, yv_ = Xt[it], yt[it], Xv[iv], yv[iv]

    if label_matrix is not None:
        W = label_matrix
    elif label_ref == "val":
        W = val_label_distance_matrix(Xv, yv, n_per_class_label, seed)
    elif label_ref == "pair":
        W = label_distance_matrix(Xt_, yt_, Xv_, yv_, n_per_class_label, seed)
    else:
        raise ValueError(label_ref)
    M_feat = ot.dist(Xt_, Xv_, metric="euclidean")
    M_lab = W[yt_][:, yv_]
    C = M_feat + label_weight * M_lab

    a = ot.unif(len(it)) if weights_t is None else (weights_t[it] / weights_t[it].sum())
    b = ot.unif(len(iv))
    scale = C.max()

    if method == "emd":
        P = ot.emd(a, b, C, numItermax=1_000_000)
    elif method == "sinkhorn":
        P = ot.sinkhorn(a, b, C / scale, reg, numItermax=2000, stopThr=1e-6)
    else:
        raise ValueError(method)

    out = {
        "distance": float((P * C).sum()),
        "feature_part": float((P * M_feat).sum()),
        "label_part": float(label_weight * (P * M_lab).sum()),
        "n_train": int(len(it)),
        "n_val": int(len(iv)),
    }
    if return_point_costs:
        mass = P.sum(1)
        out["point_costs"] = (P * C).sum(1) / np.maximum(mass, 1e-300)
        out["point_mass"] = mass
        out["train_indices"] = base[it]
    if return_gradients:
        # LAVA computes gradients from the entropic (Sinkhorn) dual: exact-OT
        # duals are degenerate (blocky, non-unique), Sinkhorn's are smooth.
        _, glog = ot.sinkhorn(a, b, C / scale, grad_reg, numItermax=2000,
                              stopThr=1e-6, log=True)
        f = grad_reg * scale * np.log(np.maximum(glog["u"], 1e-300))  # dual potential
        N = len(f)
        out["gradients"] = f - (f.sum() - f) / (N - 1)      # LAVA eq. (3)
        out["train_indices"] = base[it]          # indices into the original Xt

    return out


def score_set(parts: list[tuple], val: tuple[np.ndarray, np.ndarray],
              weighting: str = "image", **kw) -> float:
    """OT distance of the pooled datasets in `parts` to the validation set.

    parts     : list of (features, labels) or (features, labels, ids), one per
                dataset in the set S. With ids, images shared between datasets
                are merged (see collapse_duplicates).
    weighting : "image"   -> every image gets equal mass (default)
                "dataset" -> every dataset gets equal total mass
    Lower is better.
    """
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    w = None
    if weighting == "dataset":
        w = np.concatenate([np.full(len(p[1]), 1.0 / len(p[1])) for p in parts])
    elif weighting != "image":
        raise ValueError(weighting)
    ids = (np.concatenate([p[2] for p in parts])
           if all(len(p) > 2 for p in parts) else None)
    return class_wise_ot(X, y, val[0], val[1], weights_t=w, ids_t=ids, **kw)["distance"]
