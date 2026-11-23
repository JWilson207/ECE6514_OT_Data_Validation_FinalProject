"""The selection pipeline's shared machinery: LAVA-style OT scoring of whole
combinations of datasets, near-duplicate detection, and cleaning.

    select : choose k of n datasets by the class-wise OT distance of the pooled
             combination to the clean validation set (lower = better). Near-
             duplicate images inside a combination are merged before scoring,
             so data the combination already has earns no credit.
    clean  : inside the chosen combination, keep one copy of each group of
             near-duplicates, and drop points that are unusually costly to
             match to clean data (likely mislabeled or off-task).
    train  : on the result (scripts/train_subsets.py).

RepoContext loads a built repository's embeddings once and caches the two
expensive per-repository computations under repos/<name>/cache/<encoder>/:
    dup_pairs_<features>.npz  every pair of images (across all datasets)
                     whose duplicate-detection features (pixel fingerprints by
                     default) have cosine >= 0.90, with that similarity and
                     the pair's embedding cosine
    point_costs.npz  every image's transport cost against a fixed clean
                     reference (independent of which datasets are combined)

Nothing here reads truth.npz: the method only sees images, labels and
embeddings, as it would with real downloads.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .dedupe import find_pairs, fingerprint, groups_from_pairs
from .ot import class_wise_ot, clean_reference
from .repo import dataset_ids, load_manifest, load_repo_dataset

PAIR_FLOOR = 0.90          # pairs are cached down to this similarity


def validation_set(encoder: str, emb_root="emb"):
    """Features and labels of the fixed 5,000-image CIFAR-10 validation slice."""
    from .data import cifar10_train_val_split
    d = np.load(Path(emb_root) / encoder / "cifar10_train.npz", allow_pickle=False)
    X, y = d["feats"], d["y"]
    _pool, val = cifar10_train_val_split(len(y))
    return X[val], y[val]


class RepoContext:
    def __init__(self, repo, encoder="dinov2_vits14", emb_root="emb",
                 dup_threshold=0.95, clean_q=0.95, n_max=3000,
                 label_weight=1.0, val=None, verbose=True,
                 dup_features="pixels", dup_min_emb=0.0):
        self.repo, self.encoder = Path(repo), encoder
        self.dup_threshold, self.clean_q, self.n_max = dup_threshold, clean_q, n_max
        self.label_weight, self.verbose = label_weight, verbose
        self.dup_features, self.dup_min_emb = dup_features, dup_min_emb
        self.ids = dataset_ids(repo)
        self.manifest = {r["id"]: r for r in load_manifest(repo)}
        feats, ys = [], []
        for did in self.ids:
            f, y, _ = load_repo_dataset(repo, did, "emb", encoder)
            feats.append(f.astype(np.float32))
            ys.append(y)
        self.sizes = np.array([len(y) for y in ys])
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.F = np.concatenate(feats)
        self.Y = np.concatenate(ys)
        self.owner = np.repeat(np.arange(len(self.ids)), self.sizes)
        self.Xv, self.yv = val if val is not None else validation_set(encoder, emb_root)
        self.cache = self.repo / "cache" / encoder
        self.cache.mkdir(parents=True, exist_ok=True)
        self._pairs = None
        self._costs = None
        self._ref = None

    # ---------------------------------------------------------------- helpers
    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def index(self, selection) -> np.ndarray:
        """Global indices of all images in the selected datasets."""
        return np.concatenate([np.arange(self.offsets[self.ids.index(d)],
                                         self.offsets[self.ids.index(d) + 1])
                               for d in selection])

    # ------------------------------------------------------------ duplicates
    def dup_vectors(self) -> np.ndarray:
        """Features used for duplicate detection: pixel fingerprints of the
        repository images ("pixels", default) or the embeddings ("emb")."""
        if self.dup_features == "emb":
            return self.F
        if self.dup_features == "pixels":
            return np.concatenate([fingerprint(load_repo_dataset(self.repo, d, "images")[0])
                                   for d in self.ids])
        raise ValueError(self.dup_features)

    def pairs(self):
        """(i, j, sim) over global indices with sim >= PAIR_FLOOR in the
        duplicate-detection features (cached). Pairs whose embedding cosine is
        below dup_min_emb are dropped (optional semantic sanity check)."""
        if self._pairs is None:
            f = self.cache / f"dup_pairs_{self.dup_features}.npz"
            if f.exists():
                d = np.load(f)
                i, j, sim, emb = d["i"], d["j"], d["sim"], d["emb_sim"]
            else:
                t = time.perf_counter()
                self.log(f"finding near-duplicate pairs among {len(self.F)} images "
                         f"({self.dup_features}) ...")
                i, j, sim = find_pairs(self.dup_vectors(), None, PAIR_FLOOR)
                En = self.F / np.maximum(np.linalg.norm(self.F, axis=1, keepdims=True), 1e-12)
                emb = np.einsum("nd,nd->n", En[i], En[j]).astype(np.float32)
                np.savez(f, i=i, j=j, sim=sim, emb_sim=emb)
                self.log(f"  {len(i)} pairs >= {PAIR_FLOOR} ({time.perf_counter() - t:.0f} s) -> {f}")
            keep = emb >= self.dup_min_emb
            self._pairs = (i[keep], j[keep], sim[keep])
        return self._pairs

    def groups(self, gidx: np.ndarray, threshold=None) -> np.ndarray:
        """Near-duplicate group label for each of the given global indices."""
        thr = self.dup_threshold if threshold is None else threshold
        i, j, s = self.pairs()
        pos = np.full(len(self.F), -1)
        pos[gidx] = np.arange(len(gidx))
        keep = (s >= thr) & (pos[i] >= 0) & (pos[j] >= 0)
        return groups_from_pairs(len(gidx), pos[i[keep]], pos[j[keep]])

    def detected_overlap(self, threshold=None) -> np.ndarray:
        """M[a, b] = share of dataset a's images that have a near-duplicate in
        dataset b (detected from embeddings, not from the truth)."""
        thr = self.dup_threshold if threshold is None else threshold
        i, j, s = self.pairs()
        keep = (s >= thr) & (self.owner[i] != self.owner[j])
        i, j = i[keep], j[keep]
        n = len(self.ids)
        pts = np.concatenate([i, j])
        other = np.concatenate([self.owner[j], self.owner[i]])
        key = np.unique(pts * n + other)                 # each (image, other dataset) once
        M = np.zeros((n, n))
        np.add.at(M, (self.owner[key // n], key % n), 1.0)
        return M / self.sizes[:, None]

    # ----------------------------------------------------------- point costs
    def reference(self):
        if self._ref is None:
            self._ref = clean_reference(self.Xv, self.yv, q=self.clean_q, seed=0,
                                        n_max=self.n_max, label_weight=self.label_weight,
                                        label_ref="val")
        return self._ref

    def point_costs(self) -> np.ndarray:
        """Every image's transport cost against the fixed clean reference
        (cached). Each dataset is matched in chunks of the reference's size,
        so costs are on the same scale as the clean threshold."""
        if self._costs is None:
            f = self.cache / f"point_costs_q{self.clean_q}_strat.npz"
            if f.exists():
                d = np.load(f)
                self._costs, thr = d["costs"], float(d["threshold"])
                self.reference()["threshold"] = thr
            else:
                ref = self.reference()
                T, n = ref["T"], ref["n"]
                costs = np.zeros(len(self.F))
                t = time.perf_counter()
                for k, did in enumerate(self.ids):
                    g = np.arange(self.offsets[k], self.offsets[k + 1])
                    rng = np.random.default_rng(k)
                    m = int(np.ceil(len(g) / n))
                    # class-stratified chunks, like the reference halves: a chunk
                    # with too many of one class would be forced to match some of
                    # them to other classes and look costly for no real reason
                    buckets = [[] for _ in range(m)]
                    for c in np.unique(self.Y[g]):
                        members = rng.permutation(g[self.Y[g] == c])
                        for b, part in enumerate(np.array_split(members, m)):
                            buckets[b].extend(part.tolist())
                    for chunk in (np.array(b) for b in buckets if b):
                        r = class_wise_ot(self.F[chunk], self.Y[chunk], self.Xv[T], self.yv[T],
                                          seed=0, n_max=n, return_point_costs=True, **ref["kw"])
                        costs[chunk[r["train_indices"]]] = r["point_costs"]
                    self.log(f"  point costs {did}: {np.mean(costs[g] > ref['threshold']):.1%} "
                             f"above clean threshold")
                np.savez(f, costs=costs, threshold=ref["threshold"])
                self.log(f"point costs done ({time.perf_counter() - t:.0f} s) -> {f}")
                self._costs = costs
        return self._costs

    # ----------------------------------------------------------------- score
    def score(self, selection, seed=0, dedupe=True) -> float:
        """Class-wise OT distance of the pooled selection to the validation set
        (lower = better). With dedupe, near-duplicates are merged into one
        point carrying their combined mass, so redundant data adds nothing."""
        g = self.index(selection)
        ids = self.groups(g) if dedupe else None
        return class_wise_ot(self.F[g], self.Y[g], self.Xv, self.yv, ids_t=ids, seed=seed,
                             n_max=self.n_max, label_weight=self.label_weight,
                             label_ref="pair")["distance"]

    # ----------------------------------------------------------------- clean
    def clean(self, selection, dedupe=True, drop_costly=True):
        """Global indices to keep after cleaning the pooled selection.

        dedupe      : keep one image per near-duplicate group (the one with
                      the lowest transport cost)
        drop_costly : drop images costlier than the clean threshold (about
                      5% of clean data; far more of mislabeled data)
        """
        g = self.index(selection)
        costs = self.point_costs()[g]
        keep = np.ones(len(g), dtype=bool)
        n_dup = n_costly = 0
        if dedupe:
            grp = self.groups(g)
            order = np.lexsort((costs, grp))              # by group, then cost
            first = np.ones(len(g), dtype=bool)
            first[order[1:]] = grp[order[1:]] != grp[order[:-1]]
            n_dup = int((~first).sum())
            keep &= first
        if drop_costly:
            costly = costs > self.reference()["threshold"]
            n_costly = int((costly & keep).sum())
            keep &= ~costly
        stats = {"n_in": int(len(g)), "removed_duplicates": n_dup,
                 "removed_costly": n_costly, "n_out": int(keep.sum())}
        return g[keep], stats

    def training_set(self, selection, arm="raw"):
        """Images and labels to train on: arm "raw" = the datasets as
        downloaded; "clean" = after dedupe + dropping costly points."""
        if arm == "raw":
            parts = [load_repo_dataset(self.repo, d, "images") for d in selection]
            return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), None
        if arm == "clean":
            keep, stats = self.clean(selection)
            X = np.concatenate([load_repo_dataset(self.repo, d, "images")[0] for d in selection])
            pos = np.full(len(self.F), -1)
            g = self.index(selection)
            pos[g] = np.arange(len(g))
            return X[pos[keep]], self.Y[keep], stats
        raise ValueError(arm)
