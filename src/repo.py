"""Build and load a controlled dataset repository (a stand-in for Hugging Face).

A repository is a collection of candidate datasets, each one something a
practitioner might download. Each dataset is stored as its OWN images with
FRESH ids ("<dataset>:<i>"), exactly like separate downloads: nothing in the
data tells the method which images are shared between datasets. The true
provenance of every image is kept separately in truth.npz and is used only to
grade the method (overlap, redundancy, harmful selections), never by it.

Dataset types (config "type"):
    clean          CIFAR-10 pool images, clean labels
    complementary  different-but-useful data: CINIC-10 (ImageNet) images split
                   into groups by original ImageNet category, or STL-10
    overlap        a fraction `frac` of a parent's images + new images, all
                   re-encoded (`transform`, default JPEG quality 75) the way
                   re-uploaded data usually is
    noisy          CIFAR-10 pool images with CIFAR-10N labels ("worse" ~40%
                   wrong, "aggre" ~9%, "random1" ~18%); with "same_images_as"
                   it reuses a parent's images (same pictures, worse labels)
    ood            out-of-domain: MNIST or SVHN (digit labels mean nothing here)

Images are drawn class-balanced and WITHOUT replacement across the whole
repository: datasets share images only where the config plants overlap. The
CIFAR-10 validation slice (the OT reference) is never drawn. Before drawing,
any source image that is a near-copy of a CIFAR-10 test or CIFAR-10.1 image is
excluded (leakage filter). By default copies are found with pixel fingerprints
(src/dedupe.py), which catch re-encoded copies that embeddings miss.

Layout of repos/<name>/:
    config.json                  the config used
    manifest.csv                 one row per dataset (type, source, size, planned
                                 and measured overlap, label noise, transform)
    datasets/<id>.npz            X uint8 (N,32,32,3), y int64, ids str
    truth.npz                    <id> -> "source:index" key per image (grading only)
    leakage.json                 how many source images the leakage filter removed
    emb/<encoder>/<id>.npz       written by `python -m src.embed --repo ...`
    cache/<encoder>/...          written by scripts/run_selection.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np

from .data import (DATA_ROOT, cifar10_train_val_split, load_cifar10n_labels,
                   load_dataset)

N_CLASSES = 10
NOISY_KEYS = {"worse": "worse_label", "aggre": "aggre_label", "random1": "random_label1"}


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------

class Source:
    """A pool of images one or more datasets are drawn from."""

    def __init__(self, name, X, y, keys, allowed, extra=None):
        self.name, self.X, self.y, self.keys = name, X, y, keys
        self.available = np.zeros(len(y), dtype=bool)
        self.available[allowed] = True
        self.extra = extra or {}

    def take(self, idx):
        assert self.available[idx].all(), f"{self.name}: image drawn twice"
        self.available[idx] = False


def _emb_feats(encoder, name, emb_root):
    f = Path(emb_root) / encoder / f"{name}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run python -m src.embed --encoder {encoder} "
                                f"--datasets {name}")
    return np.load(f, allow_pickle=False)["feats"]


def load_source(name, data_root=DATA_ROOT):
    """name: cifar | cinic | stl | mnist | svhn. Returns a Source and the list of
    embedding files (dataset names) that cover its images, in order."""
    if name == "cifar":
        X, y, _ = load_dataset("cifar10_train", data_root)
        pool, _val = cifar10_train_val_split(len(y))       # validation never drawn
        keys = np.array([f"cifar10_train:{i}" for i in range(len(y))])
        return Source(name, X, y, keys, pool), ["cifar10_train"]
    if name == "cinic":
        X, y, ids = load_dataset("cinic10_imagenet_train", data_root)
        syn = np.array([s.split(":", 1)[1].split("_")[0] for s in ids])
        return Source(name, X, y, ids, np.arange(len(y)), {"synset": syn}), ["cinic10_imagenet_train"]
    if name == "stl":
        Xa, ya, ia = load_dataset("stl10_train", data_root)
        Xb, yb, ib = load_dataset("stl10_test", data_root)
        X, y, keys = np.concatenate([Xa, Xb]), np.concatenate([ya, yb]), np.concatenate([ia, ib])
        return Source(name, X, y, keys, np.arange(len(y))), ["stl10_train", "stl10_test"]
    if name in ("mnist", "svhn"):
        X, y, ids = load_dataset(f"{name}_train", data_root)
        return Source(name, X, y, ids, np.arange(len(y))), [f"{name}_train"]
    raise ValueError(f"unknown source {name}")


def leakage_filter(src: Source, feats, threshold, test_feats):
    """Mark source images that nearly match a test image as unavailable.
    feats / test_feats: fingerprints or embeddings of the source and test images."""
    from .dedupe import near_matches
    assert len(feats) == len(src.y), f"{src.name}: features do not match images"
    cand = np.flatnonzero(src.available)
    hit = near_matches(feats[cand], test_feats, threshold)
    src.available[cand[hit]] = False
    return int(hit.sum()), int(len(cand))


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def _balanced_draw(src: Source, size, rng, mask=None, labels=None):
    """Draw `size` available images, as class-balanced as possible."""
    y = src.y if labels is None else labels
    ok = src.available if mask is None else (src.available & mask)
    classes = [c for c in range(N_CLASSES) if (ok & (y == c)).any()]
    quota = {c: size // len(classes) for c in classes}
    for c in classes[: size - sum(quota.values())]:
        quota[c] += 1
    idx = []
    for c in classes:
        pool = np.flatnonzero(ok & (y == c))
        if len(pool) < quota[c]:
            raise RuntimeError(f"{src.name}: only {len(pool)} images of class {c} left, "
                               f"need {quota[c]}; reduce sizes or overlap")
        idx.append(rng.choice(pool, quota[c], replace=False))
    idx = np.sort(np.concatenate(idx))
    src.take(idx)
    return idx


def _cinic_group_mask(src: Source, group, n_groups):
    """Images whose ImageNet category falls in `group`: within each class the
    categories are sorted and dealt round-robin into n_groups groups."""
    syn = src.extra["synset"]
    mask = np.zeros(len(syn), dtype=bool)
    for c in range(N_CLASSES):
        cats = np.unique(syn[src.y == c])
        mine = set(cats[group::n_groups])
        mask |= (src.y == c) & np.isin(syn, list(mine))
    return mask


def _transform(X, name):
    if name in (None, "none"):
        return X
    if name == "flip":
        return X[:, :, ::-1].copy()
    if name.startswith("jpeg"):
        from PIL import Image
        q = int(name[4:] or 75)
        out = np.empty_like(X)
        for i, img in enumerate(X):
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=q)
            out[i] = np.asarray(Image.open(buf).convert("RGB"))
        return out
    raise ValueError(f"unknown transform {name}")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_repo(config: dict, out_root="repos", data_root=DATA_ROOT,
               leakage_encoder: str | None = None, emb_root="emb",
               leakage_threshold: float | None = None, verbose: bool = True,
               leakage_features: str = "pixels") -> Path:
    """leakage_features: "pixels" (default; fingerprints, threshold 0.98),
    "emb" (embeddings of leakage_encoder, threshold 0.95) or "none"."""
    rng = np.random.default_rng(config.get("seed", 0))
    size = int(config.get("size", 5000))
    default_tf = config.get("overlap_transform", "jpeg75")
    out = Path(out_root) / config["name"]
    (out / "datasets").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))

    sources, emb_names = {}, {}
    for d in config["datasets"]:
        s = d["source"]
        if s not in sources:
            sources[s], emb_names[s] = load_source(s, data_root)

    leak = {}
    if leakage_threshold is None:
        leakage_threshold = 0.98 if leakage_features == "pixels" else 0.95
    if leakage_features == "pixels":
        from .dedupe import fingerprint
        tests = [load_dataset(n, data_root)[0] for n in ("cifar10_test", "cifar101")]
        test_feats = fingerprint(np.concatenate(tests))
        feats_of = {s: fingerprint(src.X) for s, src in sources.items()}
    elif leakage_features == "emb":
        if not leakage_encoder:
            raise ValueError("leakage_features='emb' needs leakage_encoder")
        test_feats = np.concatenate([_emb_feats(leakage_encoder, n, emb_root)
                                     for n in ("cifar10_test", "cifar101")])
        feats_of = {s: np.concatenate([_emb_feats(leakage_encoder, n, emb_root)
                                       for n in emb_names[s]]) for s in sources}
    elif leakage_features == "none":
        print("WARNING: no leakage check")
    else:
        raise ValueError(leakage_features)
    if leakage_features != "none":
        for s, src in sources.items():
            removed, total = leakage_filter(src, feats_of[s], leakage_threshold, test_feats)
            leak[s] = {"removed": removed, "checked": total}
            if verbose:
                print(f"leakage filter {s} ({leakage_features}): removed {removed} of {total} "
                      f"images (cosine >= {leakage_threshold} to a CIFAR-10 test / 10.1 image)")
    (out / "leakage.json").write_text(json.dumps(
        {"features": leakage_features, "encoder": leakage_encoder,
         "threshold": leakage_threshold, "sources": leak}, indent=2))

    drawn, truth, rows = {}, {}, []
    for d in config["datasets"]:
        did, typ, src = d["id"], d["type"], sources[d["source"]]
        n = int(d.get("size", size))
        tf = d.get("transform", default_tf if typ == "overlap" else "none")
        labels_clean = src.y
        if typ in ("clean", "ood"):
            idx = _balanced_draw(src, n, rng)
            y = src.y[idx]
        elif typ == "complementary":
            mask = (_cinic_group_mask(src, d["group"], d["n_groups"])
                    if d["source"] == "cinic" else None)
            idx = _balanced_draw(src, n, rng, mask)
            y = src.y[idx]
        elif typ == "overlap":
            p_idx = drawn[d["parent"]][0]
            n_old = int(round(d["frac"] * n))
            old = np.sort(rng.choice(p_idx, n_old, replace=False))
            new = _balanced_draw(src, n - n_old, rng)
            idx = np.sort(np.concatenate([old, new]))
            y = src.y[idx]
        elif typ == "noisy":
            noisy = load_cifar10n_labels(NOISY_KEYS[d["labels"]], Path(data_root))
            if "same_images_as" in d:
                idx = drawn[d["same_images_as"]][0]
            else:
                idx = _balanced_draw(src, n, rng)
            y = noisy[idx]
        else:
            raise ValueError(f"unknown type {typ}")

        X = _transform(src.X[idx], tf)
        perm = rng.permutation(len(idx))                  # no ordering hints
        idx, X, y = idx[perm], X[perm], y[perm]
        ids = np.array([f"{did}:{i}" for i in range(len(idx))])
        np.savez(out / "datasets" / f"{did}.npz", X=X, y=y.astype(np.int64), ids=ids)
        drawn[did] = (idx, src)
        truth[did] = np.array([f"{d['source']}:{src.keys[i]}" for i in idx])

        parent = d.get("parent", d.get("same_images_as", ""))
        measured = (float(np.isin(truth[did], truth[parent]).mean()) if parent else 0.0)
        rows.append({
            "id": did, "type": typ, "source": d["source"], "size": len(idx),
            "parent": parent, "planned_overlap": d.get("frac", 1.0 if "same_images_as" in d else 0.0),
            "measured_overlap": round(measured, 4),
            "labels": d.get("labels", "clean"),
            "label_noise": round(float((y != labels_clean[idx]).mean()), 4),
            "transform": tf, "classes": int(len(np.unique(y))),
            "group": d.get("group", ""),
        })
        if verbose:
            r = rows[-1]
            print(f"  {did:14s} {typ:13s} {r['size']:5d} imgs  overlap {r['measured_overlap']:.0%}"
                  f"  label noise {r['label_noise']:.1%}  transform {tf}")

    np.savez(out / "truth.npz", **truth)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print_overlap_matrix(out)
    return out


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------

def load_manifest(repo) -> list[dict]:
    with open(Path(repo) / "manifest.csv") as f:
        return list(csv.DictReader(f))


def dataset_ids(repo) -> list[str]:
    return [r["id"] for r in load_manifest(repo)]


def load_repo_dataset(repo, did, kind="images", encoder="dinov2_vits14"):
    """kind="images" -> (X uint8, y, ids); kind="emb" -> (feats, y, ids)."""
    repo = Path(repo)
    if kind == "images":
        d = np.load(repo / "datasets" / f"{did}.npz", allow_pickle=False)
        return d["X"], d["y"], d["ids"]
    if kind == "emb":
        f = repo / "emb" / encoder / f"{did}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run python -m src.embed --repo {repo} "
                                    f"--encoder {encoder}")
        d = np.load(f, allow_pickle=False)
        return d["feats"], d["y"], d["ids"]
    raise ValueError(kind)


def load_truth(repo) -> dict:
    """GRADING ONLY: provenance key of every image, per dataset."""
    d = np.load(Path(repo) / "truth.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def overlap_matrix(repo):
    """M[a][b] = share of dataset a's images that also appear in b (truth)."""
    t = load_truth(repo)
    names = dataset_ids(repo)
    return names, {a: {b: float(np.isin(t[a], t[b]).mean()) for b in names if b != a}
                   for a in names}


def print_overlap_matrix(repo):
    names, M = overlap_matrix(repo)
    pairs = [(a, b, v) for a in names for b, v in M[a].items() if v > 0]
    print("true overlap (share of A's images also in B):"
          + ("" if pairs else " none"))
    for a, b, v in pairs:
        print(f"  {a:14s} in {b:14s} {v:.1%}")
