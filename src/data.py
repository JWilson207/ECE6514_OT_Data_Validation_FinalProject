"""Unified dataset loading (step 2).

Every dataset comes back in the same format:
    X   : uint8 array, shape (N, 32, 32, 3), RGB
    y   : int64 array, shape (N,), CIFAR-10 class indices 0-9
    ids : array of strings, shape (N,), stable image ids like "cifar10_train:123"
          (used later to trace overlap between datasets)

CIFAR-10 classes: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
                  5 dog, 6 frog, 7 horse, 8 ship, 9 truck

Auto-downloaded through torchvision: CIFAR-10, STL-10, SVHN, MNIST.
Manual downloads (put them under data/):
    data/CINIC-10/                 https://github.com/BayesWatch/cinic-10
    data/CIFAR-10_human.pt         https://github.com/UCSC-REAL/cifar-10-100n (data/ folder)
    data/cifar10.1_v6_data.npy     https://github.com/modestyachts/CIFAR-10.1 (datasets/ folder)
    data/cifar10.1_v6_labels.npy

Loaded datasets are cached to data/cache/<name>.npz so later loads are fast.

Usage:
    python -m src.data --download          # fetch the torchvision datasets
    python -m src.data --check             # load everything available, print sizes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA_ROOT = Path("data")
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]

# STL-10 order: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
# Map to CIFAR indices; monkey (7) has no CIFAR class and is dropped.
STL_TO_CIFAR = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 8: 8, 9: 9}

# Fixed held-out slice of CIFAR-10 *train* used as the OT validation set.
# The CIFAR-10 test set is never used for selection, only final accuracy.
N_VAL = 5000
VAL_SEED = 0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _resize_nhwc(X: np.ndarray, size: int = 32) -> np.ndarray:
    """Resize uint8 NHWC images to size x size with antialiasing."""
    if X.shape[1] == size and X.shape[2] == size:
        return X
    import torch
    import torch.nn.functional as F
    out = []
    for i in range(0, len(X), 2048):
        t = torch.from_numpy(X[i:i + 2048]).permute(0, 3, 1, 2).float()
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                          antialias=True, align_corners=False)
        out.append(t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy())
    return np.concatenate(out)


def _ids(prefix: str, n: int) -> np.ndarray:
    return np.array([f"{prefix}:{i}" for i in range(n)])


def _cache_path(name: str, root: Path) -> Path:
    return root / "cache" / f"{name}.npz"


# --------------------------------------------------------------------------
# individual loaders
# --------------------------------------------------------------------------

def load_cifar10(split: str, root: Path = DATA_ROOT, download: bool = False):
    from torchvision.datasets import CIFAR10
    ds = CIFAR10(root, train=(split == "train"), download=download)
    X = np.asarray(ds.data, dtype=np.uint8)                      # N,32,32,3
    y = np.asarray(ds.targets, dtype=np.int64)
    return X, y, _ids(f"cifar10_{split}", len(y))


def load_cifar10n_labels(key: str = "worse_label", root: Path = DATA_ROOT) -> np.ndarray:
    """CIFAR-10N noisy labels for the 50K CIFAR-10 *train* images (same order).

    keys: clean_label, aggre_label (~9% noise), random_label1/2/3 (~18%),
          worse_label (~40%)
    """
    f = root / "CIFAR-10_human.pt"
    if not f.exists():
        # fall back to labels cached by an earlier load_dataset("cifar10n_*")
        name = {"aggre_label": "cifar10n_aggre", "random_label1": "cifar10n_random1",
                "worse_label": "cifar10n_worse"}.get(key)
        cache = _cache_path(name, root) if name else None
        if cache is not None and cache.exists():
            return np.load(cache, allow_pickle=False)["y"]
        raise FileNotFoundError(f"{f} missing; download it from "
                                "https://github.com/UCSC-REAL/cifar-10-100n")
    import torch
    d = torch.load(f, weights_only=False)
    labels = np.asarray(d[key], dtype=np.int64)
    # sanity: the file's clean labels must match torchvision's train order
    _, y_clean, _ = load_cifar10("train", root)
    assert np.array_equal(np.asarray(d["clean_label"]), y_clean), \
        "CIFAR-10N label order does not match torchvision CIFAR-10 train order"
    return labels


def load_cifar10n(key: str = "worse_label", root: Path = DATA_ROOT):
    """CIFAR-10 train images with CIFAR-10N noisy labels. Same image ids as
    cifar10_train, so overlap with clean shards is traceable."""
    X, _, ids = load_cifar10("train", root)
    return X, load_cifar10n_labels(key, root), ids


def load_stl10(split: str = "train", root: Path = DATA_ROOT, download: bool = False):
    from torchvision.datasets import STL10
    ds = STL10(root, split=split, download=download)
    X = np.transpose(ds.data, (0, 2, 3, 1))                       # N,96,96,3
    y_stl = np.asarray(ds.labels, dtype=np.int64)
    keep = y_stl != 7                                             # drop monkey
    y = np.array([STL_TO_CIFAR[int(c)] for c in y_stl[keep]], dtype=np.int64)
    X = _resize_nhwc(np.ascontiguousarray(X[keep]))
    ids = _ids(f"stl10_{split}", len(y_stl))[keep]
    return X, y, ids


def load_svhn(split: str = "train", root: Path = DATA_ROOT, download: bool = False):
    """Out-of-domain source. Digit d is given CIFAR label d (deliberately wrong
    semantics: this dataset should be judged irrelevant)."""
    from torchvision.datasets import SVHN
    ds = SVHN(root, split=split, download=download)
    X = np.transpose(ds.data, (0, 2, 3, 1)).astype(np.uint8)
    y = np.asarray(ds.labels, dtype=np.int64)
    return np.ascontiguousarray(X), y, _ids(f"svhn_{split}", len(y))


def load_mnist(split: str = "train", root: Path = DATA_ROOT, download: bool = False):
    """Out-of-domain source: grayscale 28x28 -> RGB 32x32, digit d -> label d."""
    from torchvision.datasets import MNIST
    ds = MNIST(root, train=(split == "train"), download=download)
    X = np.asarray(ds.data, dtype=np.uint8)                       # N,28,28
    X = np.pad(X, ((0, 0), (2, 2), (2, 2)))                       # -> 32x32
    X = np.repeat(X[..., None], 3, axis=-1)
    y = np.asarray(ds.targets, dtype=np.int64)
    return X, y, _ids(f"mnist_{split}", len(y))


def load_cinic10_imagenet(split: str = "train", root: Path = DATA_ROOT):
    """The ImageNet-derived part of CINIC-10 (CIFAR-derived files are skipped).

    Returns X, y, ids, synsets. `synsets` (e.g. 'n02123045') lets you split the
    data into complementary sources by original ImageNet category.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image
    from tqdm import tqdm
    base = root / "CINIC-10" / split
    if not base.exists():
        raise FileNotFoundError(f"{base} missing; download CINIC-10 from "
                                "https://github.com/BayesWatch/cinic-10")
    files, y = [], []
    for c, cname in enumerate(CIFAR_CLASSES):
        names = sorted(n for n in os.listdir(base / cname)
                       if n.endswith(".png") and not n.startswith("cifar10"))
        files += [base / cname / n for n in names]
        y += [c] * len(names)

    def read(f):
        with Image.open(f) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)

    # Reading many small files is I/O-bound (slow on network filesystems),
    # so read them in parallel threads and show progress.
    with ThreadPoolExecutor(max_workers=32) as pool:
        X = list(tqdm(pool.map(read, files, chunksize=256), total=len(files),
                      desc=f"CINIC-10 {split}", unit="img"))
    stems = [f.stem for f in files]
    return (np.stack(X), np.asarray(y, dtype=np.int64),
            np.asarray([f"cinic10_{split}:{s}" for s in stems]),
            np.asarray([s.split("_")[0] for s in stems]))


def load_cifar101(root: Path = DATA_ROOT):
    """CIFAR-10.1 v6 (2,000 new test images). Evaluation only."""
    X = np.load(root / "cifar10.1_v6_data.npy").astype(np.uint8)
    y = np.load(root / "cifar10.1_v6_labels.npy").astype(np.int64)
    return X, y, _ids("cifar101", len(y))


# --------------------------------------------------------------------------
# unified entry point
# --------------------------------------------------------------------------

LOADERS = {
    "cifar10_train":        lambda r: load_cifar10("train", r),
    "cifar10_test":         lambda r: load_cifar10("test", r),
    "cifar10n_aggre":       lambda r: load_cifar10n("aggre_label", r),
    "cifar10n_random1":     lambda r: load_cifar10n("random_label1", r),
    "cifar10n_worse":       lambda r: load_cifar10n("worse_label", r),
    "stl10_train":          lambda r: load_stl10("train", r),
    "stl10_test":           lambda r: load_stl10("test", r),
    "svhn_train":           lambda r: load_svhn("train", r),
    "mnist_train":          lambda r: load_mnist("train", r),
    "cinic10_imagenet_train": lambda r: load_cinic10_imagenet("train", r)[:3],
    "cifar101":             lambda r: load_cifar101(r),
}


def load_dataset(name: str, root: Path = DATA_ROOT, use_cache: bool = True):
    """Load any dataset by name -> (X uint8 N,32,32,3; y int64; ids str)."""
    root = Path(root)
    cache = _cache_path(name, root)
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=False)
        return d["X"], d["y"], d["ids"]
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    X, y, ids = LOADERS[name](root)
    assert X.dtype == np.uint8 and X.shape[1:] == (32, 32, 3), (name, X.shape)
    assert y.min() >= 0 and y.max() <= 9, name
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, y=y, ids=ids)
    return X, y, ids


def cifar10_train_val_split(n_train_total: int = 50000, n_val: int = N_VAL,
                            seed: int = VAL_SEED):
    """Fixed split of CIFAR-10 train indices into (pool, val).

    val  -> the OT validation set (and early checks)
    pool -> everything candidate shards may be drawn from
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train_total)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DATA_ROOT))
    p.add_argument("--download", action="store_true",
                   help="download CIFAR-10, STL-10, SVHN, MNIST via torchvision")
    p.add_argument("--check", action="store_true",
                   help="load every available dataset and print its size")
    a = p.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)

    if a.download:
        load_cifar10("train", root, download=True)
        load_cifar10("test", root, download=True)
        load_stl10("train", root, download=True)
        load_svhn("train", root, download=True)
        load_mnist("train", root, download=True)
        print("torchvision datasets downloaded. Manual downloads still needed: "
              "CINIC-10, CIFAR-10_human.pt, CIFAR-10.1 (see module docstring).")

    if a.check:
        for name in LOADERS:
            try:
                X, y, _ = load_dataset(name, root)
                counts = np.bincount(y, minlength=10)
                print(f"{name:24s} {len(y):7d} images  per-class min/max "
                      f"{counts.min()}/{counts.max()}")
            except Exception as e:                                # noqa: BLE001
                print(f"{name:24s} not available ({type(e).__name__}: {e})")


if __name__ == "__main__":
    _main()
