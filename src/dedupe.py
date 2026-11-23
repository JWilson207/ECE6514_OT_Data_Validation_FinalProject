"""Near-duplicate detection.

Real repositories do not share image ids: the same photo reappears under
another file name, often resized or re-compressed. Duplicates are therefore
found from the images themselves: two images whose feature vectors have cosine
similarity >= threshold are treated as the same image.

Which features: pixel fingerprints (fingerprint(); default for deduplication)
or embeddings. On 32x32 images, JPEG re-encoding barely changes the pixels but
changes DINOv2 embeddings a lot (DINOv2 sees the image enlarged to 224x224,
artifacts included): on the starter repository embeddings found under 20% of
the true copies even at cosine 0.90. Fingerprints are what practical
deduplication tools compare; embeddings measure meaning, not identity.

Used for
  * deduplication inside a selected combination of datasets (src/clean.py),
  * overlap-aware scoring: duplicates are merged into one point before OT
    (src/select.py),
  * the test-set leakage filter when building a repository (src/repo.py).

NumPy only by default; uses a CUDA GPU through PyTorch when available, which
is much faster for repositories with tens of thousands of images.
"""

from __future__ import annotations

import numpy as np


def fingerprint(X: np.ndarray, size: int = 16) -> np.ndarray:
    """Pixel fingerprint of uint8 images (N, H, W, 3): grayscale, average-pooled
    to size x size, mean-centered and L2-normalized. Robust to re-compression
    and brightness/contrast shifts; two copies of an image have cosine near 1."""
    X = np.asarray(X)
    n, h, w = X.shape[:3]
    out = np.empty((n, size * size), dtype=np.float32)
    for s in range(0, n, 8192):
        g = X[s:s + 8192].astype(np.float32).mean(-1)
        g = g.reshape(len(g), size, h // size, size, w // size).mean((2, 4)).reshape(len(g), -1)
        g -= g.mean(1, keepdims=True)
        out[s:s + 8192] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    return out


def _normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def _torch_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch, torch.device("cuda")
    except ImportError:
        pass
    return None, None


def find_pairs(A: np.ndarray, B: np.ndarray | None = None, threshold: float = 0.95,
               chunk: int = 1024, use_gpu: bool = True):
    """All pairs with cosine similarity >= threshold.

    A, B : features (N, D), (M, D). With B=None, pairs within A (i < j).
    Returns (i, j, sim): int64, int64, float32 arrays.
    """
    within = B is None
    An = _normalize(A)
    Bn = An if within else _normalize(B)
    torch, dev = _torch_device() if use_gpu else (None, None)
    if torch is not None:
        Bt = torch.from_numpy(Bn).to(dev)
    I, J, S = [], [], []
    for s in range(0, len(An), chunk):
        if torch is not None:
            sim = (torch.from_numpy(An[s:s + chunk]).to(dev) @ Bt.T).cpu().numpy()
        else:
            sim = An[s:s + chunk] @ Bn.T
        if within:                                   # keep i < j only
            rows = np.arange(s, s + len(sim))[:, None]
            sim[np.arange(len(Bn))[None, :] <= rows] = -1.0
        i, j = np.nonzero(sim >= threshold)
        I.append(i + s)
        J.append(j)
        S.append(sim[i, j])
    return (np.concatenate(I).astype(np.int64), np.concatenate(J).astype(np.int64),
            np.concatenate(S).astype(np.float32))


def groups_from_pairs(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected components (union-find): group label per point, 0..G-1.
    Points with no duplicate get their own group."""
    parent = np.arange(n)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(i.tolist(), j.tolist()):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(x) for x in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def near_matches(A: np.ndarray, B: np.ndarray, threshold: float = 0.95, **kw) -> np.ndarray:
    """Boolean mask over A: True where some row of B is a near-duplicate."""
    i, _, _ = find_pairs(A, B, threshold, **kw)
    mask = np.zeros(len(A), dtype=bool)
    mask[i] = True
    return mask
