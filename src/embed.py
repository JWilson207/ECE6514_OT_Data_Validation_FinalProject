"""Embed datasets once with a frozen pretrained encoder and save to disk.

Output: emb/<encoder>/<dataset>.npz with
    feats : float32 (N, D)
    y     : int64 (N,)
    ids   : str (N,)

Encoders:
    resnet50       torchvision ImageNet-1k weights (IMAGENET1K_V2), 2048-d.
                   Caveat: CINIC-10's ImageNet part overlaps this encoder's
                   training data. Fine for a first pass; compare with a second
                   encoder later.
    dinov2_vits14  self-supervised DINOv2 ViT-S/14 via torch.hub, 384-d.

CIFAR-10N datasets reuse the cifar10_train features (identical images) and
just swap in the noisy labels, so they cost nothing extra.

Usage (from repo root):
    python -m src.embed --encoder resnet50 --datasets cifar10_train mnist_train
    python -m src.embed --encoder resnet50 --datasets cifar10n_worse cifar10n_aggre
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_cifar10n_labels, load_dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CIFAR10N_KEYS = {"cifar10n_aggre": "aggre_label",
                 "cifar10n_random1": "random_label1",
                 "cifar10n_worse": "worse_label"}


def build_encoder(name: str, device):
    """Return (model, input_size). Model maps normalized images -> features."""
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Identity()
        size = 224
    elif name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        size = 224                                              # multiple of 14
    else:
        raise ValueError(f"unknown encoder {name}")
    return m.eval().to(device), size


@torch.no_grad()
def embed_array(X: np.ndarray, model, size: int, device, bs: int = 256) -> np.ndarray:
    """uint8 (N,32,32,3) -> float32 (N,D)."""
    use_amp = device.type == "cuda"
    out = []
    for s in range(0, len(X), bs):
        x = torch.from_numpy(X[s:s + bs]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        x = (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            f = model(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def emb_path(encoder: str, dataset: str, root: str | Path = "emb") -> Path:
    return Path(root) / encoder / f"{dataset}.npz"


def load_embeddings(encoder: str, dataset: str, root: str | Path = "emb"):
    d = np.load(emb_path(encoder, dataset, root), allow_pickle=False)
    return d["feats"], d["y"], d["ids"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default="resnet50")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="emb")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    model, size = None, None

    for name in a.datasets:
        out = emb_path(a.encoder, name, a.out)
        if out.exists() and not a.overwrite:
            print(f"{name}: exists, skipping ({out})")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        if name in CIFAR10N_KEYS:                      # reuse clean features
            base = emb_path(a.encoder, "cifar10_train", a.out)
            if not base.exists():
                raise SystemExit(f"embed cifar10_train first (needed for {name})")
            feats, _, ids = load_embeddings(a.encoder, "cifar10_train", a.out)
            y = load_cifar10n_labels(CIFAR10N_KEYS[name], Path(a.data_root))
            np.savez(out, feats=feats, y=y, ids=ids)
            print(f"{name}: reused cifar10_train features, noisy labels "
                  f"({(y != np.load(base)['y']).mean():.1%} differ)")
            continue

        if model is None:
            model, size = build_encoder(a.encoder, device)
            print(f"encoder {a.encoder} on {device}")
        X, y, ids = load_dataset(name, a.data_root)
        feats = embed_array(X, model, size, device, a.batch_size)
        np.savez(out, feats=feats, y=y, ids=ids)
        print(f"{name}: {feats.shape} -> {out}")


if __name__ == "__main__":
    main()
