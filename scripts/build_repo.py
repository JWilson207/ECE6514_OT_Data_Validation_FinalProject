"""Step 5: build a controlled dataset repository from a config.

Each candidate dataset is written as its own images with fresh ids, as if
downloaded separately; true provenance goes to truth.npz for grading only.
See src/repo.py for the dataset types and file layout.

The leakage filter (on by default) drops source images that are near-copies
of CIFAR-10 test or CIFAR-10.1 images, found with pixel fingerprints. It needs
no embeddings.

Usage (from repo root):
    python scripts/build_repo.py configs/starter.json
    python scripts/build_repo.py configs/full.json
    python -m src.embed --repo repos/full             # then embed the repository
    python scripts/build_repo.py configs/full.json --leakage-features emb \
        --leakage-encoder dinov2_vits14               # old embedding-based filter
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.repo import build_repo  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="JSON config, e.g. configs/starter.json")
    p.add_argument("--out", default="repos", help="parent folder for repositories")
    p.add_argument("--data-root", default="data")
    p.add_argument("--emb-root", default="emb")
    p.add_argument("--leakage-features", default="pixels", choices=["pixels", "emb", "none"],
                   help="how test-image copies are found: pixel fingerprints (default), "
                        "embeddings, or no leakage check")
    p.add_argument("--leakage-encoder", default=None,
                   help="encoder for --leakage-features emb (e.g. dinov2_vits14)")
    p.add_argument("--leakage-threshold", type=float, default=None,
                   help="cosine treated as a test-image copy (default 0.98 pixels, 0.95 emb)")
    a = p.parse_args()

    config = json.loads(Path(a.config).read_text())
    features = a.leakage_features
    if a.leakage_encoder and features == "pixels" and "--leakage-features" not in sys.argv:
        print("note: --leakage-encoder is only used with --leakage-features emb; "
              "using pixel fingerprints")
    out = build_repo(config, a.out, Path(a.data_root), a.leakage_encoder, a.emb_root,
                     a.leakage_threshold, leakage_features=features)
    print(f"\nrepository written to {out}")
    print(f"next: python -m src.embed --repo {out}")


if __name__ == "__main__":
    main()
