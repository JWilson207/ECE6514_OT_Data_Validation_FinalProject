"""Step 3: time one ResNet-18 training run and project the cost of the
exhaustive k-of-n study.

Examples (run from the repo root):
    python scripts/time_one_run.py --fake --epochs 2           # smoke test, any machine
    python scripts/time_one_run.py --epochs 30                 # real timing on CIFAR-10
    python scripts/time_one_run.py --epochs 30 --n-train 15000 --n 12

The number that matters is "seconds per run" ON THE MACHINE YOU WILL TRAIN ON
(the NVIDIA server). Results are also saved to results/timing_<host>.json.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train import get_device, projected_hours, train_eval  # noqa: E402


def fake_data(n, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8),
            rng.integers(0, 10, n).astype(np.int64))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train", type=int, default=15000,
                   help="training images per run (about the size of a merged k-subset)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    p.add_argument("--n", type=int, default=12, help="repository size for projection")
    p.add_argument("--seeds", type=int, default=1, help="seeds per subset for projection")
    p.add_argument("--fake", action="store_true", help="random data, no downloads")
    p.add_argument("--data-root", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--labels", default="clean",
                   choices=["clean", "aggre", "random1", "worse"],
                   help="training labels: clean CIFAR-10, or CIFAR-10N aggregate "
                        "(~9%% wrong), random1 (~18%%), worse (~40%%). Same images "
                        "either way; test labels are always clean.")
    a = p.parse_args()

    if a.fake:
        Xtr, ytr = fake_data(a.n_train, a.seed)
        tests = {"fake_test": fake_data(2000, a.seed + 1)}
    else:
        from src.data import cifar10_train_val_split, load_dataset
        X, y, _ = load_dataset("cifar10_train", a.data_root)
        pool, _val = cifar10_train_val_split(len(y))
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(pool, size=a.n_train, replace=False)
        Xtr, ytr = X[idx], y[idx]
        if a.labels != "clean":
            from src.data import load_cifar10n_labels
            key = {"aggre": "aggre_label", "random1": "random_label1",
                   "worse": "worse_label"}[a.labels]
            ytr = load_cifar10n_labels(key, Path(a.data_root))[idx]
            print(f"labels: CIFAR-10N {a.labels} "
                  f"({(ytr != y[idx]).mean():.1%} of training labels wrong)")
        tests = {"cifar10_test": load_dataset("cifar10_test", a.data_root)[:2]}
        try:
            tests["cifar101"] = load_dataset("cifar101", a.data_root)[:2]
        except FileNotFoundError:
            print("(CIFAR-10.1 not found; skipping it)")

    device = get_device(a.device)
    print(f"device: {device}  |  {a.n_train} images, {a.epochs} epochs, "
          f"batch {a.batch_size}")
    r = train_eval(Xtr, ytr, tests, epochs=a.epochs, batch_size=a.batch_size,
                   lr=a.lr, seed=a.seed, device=device,
                   amp=False if a.no_amp else None)

    per_run = r["train_time_s"] + r["eval_time_s"]
    proj = projected_hours(per_run, a.n, seeds=a.seeds)
    r.update({"seconds_per_run": per_run, "projection": proj,
              "host": platform.node(), "fake": a.fake,
              "labels": a.labels, "seed": a.seed})

    print("\n=== result ===")
    print(f"GPU/CPU          : {r['device']}  (amp={r['amp']})")
    for k, v in r["acc"].items():
        print(f"accuracy {k:12s}: {v:.4f}")
    print(f"first epoch      : {r['first_epoch_s']:.1f}s   steady epoch: {r['steady_epoch_s']:.1f}s")
    print(f"seconds per run  : {per_run:.0f}s  ({per_run / 60:.1f} min)")
    for key, v in proj.items():
        print(f"exhaustive {key:10s}: {v['subsets']:4d} runs x {a.seeds} seed(s) "
              f"= {v['hours']:.1f} GPU-hours")

    tag = "_fake" if a.fake else f"_{a.labels}_s{a.seed}"
    out = Path("results") / f"timing_{platform.node()}{tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
