# Code Documentation

Reference for every module and script in the repository: what it does, how to
run it, the functions it exposes, the files it reads and writes, and why it is
built the way it is. The README is the quick start. Results and decisions are
recorded in the project log (Claude Docs: "Project Log: Training-Free k-of-n
Dataset Selection").

**Project question:** can a training-free optimal transport (OT) score identify
which k-of-n combination of repository datasets trains the best model, and does
accounting for overlap between datasets matter?

## Contents

1. [Overview](#1-overview)
2. [Running on the server](#2-running-on-the-server)
3. [Conventions](#3-conventions)
4. [Files on disk](#4-files-on-disk)
5. [`src/data.py`: dataset loading](#5-srcdatapy-dataset-loading)
6. [`src/models.py`: ResNet-18](#6-srcmodelspy-resnet-18)
7. [`src/train.py`: training and evaluation](#7-srctrainpy-training-and-evaluation)
8. [`src/embed.py`: frozen-encoder features](#8-srcembedpy-frozen-encoder-features)
9. [`src/ot.py`: OT score and point-level signals](#9-srcotpy-ot-score-and-point-level-signals)
10. [`scripts/time_one_run.py`: timing and label-noise runs](#10-scriptstime_one_runpy-timing-and-label-noise-runs)
11. [`scripts/ot_sanity.py`: OT sanity checks](#11-scriptsot_sanitypy-ot-sanity-checks)
12. [The selection pipeline: repository, select, clean, train, analyze](#12-the-selection-pipeline)
13. [Testing status and known limitations](#13-testing-status-and-known-limitations)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Overview

```
repo/
  environment.yml          conda env "otproject" (Python 3.11 + PyTorch + requirements.txt)
  requirements.txt         numpy, scipy, pot, pillow, tqdm (PyTorch installed separately)
  src/
    data.py                load any dataset as (images uint8 N,32,32,3; labels 0-9; ids)
    models.py              ResNet-18 for 32x32 images
    train.py               train_eval(): train ResNet-18 on any set, report accuracy + time
    embed.py               frozen encoder (DINOv2 default) -> emb/<encoder>/<dataset>.npz
    ot.py                  class-wise OT distance, duplicate merging, point-level signals
    dedupe.py              near-duplicate detection from embeddings
    repo.py                build / load a controlled dataset repository
    pipeline.py            RepoContext: set score, duplicate groups, point costs, cleaning
    selectors.py           one-shot, greedy, MMR, dedup+one-shot, exhaustive, random
  scripts/
    time_one_run.py        one training run: timing, projection, clean vs noisy labels
    ot_sanity.py           checks the OT score on duplicate / OOD / noisy-label data
    build_repo.py          step 5: build a repository from a config
    run_selection.py       steps 6-7: every selector + cleaning + dedupe check
    train_subsets.py       step 8: ground truth, raw and cleaned, resumable
    analyze.py             correlation, proxy/search error, regret, redundancy
  configs/
    starter.json           6-dataset repository (end-to-end check)
    full.json              12-dataset repository (main study)
  docs/DOCUMENTATION.md    this file
```

How the pieces connect:

```
             downloads in data/
                    |
               src/data.py  ── load_dataset(name) -> images, labels, ids
                /         \
       src/train.py      src/embed.py
   (train ResNet-18)     (frozen encoder -> features in emb/)
          |                    |
          |               src/ot.py
          |         (OT distance + point signals on features)
          |                    |
 scripts/time_one_run.py   scripts/ot_sanity.py
   ground-truth accuracy     training-free score
```

The project compares two quantities for each candidate combination of datasets:

* **Ground truth:** test accuracy of a ResNet-18 actually trained on the merged
  datasets (`src/train.py`). About 20 s per run on an L40S.
* **Training-free score:** class-wise OT distance between the merged datasets'
  features and a clean validation set (`src/ot.py`). Lower is better.

The method is LAVA extended from one dataset to a combination of datasets:
**select** k of n datasets by the OT score of their pooled, deduplicated
images; **clean** the chosen combination (merge near-duplicates, drop points
that are unusually costly to match to clean data); **train** on the result.
Section 12 describes the pipeline.

Every command is run **from the repository root**.

---

## 2. Running on the server

Linux with an NVIDIA GPU. Run long jobs inside `tmux` (`tmux new -s ot`;
detach with Ctrl-b d; reattach with `tmux attach -t ot`). Expected values below
are the project's measured results on an NVIDIA L40S.

### 2.1 Environment (once)

```bash
nvidia-smi
conda env create -f environment.yml
conda activate otproject                 # every new shell, on every node
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Look for `True` and the GPU name. The prompt should show `(otproject)` only:
if a `(.venv)` or `(base)` is also shown, run `deactivate` / `conda activate
otproject` so a single environment is active. After `requirements.txt`
changes: `conda env update -f environment.yml`.

### 2.2 Smoke tests (about 3 minutes)

```bash
python scripts/ot_sanity.py --fake
python scripts/time_one_run.py --fake --epochs 2
```

* `ot_sanity.py --fake`: two tables ("pair" and "val"), every line `[PASS]`.
* `time_one_run.py --fake`: `device: cuda`, accuracy near 0.10 (random labels,
  so chance level is correct). A crash here is an install or code problem.

### 2.3 Data

```bash
python -m src.data --download
wget -P data https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/main/data/CIFAR-10_human.pt
wget -P data https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy
wget -P data https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy
mkdir -p data/CINIC-10
wget -O data/CINIC-10/CINIC-10.tar.gz https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
tar -xzf data/CINIC-10/CINIC-10.tar.gz -C data/CINIC-10 --wildcards '*train/*'
rm -f data/cache/cinic10_imagenet_train.npz
python -m src.data --check
```

Wait for `tar` to return (it prints nothing) before `--check`. If a `wget`
returns 404, open the file on GitHub, click "Raw" and copy that link.

Expected `--check` output:

| Dataset | Images | Per-class min / max |
| --- | --- | --- |
| cifar10_train | 50,000 | 5000 / 5000 |
| cifar10_test | 10,000 | 1000 / 1000 |
| cifar10n_aggre | 50,000 | 4349 / 5509 |
| cifar10n_random1 | 50,000 | 4445 / 5558 |
| cifar10n_worse | 50,000 | 4040 / 6053 |
| stl10_train / stl10_test | 4,500 / 7,200 | 0 / 500, 0 / 800 (no frogs in STL-10) |
| svhn_train | 73,257 | 4659 / 13861 |
| mnist_train | 60,000 | 5421 / 6742 |
| cinic10_imagenet_train | 70,000 | 7000 / 7000 (progress bar on first load) |
| cifar101 | 2,000 | 200 / 200 |

### 2.4 Training time and noise floor

```bash
python scripts/time_one_run.py --epochs 30 --seed 0
python scripts/time_one_run.py --epochs 30 --seed 1
```

Expected: CIFAR-10 test about 87.0–87.4%, CIFAR-10.1 about 77–78%, about
18–20 s per run, and the two seeds within about 0.4 points (the training noise
floor). Below about 70% test accuracy means something is broken.

### 2.5 Label-noise training test

```bash
for lab in clean aggre worse; do for s in 0 1; do
  python scripts/time_one_run.py --epochs 30 --labels $lab --seed $s
done; done
```

Expected: about 9.1% and 40.5% of training labels wrong for `aggre` and
`worse`; mean CIFAR-10 test about 87.2% (clean), 82.7% (9%), 67.8% (40%).

### 2.6 Embeddings and OT checks

```bash
python -m src.embed --datasets cifar10_train mnist_train cifar10n_worse cifar10n_aggre
python scripts/ot_sanity.py
```

The first command downloads DINOv2 through `torch.hub` (needs internet) and
embeds 110,000 images in a few minutes. The CIFAR-10N lines should report
40.2% and 9.0% of labels differing. The ResNet-50 ablation:

```bash
python -m src.embed --encoder resnet50 --datasets cifar10_train mnist_train cifar10n_worse cifar10n_aggre
python scripts/ot_sanity.py --encoder resnet50
```

Expected with DINOv2 (section 11 explains every column): every check passes in
both tables; "pair" distance gaps of about 411× (OOD), 52× (40% noise) and
7.7× (9% noise) the noise floor.

### 2.7 The selection pipeline (steps 5–8)

```bash

# starter repository: build, embed, select + clean, ground truth, analysis
python scripts/build_repo.py configs/starter.json
python -m src.embed --repo repos/starter
python scripts/run_selection.py --repo repos/starter --k 3 --workers 8 --dup-threshold 0.98
python scripts/train_subsets.py --repo repos/starter --k 3 --dup-threshold 0.98
python scripts/analyze.py --repo repos/starter --k 3
```

Then the same five commands with `configs/full.json` / `repos/full` for the
main study. `train_subsets.py` on the full repository is 220 combinations × 2
arms × 3 seeds ≈ 7 GPU-hours; split it with `--part 1/4` … `--part 4/4`.

What to look for: section 12.6.

---

## 3. Conventions

### Image and label format

Every loader returns the same triple:

| Name | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `X` | `uint8` | `(N, 32, 32, 3)` | RGB images, channels last, 0–255 |
| `y` | `int64` | `(N,)` | label in CIFAR-10 order, 0–9 |
| `ids` | `str` | `(N,)` | stable image id, e.g. `"cifar10_train:123"` |

CIFAR-10 label order: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog,
6 frog, 7 horse, 8 ship, 9 truck.

Image ids identify the same image across datasets. They let `src/ot.py` merge
repeated images and will let the benchmark measure overlap exactly.

### Label mapping for non-CIFAR datasets

* **STL-10:** "monkey" is dropped; the other nine classes are re-indexed to
  CIFAR order (`STL_TO_CIFAR`); images resized from 96×96 to 32×32.
* **SVHN, MNIST:** digit d keeps label d. Deliberately out-of-domain: the
  labels mean nothing for CIFAR, so a good score must reject them. MNIST is
  padded to 32×32 and repeated to 3 channels.
* **CINIC-10:** only the ImageNet-derived images are loaded (files whose names
  start with `cifar10` are CIFAR copies and are skipped).
* **CIFAR-10N:** the same images as CIFAR-10 train with human-annotated noisy
  labels; ids equal CIFAR-10 train ids.

### Splits

| Set | Size | Used for | Never used for |
| --- | --- | --- | --- |
| CIFAR-10 train, validation slice | 5,000 (fixed, seed 0) | the OT reference set | training |
| CIFAR-10 train, pool | 45,000 | building candidate datasets | OT reference |
| CIFAR-10 test | 10,000 | final accuracy | any selection |
| CIFAR-10.1 | 2,000 | second test set | any selection |

The split comes from `cifar10_train_val_split()`. **Do not change `N_VAL` or
`VAL_SEED`**: every score and embedding-based result depends on them.

---

## 4. Files on disk

`data/`, `emb/` and `results/` are git-ignored; each person regenerates them.

```
data/
  cifar-10-batches-py/, stl10_binary/, ...   torchvision downloads
  CINIC-10/train/<class>/*.png               manual download
  CIFAR-10_human.pt                          CIFAR-10N labels
  cifar10.1_v6_data.npy, cifar10.1_v6_labels.npy
  cache/<dataset>.npz                        load_dataset cache (keys X, y, ids)
emb/
  <encoder>/<dataset>.npz                    src.embed output (keys feats, y, ids)
results/
  timing_<host>_<labels>_s<seed>.json        time_one_run.py (real runs)
  timing_<host>_fake.json                    time_one_run.py --fake
```

**Cache invalidation:** `load_dataset` reuses `data/cache/<name>.npz` whenever
it exists. After changing a loader, or if a dataset was read while incomplete,
delete its cache file.

---

## 5. `src/data.py`: dataset loading

### Command line

```bash
python -m src.data --download      # CIFAR-10, STL-10, SVHN, MNIST via torchvision
python -m src.data --check         # load everything available, print counts
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--root` | `data` | folder for downloads and cache |
| `--download` | off | fetch the torchvision datasets |
| `--check` | off | load every dataset, print image counts and per-class min/max; missing ones are reported, not fatal |

### `load_dataset(name, root="data", use_cache=True) -> (X, y, ids)`

Loads any dataset by name, checks the format (uint8, 32×32×3, labels 0–9),
and caches it. Names: `cifar10_train`, `cifar10_test`, `cifar10n_aggre`
(~9% wrong), `cifar10n_random1` (~18%), `cifar10n_worse` (~40%),
`stl10_train`, `stl10_test`, `svhn_train`, `mnist_train`,
`cinic10_imagenet_train`, `cifar101`. Raises `KeyError` for unknown names and
`FileNotFoundError` for missing manual downloads.

### `cifar10_train_val_split(n_train_total=50000, n_val=5000, seed=0) -> (pool, val)`

Sorted index arrays into CIFAR-10 train: 45,000 pool indices and 5,000
validation indices. Identical on every machine.

### Individual loaders

| Function | Returns | Notes |
| --- | --- | --- |
| `load_cifar10(split, root, download=False)` | X, y, ids | `split` = `"train"` or `"test"` |
| `load_cifar10n_labels(key="worse_label", root)` | labels only | keys: `clean_label`, `aggre_label`, `random_label1/2/3`, `worse_label`. Asserts the file's clean labels match torchvision's order |
| `load_cifar10n(key, root)` | X, y, ids | CIFAR-10 train images with noisy labels |
| `load_stl10(split="train", root, download=False)` | X, y, ids | drops monkey, maps labels, resizes with antialiasing |
| `load_svhn(split="train", root, download=False)` | X, y, ids | digit d → label d |
| `load_mnist(split="train", root, download=False)` | X, y, ids | pad, gray → RGB, digit d → label d |
| `load_cinic10_imagenet(split="train", root)` | X, y, ids, synsets | reads PNGs in 32 threads with a progress bar; `synsets` (e.g. `n02123045`) is each image's ImageNet category, for splitting CINIC into sources |
| `load_cifar101(root)` | X, y, ids | evaluation only |

Constants: `DATA_ROOT`, `CIFAR_CLASSES`, `STL_TO_CIFAR`, `N_VAL = 5000`,
`VAL_SEED = 0`, `LOADERS` (name → loader).

---

## 6. `src/models.py`: ResNet-18

`ResNet18(num_classes=10)`: the CIFAR variant (about 11 M parameters). It
differs from the ImageNet version by a 3×3 stride-1 first convolution and no
max-pool, which suits 32×32 inputs. Four stages of two `BasicBlock`s (64, 128,
256, 512 channels), global average pooling, linear classifier.

---

## 7. `src/train.py`: training and evaluation

### `train_eval(X_train, y_train, test_sets, ...) -> dict`

Trains ResNet-18 from scratch and evaluates. The ground-truth study calls it
once per combination of datasets.

| Argument | Default | Meaning |
| --- | --- | --- |
| `X_train`, `y_train` | required | uint8 N×32×32×3 images and labels |
| `test_sets` | required | dict name → `(X, y)` |
| `epochs` | 30 | |
| `batch_size` | 256 | last partial batch dropped |
| `lr` | 0.1 | peak learning rate |
| `weight_decay` | 5e-4 | |
| `seed` | 0 | model init, shuffling, augmentation |
| `device` | `"auto"` | `cuda` if available, else `cpu` |
| `amp` | `None` | mixed precision; on for GPU, off for CPU |
| `verbose` | `True` | prints epoch 1, every 5th and the last |

Recipe: SGD, Nesterov momentum 0.9, one-cycle learning rate (25% warm-up),
cross-entropy, random crop (4-px padding) and horizontal flip, channels-last.

Returns: `acc` (test-set name → accuracy), `train_time_s`, `eval_time_s`,
`first_epoch_s`, `steady_epoch_s` (mean of epochs 2+), `n_train`, `epochs`,
`batch_size`, `amp`, `device` (GPU name).

**Design:** the training set is copied to the GPU once as uint8 (15,000 images
≈ 46 MB) and batches are sliced and augmented on the GPU (`gpu_augment`). No
`DataLoader`: for CIFAR-sized images, CPU loading would be the bottleneck.

Helpers: `get_device`, `device_name`, `gpu_augment(x, pad=4)`,
`evaluate(model, X_u8, y, amp, bs=1000)`, and
`projected_hours(seconds_per_run, n, ks=(2, 3), seeds=1)` (number of subsets
C(n, k) and GPU-hours).

---

## 8. `src/embed.py`: frozen-encoder features

Embeds each dataset once with a pretrained encoder that is never fine-tuned and
saves the features; all OT work reads these files.

```bash
python -m src.embed --datasets cifar10_train mnist_train cifar10n_worse cifar10n_aggre
python -m src.embed --encoder resnet50 --datasets cifar10_train
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--encoder` | `dinov2_vits14` | `dinov2_vits14` or `resnet50` |
| `--datasets` | required | any `load_dataset` names |
| `--data-root` | `data` | |
| `--out` | `emb` | output folder |
| `--batch-size` | 256 | |
| `--device` | `auto` | |
| `--overwrite` | off | re-embed even if the output exists |

| Encoder | Source | Features | Status |
| --- | --- | --- | --- |
| `dinov2_vits14` | `torch.hub` `facebookresearch/dinov2`, self-supervised ViT-S/14 | 384 | **default**: best on every OT check |
| `resnet50` | torchvision, ImageNet-1k `IMAGENET1K_V2` | 2048 | ablation: misses mild label noise |

Images are upsampled from 32×32 to 224×224 and normalized with ImageNet
statistics. Mixed precision on GPU.

**CIFAR-10N shortcut:** `cifar10n_*` datasets share `cifar10_train`'s images,
so their features are copied and only the labels swapped; the command prints
the share of labels that differ. Embed `cifar10_train` first (listing it first
in the same command is enough).

**Caveat:** both encoders were pretrained on data overlapping ImageNet, and
CINIC-10's extra images come from ImageNet.

API: `build_encoder(name, device) -> (model, input_size)`,
`embed_array(X, model, size, device, bs=256) -> feats`,
`emb_path(encoder, dataset, root="emb")`,
`load_embeddings(encoder, dataset, root="emb") -> (feats, y, ids)`.

---

## 9. `src/ot.py`: OT score and point-level signals

NumPy + POT on CPU (no PyTorch), so it can be tested without a GPU.

### 9.1 The dataset score

Following LAVA (Just et al., ICLR 2023), matching training point (x, y) to
validation point (x′, y′) costs

```
C((x, y), (x', y')) = ||x - x'|| + label_weight * W(y, y')
```

`||x - x'||` is the Euclidean distance between features; `W(y, y')` is the
exact OT distance between the features of class y and class y′. The score is
the minimum total cost of transporting the candidate's mass onto the
validation set's. **Lower is better.**

Properties the project relies on:

* **Duplicates add nothing** (after merging repeated ids; see 9.3).
* **It is a set function:** pooling datasets and re-scoring captures overlap
  and complementarity.

### 9.2 `class_wise_ot(Xt, yt, Xv, yv, ...) -> dict`

| Argument | Default | Meaning |
| --- | --- | --- |
| `Xt`, `yt` | required | candidate features (N, D) and labels |
| `Xv`, `yv` | required | validation features (M, D) and labels |
| `weights_t` | `None` | per-point mass (uniform if `None`) |
| `ids_t` | `None` | image ids; repeated ids are merged into one point with summed mass. **Pass them whenever datasets are pooled** |
| `label_weight` | 1.0 | λ; 0 = plain feature OT |
| `n_max` | 3000 | class-stratified subsample per side; `None` = all |
| `n_per_class_label` | 300 | points per class for each label distance |
| `method` | `"emd"` | exact network simplex, or `"sinkhorn"` |
| `reg` | 0.05 | Sinkhorn regularization relative to the largest cost (`sinkhorn` only) |
| `seed` | 0 | subsampling seed |
| `label_ref` | `"pair"` | how W is measured (9.4) |
| `label_matrix` | `None` | precomputed 10×10 W; overrides `label_ref` |
| `return_point_costs` | `False` | also return per-point transport costs |
| `return_gradients` | `False` | also return LAVA calibrated gradients |
| `grad_reg` | 0.05 | Sinkhorn regularization for the gradients |

Returned keys:

| Key | Meaning |
| --- | --- |
| `distance` | the score |
| `feature_part`, `label_part` | its two components (`label_part / distance` = label term share) |
| `n_train`, `n_val` | points used after merging and subsampling |
| `point_costs`, `point_mass` | (if requested) each training point's matched cost per unit mass, in absolute units, and its mass |
| `gradients` | (if requested) LAVA calibrated gradient per point; positive = pushes the data away from the task |
| `train_indices` | (with either option) indices into the original `Xt` that the per-point values refer to |

**Solver speed** (2,000 × 2,000 points, 64-d, 2-core CPU): exact `emd` 0.5 s
(true value); Sinkhorn reg 0.05 1.2 s (biased upward). Exact is the default.
Cost grows roughly with the square of `n_max`.

### 9.3 Duplicates: `collapse_duplicates(ids, weights=None) -> (keep, mass)`

Merges repeated ids into one point carrying their summed mass. Without it,
subsampling a pool that contains each image twice yields fewer distinct points
and inflates the distance (+0.12 against a 0.096 noise floor on ResNet-50
features). With it, a dataset pooled with a copy of itself scores exactly the
same as the original.

### 9.4 Label-distance reference

| `label_ref` | W(y, y′) between | Use |
| --- | --- | --- |
| `"pair"` | candidate class y and validation class y′ (LAVA) | **dataset distance** (default) |
| `"val"` | validation class y and validation class y′; W(y, y) = 0 | **suspicious share**: a fixed clean reference, identical for every candidate |

`val_label_distance_matrix(Xv, yv, n_per_class=300, seed=0)` computes the
`"val"` matrix; pass it as `label_matrix` to avoid recomputing.
`label_distance_matrix(Xt, yt, Xv, yv, ...)` computes the `"pair"` matrix
(classes missing from a set get `inf` and are never used).

### 9.5 Point-level signals

Two per-point signals, with different uses:

| Signal | Units | Use |
| --- | --- | --- |
| LAVA calibrated gradient (`return_gradients`) | relative: f_i minus the mean of the other points' dual potentials | rank points **within** one dataset |
| per-point transport cost (`return_point_costs`) | absolute cost units | compare datasets, via the suspicious share |

Gradients cannot compare datasets: in a noisy dataset the mislabeled points
raise the mean and shift every other point's gradient down, so clean datasets
can look worse than noisy ones. Gradients are computed from the Sinkhorn dual,
as in LAVA (exact-OT duals are degenerate).

**Suspicious share** (secondary diagnostic, not used for selection):

* `clean_reference(Xv, yv, q=0.95, seed=0, **kw) -> ref`: splits the clean
  validation set into class-stratified halves R and T, matches R to T, and
  stores T, the subsample size and the q-quantile of R's per-point costs.
  Call once per seed and setting; `kw` go to `class_wise_ot` (`label_ref`,
  `label_weight`, `n_max`).
* `point_signal(Xt, yt, Xv, yv, ref, ids_t=None) -> dict`: matches a candidate
  to the same T at the same size; returns `suspicious_share` (mass share of
  points costlier than the threshold; clean data about 5–7%), `point_costs`
  and `train_indices`.
* `suspicious_share(values, threshold, weights=None)`: the underlying share.

### 9.6 `score_set(parts, val, weighting="image", **kw) -> float`

Scores a combination of datasets: pools them and returns `class_wise_ot`'s
distance. `parts` = list of `(features, labels)` or `(features, labels, ids)`;
with ids on every part, shared images are merged. `weighting="image"` gives
every image equal mass; `"dataset"` gives every dataset equal total mass.

### 9.7 Other helpers

`stratified_subsample(y, n_max, rng)`: class-proportional subsample indices.
`auroc(scores, positives)`: probability a random positive outscores a random
negative (0.5 = chance, 1.0 = perfect).

---

## 10. `scripts/time_one_run.py`: timing and label-noise runs

Trains one ResNet-18 on 15,000 CIFAR-10 pool images (about the size of a
merged 3-dataset combination), reports accuracy and time, and projects the
cost of the exhaustive study.

```bash
python scripts/time_one_run.py --fake --epochs 2
python scripts/time_one_run.py --epochs 30 --seed 1
python scripts/time_one_run.py --epochs 30 --labels aggre
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n-train` | 15000 | training images |
| `--epochs` | 30 | |
| `--batch-size` | 256 | |
| `--lr` | 0.1 | |
| `--device` | `auto` | |
| `--no-amp` | off | disable mixed precision |
| `--n` | 12 | repository size for the projection |
| `--seeds` | 1 | seeds per subset for the projection |
| `--fake` | off | random images and labels |
| `--data-root` | `data` | |
| `--seed` | 0 | |
| `--labels` | `clean` | `clean`, `aggre` (~9% wrong), `random1` (~18%), `worse` (~40%): same images, CIFAR-10N labels; test labels always clean |

Output: device, accuracy per test set (CIFAR-10 test, and CIFAR-10.1 if
present), first and steady epoch time, seconds per run, and GPU-hours for all
C(n, k) subsets at k = 2 and 3. With `--labels` other than `clean` it first
prints the share of training labels that are wrong. Results save to
`results/timing_<host>_<labels>_s<seed>.json`.

---

## 11. `scripts/ot_sanity.py`: OT sanity checks

Builds a few candidate datasets and checks that the OT distance and the
suspicious share rank them as the project assumes.

```bash
python scripts/ot_sanity.py --fake                    # synthetic, CPU only
python scripts/ot_sanity.py                           # DINOv2 embeddings
python scripts/ot_sanity.py --encoder resnet50        # ablation
python scripts/ot_sanity.py --label-ref pair --label-weight 3
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--fake` | off | Gaussian class clusters instead of embeddings |
| `--encoder` | `dinov2_vits14` | which embeddings to read |
| `--emb-root`, `--data-root` | `emb`, `data` | |
| `--shard` | 4000 | images per candidate |
| `--n-max` | 3000 | OT subsample per side |
| `--seeds` | 3 | repetitions with different shards and subsamples |
| `--label-weight` | 1.0 | λ |
| `--label-ref` | `both` | `pair`, `val` or `both` (one table each) |
| `--clean-q` | 0.95 | suspicious threshold = this quantile of clean points' costs |

Real mode needs `emb/<encoder>/cifar10_train.npz` and `mnist_train.npz`, and
optionally `cifar10n_worse.npz` and `cifar10n_aggre.npz`.

### Candidates

| Name | Built from | Expected |
| --- | --- | --- |
| `clean_A`, `clean_B` | two disjoint 4,000-image CIFAR pool shards | low; their gap is the noise floor |
| `duplicate_A` | shard A pooled with itself (ids passed) | identical to `clean_A` |
| `ood_mnist` | an MNIST shard | far worse |
| `noisy40_A` | A's images, CIFAR-10N worst labels | worse |
| `noisy9_A` | A's images, CIFAR-10N aggregate labels | worse |

### Output columns

| Column | Meaning |
| --- | --- |
| distance, +/- sd | mean and spread over seeds of the OT distance (lower = better) |
| suspicious, +/- sd | suspicious share (lower = better; clean about 5–7%) |
| AUROC grad | (noisy only) how well LAVA gradients rank the actually-wrong labels above correct ones |
| AUROC cost | (noisy only) the same for per-point costs |
| noise floor | mean \|A − B\| between the two clean shards, for distance and for share |
| label term share | `label_part / distance` for clean A |

### Checks

"N× floor" = the gap from clean A divided by the noise floor. Thresholds are
the project's own conventions.

| Check | Passes when |
| --- | --- |
| duplicate ~ clean (distance) | gap < 1× floor |
| ood worse (distance), (suspicious) | gap > 5× floor |
| noisy 40% worse (distance), (suspicious) | gap > 2× floor |
| noisy 9% worse (distance), (suspicious) | gap > 2× floor |

### Reference results (L40S, 3 seeds)

| Check | ResNet-50 pair distance | DINOv2 pair distance | DINOv2 val suspicious |
| --- | --- | --- | --- |
| Duplicate = clean | 0.000 | 0.000 | pass |
| OOD worse | 135× | 411× | 151× |
| 40% noise worse | 11× | 52× | 29× |
| 9% noise worse | 1.5× (fail) | 7.7× | 3.4× |

---

## 12. The selection pipeline

LAVA scores the points of one training set. This pipeline extends it to
choosing **which k of n candidate datasets** to download and combine, and then
cleans the result. It runs in five steps:

```
configs/*.json ─► build_repo.py ─► src.embed --repo ─► run_selection.py ─► train_subsets.py ─► analyze.py
                  (repository)      (features)         (select + clean)     (ground truth)       (metrics)
```

### 12.1 `src/repo.py` and `scripts/build_repo.py`: the repository

A stand-in for Hugging Face whose structure is known, so the method can be
graded. Each candidate dataset is written as **its own images with fresh ids**
(`<dataset>:<i>`), as if downloaded separately: nothing tells the method which
images are shared. True provenance goes to `truth.npz` and is used only for
grading.

| Type | Built from | Tests whether the method… |
| --- | --- | --- |
| `clean` | CIFAR-10 pool images | (reference: good data) |
| `complementary` | CINIC-10 ImageNet images split by ImageNet category (`group` of `n_groups`), or STL-10 | values different-but-useful data |
| `overlap` | `frac` of a parent's images + new images, all re-encoded (`transform`, default `jpeg75`) | credits only what is new |
| `noisy` | CIFAR-10 pool images with CIFAR-10N labels (`labels`: `worse` ~40%, `aggre` ~9%, `random1` ~18%); `same_images_as` reuses a parent's images | avoids mislabeled data |
| `ood` | MNIST or SVHN | rejects irrelevant data |

Rules: class-balanced draws; no image drawn twice except where `overlap` or
`same_images_as` plants it; the CIFAR-10 validation slice is never drawn;
source images that are near-copies of a CIFAR-10 test or CIFAR-10.1 image are
excluded first (leakage filter). The filter compares pixel fingerprints at
cosine ≥ 0.98 by default (it caught 299 of 300 planted JPEG test copies in a
synthetic test); `--leakage-features emb` uses embeddings instead (cosine
0.95; misses re-encoded copies).

| Flag (`build_repo.py`) | Default | Meaning |
| --- | --- | --- |
| `config` | required | JSON config (`configs/starter.json`, `configs/full.json`) |
| `--out` | `repos` | parent folder |
| `--data-root`, `--emb-root` | `data`, `emb` | |
| `--leakage-features` | `pixels` | `pixels` (fingerprints), `emb` (embeddings) or `none` |
| `--leakage-encoder` | none | encoder for `--leakage-features emb` |
| `--leakage-threshold` | 0.98 pixels, 0.95 emb | cosine treated as a test-image copy |

Config keys: `name`, `seed`, `size` (images per dataset, 5,000), 
`overlap_transform`, and `datasets` (each with `id`, `type`, `source` and the
type's options). Parents must come before the datasets that reference them.

Output in `repos/<name>/`: `config.json`, `manifest.csv` (type, source, size,
parent, planned and measured overlap, label noise, transform), `datasets/<id>.npz`
(`X`, `y`, `ids`), `truth.npz`, `leakage.json`. The build prints every dataset
and the true overlap matrix.

API: `build_repo(config, out_root, data_root, leakage_encoder, emb_root,
leakage_threshold)`, `load_manifest(repo)`, `dataset_ids(repo)`,
`load_repo_dataset(repo, id, kind="images" | "emb", encoder)`,
`load_truth(repo)` (grading only), `overlap_matrix(repo)`.

`python -m src.embed --repo repos/<name>` embeds every dataset into
`repos/<name>/emb/<encoder>/<id>.npz`.

### 12.2 `src/dedupe.py`: near-duplicates

Two images count as the same when their feature vectors have cosine similarity
≥ a threshold. **The features are pixel fingerprints by default**
(`fingerprint(X, size=16)`: grayscale, 16×16 average-pooled, mean-centered,
L2-normalized). On the starter repository, DINOv2 embeddings found under 20%
of the JPEG-re-encoded copies even at cosine 0.90: DINOv2 sees each 32×32
image enlarged to 224×224, compression artifacts included. In a test on
natural-image thumbnails, fingerprints matched 99.6% of JPEG-75 copies at
cosine 0.95. Embeddings measure what an image shows; fingerprints measure
whether it is the same picture. Uses a CUDA GPU through PyTorch when
available, else NumPy in chunks.

* `find_pairs(A, B=None, threshold=0.95)`: all pairs above the threshold
  (within A when B is `None`) as `(i, j, sim)`.
* `groups_from_pairs(n, i, j)`: union-find group label per point.
* `near_matches(A, B, threshold)`: mask of A's rows with a near-duplicate in B.

### 12.3 `src/pipeline.py`: `RepoContext`

Loads a repository's embeddings and the validation set once, and provides:

| Method | What it does |
| --- | --- |
| `score(selection, seed, dedupe=True)` | class-wise OT distance ("pair", λ = 1) of the pooled selection to the validation set; near-duplicates merged into one point with combined mass (lower = better) |
| `groups(gidx, threshold)` | near-duplicate group of each image |
| `detected_overlap()` | matrix: share of dataset a's images with a near-duplicate in b (from embeddings) |
| `point_costs()` | every image's transport cost against the fixed clean reference ("val" labels), matched in reference-sized chunks so it is comparable with the clean threshold |
| `clean(selection)` | keep one image per duplicate group (lowest cost), drop images above the clean threshold; returns kept indices + counts |
| `training_set(selection, arm)` | images and labels to train on: `"raw"` (as downloaded) or `"clean"` |

Cached under `repos/<name>/cache/<encoder>/`: `dup_pairs_<features>.npz`
(all pairs with fingerprint or embedding cosine ≥ 0.90, so any threshold
≥ 0.90 is a filter; also stores each pair's embedding cosine) and
`point_costs_q<q>_strat.npz` (computed in class-stratified chunks, so a chunk's
class mix matches the reference). Delete them after changing the images or
embeddings.

Settings: `dup_features` (`"pixels"` default, or `"emb"`), `dup_min_emb`
(optional: also require this embedding cosine for a duplicate pair; 0 = off),
`dup_threshold` (0.95; choose it from the dedupe check, 12.6),
`clean_q` (0.95: drops about 5–7% of clean data), `n_max` (3000),
`label_weight` (1.0).

Nothing in `pipeline.py` or `selectors.py` reads `truth.npz`.

### 12.4 `src/selectors.py`: choosing k of n

| Selector | How it chooses | Role |
| --- | --- | --- |
| `one_shot` | score each dataset alone; take the k best | LAVA one dataset at a time; blind to overlap between picks |
| `greedy` | k times, add the dataset that gives the lowest pooled score | overlap-aware |
| `mmr` | one-shot score + λ × similarity to the most similar pick; similarity = shared images (`overlap_similarity`) or distribution similarity (`ot_similarity`: OT distance between datasets, rescaled to 0–1); λ = multiplier × spread of single scores | diversity heuristic; at large λ it can favour irrelevant data |
| `dedup_oneshot` | drop datasets whose detected overlap with a better one exceeds θ (0.5), then one-shot | deduplication baseline |
| `exhaustive` | score every k-subset | the OT optimum (reference for search error) |
| `random_draws` | 20 random k-subsets | floor |

`Scorer(ctx, seeds, dedupe)` memoizes the mean score over OT seeds, so
selectors share work; `exhaustive(..., workers=N)` scores in N processes.

### 12.5 Scripts

**`run_selection.py`**: builds the caches, runs every selector, exhaustive
scoring with and without duplicate merging, cleaning of each pick, and the
dedupe check. Writes `repos/<name>/results/<encoder>_k<k>/`:
`combo_scores.csv` (every combination: score, sd, unique images, per-seed,
no-merge score), `selections.json`, `dedupe_check.json`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--repo`, `--k` | required, 3 | |
| `--encoder` | `dinov2_vits14` | |
| `--seeds` | 3 | OT subsample seeds per combination |
| `--dup-features` | `pixels` | duplicate detection on pixel fingerprints or embeddings (`emb`) |
| `--dup-min-emb` | 0 | also require this embedding cosine for a duplicate pair |
| `--dup-threshold` | 0.95 | cosine for "same image" |
| `--clean-q` | 0.95 | cleaning threshold quantile |
| `--n-max` | 3000 | |
| `--mmr-similarity` | `overlap` | `overlap` and/or `ot`; one MMR selector per similarity × λ |
| `--mmr-lambdas` | 1.0 | penalty multipliers, e.g. `0 0.25 0.5 1 2`; selectors are named `mmr_<similarity>_x<λ>` (plain `mmr` for the default single setting) |
| `--dedup-theta` | 0.5 | dedup+one-shot setting |
| `--no-exhaustive`, `--no-dedupe-ablation` | off | skip parts |
| `--workers` | 1 | processes for exhaustive scoring |

**`train_subsets.py`**: trains ResNet-18 on every k-subset (or `--only-selected`)
for each arm (`raw`, `clean`) and seed (0 1 2); appends each run to
`ground_truth.jsonl` and skips finished runs on restart. `--part i/N` splits
the job list across GPUs or jobs. Flags: `--repo`, `--k`, `--encoder`,
`--arms`, `--seeds`, `--epochs` (30), `--only-selected`, `--part`,
`--dup-threshold`, `--dup-features`, `--dup-min-emb`, `--clean-q` (use the
same values as `run_selection.py`).

**`analyze.py`**: reads the three result files and prints the metrics below;
writes `summary.json`. Flags: `--repo`, `--k`, `--encoder`, `--metric`
(`acc_cifar10` or `acc_cifar101`), `--ground-truth PATH` (grade this
encoder's or this run's selections with another run's `ground_truth.jsonl`;
only its raw arm is used, since cleaning depends on the encoder). Use it for
the encoder ablation and MMR sweeps, which need no new training:

```bash
python -m src.embed --repo repos/full --encoder resnet50
python scripts/run_selection.py --repo repos/full --k 3 --encoder resnet50 --dup-threshold 0.98 --workers 8
python scripts/analyze.py --repo repos/full --k 3 --encoder resnet50 \
    --ground-truth repos/full/results/dinov2_vits14_k3/ground_truth.jsonl
```

### 12.6 Reading the results

| Output | Meaning | Good result |
| --- | --- | --- |
| dedupe check precision / recall per threshold | how well embedding similarity finds the planted shared images | pick the lowest threshold with precision ≈ 1 on the starter repository; then freeze it |
| Spearman / Kendall (score vs accuracy) | does lower OT distance mean higher accuracy across all combinations | strongly negative, CI excluding 0 |
| Spearman without duplicate merging | the same for the plain score | merging should not hurt; compare |
| Spearman of unique images vs accuracy | does the amount of distinct data matter | if clearly positive, the score (which measures match, not quantity) misses something |
| proxy error | best true accuracy − accuracy of the best-scoring combination | within about 1 point (training noise) |
| per selector: regret, search error, rank | accuracy given up vs the truly best / vs the OT optimum; position among all combinations | low regret; greedy ≤ one-shot |
| clean gain | accuracy after cleaning − before, for the pick | positive, larger for picks containing noisy data |
| redundancy | share of the pick's images that are copies (truth) | low; greedy and dedup should beat one-shot |
| harmful | share of picked datasets that are noisy or out-of-domain | low |

**The OT distance measures how well data matches the task, not how much
there is.** Exact duplicates leave it unchanged, so merging them stops copies
from earning credit but does not penalize them, and with equal-size datasets
an overlapping copy of clean data can tie with fresh clean data. The
unique-images correlation tests whether that matters for accuracy.

## 13. Testing status and known limitations

| Component | Status |
| --- | --- |
| `src/data.py` | Run on the server; all counts as expected. CINIC-10 full reload (70,000) still to confirm |
| `src/train.py`, `time_one_run.py` | Run on the server, including `--labels` |
| `src/embed.py` | Run on the server with both encoders |
| `src/ot.py`, `ot_sanity.py` | Run on synthetic data and on both encoders' real features |
| Selection pipeline (`repo`, `dedupe`, `pipeline`, `selectors`, `build_repo`, `run_selection`, `analyze`) | End to end on a synthetic repository (fake images and features): planted overlap and noise measured correctly; duplicates found exactly at cosine 0.97; cleaning removes all planted copies; analysis runs. **Not yet run on real data** |
| `train_subsets.py` | Data assembly tested on synthetic data; training itself not yet run (needs the server) |

Known limitations:

* **Mild label noise is under-weighted.** With DINOv2, 9% noise gives 15% of
  the 40%-noise distance gap, while its accuracy cost is 23% of the 40% cost.
* **Human label noise is hard to spot point by point** (AUROC 0.79–0.89 on
  real features vs 0.94–1.00 on synthetic): CIFAR-10N errors concentrate on
  ambiguous images.
* **Suspicious share underestimates the noise rate** (about +2 points for 9%
  noise, +18 for 40% with DINOv2).
* **Encoders and ImageNet:** both encoders' pretraining overlaps ImageNet, as
  do CINIC-10's extra images.
* **Pass thresholds** are project conventions.
* **Validation split is permanent:** changing it invalidates all scores.

---

## 14. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `torch.cuda.is_available()` is `False` | PyTorch's CUDA build doesn't match the driver, or no GPU allocated |
| `ModuleNotFoundError` (`ot`, `tqdm`, ...) or prompt shows `(base)` / `(.venv)` | `conda activate otproject`; on a new node, activate again |
| `CondaValueError: prefix already exists` | use `conda env update -f environment.yml` |
| `RuntimeError: Dataset not found` | run `python -m src.data --download` |
| `FileNotFoundError` for CINIC-10, CIFAR-10N or CIFAR-10.1 | manual download missing (section 2.3) |
| `--check` looks frozen on CINIC-10 | first load reads 70,000 PNGs; watch the progress bar |
| CINIC-10 count is not 70,000 | archive not fully unpacked; re-run `tar`, delete `data/cache/cinic10_imagenet_train.npz` |
| `CIFAR-10N label order does not match` | wrong or corrupt `CIFAR-10_human.pt`; re-download |
| `embed cifar10_train first` | the CIFAR-10N shortcut needs clean features |
| DINOv2 download fails | the node needs internet for `torch.hub` the first time |
| OT is slow | lower `--n-max`; time grows with its square |
| Loss becomes `nan` | try `--no-amp` or a lower `--lr` |
| Out of GPU memory | lower `--batch-size` |
| Job dies when SSH disconnects | run inside `tmux` |
| Timings vary between runs | another job shares the GPU; check `nvidia-smi` |
| Strange relative paths | run from the repository root |
| `... missing: run python -m src.embed --repo ...` | embed the repository after building it |
| `only N images of class c left` (build_repo) | the sources ran out; lower `size` or overlap fractions |
| Duplicate pairs or cleaning look wrong after re-embedding | delete `repos/<name>/cache/` |
| Ground truth for the clean arm doesn't match the selection's cleaning | use the same `--dup-threshold` and `--clean-q` in both scripts |
