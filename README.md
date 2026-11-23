# ECE6514_OT_Data_Validation_FinalProject
ECE6514 Optimal transport and data validation final project

**Question:** can a training-free optimal transport (OT) score identify which
k-of-n combination of repository datasets trains the best model, and does
accounting for overlap between datasets matter?

## Layout

```
src/data.py              step 2: load any dataset as (uint8 N,32,32,3; labels 0-9; ids)
src/models.py            ResNet-18 for 32x32 images
src/train.py             train_eval(): train ResNet-18 on any merged set, report accuracy + time
src/embed.py             step 4: frozen-encoder features -> emb/<encoder>/<dataset>.npz
src/ot.py                step 4: LAVA-style class-wise OT distance, gradients, score_set()
src/dedupe.py            near-duplicate detection from embeddings
src/repo.py              build / load a controlled dataset repository
src/pipeline.py          set score, duplicate groups, point costs, cleaning
src/selectors.py         one-shot, greedy, MMR, dedup+one-shot, exhaustive, random
scripts/time_one_run.py  step 3: time one training run, project cost of the k-of-n study
scripts/ot_sanity.py     step 4: check the OT score on duplicate / OOD / noisy-label data
scripts/build_repo.py    step 5: build a repository from configs/*.json
scripts/run_selection.py steps 6-7: select k of n, clean, dedupe check
scripts/train_subsets.py step 8: ground-truth training, raw vs cleaned
scripts/analyze.py       correlation, proxy/search error, regret, redundancy
```

The method is LAVA extended from scoring points in one dataset to choosing a
combination of datasets: select k of n by OT score with near-duplicates merged,
clean the pick (dedupe + drop costly points), train.

`data/`, `emb/` and `results/` are git-ignored.

## Setup

Linux server with an NVIDIA GPU, using a conda environment named `otproject`:

```
conda env create -f environment.yml    # once; installs Python 3.11, PyTorch and the rest
conda activate otproject               # every new shell or node
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The environment lives in your conda folder, not the repo, so it works on every
node that shares your home directory. To update after `requirements.txt` changes:
`conda env update -f environment.yml`.

Run every command from the repo root. Step-by-step server guide with expected
output: [docs/DOCUMENTATION.md, section 2](docs/DOCUMENTATION.md#2-running-on-the-server).

## Step 2: data

```
python -m src.data --download      # CIFAR-10, STL-10, SVHN, MNIST via torchvision
python -m src.data --check         # loads everything available, prints sizes
```

Manual downloads, placed in `data/`:

| File | From |
| --- | --- |
| `data/CINIC-10/` (unzipped) | https://github.com/BayesWatch/cinic-10 |
| `data/CIFAR-10_human.pt` | https://github.com/UCSC-REAL/cifar-10-100n (`data/` folder) |
| `data/cifar10.1_v6_data.npy`, `data/cifar10.1_v6_labels.npy` | https://github.com/modestyachts/CIFAR-10.1 (`datasets/` folder) |

Conventions: all images 32x32 RGB, CIFAR-10 label order. STL-10 drops "monkey".
SVHN/MNIST keep digit d as label d (they are deliberately out-of-domain).
5,000 fixed CIFAR-10 *train* images are the OT validation set
(`cifar10_train_val_split`); the CIFAR-10 test set is used only for final accuracy.

## Step 3: timing

```
python scripts/time_one_run.py --fake --epochs 2     # smoke test (seconds)
python scripts/time_one_run.py --epochs 30           # real run on 15K CIFAR images
```

Prints seconds per run and projected GPU-hours for n = 12 with k = 2 and 3.
Only the number from the machine you train on (the server) matters.

## Step 4: OT score

```
python scripts/ot_sanity.py --fake                   # synthetic check, CPU, ~15 s
python -m src.embed --datasets cifar10_train mnist_train        # DINOv2 (default encoder)
python -m src.embed --datasets cifar10n_worse cifar10n_aggre
python scripts/ot_sanity.py                                     # both label variants
python -m src.embed --encoder resnet50 --datasets cifar10_train mnist_train cifar10n_worse cifar10n_aggre
python scripts/ot_sanity.py --encoder resnet50                  # ablation
```

Checks: a duplicated shard scores the same as the original, MNIST scores far
worse, and CIFAR-10N noisy labels (40% and 9%) score worse than clean labels.
OT runs on CPU with POT (exact solver, about 0.5 s at 2,000 x 2,000 points).

Reference: Just et al., *LAVA: Data Valuation without Pre-Specified Learning
Algorithms*, ICLR 2023 (https://github.com/ruoxi-jia-group/LAVA). We implement
its class-wise distance directly with POT rather than depending on that repo.

## Steps 5-8: selection pipeline

```
python scripts/build_repo.py configs/starter.json
python -m src.embed --repo repos/starter
python scripts/run_selection.py --repo repos/starter --k 3 --workers 8 --dup-threshold 0.98
python scripts/train_subsets.py --repo repos/starter --k 3 --dup-threshold 0.98
python scripts/analyze.py --repo repos/starter --k 3
```

Full reference for every module and script: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).
