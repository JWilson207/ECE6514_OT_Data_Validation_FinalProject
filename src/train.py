"""Training and evaluation of ResNet-18 on a merged set of datasets.

Design choice: the whole training set lives on the GPU as uint8; batches are
sliced and augmented on the GPU (random crop with 4-px padding + horizontal
flip). For CIFAR-sized images this removes the CPU data-loading bottleneck, so
no DataLoader is needed.

Main entry point for the project: train_eval(...) -> dict of results.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .models import ResNet18

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_device_uint8(X: np.ndarray, device) -> torch.Tensor:
    """uint8 NHWC numpy -> uint8 NCHW tensor on device."""
    return torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).contiguous().to(device)


def _normalize(xb_uint8: torch.Tensor) -> torch.Tensor:
    x = xb_uint8.float().div_(255.0)
    return (x - CIFAR_MEAN.to(x.device)) / CIFAR_STD.to(x.device)


def gpu_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random crop (zero padding) + random horizontal flip, on device.
    x: float NCHW batch."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    i = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    j = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (torch.arange(H, device=x.device)[None, :] + i[:, None])          # B,H
    cols = (torch.arange(W, device=x.device)[None, :] + j[:, None])          # B,W
    b = torch.arange(B, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[b, rows[:, :, None], cols[:, None, :]]      # B,H,W,C
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


@torch.no_grad()
def evaluate(model, X_u8: torch.Tensor, y: torch.Tensor, amp: bool, bs: int = 1000) -> float:
    model.eval()
    correct = 0
    for s in range(0, len(y), bs):
        xb = _normalize(X_u8[s:s + bs])
        with torch.autocast(device_type=xb.device.type, dtype=torch.float16, enabled=amp):
            pred = model(xb).argmax(1)
        correct += (pred == y[s:s + bs]).sum().item()
    return correct / len(y)


def train_eval(X_train: np.ndarray, y_train: np.ndarray,
               test_sets: dict[str, tuple[np.ndarray, np.ndarray]],
               epochs: int = 30, batch_size: int = 256, lr: float = 0.1,
               weight_decay: float = 5e-4, seed: int = 0,
               device: str | torch.device = "auto", amp: bool | None = None,
               verbose: bool = True) -> dict:
    """Train ResNet-18 from scratch on (X_train, y_train) and evaluate.

    X_* : uint8 (N,32,32,3); y_* : int64 (N,)
    test_sets : {"cifar10_test": (X, y), "cifar101": (X, y), ...}
    Returns accuracies per test set plus timing.
    """
    device = get_device(device) if isinstance(device, str) else device
    if amp is None:
        amp = device.type == "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = _to_device_uint8(X_train, device)
    ytr = torch.as_tensor(y_train, dtype=torch.long, device=device)
    tests = {k: (_to_device_uint8(X, device), torch.as_tensor(y, dtype=torch.long, device=device))
             for k, (X, y) in test_sets.items()}

    model = ResNet18().to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          nesterov=True, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(ytr) // batch_size)               # drop_last
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.25)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    gen = torch.Generator(device=device).manual_seed(seed)

    epoch_times = []
    _sync(device)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        te = time.perf_counter()
        perm = torch.randperm(len(ytr), device=device, generator=gen)
        loss_sum = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            xb = gpu_augment(_normalize(Xtr[idx])).contiguous(memory_format=torch.channels_last)
            yb = ytr[idx]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() if s % 20 == 0 else 0.0
        _sync(device)
        epoch_times.append(time.perf_counter() - te)
        if verbose and (ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep + 1:3d}/{epochs}  {epoch_times[-1]:6.2f}s")
    train_time = time.perf_counter() - t0

    te = time.perf_counter()
    accs = {k: evaluate(model, X, y, amp) for k, (X, y) in tests.items()}
    _sync(device)
    eval_time = time.perf_counter() - te

    # steady-state time: ignore epoch 1 (kernel autotuning / warm-up)
    steady = float(np.mean(epoch_times[1:])) if len(epoch_times) > 1 else epoch_times[0]
    return {
        "acc": accs,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "first_epoch_s": epoch_times[0],
        "steady_epoch_s": steady,
        "n_train": int(len(ytr)),
        "epochs": epochs,
        "batch_size": batch_size,
        "amp": bool(amp),
        "device": device_name(device),
    }


def projected_hours(seconds_per_run: float, n: int, ks=(2, 3), seeds: int = 1) -> dict:
    return {f"n={n},k={k}": {"subsets": math.comb(n, k),
                             "hours": math.comb(n, k) * seeds * seconds_per_run / 3600}
            for k in ks}
