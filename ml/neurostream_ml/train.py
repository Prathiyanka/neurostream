"""EEGNet training loop with class weighting and early stopping on a validation set.

Splits are passed as *index arrays* into one shared window array, never as copies:
``X[train_idx]`` on the 64-channel set would duplicate about 1.4 GB. Only one batch at a
time is gathered and converted to float32.

With ``resume_path`` the full training state (model, optimizer, both RNGs, early-stopping
bookkeeping) is written after every epoch, so a killed run continues from its last
finished epoch and produces exactly the same model as an uninterrupted one.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from neurostream_ml.eegnet import EEGNet

log = logging.getLogger("ml.train")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    patience: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    threads: int = 4


@dataclass
class TrainResult:
    model: EEGNet
    best_epoch: int
    best_val_bal_acc: float
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


IndexArray = npt.NDArray[np.int64]


def _batches(idx: IndexArray, batch_size: int, rng: np.random.Generator | None) -> list[IndexArray]:
    order = idx[rng.permutation(len(idx))] if rng is not None else idx
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def as_index(sel: npt.NDArray[np.bool_] | IndexArray | None, n: int) -> IndexArray:
    """Boolean mask, index array or None (= everything) -> sorted int64 indices."""
    if sel is None:
        return np.arange(n, dtype=np.int64)
    arr = np.asarray(sel)
    return np.flatnonzero(arr).astype(np.int64) if arr.dtype == bool else arr.astype(np.int64)


def _tensor(X: npt.NDArray[np.floating], idx: npt.NDArray[np.int64]) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).unsqueeze(1)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    X: npt.NDArray[np.floating],
    sel: npt.NDArray[np.bool_] | IndexArray | None = None,
    batch_size: int = 512,
) -> npt.NDArray[np.float32]:
    """Class probabilities for ``X[sel]`` (all of ``X`` if ``sel`` is None)."""
    model.eval()
    idx = as_index(sel, len(X))
    out = [torch.softmax(model(_tensor(X, b)), 1).numpy() for b in _batches(idx, batch_size, None)]
    return np.concatenate(out) if out else np.empty((0, 3), np.float32)


def _save_atomic(obj: dict[str, Any], path: Path) -> None:
    """Write to a temp file then rename, so a kill mid-write never leaves a corrupt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train_eegnet(
    X: npt.NDArray[np.floating],
    y: npt.NDArray[np.integer],
    train_sel: npt.NDArray[np.bool_] | IndexArray,
    val_sel: npt.NDArray[np.bool_] | IndexArray,
    cfg: TrainConfig,
    resume_path: Path | None = None,
    on_epoch_end: Callable[[int], None] | None = None,
) -> TrainResult:
    """Train on ``X[train_sel]``, early-stop on ``X[val_sel]``. ``X`` is never copied.

    ``resume_path``: per-epoch training state; resumed from if it exists. The caller
    deletes it once the final model is saved. ``on_epoch_end`` runs after each epoch's
    state is saved (tests use it to simulate a kill).
    """
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    train_idx, val_idx = as_index(train_sel, len(X)), as_index(val_sel, len(X))
    y_all = torch.as_tensor(np.asarray(y, dtype=np.int64))
    y_val = np.asarray(y[val_idx])
    model = EEGNet(n_channels=X.shape[1], n_times=X.shape[2])
    counts = np.bincount(np.asarray(y[train_idx]), minlength=3).astype(np.float64)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)  # balanced: rest is ~half the windows
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state, best_epoch, best_score, since_best = None, -1, -1.0, 0
    history: list[dict[str, float]] = []
    first_epoch, prior_seconds = 0, 0.0
    if resume_path is not None and resume_path.exists():
        st = torch.load(resume_path, weights_only=False)  # our own file: holds RNG states
        if st["train_config"] != asdict(cfg):
            raise SystemExit(f"{resume_path} was written with {st['train_config']}, not {cfg}")
        if st["n_train"] != len(train_idx) or st["n_val"] != len(val_idx):
            raise SystemExit(f"{resume_path} was written for a different split")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        rng.bit_generator.state = st["numpy_rng"]
        torch.set_rng_state(st["torch_rng"])
        best_state, best_epoch, best_score = st["best_state"], st["best_epoch"], st["best_score"]
        since_best, history = st["since_best"], st["history"]
        first_epoch, prior_seconds = st["epoch"] + 1, st["seconds"]
        log.info("resumed from %s after epoch %d", resume_path, st["epoch"])
    start = time.perf_counter() - prior_seconds
    for epoch in range(first_epoch, cfg.epochs):
        if since_best >= cfg.patience:
            break  # the saved state had already early-stopped
        model.train()
        total, n = 0.0, 0
        for b in _batches(train_idx, cfg.batch_size, rng):
            opt.zero_grad()
            loss = loss_fn(model(_tensor(X, b)), y_all[torch.from_numpy(b)])
            loss.backward()
            opt.step()
            model.apply_max_norm()
            total += loss.item() * len(b)
            n += len(b)
        val_pred = predict_proba(model, X, val_idx).argmax(1)
        score = float(balanced_accuracy_score(y_val, val_pred))
        history.append({"epoch": epoch, "train_loss": total / n, "val_bal_acc": score})
        log.info(
            "epoch %d loss %.4f val_bal_acc %.4f (%.0fs)",
            epoch,
            total / n,
            score,
            time.perf_counter() - start,
        )
        if score > best_score:
            best_state, best_epoch, best_score, since_best = (
                copy.deepcopy(model.state_dict()),
                epoch,
                score,
                0,
            )
        else:
            since_best += 1
        if resume_path is not None:
            _save_atomic(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "since_best": since_best,
                    "history": history,
                    "seconds": time.perf_counter() - start,
                    "train_config": asdict(cfg),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                },
                resume_path,
            )
        if on_epoch_end is not None:
            on_epoch_end(epoch)
        if since_best >= cfg.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, best_epoch, best_score, history, time.perf_counter() - start)
