"""EEGNet training loop with class weighting and early stopping on a validation set."""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

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


def _batches(
    n: int, batch_size: int, rng: np.random.Generator | None
) -> list[npt.NDArray[np.int64]]:
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def _tensor(X: npt.NDArray[np.floating], idx: npt.NDArray[np.int64]) -> torch.Tensor:
    return torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).unsqueeze(1)


@torch.no_grad()
def predict_proba(
    model: nn.Module, X: npt.NDArray[np.floating], batch_size: int = 512
) -> npt.NDArray[np.float32]:
    model.eval()
    out = [
        torch.softmax(model(_tensor(X, b)), 1).numpy() for b in _batches(len(X), batch_size, None)
    ]
    return np.concatenate(out) if out else np.empty((0, 3), np.float32)


def train_eegnet(
    X_train: npt.NDArray[np.floating],
    y_train: npt.NDArray[np.integer],
    X_val: npt.NDArray[np.floating],
    y_val: npt.NDArray[np.integer],
    cfg: TrainConfig,
) -> TrainResult:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    model = EEGNet(n_channels=X_train.shape[1], n_times=X_train.shape[2])
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)  # balanced: rest is ~half the windows
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    y_t = torch.as_tensor(np.asarray(y_train, dtype=np.int64))

    best_state, best_epoch, best_score, since_best = None, -1, -1.0, 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        total, n = 0.0, 0
        for b in _batches(len(X_train), cfg.batch_size, rng):
            opt.zero_grad()
            loss = loss_fn(model(_tensor(X_train, b)), y_t[b])
            loss.backward()
            opt.step()
            model.apply_max_norm()
            total += float(loss) * len(b)
            n += len(b)
        val_pred = predict_proba(model, X_val).argmax(1)
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
            if since_best >= cfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, best_epoch, best_score, history, time.perf_counter() - start)
