"""Metrics, computed the same way for every model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

N_CLASSES = 3


def summarize(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, Any]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    counts = np.bincount(yt, minlength=N_CLASSES)
    return {
        "n": int(len(yt)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=range(N_CLASSES))),
        # Plain accuracy's real chance level is the majority-class rate, not 1/3.
        "majority_baseline": float(counts.max() / counts.sum()),
        "class_counts": counts.tolist(),
        "confusion": confusion_matrix(yt, yp, labels=range(N_CLASSES)).tolist(),
    }


def per_subject(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, subjects: npt.ArrayLike
) -> dict[int, float]:
    yt, yp, s = np.asarray(y_true), np.asarray(y_pred), np.asarray(subjects)
    return {int(u): float(balanced_accuracy_score(yt[s == u], yp[s == u])) for u in np.unique(s)}


def bootstrap_ci(values: Sequence[float], n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean over subjects (subjects are the independent unit, not windows)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
