"""Evaluation splits. Overlapping windows from the same recording are strongly correlated,
so no split ever puts windows from the same subject (cross-subject) or the same run
(within-subject) on both sides."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

WITHIN_TRAIN_RUNS: tuple[int, ...] = (4, 8)
WITHIN_TEST_RUNS: tuple[int, ...] = (12,)  # chronologically last run, like a real session


@dataclass(frozen=True)
class SubjectFold:
    index: int
    train: tuple[int, ...]
    val: tuple[int, ...]  # carved out of the training subjects, for early stopping
    test: tuple[int, ...]


def subject_folds(
    subjects: Sequence[int], k: int = 5, val_fraction: float = 0.1, seed: int = 0
) -> list[SubjectFold]:
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(sorted(subjects)))
    groups = [sorted(int(s) for s in order[i::k]) for i in range(k)]
    folds = []
    for i, test in enumerate(groups):
        rest = [s for s in order if s not in test]
        n_val = max(1, round(len(rest) * val_fraction))
        val, train = rest[:n_val], rest[n_val:]
        folds.append(
            SubjectFold(
                i,
                tuple(sorted(int(s) for s in train)),
                tuple(sorted(int(s) for s in val)),
                tuple(test),
            )
        )
    return folds
