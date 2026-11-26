"""Ground-truth labels for the EEGMMIDB motor-imagery runs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

REST, LEFT, RIGHT = 0, 1, 2
UNKNOWN = -1  # samples we have no annotation for (or that were gap-filled)
CLASS_NAMES: tuple[str, ...] = ("rest", "left", "right")
ANNOTATION_TO_LABEL: dict[str, int] = {"T0": REST, "T1": LEFT, "T2": RIGHT}


def annotations_to_labels(
    n_samples: int,
    sfreq: float,
    onsets_s: Sequence[float],
    durations_s: Sequence[float],
    descriptions: Sequence[str],
) -> npt.NDArray[np.int8]:
    """Expand EDF annotations (onset, duration, label) into one label per sample.

    Samples not covered by any known annotation stay ``UNKNOWN``. Kept free of MNE so the
    replayer and the training pipeline share it without extra dependencies.
    """
    labels = np.full(n_samples, UNKNOWN, dtype=np.int8)
    for onset, duration, desc in zip(onsets_s, durations_s, descriptions, strict=True):
        label = ANNOTATION_TO_LABEL.get(desc)
        if label is None:
            continue
        start = max(0, round(onset * sfreq))
        stop = min(n_samples, round((onset + duration) * sfreq))
        labels[start:stop] = label
    return labels
