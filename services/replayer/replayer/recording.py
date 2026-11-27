"""Recordings the replayer can stream: real EEGMMIDB data or a synthetic signal."""

from __future__ import annotations

import numpy as np

from neurostream_common.eegmmidb import (
    MOTOR_IMAGERY_RUNS,
    Recording,
    edf_path,
    excluded_subjects,
    load_subject,
)

__all__ = [
    "MOTOR_IMAGERY_RUNS",
    "Recording",
    "edf_path",
    "excluded_subjects",
    "load_subject",
    "synthetic_recording",
]


def synthetic_recording(
    n_channels: int = 8, seconds: float = 10.0, sfreq: float = 160.0, seed: int = 0
) -> Recording:
    """Deterministic fake recording for tests and demos without the dataset."""
    rng = np.random.default_rng(seed)
    n = round(seconds * sfreq)
    t = np.arange(n) / sfreq
    data = (10 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 5, (n_channels, n))).astype(np.float32)
    labels = ((np.arange(n) // round(4 * sfreq)) % 3).astype(np.int8)  # 4 s blocks of 0,1,2
    return Recording(None, sfreq, [f"CH{i}" for i in range(n_channels)], data, labels)
