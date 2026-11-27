"""Builds the training windows, using the *same* windowing and filter code as serving.

For every usable subject and run, windows come from ``extract_windows`` (the offline twin
of ingest's ``StreamingWindower``) and are filtered with ``filter_and_crop`` (the same
function the workers call). Only *pure* windows (the whole classified 2 s has one known
label) are kept for training and evaluation.

Filtered 64-channel cores are cached to disk as float16 (about 2 GB instead of 4 GB). Channel
selection and normalization are applied when a channel set is loaded. The cache is only
for speed: the final test numbers are re-checked through the exact float32 serving path
(see ``experiments.py``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from neurostream_common.channels import channel_indices
from neurostream_common.eegmmidb import MOTOR_IMAGERY_RUNS, load_run
from neurostream_common.preprocessing import PreprocessConfig, filter_and_crop, normalize_window
from neurostream_common.windowing import WindowSpec, extract_windows

log = logging.getLogger("ml.dataset")


@dataclass
class WindowSet:
    X: npt.NDArray[np.float16]  # (N, n_channels, n_times)
    y: npt.NDArray[np.int8]
    subject: npt.NDArray[np.int16]
    run: npt.NDArray[np.int8]
    ch_names: list[str]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def mask(
        self, subjects: Sequence[int] | None = None, runs: Sequence[int] | None = None
    ) -> npt.NDArray[np.bool_]:
        m = np.ones(len(self), dtype=bool)
        if subjects is not None:
            m &= np.isin(self.subject, np.asarray(subjects))
        if runs is not None:
            m &= np.isin(self.run, np.asarray(runs))
        return m


def pure_windows_for_run(
    data: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int8],
    spec: WindowSpec,
    cfg: PreprocessConfig,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8]]:
    """Filtered cores (unnormalized) and labels of the pure windows of one run."""
    xs, ys = [], []
    for w in extract_windows(data, labels, spec):
        if w.label_pure:
            xs.append(filter_and_crop(w.data, w.pad_samples, cfg))
            ys.append(w.label_end)
    if not xs:
        return np.empty((0, data.shape[0], spec.window_samples), np.float32), np.empty(0, np.int8)
    return np.stack(xs), np.asarray(ys, dtype=np.int8)


def _cache_key(spec: WindowSpec, cfg: PreprocessConfig, subjects: Sequence[int]) -> str:
    blob = json.dumps(
        {
            "spec": spec.__dict__,
            "cfg": {**cfg.to_dict(), "channels": None},
            "subjects": list(subjects),
            "v": 1,
        },
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def build_cache(
    subjects: Sequence[int],
    data_dir: Path,
    spec: WindowSpec,
    cfg: PreprocessConfig,
    runs: Sequence[int] = MOTOR_IMAGERY_RUNS,
) -> Path:
    """Filter every pure window of every subject (all channels) and cache it."""
    cache_dir = data_dir / "cache" / _cache_key(spec, cfg, subjects)
    if (cache_dir / "done.json").exists():
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_cfg = dataclasses.replace(cfg, channels=None)
    parts: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    ch_names: list[str] | None = None
    for subject in subjects:
        for run in runs:
            rec = load_run(subject, run, data_dir)
            if ch_names is None:
                ch_names = rec.ch_names
            elif rec.ch_names != ch_names:
                raise ValueError(f"subject {subject} run {run}: channel layout differs")
            X, y = pure_windows_for_run(rec.data, rec.labels, spec, all_cfg)
            parts.append((X.astype(np.float16), y, subject, run))
        log.info("cached subject %d (%d windows so far)", subject, sum(len(p[1]) for p in parts))
    assert ch_names is not None
    n = sum(len(p[1]) for p in parts)
    X_mm = np.lib.format.open_memmap(
        cache_dir / "X.npy",
        mode="w+",
        dtype=np.float16,
        shape=(n, len(ch_names), spec.window_samples),
    )
    i = 0
    for X, _, _, _ in parts:
        X_mm[i : i + len(X)] = X
        i += len(X)
    X_mm.flush()
    np.save(cache_dir / "y.npy", np.concatenate([p[1] for p in parts]))
    np.save(
        cache_dir / "subject.npy",
        np.concatenate([np.full(len(p[1]), p[2], np.int16) for p in parts]),
    )
    np.save(
        cache_dir / "run.npy", np.concatenate([np.full(len(p[1]), p[3], np.int8) for p in parts])
    )
    (cache_dir / "done.json").write_text(
        json.dumps(
            {
                "n_windows": n,
                "ch_names": ch_names,
                "subjects": list(subjects),
                "spec": spec.__dict__,
                "cfg": all_cfg.to_dict(),
            },
            indent=1,
        )
    )
    return cache_dir


def _normalized_path(cache_dir: Path, ch_names: Sequence[str]) -> Path:
    key = hashlib.sha1(json.dumps(list(ch_names)).encode()).hexdigest()[:10]
    return cache_dir / f"X_norm_{len(ch_names)}ch_{key}.npy"


def load_window_set(
    cache_dir: Path, channels: Sequence[str] | None, chunk: int = 4096
) -> WindowSet:
    """Load one channel set, normalizing each window with the shared serving function.

    The normalized windows are written to disk once per channel set and returned as a
    read-only memory map, not as an in-process array. The OS keeps the pages it can in
    its file cache and can evict them under memory pressure, instead of the training
    process holding ~1.7 GB (64 channels) of committed memory and being killed.
    """
    meta = json.loads((cache_dir / "done.json").read_text())
    ch_all: list[str] = meta["ch_names"]
    idx = channel_indices(ch_all, channels)
    ch_names = [ch_all[i] for i in idx]
    out = _normalized_path(cache_dir, ch_names)
    if not out.exists():
        X_src = np.load(cache_dir / "X.npy", mmap_mode="r")
        tmp = out.with_suffix(".tmp.npy")
        X = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float16, shape=(X_src.shape[0], len(idx), X_src.shape[2])
        )
        for start in range(0, X_src.shape[0], chunk):
            block = np.asarray(X_src[start : start + chunk][:, idx], dtype=np.float32)
            X[start : start + chunk] = np.stack([normalize_window(w) for w in block])
        X.flush()
        del X, X_src  # close the maps before renaming (required on Windows)
        tmp.replace(out)  # atomic: a half-written file is never mistaken for a done one
    return WindowSet(
        X=np.load(out, mmap_mode="r"),
        y=np.load(cache_dir / "y.npy"),
        subject=np.load(cache_dir / "subject.npy"),
        run=np.load(cache_dir / "run.npy"),
        ch_names=ch_names,
    )
