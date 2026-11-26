"""Loading EEGMMIDB recordings into memory as (channels x samples) microvolts + labels."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurostream_common.labels import annotations_to_labels
from neurostream_common.windowing import FloatArray, LabelArray

MOTOR_IMAGERY_RUNS = (4, 8, 12)


@dataclass(frozen=True)
class Recording:
    subject: int | None
    sfreq: float
    ch_names: list[str]
    data: FloatArray  # (n_channels, n_samples), microvolts
    labels: LabelArray  # (n_samples,)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])


def edf_path(data_dir: Path, subject: int, run: int) -> Path:
    return (
        data_dir
        / "MNE-eegbci-data"
        / "files"
        / "eegmmidb"
        / "1.0.0"
        / f"S{subject:03d}"
        / f"S{subject:03d}R{run:02d}.edf"
    )


def excluded_subjects(data_dir: Path) -> set[int]:
    """Subjects the download script's verifier rejected (empty if no report yet)."""
    report = data_dir / "subject_report.json"
    if not report.exists():
        return set()
    return set(json.loads(report.read_text())["summary"]["excluded"])


def load_subject(
    subject: int, data_dir: Path, runs: Sequence[int] = MOTOR_IMAGERY_RUNS
) -> Recording:
    """Concatenate the given runs of one subject.

    Runs are joined end to end, so there is a small discontinuity at each boundary. A real
    headset session would have one too (the subject rests between runs), and the windows
    straddling it get a mixed label, so they are not 'pure' and the evaluation can skip them.
    """
    import mne

    if subject in excluded_subjects(data_dir):
        raise ValueError(f"subject {subject} failed verification (see subject_report.json)")
    datas: list[FloatArray] = []
    labels: list[LabelArray] = []
    sfreq: float | None = None
    ch_names: list[str] | None = None
    for run in runs:
        path = edf_path(data_dir, subject, run)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run: python scripts/download_data.py")
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        mne.datasets.eegbci.standardize(raw)  # 'C3..' -> 'C3'
        if sfreq is None:
            sfreq, ch_names = float(raw.info["sfreq"]), list(raw.ch_names)
        elif (float(raw.info["sfreq"]), list(raw.ch_names)) != (sfreq, ch_names):
            raise ValueError(f"subject {subject} run {run} differs in rate or channels")
        datas.append((raw.get_data() * 1e6).astype(np.float32))  # volts -> microvolts
        ann = raw.annotations
        labels.append(
            annotations_to_labels(
                raw.n_times, sfreq, list(ann.onset), list(ann.duration), list(ann.description)
            )
        )
    assert sfreq is not None and ch_names is not None
    return Recording(
        subject, sfreq, ch_names, np.concatenate(datas, axis=1), np.concatenate(labels)
    )


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
