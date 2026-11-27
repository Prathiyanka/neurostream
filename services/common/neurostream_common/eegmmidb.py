"""EEGMMIDB loading, shared by the replayer (streaming) and ``ml/`` (training).

MNE is imported lazily, so services that never read EDF files don't need it installed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurostream_common.labels import annotations_to_labels
from neurostream_common.windowing import FloatArray, LabelArray

MOTOR_IMAGERY_RUNS: tuple[int, ...] = (4, 8, 12)
ALL_SUBJECTS: tuple[int, ...] = tuple(range(1, 110))


@dataclass(frozen=True)
class Recording:
    subject: int | None
    sfreq: float
    ch_names: list[str]
    data: FloatArray  # (n_channels, n_samples), microvolts
    labels: LabelArray  # (n_samples,)
    run: int | None = None  # None when several runs are concatenated

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
    """Subjects rejected by ``scripts/download_data.py`` (empty if no report yet)."""
    report = data_dir / "subject_report.json"
    if not report.exists():
        return set()
    return set(json.loads(report.read_text())["summary"]["excluded"])


def usable_subjects(data_dir: Path) -> list[int]:
    report = data_dir / "subject_report.json"
    if not report.exists():
        raise FileNotFoundError(f"{report} missing; run: python scripts/download_data.py")
    return sorted(json.loads(report.read_text())["summary"]["usable"])


def load_run(subject: int, run: int, data_dir: Path) -> Recording:
    import mne

    path = edf_path(data_dir, subject, run)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run: python scripts/download_data.py")
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    mne.datasets.eegbci.standardize(raw)  # 'C3..' -> 'C3'
    sfreq = float(raw.info["sfreq"])
    ann = raw.annotations
    return Recording(
        subject=subject,
        sfreq=sfreq,
        ch_names=list(raw.ch_names),
        data=(raw.get_data() * 1e6).astype(np.float32),  # volts -> microvolts
        labels=annotations_to_labels(
            raw.n_times, sfreq, list(ann.onset), list(ann.duration), list(ann.description)
        ),
        run=run,
    )


def load_subject(
    subject: int, data_dir: Path, runs: Sequence[int] = MOTOR_IMAGERY_RUNS
) -> Recording:
    """Concatenate the given runs of one subject (what the replayer streams).

    Runs are joined end to end, so each boundary has a small discontinuity. A real
    headset session would have one too (the subject rests between runs). Windows that
    straddle it have mixed labels, so they are not 'pure'.
    """
    if subject in excluded_subjects(data_dir):
        raise ValueError(f"subject {subject} failed verification (see subject_report.json)")
    recs = [load_run(subject, run, data_dir) for run in runs]
    first = recs[0]
    for r in recs[1:]:
        if (r.sfreq, r.ch_names) != (first.sfreq, first.ch_names):
            raise ValueError(f"subject {subject} run {r.run} differs in rate or channels")
    return Recording(
        subject=subject,
        sfreq=first.sfreq,
        ch_names=first.ch_names,
        data=np.concatenate([r.data for r in recs], axis=1),
        labels=np.concatenate([r.labels for r in recs]),
    )
