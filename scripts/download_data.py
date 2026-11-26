"""Download and verify the PhysioNet EEG Motor Movement/Imagery dataset (EEGMMIDB).

Only the motor-imagery runs we use (4, 8, 12: imagined left vs. right fist) are fetched.
After downloading, every recording is checked and a report is written to
``<data-dir>/subject_report.json``. Later phases (training, replayer) read that report
instead of hard-coding which subjects to skip.

Usage (works the same on Windows, macOS, Linux):
    python scripts/download_data.py                  # all 109 subjects
    python scripts/download_data.py --subjects 1-10  # a subset
    python scripts/download_data.py --verify-only    # re-check files already on disk
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("download_data")

MOTOR_IMAGERY_RUNS: tuple[int, ...] = (4, 8, 12)
ALL_SUBJECTS: tuple[int, ...] = tuple(range(1, 110))
EXPECTED_SFREQ = 160.0
EXPECTED_N_CHANNELS = 64
EXPECTED_LABELS = frozenset({"T0", "T1", "T2"})
# Each MI run is ~2 minutes with ~15 cued trials; anything far below that is suspicious.
MIN_DURATION_S = 100.0
MIN_TRIALS_PER_RUN = 10
# Subjects the literature commonly excludes. We don't trust this list blindly:
# the verifier re-derives exclusions from the files and reports any disagreement.
COMMONLY_EXCLUDED: frozenset[int] = frozenset({88, 92, 100, 104})


@dataclass
class RecordingInfo:
    """The few facts about one EDF file that decide whether it is usable."""

    sfreq: float
    n_channels: int
    duration_s: float
    annotations: list[str]


@dataclass
class SubjectReport:
    subject: int
    usable: bool
    issues: list[str] = field(default_factory=list)
    runs: dict[int, dict[str, object]] = field(default_factory=dict)


def check_recording(info: RecordingInfo) -> list[str]:
    """Return a list of human-readable problems with one recording (empty == OK).

    Pure function (no file I/O) so it can be unit-tested with synthetic inputs.
    """
    issues: list[str] = []
    if info.sfreq != EXPECTED_SFREQ:
        issues.append(f"sampling rate {info.sfreq:g} Hz != {EXPECTED_SFREQ:g} Hz")
    if info.n_channels != EXPECTED_N_CHANNELS:
        issues.append(f"{info.n_channels} channels != {EXPECTED_N_CHANNELS}")
    if info.duration_s < MIN_DURATION_S:
        issues.append(f"recording only {info.duration_s:.1f} s (< {MIN_DURATION_S:g} s)")
    counts = Counter(info.annotations)
    unknown = set(counts) - EXPECTED_LABELS
    if unknown:
        issues.append(f"unexpected annotation labels {sorted(unknown)}")
    missing = EXPECTED_LABELS - set(counts)
    if missing:
        issues.append(f"missing annotation labels {sorted(missing)}")
    n_trials = counts.get("T1", 0) + counts.get("T2", 0)
    if n_trials < MIN_TRIALS_PER_RUN:
        issues.append(f"only {n_trials} imagery trials (< {MIN_TRIALS_PER_RUN})")
    return issues


def parse_subjects(spec: str) -> list[int]:
    """Parse '1-10,15,20-22' into a sorted list of subject ids in [1, 109]."""
    subjects: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"bad range {part!r}")
            subjects.update(range(lo, hi + 1))
        else:
            subjects.add(int(part))
    bad = [s for s in subjects if s not in ALL_SUBJECTS]
    if bad:
        raise ValueError(f"subject ids out of range 1-109: {sorted(bad)}")
    return sorted(subjects)


def read_recording_info(path: Path) -> RecordingInfo:
    import mne  # imported lazily: heavy, and unit tests of pure functions don't need it

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    return RecordingInfo(
        sfreq=float(raw.info["sfreq"]),
        n_channels=len(raw.ch_names),
        duration_s=float(raw.times[-1]) if len(raw.times) else 0.0,
        annotations=[str(d) for d in raw.annotations.description],
    )


def edf_path(data_dir: Path, subject: int, run: int) -> Path:
    # Layout produced by mne.datasets.eegbci with path=data_dir.
    return (
        data_dir
        / "MNE-eegbci-data"
        / "files"
        / "eegmmidb"
        / "1.0.0"
        / f"S{subject:03d}"
        / f"S{subject:03d}R{run:02d}.edf"
    )


def download(subjects: Iterable[int], data_dir: Path) -> None:
    import mne

    data_dir.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        log.info("downloading subject %d", subject)
        mne.datasets.eegbci.load_data(
            subject,
            list(MOTOR_IMAGERY_RUNS),
            path=str(data_dir),
            update_path=False,
            verbose="ERROR",
        )


def verify(subjects: Iterable[int], data_dir: Path) -> list[SubjectReport]:
    reports: list[SubjectReport] = []
    for subject in subjects:
        report = SubjectReport(subject=subject, usable=True)
        for run in MOTOR_IMAGERY_RUNS:
            path = edf_path(data_dir, subject, run)
            if not path.exists():
                issues = ["file missing"]
                report.runs[run] = {"issues": issues}
            else:
                info = read_recording_info(path)
                issues = check_recording(info)
                report.runs[run] = {
                    "sfreq": info.sfreq,
                    "n_channels": info.n_channels,
                    "duration_s": round(info.duration_s, 2),
                    "label_counts": dict(Counter(info.annotations)),
                    "issues": issues,
                }
            report.issues.extend(f"run {run}: {msg}" for msg in issues)
        report.usable = not report.issues
        reports.append(report)
    return reports


def summarize(reports: Sequence[SubjectReport]) -> dict[str, object]:
    checked = {r.subject for r in reports}
    excluded = sorted(r.subject for r in reports if not r.usable)
    expected = sorted(COMMONLY_EXCLUDED & checked)
    return {
        "checked": len(reports),
        "usable": sorted(r.subject for r in reports if r.usable),
        "excluded": excluded,
        "commonly_excluded_in_checked_set": expected,
        "excluded_but_not_commonly": sorted(set(excluded) - set(expected)),
        "commonly_excluded_but_passed": sorted(set(expected) - set(excluded)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--subjects", default="1-109", help="e.g. '1-109' or '1-10,42'")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--verify-only", action="store_true", help="skip download")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    subjects = parse_subjects(args.subjects)
    if not args.verify_only:
        download(subjects, args.data_dir)
    reports = verify(subjects, args.data_dir)
    summary = summarize(reports)

    out = args.data_dir / "subject_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "subjects": [asdict(r) for r in reports]}, indent=2)
    )

    log.info("checked %d subjects, %d usable", summary["checked"], len(summary["usable"]))  # type: ignore[arg-type]
    for r in reports:
        if not r.usable:
            log.warning("subject %d excluded: %s", r.subject, "; ".join(r.issues))
    log.info("report written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
