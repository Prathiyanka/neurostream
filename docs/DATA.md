# Dataset notes

**PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)**: 109 subjects, 64 channels,
160 Hz, EDF. We use the motor-imagery runs **4, 8, 12** (imagined left vs. right fist):

| Annotation | Label | Meaning |
|---|---|---|
| T0 | 0 `rest` | rest |
| T1 | 1 `left` | imagined left fist |
| T2 | 2 `right` | imagined right fist |

Download + verify: `python scripts/download_data.py` (writes `data/subject_report.json`;
`data/` is gitignored and never committed).

## Subject verification (actual results, all 109 subjects, runs 4/8/12)

Checks per recording: sampling rate == 160 Hz, 64 channels, duration >= 100 s, only
T0/T1/T2 labels present (all three), >= 10 imagery trials.

- **106 usable.**
- **Excluded: 88, 92, 100.** All three runs are recorded at **128 Hz**, not 160 Hz.
- **Subject 104: kept.** It is on the commonly cited exclusion list (88, 92, 100, 104), but
  passes every check on these runs. Its only irregularity: run 8 is shorter (106 s, 13
  imagery trials vs. 125 s and 15 for typical subjects). Runs 4 and 12 are 123 s, 15 trials.
  Shorter isn't invalid, so we keep the data. Decision confirmed with the project owner.
- Across usable subjects: run duration 106-125 s (median 123 s), imagery trials per run
  13-15 (median 15).

Excluded subjects are read from `subject_report.json` by the replayer (and, from Phase 3,
the training pipeline), not hard-coded.

## Units and labels in the pipeline

- Signal is converted from volts to **microvolts** (float32) by the replayer and sent with
  3 decimals.
- Every sample carries a label. Samples outside any annotation, and samples synthesized to
  fill a gap, are `-1` (unknown).
- Runs are concatenated per subject, so each run boundary has a discontinuity. Windows
  straddling one have mixed labels, so `label_pure` is false.
