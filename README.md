# NeuroStream

> Real-time EEG streaming & mental-state classification platform.
> **Research / portfolio project — not a medical device.**

NeuroStream streams brain (EEG) signals in real time, runs them through a low-latency
processing pipeline, classifies motor imagery (rest / imagined left fist / imagined right
fist) with a model trained on the PhysioNet EEGMMIDB dataset, and shows everything on a live
dashboard. A replayer service plays recorded EEG as if it came from a live headset, so the
rest of the system works the same with real hardware.

**Status:** Phase 1 (foundation) of 8. See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full plan.

## Architecture (planned)

```
[Replayer] --WebSocket--> [Ingest] --Redis Streams--> [Inference Workers xN]
                                                              |
[React Dashboard] <--WebSocket-- [API] <--Redis pub/sub-------+
                                   |
                               [Postgres]
```

## Quickstart (so far)

Requirements: Python 3.11+, Docker Desktop.

```powershell
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env

docker compose up -d --wait       # Postgres + Redis

python scripts/download_data.py --subjects 1-5   # or omit --subjects for all 109 (~1 GB)
pytest                            # unit + infrastructure smoke tests
```

Example output of the download/verify step:

```
INFO download_data: checked 5 subjects, 5 usable
INFO download_data: report written to data\subject_report.json
```

## Repository layout

| Path | Purpose |
|---|---|
| `services/replayer` | Streams EDF recordings at real-time rate, with fault injection |
| `services/ingest` | WebSocket ingest, validation, gap/duplicate detection → Redis Streams |
| `services/worker` | Streaming filter, windowing, ONNX inference (consumer groups) |
| `services/api` | REST + WebSocket API, Postgres persistence, model hot-swap |
| `dashboard/` | React + TypeScript + Vite live dashboard |
| `ml/` | CSP+LDA baseline, EEGNet training, subject-wise evaluation, ONNX export |
| `scripts/` | Cross-platform helper scripts (Python, so they work on Windows) |
| `tests/` | `unit/` and `integration/` pytest suites |
| `docs/` | Model results, benchmarks |

## Design decisions (so far)

- **Data subjects are verified, not assumed.** `download_data.py` checks every recording
  (sampling rate, channel count, duration, annotation labels, trial count) and writes
  `data/subject_report.json`. Training and replay read that report instead of a
  hard-coded exclusion list, and the report shows where it agrees or disagrees with the
  commonly cited exclusions (88, 92, 100, 104).
- **Python helper scripts instead of a Makefile** so everything runs on Windows.
- **Integration tests are marked** (`-m integration`) so unit tests run without Docker;
  CI runs both, with Redis and Postgres as service containers.

## Tech used

Python 3.11, MNE-Python, Redis Streams, PostgreSQL, Docker Compose, pytest, ruff, mypy,
GitHub Actions. (Later: FastAPI, PyTorch, ONNX Runtime, React, TypeScript, Vite.)

## What I learned

_To be filled in as the project progresses._

## License

[MIT](LICENSE)
