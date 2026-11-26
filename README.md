# NeuroStream

> Real-time EEG streaming & mental-state classification platform.
> **Research / portfolio project — not a medical device.**

NeuroStream streams brain (EEG) signals in real time, runs them through a low-latency
processing pipeline, classifies motor imagery (rest / imagined left fist / imagined right
fist) with a model trained on the PhysioNet EEGMMIDB dataset, and shows everything on a live
dashboard. A replayer service plays recorded EEG as if it came from a live headset, so the
rest of the system works the same with real hardware.

**Status:** Phase 2 (replayer + ingest) of 8. See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full plan.

## Architecture

```
[Replayer] --WebSocket--> [Ingest] --Redis Streams--> [Inference Workers xN]
 EDF @ 160 Hz           validate, seq check,     ns:windows (consumer group)
 fault injection        build window jobs        stateless: filter + classify
                             |                                |
                             +--> ns:raw:{session} (traces)   v
[React Dashboard] <--WebSocket-- [API] <--Redis pub/sub-- predictions + metrics
                                   |
                               [Postgres]
```

Built so far: replayer, ingest, the shared `neurostream_common` package, Redis, Postgres.
Workers, API and dashboard come in later phases.

## Quickstart (so far)

Requirements: Python 3.11+, Docker Desktop.

```powershell
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env

docker compose up -d --wait       # Postgres, Redis, ingest (http://localhost:8001/health)

python scripts/download_data.py --subjects 1-5   # or omit --subjects for all 109 (~1 GB)
pytest                            # unit + integration tests (integration needs the stack up)
pytest -m "not integration"       # unit tests only, no Docker needed
```

Stream EEG into the pipeline:

```powershell
# Two real subjects at 10x speed with 2% dropped and 2% duplicated packets
docker compose run --rm replayer --url ws://ingest:8001/ws/device --data-dir /data `
    --subjects 1 2 --speed 10 --max-seconds 30 --drop-prob 0.02 --duplicate-prob 0.02 --seed 1

# Or from the host, without the dataset (synthetic signal)
python -m replayer --synthetic 3 --speed 5 --max-seconds 20
```

Example (real output from the run above): the replayer reports what it injected, and
ingest's counters in Redis (`ns:metrics:ingest:<session>`) match it exactly:

| session | sent | dropped | duplicated | ingest accepted | ingest duplicates | ingest gaps | windows |
|---|---|---|---|---|---|---|---|
| sub001 | 294 | 6 | 5 | 294 | 5 | 6 | 55 |
| sub002 | 291 | 9 | 4 | 291 | 4 | 9 | 55 |

Replayer flags: `--speed`, `--loop`, `--max-seconds`, `--chunk-size`, and fault injection
with `--drop-prob`, `--duplicate-prob`, `--reorder-prob`, `--jitter-ms`,
`--disconnect-every-s`, `--reconnect-delay-s` and `--seed`.

## Repository layout

| Path | Purpose |
|---|---|
| `services/common` | `neurostream_common`: schemas, sequence tracking, windowing, codecs, shared by services **and** `ml/` |
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
  commonly cited exclusions (88, 92, 100, 104). **Result on runs 4/8/12:** 106 of 109
  subjects pass. Subjects 88, 92 and 100 are excluded: they were recorded at 128 Hz, not 160 Hz.
  Subject 104 passes every check. Its run 8 is shorter (106 s, 13 trials, vs. 125 s and 15
  trials for most subjects), but the data is valid, so it is kept.
- **Stateless workers via self-contained window jobs.** The spec had workers read raw packets
  through a consumer group. But a consumer group hands each message to *any* worker, while
  causal filtering and sliding windows need the session's recent history. So ingest builds
  the windows instead. Each job carries the 2 s window plus 1 s of earlier samples, so the
  filter has settled before it reaches the part that gets classified. Any worker can then
  process any job: scaling is just "start more workers", and a crashed worker's pending
  jobs can be reclaimed by another with nothing to rebuild.
  - *Cost:* data amplification. Each 64 x 480 float32 job is 123 KB and there are 2 per
    second, about 246 KB/s per session versus 41 KB/s of raw signal (6x, i.e.
    span / step = 480 / 80). At 50 sessions that's about 12 MB/s through Redis: fine on
    one host, but it's the first thing to optimize (channel subset, float16, shorter
    padding once Phase 3 measures the filter's real settling time).
  - *Alternative rejected:* sticky session-to-worker assignment (leases). It's less data,
    but rebalancing and crash recovery are harder (the new owner must rebuild filter
    state), and that's exactly the part the project needs to show is correct.
  - Windows have a deterministic id `session:end_sample`, so a window reprocessed after
    a crash can be de-duplicated downstream.
- **Training and serving share one windowing implementation.** `neurostream_common.windowing`
  has an offline `extract_windows` (training) and a `StreamingWindower` (ingest). A test feeds
  the same signal through both in chunk sizes from 1 to 500 and requires byte-identical
  windows. An end-to-end test checks that what lands in Redis equals the offline windows.
  This prevents train/serve skew, a classic silent accuracy killer.
- **Sequence policy: detect, don't reorder.** Packet seq ahead of expected = gap (packets
  presumed lost); behind = *late* if it was marked missing, otherwise *duplicate*. There is
  no reorder buffer, because it would add latency to every packet to fix a rare event.
  Short gaps (<= 0.25 s) are filled by holding the last value. Those samples are labeled
  unknown and counted per window. Longer gaps restart the window grid instead of
  inventing signal.
- **Backpressure: bounded per-session buffers, drop oldest.** The WebSocket reader never
  waits on Redis. If Redis stalls, a session's buffer fills and its oldest items are dropped
  and counted. For real-time data, fresh beats stale, and one session can't starve another.
  Writes are batched into one Redis pipeline round trip per batch.
- **Replayer timing uses an absolute schedule** (packet k due at `t0 + k * period`), so
  late wake-ups (Windows timers have ~15 ms granularity) never accumulate into drift. Like
  a real headset, it doesn't buffer while disconnected: packets due during an outage are
  lost, and ingest sees the gap.
- **Ground truth travels beside the signal, never inside it.** `truth.labels` is a
  separate field of each packet and of each window job. The model only ever gets `data`.
- **Wire formats:** JSON over WebSocket from the device (easy to debug, and Pydantic
  validates it). Raw little-endian float32 bytes inside Redis (compact, and decoding is a
  zero-copy view). Whether JSON parsing becomes the bottleneck is measured in Phase 7.
- **Python helper scripts instead of a Makefile** so everything runs on Windows.
- **Integration tests are marked** (`-m integration`) so unit tests run without Docker;
  CI runs both, with Redis and Postgres as service containers.

## Tech used

Python 3.11, asyncio, FastAPI, WebSockets, Pydantic v2, NumPy, MNE-Python, Redis Streams,
PostgreSQL, Docker Compose, pytest (+ pytest-asyncio), ruff, mypy (strict), GitHub Actions.
(Later: PyTorch, ONNX Runtime, SQLAlchemy/Alembic, React, TypeScript, Vite.)

## What I learned

- An integration test caught a real timing bug: the replayer started its clock *before*
  the first connection, so a slow handshake made it skip the first packets as if they'd
  been lost in an outage. It failed only about one run in three. Running it in a loop and
  logging *which* window was missing (start sample 96, not 0) pointed straight at it.
- Consumer groups and stateful stream processing don't mix without extra design work.
  That one realization drove the whole window-job architecture.

## License

[MIT](LICENSE)
