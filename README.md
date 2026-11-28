# NeuroStream

> Real-time EEG streaming & mental-state classification platform.
> **Research / portfolio project — not a medical device.**

NeuroStream streams brain (EEG) signals in real time, runs them through a low-latency
processing pipeline, classifies motor imagery (rest / imagined left fist / imagined right
fist) with a model trained on the PhysioNet EEGMMIDB dataset, and shows everything on a live
dashboard. A replayer service plays recorded EEG as if it came from a live headset, so the
rest of the system works the same with real hardware.

**Status:** Phase 4 (inference workers) of 8 done. See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full plan.

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

Built so far: replayer, ingest, inference workers, the shared `neurostream_common`
package, Redis, Postgres, and the trained, ONNX-exported model. Predictions land in the
`ns:predictions` stream (durable) and on the `ns:predictions:live` pub/sub channel. The API
and dashboard come in Phases 5 and 6.

## System latency (Phase 4)

10 held-out subjects streamed concurrently at real-time speed through the whole stack
(replayer → ingest → Redis → worker → published prediction), 3 runs per configuration.
Full method and analysis: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| Workers | Latency p50 | p95 (target < 50 ms) | p99 | Windows lost |
|---|---|---|---|---|
| 1 | 14.2 to 18.9 ms | **32.8 to 37.9 ms** | 40.2 to 47.7 ms | 0 of 3,480 |
| 3 | 8.8 to 10.1 ms | **21.6 to 45.8 ms** | 28.4 to 108.0 ms | 0 of 3,480 |

Ranges are over the 3 runs. Most of the time is queueing behind a burst (all 10 sessions'
windows complete together every 0.5 s), not compute: inference is about 2 ms per batch. The
3-worker tail is noisy because of host-level CPU stalls on this laptop (Docker Desktop on
Windows); the benchmark page shows how that was diagnosed.

## Model results (Phase 3)

Three classes (rest / imagined left fist / imagined right fist), evaluated on **people the
model never saw** (5-fold cross-subject over 106 subjects). Full write-up, including what
did not work: [`docs/MODEL_RESULTS.md`](docs/MODEL_RESULTS.md).

| Model | Balanced accuracy (chance 33.3%) | 95% CI over subjects |
|---|---|---|
| CSP+LDA, 21 motor channels (baseline) | 43.3% | 41.4% to 45.1% |
| **EEGNet, 21 motor channels (served)** | **45.7%** | 43.7% to 47.7% |
| EEGNet, all 64 channels | 46.7% | 44.8% to 48.7% |

EEGNet beats the baseline by +2.4 pt (paired over subjects, CI +1.1 to +3.8). All 64
channels add +1.1 pt, which is not distinguishable from zero (CI −0.2 to +2.4), at 3x the
data per window and twice the latency, so the 21-channel model is the one served. Filter +
ONNX inference takes 0.90 ms per window (p50, one CPU thread, offline benchmark).

## Quickstart (so far)

Requirements: Python 3.11+, Docker Desktop.

```powershell
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env

docker compose up -d --wait       # Postgres, Redis, ingest (http://localhost:8001/health), 1 worker
docker compose up -d --wait --scale worker=3     # more workers: same consumer group

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

# Watch predictions, and measure latency (10 held-out subjects in real time)
docker compose exec redis redis-cli XREVRANGE ns:predictions + - COUNT 1
python scripts/measure_latency.py --sessions 10 --workers 1 --seconds 60
```

Example (real output from the run above): the replayer reports what it injected, and
ingest's counters in Redis (`ns:metrics:ingest:<session>`) match it exactly:

| session | sent | dropped | duplicated | ingest accepted | ingest duplicates | ingest gaps | windows |
|---|---|---|---|---|---|---|---|
| sub001 | 294 | 6 | 5 | 294 | 5 | 6 | 56 |
| sub002 | 291 | 9 | 4 | 291 | 4 | 9 | 56 |

Replayer flags: `--speed`, `--loop`, `--max-seconds`, `--chunk-size`, and fault injection
with `--drop-prob`, `--duplicate-prob`, `--reorder-prob`, `--jitter-ms`,
`--disconnect-every-s`, `--reconnect-delay-s` and `--seed`.

## Repository layout

| Path | Purpose |
|---|---|
| `services/common` | `neurostream_common`: schemas, sequence tracking, windowing, codecs, shared by services **and** `ml/` |
| `services/replayer` | Streams EDF recordings at real-time rate, with fault injection |
| `services/ingest` | WebSocket ingest, validation, gap/duplicate detection → Redis Streams |
| `services/worker` | Stateless inference workers: batched ONNX inference via a Redis consumer group, reclaiming, dead letters, loss accounting |
| `services/api` | REST + WebSocket API, Postgres persistence, model hot-swap |
| `dashboard/` | React + TypeScript + Vite live dashboard |
| `ml/` | CSP+LDA baseline, EEGNet training, subject-wise evaluation, ONNX export |
| `scripts/` | Cross-platform helper scripts (Python, so they work on Windows) |
| `tests/` | `unit/` and `integration/` pytest suites |
| `models/` | Exported ONNX models + model cards (each ~30 KB, committed so workers run without training) |
| `benchmarks/results/` | Raw latency benchmark results (JSON), summarized in `docs/BENCHMARKS.md` |
| `docs/` | Dataset notes, model results, benchmarks |

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
  the windows instead. Each job carries the 2 s window plus 0.5 s of earlier samples (the
  pad), so the filter has settled before it reaches the part that gets classified. The pad
  was 1 s until Phase 3 measured the filter's settling time; 0.5 s costs no accuracy. Any worker can then
  process any job: scaling is just "start more workers", and a crashed worker's pending
  jobs can be reclaimed by another with nothing to rebuild.
  - *Cost:* data amplification. Each 64 x 400 float32 job is 102 KB and there are 2 per
    second, about 205 KB/s per session versus 41 KB/s of raw signal (5x, i.e.
    span / step = 400 / 80). At 50 sessions that's about 10 MB/s through Redis: fine on
    one host. Next optimizations, to be justified by Phase 7 load tests: send only the
    model's 21 channels (3x less), or float16.
  - *Alternative rejected:* sticky session-to-worker assignment (leases). It's less data,
    but rebalancing and crash recovery are harder (the new owner must rebuild filter
    state), and that's exactly the part the project needs to show is correct.
  - Windows have a deterministic id `session:end_sample`, so a window reprocessed after
    a crash can be de-duplicated downstream.
- **Workers: take what is available, publish, then acknowledge.** A worker reads up to 32
  jobs but never waits for a batch to fill: if one job is there, it processes one. Batches
  form on their own under bursts, which is exactly when one ONNX call for many windows
  pays off. Predictions are published (durable stream + live pub/sub) *before* the job is
  acknowledged, so a crash can produce a duplicate prediction but never lose one
  (at-least-once; the deterministic window id lets the API de-duplicate).
- **Crash recovery: reclaim, then dead-letter.** Every 0.5 s, each worker takes over jobs
  another worker has held unacknowledged for 1 s (`XAUTOCLAIM`). 1 s is ~100x a normal
  batch, so live workers keep their jobs, and short enough that recovered jobs are still
  fresh enough to be worth predicting. A job that can never work
  (wrong sampling rate, unknown session) is set aside in `ns:windows:dead` at once; one
  delivered more than 3 times is too, so a poison message cannot crash workers forever.
- **Stale jobs are skipped, not processed late.** Under overload, a prediction for a
  window that completed seconds ago is useless live and delays fresher ones. A worker
  acknowledges such jobs without running the model and counts them as `stale` (threshold
  `STALE_AFTER_S`, default 2 s; 0 disables). Stale is counted separately from lost: the
  data arrived, the system chose to drop it.
- **Lost jobs are counted, exactly.** `ns:windows` is capped (`MAXLEN ~ 5000`) so an
  overload cannot exhaust Redis memory, but trimming deletes jobs whether or not anyone
  acknowledged them. Redis 7 keeps a consumer group's `entries-read` as a position in
  everything ever added, so a read that jumps over trimmed jobs reveals how many it
  skipped. Workers read through a small Lua script that checks the position before and
  after the read *atomically*, which keeps the count exact with many workers reading.
  Trimmed jobs that were pending show up when they are reclaimed. Tests check that the
  books balance: `entries-added = acked + stale + lost + dead-lettered + pending + lag`.
- **Workers refuse a model whose hash does not match its card**, and warm the model up
  (dummy batches of every size) before taking traffic: measured, a cold worker's first
  inference calls took up to 101 ms instead of ~2 ms.
- **Latency is measured on every prediction, by stage.** Each prediction carries ingest,
  queue and worker times that add up to the total, plus its batch size. That is what
  showed queueing, not compute, dominates (see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).
- **Training and serving share one windowing implementation.** `neurostream_common.windowing`
  has an offline `extract_windows` (training) and a `StreamingWindower` (ingest). A test feeds
  the same signal through both in chunk sizes from 1 to 500 and requires byte-identical
  windows. End-to-end tests check that what lands in Redis equals the offline windows, and
  that the predictions the running system publishes for a held-out subject equal the
  offline pipeline's to 2e-6. This prevents train/serve skew, a classic silent accuracy
  killer.
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
- **Evaluate on unseen people, report balanced accuracy, bootstrap over subjects.** Half
  the windows are `rest`, so plain accuracy rewards "always rest" with 49.5%. Windows from
  one person are correlated, so CIs resample subjects, not windows, and model comparisons
  are paired per subject. See [`docs/MODEL_RESULTS.md`](docs/MODEL_RESULTS.md).
- **The model card is the train/serve contract.** Each exported model ships with the
  channels, filter, pad and normalization it was trained with, plus a SHA-256 of the ONNX
  file (weights embedded, so the hash covers them). Workers rebuild their input from the
  card, and the exported model is re-scored through the exact float32 serving path.
- **Long training runs are resumable and deterministic.** Results are saved after every
  fold and the full training state (model, optimizer, both RNGs) after every epoch. A
  resumed run is bit-identical to an uninterrupted one (tested). Indexing one memory-mapped
  window array instead of copying splits keeps the 64-channel set (1.7 GB) from being
  duplicated in RAM.
- **Python helper scripts instead of a Makefile** so everything runs on Windows.
- **Integration tests are marked** (`-m integration`) so unit tests run without Docker;
  CI runs both, with Redis and Postgres as service containers.

## Tech used

Python 3.11, asyncio, FastAPI, WebSockets, Pydantic v2, NumPy, MNE-Python, Redis Streams,
PostgreSQL, Docker Compose, pytest (+ pytest-asyncio), ruff, mypy (strict), GitHub Actions.
PyTorch, ONNX / ONNX Runtime, scikit-learn, Redis consumer groups and Lua scripting.
(Later: SQLAlchemy/Alembic, React, TypeScript, Vite.)

## What I learned

- An integration test caught a real timing bug: the replayer started its clock *before*
  the first connection, so a slow handshake made it skip the first packets as if they'd
  been lost in an outage. It failed only about one run in three. Running it in a loop and
  logging *which* window was missing (start sample 96, not 0) pointed straight at it.
- Consumer groups and stateful stream processing don't mix without extra design work.
  That one realization drove the whole window-job architecture.
- A model can fail for honest reasons. Within-subject EEGNet scored chance (34.7%). Before
  calling it a bug I checked the evidence: training loss stayed at ln 3 (uniform guessing),
  predictions collapsed onto one class, and each person has only ~130 windows. The same
  code reaches 46% across subjects. The lesson: tiny data needs either a low-capacity model
  (CSP+LDA gets 50 to 54%) or fine-tuning a pretrained one.
- Measure before optimizing. The obvious fix for memory pressure during training was a
  smaller batch, but measuring showed the process used the same ~1.5 GB at batch 16 and
  64, so batch size was not the lever (and changing it mid-experiment would have made
  the folds incomparable). Making the run resumable per epoch was the real fix.
- Don't trust a guess about a system you can test in a minute. Whether Redis's consumer
  group counters stay correct when `MAXLEN` trims entries decided the whole loss-accounting
  design, so I checked it against a real Redis 7.4 with exact numbers (17 skipped, 3
  pending trimmed) before writing any worker code.
- One benchmark run is not a result. Identical 10-stream runs differed by nearly 4x at p99.
  Grouping the slow windows by time showed they came in bursts where *every* stage slowed
  down at once, which points at the host (CPU taken away from the Docker VM), not the code.

## License

[MIT](LICENSE)
