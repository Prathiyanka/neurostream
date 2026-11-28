# NeuroStream

> Real-time EEG streaming & mental-state classification platform.
> **Research / portfolio project — not a medical device.**

NeuroStream streams brain (EEG) signals in real time, runs them through a low-latency
processing pipeline, classifies motor imagery (rest / imagined left fist / imagined right
fist) with a model trained on the PhysioNet EEGMMIDB dataset, and shows everything on a live
dashboard. A replayer service plays recorded EEG as if it came from a live headset, so the
rest of the system works the same with real hardware.

**Status:** Phase 5 (API + Postgres) of 8 done. See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full plan.

## Architecture

```
                 HTTP: start / stop replays
           +--------------------------------------+
           |                                      |
[Replayer service] --WebSocket--> [Ingest] --ns:windows--> [Inference workers xN]
 "the device":                 validate, seq check,  (consumer group)  batch ONNX,
 EDF @ 160 Hz, faults          build window jobs                       hot-swap model
                                    |                                      |
                         ns:raw:{session}                        ns:predictions (stream)
                         (raw EEG, ~2 min)                       ns:predictions:live (pub/sub)
                                    |                                      |
                                    v                                      v
[Dashboard, Phase 6] <--/ws/live-- [API] <------ persister (consumer group "api")
                                    |  REST: sessions, models, metrics
                                    v
                               [Postgres]  sessions, predictions (window_id UNIQUE),
                                           metrics snapshots, model events
```

Built so far: everything except the dashboard (Phase 6). The API is the only thing a
client talks to: it starts replays through the replayer service, stores every prediction
in Postgres exactly once, hot-swaps the served model, and streams live predictions,
metrics and EEG traces over a WebSocket.

**No authentication.** This is a local research/portfolio system: every endpoint is open
to whoever can reach the port. Before exposing it beyond localhost it would need auth
(and TLS), which is out of scope here.

## API (Phase 5)

Interactive docs at http://localhost:8000/docs once the stack is up.

| Endpoint | What it does |
|---|---|
| `POST /sessions` | Start a replay: `{"subject": 7, "speed": 1, "max_seconds": 60, "faults": {"drop_prob": 0.02}}` (or `"synthetic": true`). Sent to the replayer service. |
| `GET /sessions`, `GET /sessions/{id}` | Sessions from Postgres; one session's summary: predictions stored, balanced accuracy, latency p50/p95/p99, models used, ingest counters. |
| `DELETE /sessions/{id}` | Stop a replay the API started (it still ends cleanly at ingest). |
| `GET /sessions/{id}/predictions` | Stored predictions, in window order. |
| `GET /models`, `PUT /models/active` | Available models, the active one and what each worker serves; hot-swap with `{"model": "eegnet-motor-1.0.0"}`. |
| `GET /metrics`, `GET /metrics/history` | Workers, queue, job outcomes (acked / stale / lost / dead-lettered), ingest totals; history from 5 s snapshots in Postgres. |
| `WS /ws/live?session=<id>&channels=C3,Cz,C4` | Live predictions, metrics every second, and EEG traces at 40 Hz. |

Example (real run, `subject 1` at 5x speed, model hot-swapped from `eegnet-motor-1.0.1` to
`1.0.0` three seconds in): 76 of 76 windows stored, switch at window 3519 with no gap in
the window sequence, session latency p50 5.6 ms / p99 14.2 ms.

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

## Quickstart

Requirements: Python 3.11+, Docker Desktop.

```powershell
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env

docker compose up -d --build --wait   # everything; --build so code changes are picked up
docker compose up -d --wait --scale worker=3     # more workers: same consumer group
curl http://localhost:8000/health

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
| `services/replayer` | The simulated device: streams EDF recordings at real-time rate with fault injection; HTTP service (`serve`) or command line |
| `services/ingest` | WebSocket ingest, validation, gap/duplicate detection → Redis Streams |
| `services/worker` | Stateless inference workers: batched ONNX inference via a Redis consumer group, reclaiming, dead letters, loss accounting |
| `services/api` | REST + WebSocket API, Postgres persistence (SQLAlchemy Core + Alembic), model hot-swap |
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
- **The replayer is its own service, commanded over HTTP.** The API never simulates the
  device in its own process, so heavy replays (load tests) cannot slow the API. HTTP, not
  a Redis command channel, because a start command needs an immediate answer (the
  session id, or "subject 88 is excluded"), which HTTP gives as a status code. Commands
  should also fail fast when the replayer is down instead of being queued and replayed
  later. Load tests run the replayer's command-line mode in their own containers.
- **Persistence: own consumer group, unique window id, ack after commit.** The API reads
  `ns:predictions` with consumer group `api` and stores batches with
  `INSERT ... ON CONFLICT (window_id) DO NOTHING`, acknowledging only after the commit.
  Delivery is at-least-once (a worker crash can duplicate a prediction; so can an API
  restart between commit and ack); the unique key makes the table exactly-once. Tested
  with a duplicated prediction and with a persister "crashing" between read and store.
- *Trade-off:* the persister runs inside the API process. It is simple and fast enough
  here; if it ever falls behind, it moves to its own service unchanged, because it
  already has its own consumer group and its own metrics (`ns:metrics:persister`).
- **Model hot-swap without restarts.** `PUT /models/active` verifies the model's hash and
  sets `ns:model:active`. Each worker checks it about once a second, loads and warms up
  the new model on a background thread while serving with the old one, and swaps between
  two batches. A model that fails to load is reported in the worker's heartbeat and the
  old one keeps serving. New workers start on the active model.
- **Live traces are display-only.** `/ws/live` reads the raw packets ingest stores in
  `ns:raw:{session}` (validated, de-duplicated, unfiltered), keeps the channels a client
  asked for and averages blocks of 4 samples (160 to 40 Hz). The model never sees this
  path. Each client has a bounded drop-oldest queue, so a slow browser cannot slow
  anything else.
- **asyncpg for the API, psycopg for migrations.** psycopg's async mode needs a selector
  event loop, which Windows does not use by default; asyncpg works everywhere. Alembic
  runs synchronously, where psycopg is fine. A test checks that the migration and the
  schema in code never drift apart.
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
PyTorch, ONNX / ONNX Runtime, scikit-learn, Redis consumer groups and Lua scripting,
SQLAlchemy 2 (Core, asyncpg) + Alembic, httpx. (Later: React, TypeScript, Vite.)

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

- Check what is actually running. The first hot-swap test through Docker "failed": the
  worker container was still an image built before hot-swap existed, because
  `docker compose up` reuses images. The quickstart now says `--build`.
- A flaky count is often a race in the test, not the code. An end-to-end test briefly saw
  76 predictions for "73" windows: it had counted the windows before ingest finished
  flushing the session. Waiting for the session's `ended` status fixed it; the code had
  delivered and stored each window exactly once.

## License

[MIT](LICENSE)
