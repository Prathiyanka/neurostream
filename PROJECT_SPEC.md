\# NeuroStream — Real-Time EEG Streaming \& Classification Platform

&#x20;

\## What we're building

A production-style system that streams brain (EEG) signals in real time, processes them through a low-latency pipeline, classifies the user's mental state with a trained ML model, and shows everything on a live dashboard.

&#x20;

Since I don't have an EEG headset, a \*\*replayer service\*\* streams real recorded EEG data from a public dataset as if it were coming live from a device. The rest of the system must not know or care that the data is replayed — it should work the same way with a real headset later.

&#x20;

\*\*Goal of this project:\*\* a portfolio piece that demonstrates \*software engineering\* (distributed services, streaming, concurrency, fault tolerance, testing, CI, Docker, measured performance) with a \*real ML component\* I train and evaluate myself. I'm a UW CS + Neuroscience student; I need to be able to explain every design decision in an interview, so explain non-obvious choices as you go.

&#x20;

\---

&#x20;

\## Dataset

\*\*PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB)\*\* — 109 subjects, 64 channels, 160 Hz, EDF files. Download it with `mne.datasets.eegbci` (MNE-Python).

\- Use the \*\*motor imagery runs 4, 8, 12\*\* (imagined left fist vs. right fist), with annotations T0 = rest, T1 = left, T2 = right.

\- Task: \*\*3-class classification — rest / left-hand imagery / right-hand imagery\*\*.

\- A few subjects have irregular recordings (commonly excluded: 88, 92, 100, 104) — verify this and handle it in code.

\- \*\*Never commit the dataset\*\* to git. Data goes in `data/` (gitignored), with a script to download it.

\---

&#x20;

\## Architecture

&#x20;

```

&#x20;\[Replayer]  --WebSocket-->  \[Ingest Service]  --Redis Streams-->  \[Inference Workers (N)]

&#x20; EDF files                   validate, buffer,                     filter, window, classify

&#x20; real-time rate              detect gaps                                  |

&#x20;                                                                          v

&#x20;\[React Dashboard]  <--WebSocket--  \[API Service]  <--Redis pub/sub--  predictions + metrics

&#x20;                                        |

&#x20;                                    \[Postgres]  sessions, predictions, latency metrics

```

&#x20;

\### Components

1\. \*\*Replayer\*\* (`services/replayer/`, Python)

&#x20;  - Reads EDF files and sends chunks at the true sampling rate (e.g., 16 samples every 100 ms per session).

&#x20;  - Each packet: `session\_id`, `seq` (sequence number), `timestamp`, `samples` (channels × n), and the ground-truth label (kept separate so the classifier can't cheat — used only for live accuracy display).

&#x20;  - Supports multiple concurrent sessions, a speed multiplier, and \*\*fault injection\*\* flags: dropped packets, jitter, duplicate packets, disconnect/reconnect.

2\. \*\*Ingest Service\*\* (`services/ingest/`, Python, FastAPI + asyncio)

&#x20;  - Accepts device WebSocket connections; validates packet schema (Pydantic).

&#x20;  - Uses sequence numbers to \*\*detect gaps, duplicates and out-of-order packets\*\*; records them as metrics.

&#x20;  - Writes packets to a \*\*Redis Stream\*\* per session. Handles backpressure (bounded buffers; don't let one slow consumer take down the service).

3\. \*\*Inference Workers\*\* (`services/worker/`, Python)

&#x20;  - Consume from Redis Streams using \*\*consumer groups\*\* so workers can scale horizontally; if a worker crashes, its pending messages are reclaimed by another worker.

&#x20;  - Signal processing: band-pass filter (8–30 Hz, mu/beta bands), sliding windows (e.g., 2 s windows, 0.5 s step), per-window normalization. Filtering must be \*\*causal/streaming-safe\*\* (no peeking at future samples).

&#x20;  - Run the model (ONNX Runtime) and publish `{session\_id, window\_end\_ts, predicted\_class, probabilities, true\_label, latency\_ms}`.

&#x20;  - \*\*Latency budget:\*\* measure from the moment a window's last sample arrives at ingest to the moment the prediction is published. Target \*\*p95 < 50 ms\*\* at 10 concurrent streams. Report the real number, whatever it is.

4\. \*\*API Service\*\* (`services/api/`, Python, FastAPI)

&#x20;  - REST: list/create/stop sessions, session history, model info, health check, metrics.

&#x20;  - WebSocket: pushes live EEG traces (downsampled for display), predictions, and system metrics to the dashboard.

&#x20;  - Persists sessions, predictions and latency stats to \*\*Postgres\*\* (use SQLAlchemy + Alembic migrations).

5\. \*\*Dashboard\*\* (`dashboard/`, TypeScript + React + Vite)

&#x20;  - Live scrolling EEG traces for key motor channels (C3, Cz, C4), with a channel selector.

&#x20;  - Current prediction with probability bars, and the true label beside it.

&#x20;  - Rolling accuracy, latency (p50/p95) chart, dropped/duplicate packet counters, worker count, queue depth.

&#x20;  - Session controls: start a replay for a chosen subject, stop it, set speed, toggle fault injection.

&#x20;  - Clean, professional look. Must stay smooth (render with canvas, not thousands of DOM nodes).

6\. \*\*ML Training Pipeline\*\* (`ml/`, Python, PyTorch + scikit-learn + MNE)

&#x20;  - \*\*Baseline:\*\* CSP + LDA (the classic motor-imagery method).

&#x20;  - \*\*Main model:\*\* a compact CNN (EEGNet-style) in PyTorch.

&#x20;  - \*\*Evaluation must be honest:\*\* split \*\*by subject\*\* (train on some subjects, test on held-out subjects) so the model is tested on people it has never seen. Also report within-subject results separately. No data leakage — the filter/normalization stats must come from training data only. Report accuracy, macro-F1, confusion matrix, and compare to chance (33%).

&#x20;  - Export the trained model to \*\*ONNX\*\*, with a `model\_card.json` (version, training subjects, metrics, date). Workers load the model by version; the API can \*\*hot-swap\*\* the active model without restarting.

&#x20;  - Write the results to `docs/MODEL\_RESULTS.md`. Cross-subject EEG is hard — realistic accuracy is fine; never inflate numbers.

\---

&#x20;

\## Engineering requirements

\- \*\*Monorepo\*\* layout: `services/`, `dashboard/`, `ml/`, `tests/`, `docs/`, `scripts/`.

\- \*\*Docker Compose\*\* brings up everything (Postgres, Redis, ingest, workers, api, dashboard, replayer) with one command. I'm on \*\*Windows with Docker Desktop\*\*, so scripts must work on Windows (prefer Python scripts or npm scripts over bash-only Makefiles).

\- \*\*Config\*\* via environment variables with a committed `.env.example` (never commit `.env`).

\- \*\*Tests (pytest):\*\*

&#x20; - Unit: filtering, windowing, gap/duplicate detection, packet validation.

&#x20; - Integration: replayer → ingest → worker → API end-to-end produces predictions.

&#x20; - Fault tolerance: kill a worker mid-stream and show no windows are lost.

\- \*\*Load test\*\* script: run N concurrent streams (1, 10, 25, 50), measure throughput and p50/p95/p99 latency, write results to `docs/BENCHMARKS.md` with a short analysis of where the bottleneck is.

\- \*\*CI (GitHub Actions):\*\* lint (ruff), type-check where reasonable, run tests, build the dashboard on every push/PR.

\- \*\*Code quality:\*\* type hints, small focused modules, structured logging, clear error handling.

\- \*\*README.md:\*\* one-paragraph pitch, architecture diagram, demo GIF/screenshot placeholder, quickstart, design decisions \& trade-offs, results tables (model + latency), "what I'd do next". Include a note that this is a research/portfolio project, \*\*not a medical device\*\*.

\---

&#x20;

\## Build in phases

Work through these in order. \*\*At the end of each phase: run the tests, commit with clear messages, push, then stop and give me a short summary (what was built, how to run it, anything I should understand) and wait for my OK before starting the next phase.\*\*

&#x20;

1\. \*\*Foundation\*\* — repo structure, `.gitignore`, Docker Compose with Postgres + Redis, dataset download script, CI skeleton, README stub. Create the GitHub repo and push.

2\. \*\*Replayer + Ingest\*\* — stream real EEG data at real-time rate; ingest validates and writes to Redis Streams; gap/duplicate detection with tests.

3\. \*\*ML training\*\* — baseline CSP+LDA, then EEGNet; subject-wise evaluation; ONNX export; `docs/MODEL\_RESULTS.md`.

4\. \*\*Inference workers\*\* — streaming filter + windowing + ONNX inference via consumer groups; latency measurement; worker-crash test.

5\. \*\*API + Postgres\*\* — REST + WebSocket endpoints, persistence, migrations, model hot-swap.

6\. \*\*Dashboard\*\* — live traces, predictions, metrics, session controls.

7\. \*\*Hardening\*\* — fault injection end-to-end, load tests, `docs/BENCHMARKS.md`, CI green.

8\. \*\*Polish\*\* — final README with diagram and results, cleanup, and a list of 3–4 strong resume bullet points based on the \*actual measured\* results.

9\. \*\*Accuracy sprint\*\* (after Phase 8) — improve the model, one change at a time, in this order:

&#x20;  1. \*\*Early-stopping patience:\*\* check patience (and validation set size) given that validation is only 8 subjects per fold and best epochs are often 1.

&#x20;  2. \*\*Euclidean alignment:\*\* align each subject's EEG covariance before training and inference.

&#x20;  3. \*\*Temporal smoothing of predictions:\*\* smooth over consecutive windows; chart accuracy vs. added latency.

&#x20;  4. \*\*Per-user fine-tuning:\*\* fine-tune the cross-subject EEGNet on N minutes of a held-out subject's data; plot accuracy vs. calibration minutes.

&#x20;  5. \*\*Optional — extra datasets via MOABB,\*\* mapped to our 21 motor channels.

&#x20;  Rules for this phase: every change is evaluated on the same held-out subjects (the existing 5 cross-subject folds), with no tuning on test folds (choices are made on validation data only). Every result goes in `docs/MODEL\_RESULTS.md`, including changes that don't help.

\## Stretch goals (only after phase 8, and only if I ask)

\- Rewrite the ingest service in Go for higher throughput and benchmark the difference.

\- ~~Per-user calibration~~ — moved into Phase 9, step 4.

\- Deploy a live demo to the cloud.

\## Rules

\- Don't invent results. Every number in the docs must come from code you actually ran.

\- Ask me before adding paid services or anything that needs an account/API key.

\- If something in this spec turns out to be a bad idea once you're building, tell me and propose an alternative instead of silently changing it.

&#x20;

