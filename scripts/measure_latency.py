"""Measure end-to-end prediction latency through the running Docker Compose stack.

Replays N held-out EEGMMIDB subjects (from the served model's card, so none were used
for training) at real-time speed, one concurrent session each, then reads every
prediction the workers published and summarizes the latency breakdown:

    latency = ingest (window complete -> job produced)
            + queue  (job produced -> read by a worker: ingest buffer, XADD, waiting)
            + worker (read -> published: whole batch preprocessing + ONNX + encoding)

All timestamps come from containers on the same Docker VM clock, so they can be compared.
"Published" is the moment the worker sends its publish round trip (one Redis RTT, well
under a millisecond locally, is not included).

    docker compose up -d --wait
    python scripts/measure_latency.py --sessions 10 --workers 1 --seconds 60
    python scripts/measure_latency.py --summarize      # table over benchmarks/results/
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from redis import Redis

from neurostream_common import keys
from neurostream_common.codec import Prediction, decode_prediction
from neurostream_common.labels import UNKNOWN

REPO = Path(__file__).resolve().parents[1]
FIELDS = ("latency_ms", "ingest_ms", "queue_ms", "worker_ms", "preprocess_ms", "inference_ms")


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}  # keep /data a container path on Git Bash
    return subprocess.run(
        ["docker", "compose", *args], cwd=REPO, env=env, check=check, text=True,
        capture_output=True,
    )  # fmt: skip


def counters(r: Any) -> dict[str, int]:
    m = {k.decode(): int(v) for k, v in r.hgetall(keys.WORKER_METRICS).items()}
    try:
        m["entries_added"] = int(r.xinfo_stream(keys.WINDOWS_STREAM)["entries-added"])
    except Exception:  # stream does not exist yet
        m["entries_added"] = 0
    return m


def live_workers(r: Any) -> int:
    return sum(
        1 for w in r.smembers(keys.WORKERS_SET) if bool(r.exists(keys.worker_heartbeat(w.decode())))
    )


def drained(r: Any) -> bool:
    groups = r.xinfo_groups(keys.WINDOWS_STREAM)
    g = next((g for g in groups if g["name"] == keys.WORKER_GROUP.encode()), None)
    return g is not None and g["pending"] == 0 and (g["lag"] or 0) == 0


def pct(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def balanced_accuracy(preds: list[Prediction]) -> tuple[float, int]:
    known = [p for p in preds if p.true_label != UNKNOWN]
    recalls = []
    for c in (0, 1, 2):
        of_c = [p for p in known if p.true_label == c]
        if of_c:
            recalls.append(sum(p.predicted_class == c for p in of_c) / len(of_c))
    return (float(np.mean(recalls)) if recalls else float("nan")), len(known)


def summarize(results_dir: Path) -> str:
    """Markdown table over every result file, so docs only contain measured numbers."""
    rows = [
        "| Workers | Run | Windows | Lost | Stale | Total p50 | p95 | p99 | max "
        "| Queue p50 / p95 | Worker p50 / p95 | Mean batch |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    files = sorted(results_dir.glob("latency_*.json"))
    for f in sorted(files, key=lambda f: (json.loads(f.read_text())["config"]["workers"], f.name)):
        d = json.loads(f.read_text())
        lat, q, w = d["latency"]["latency_ms"], d["latency"]["queue_ms"], d["latency"]["worker_ms"]
        lost = d["lost_trimmed_undelivered"] + d["lost_trimmed_pending"] + d["dead_lettered"]
        rows.append(
            f"| {d['config']['workers']} | {f.stem.rsplit('_', 1)[-1]} | {d['unique_windows']} "
            f"| {lost} | {d.get('stale', 'n/a')} "  # n/a: run predates stale skipping
            f"| {lat['p50']:.1f} ms | {lat['p95']:.1f} ms | {lat['p99']:.1f} ms "
            f"| {lat['max']:.1f} ms | {q['p50']:.1f} / {q['p95']:.1f} ms "
            f"| {w['p50']:.1f} / {w['p95']:.1f} ms | {d['batch_size']['mean']:.2f} |"
        )
    return "\n".join(rows)


def main() -> int:
    if "--summarize" in sys.argv:
        print(summarize(REPO / "benchmarks" / "results"))
        return 0
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--workers", type=int, default=1, help="scale the worker service to this")
    p.add_argument("--seconds", type=float, default=60.0, help="recording time per session")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--model", default="eegnet-motor-1.0.1")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    card = json.loads((REPO / "models" / args.model / "model_card.json").read_text())
    subjects = card["training"]["test_subjects"][: args.sessions]
    if len(subjects) < args.sessions:
        sys.exit(f"only {len(subjects)} held-out subjects in the model card")

    r: Any = Redis.from_url(args.redis_url)  # Any: redis-py sync stubs return Awaitable unions
    compose("up", "-d", "--wait", "--scale", f"worker={args.workers}")
    deadline = time.monotonic() + 60
    while live_workers(r) != args.workers:
        if time.monotonic() > deadline:
            sys.exit(f"expected {args.workers} live workers, see {live_workers(r)}")
        time.sleep(0.5)

    before = counters(r)
    last = r.xrevrange(keys.PREDICTIONS_STREAM, count=1)
    start_id = last[0][0].decode() if last else "0-0"
    print(f"replaying subjects {subjects} for {args.seconds:g} s at {args.speed:g}x "
          f"with {args.workers} worker(s)...", flush=True)  # fmt: skip
    t0 = time.monotonic()
    compose(
        "run", "--rm", "replayer", "--url", "ws://ingest:8001/ws/device", "--data-dir", "/data",
        "--subjects", *map(str, subjects), "--speed", str(args.speed),
        "--max-seconds", str(args.seconds),
    )  # fmt: skip
    wall = time.monotonic() - t0
    deadline = time.monotonic() + 30
    while not drained(r) and time.monotonic() < deadline:
        time.sleep(0.2)
    time.sleep(0.5)
    after = counters(r)

    preds = [decode_prediction(f) for _, f in r.xrange(keys.PREDICTIONS_STREAM, min=f"({start_id}")]
    if not preds:
        sys.exit("no predictions were published")
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    bal_acc, n_known = balanced_accuracy(preds)
    batch_sizes = Counter(p.batch_size for p in preds)
    result: dict[str, Any] = {
        "config": {
            "sessions": args.sessions,
            "workers": args.workers,
            "seconds_per_session": args.seconds,
            "speed": args.speed,
            "model": args.model,
            "subjects": subjects,
            "ort_threads_per_worker": int(os.environ.get("ORT_THREADS", "1")),
        },
        "host": {"platform": platform.platform(), "cpus": os.cpu_count()},
        "wall_seconds": wall,
        "predictions": len(preds),
        "unique_windows": len({p.window_id for p in preds}),
        "jobs_added": delta["entries_added"],
        "acked": delta.get("acked", 0),
        "lost_trimmed_undelivered": delta.get("lost_trimmed_undelivered", 0),
        "lost_trimmed_pending": delta.get("lost_trimmed_pending", 0),
        "dead_lettered": delta.get("dead_lettered", 0),
        "stale": delta.get("stale", 0),
        "reclaimed": delta.get("reclaimed", 0),
        "latency": {f: pct([getattr(p, f) for p in preds]) for f in FIELDS},
        "batch_size": {
            "mean": float(np.mean([p.batch_size for p in preds])),
            "histogram": dict(sorted(batch_sizes.items())),
        },
        "live_balanced_accuracy": bal_acc,
        "live_labelled_windows": n_known,
    }
    lat = result["latency"]
    print(f"{len(preds)} predictions ({result['unique_windows']} windows), "
          f"lost {result['lost_trimmed_undelivered'] + result['lost_trimmed_pending']}, "
          f"dead {result['dead_lettered']}, stale {result['stale']}")  # fmt: skip
    print("| Stage | p50 | p95 | p99 | max |\n|---|---|---|---|---|")
    for f in FIELDS:
        x = lat[f]
        print(f"| {f} | {x['p50']:.2f} | {x['p95']:.2f} | {x['p99']:.2f} | {x['max']:.2f} |")
    print(
        f"batch size: mean {result['batch_size']['mean']:.2f}, {result['batch_size']['histogram']}"
    )
    print(f"live balanced accuracy {bal_acc:.3f} on {n_known} labelled windows")
    out = args.out or REPO / "benchmarks" / "results" / (
        f"latency_{args.sessions}s_{args.workers}w.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
