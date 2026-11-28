"""Replayer -> ingest -> workers -> predictions, end to end against a real Redis.

* every window ingest produces gets a prediction, identical to the offline pipeline;
* a worker killed mid-stream (hard kill, like ``kill -9``) loses no windows;
* under overload, jobs trimmed by the ns:windows cap are counted as lost, and the stream
  accounting still balances.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from redis import Redis as SyncRedis

from neurostream_common import keys
from neurostream_common.codec import Prediction, decode_prediction, decode_window_job
from neurostream_common.model_card import load_model_card
from neurostream_common.windowing import WindowSpec, extract_windows
from replayer.recording import Recording, edf_path, load_subject, synthetic_recording
from replayer.streamer import SessionConfig, SessionStreamer
from tests.integration.conftest import (
    MODEL_DIR,
    REPO_ROOT,
    SERVED_MODEL,
    IngestServer,
    WorkerThread,
    assert_conserved,
    make_worker,
    redis_test_url,
    stream_accounting,
    worker_metrics,
)
from worker.model import InferenceModel

pytestmark = pytest.mark.integration


async def replay(
    server: IngestServer, rec: Recording, session_id: str, speed: float = 40.0, **kw: object
) -> None:
    cfg = SessionConfig(
        url=server.ws_url,
        session_id=session_id,
        speed=speed,
        max_connect_attempts=5,
        **kw,  # type: ignore[arg-type]
    )
    await SessionStreamer(cfg, rec).run()


async def wait_until(cond: Callable[[], bool], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def drained(r: SyncRedis) -> bool:
    acc = stream_accounting(r)
    return acc["lag"] == 0 and acc["pending"] == 0


def emitted_window_ids(r: SyncRedis) -> set[str]:
    return {decode_window_job(f).window_id for _, f in r.xrange(keys.WINDOWS_STREAM)}


def all_predictions(r: SyncRedis) -> list[Prediction]:
    return [decode_prediction(f) for _, f in r.xrange(keys.PREDICTIONS_STREAM)]


def offline_probabilities(rec: Recording, pad_samples: int) -> dict[int, np.ndarray]:
    """What the training-side pipeline computes for each window, keyed by end sample.

    The replayer rounds samples to 0.001 uV for the JSON wire format, so the reference is
    computed on identically rounded input: any remaining difference is real skew."""
    card = load_model_card(MODEL_DIR / SERVED_MODEL)
    model = InferenceModel(card)
    spec = WindowSpec(sfreq=rec.sfreq, pad_s=pad_samples / rec.sfreq)
    wire = np.round(rec.data.astype(np.float64), 3).astype(np.float32)
    out: dict[int, np.ndarray] = {}
    for w in extract_windows(wire, rec.labels, spec):
        x = model._preprocessor(rec.ch_names)(w.data, w.pad_samples)
        out[w.end_sample] = model.infer([np.asarray(x, np.float32)])[0]
    return out


# ---- every window, same answer as offline ---------------------------------------------


async def check_matches_offline(
    server: IngestServer, sync_redis: SyncRedis, rec: Recording
) -> None:
    with WorkerThread(make_worker()):
        await replay(server, rec, "s1")
        n_emitted = len(emitted_window_ids(sync_redis))
        await wait_until(
            lambda: len(all_predictions(sync_redis)) >= n_emitted and drained(sync_redis),
            30,
            "all windows predicted",
        )
    preds = all_predictions(sync_redis)
    assert n_emitted > 0
    assert {p.window_id for p in preds} == emitted_window_ids(sync_redis)
    assert len(preds) == n_emitted  # no duplicates without a crash
    pad = decode_window_job(
        sync_redis.xrange(keys.WINDOWS_STREAM, count=1)[0][1]
    ).window.pad_samples
    offline = offline_probabilities(rec, pad)
    for p in preds:
        np.testing.assert_allclose(p.probabilities, offline[p.window_end_sample], atol=2e-6)
        assert p.latency_ms > 0 and p.batch_size >= 1


async def test_every_window_is_predicted_exactly_as_offline(
    ingest_server: IngestServer, sync_redis: SyncRedis
) -> None:
    await check_matches_offline(
        ingest_server, sync_redis, synthetic_recording(n_channels=64, seconds=12.0)
    )


async def test_real_eeg_predictions_match_offline(
    ingest_server: IngestServer, sync_redis: SyncRedis
) -> None:
    data_dir = REPO_ROOT / "data"
    card = load_model_card(MODEL_DIR / SERVED_MODEL)
    subject = int(card.raw["training"]["test_subjects"][0])  # held out from training
    if not edf_path(data_dir, subject, 4).exists():
        pytest.skip("EEGMMIDB data not downloaded")
    rec = load_subject(subject, data_dir)
    n = round(40 * rec.sfreq)  # first 40 s
    rec = Recording(rec.subject, rec.sfreq, rec.ch_names, rec.data[:, :n], rec.labels[:n])
    await check_matches_offline(ingest_server, sync_redis, rec)


# ---- hard-killed worker loses nothing -------------------------------------------------


def spawn_worker(worker_id: str, log_path: Path, **env: str) -> subprocess.Popen[bytes]:
    full_env = {
        **os.environ,
        "REDIS_URL": redis_test_url(),
        "MODEL_DIR": str(MODEL_DIR),
        "MODEL": SERVED_MODEL,
        "WORKER_ID": worker_id,
        "WORKER_BLOCK_MS": "100",
        "RECLAIM_INTERVAL_S": "0.2",
        "RECLAIM_MIN_IDLE_MS": "1000",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "services" / "worker"), os.environ.get("PYTHONPATH", "")]
        ),
        **env,
    }
    log = log_path.open("wb")
    return subprocess.Popen(
        [sys.executable, "-m", "worker.main"], env=full_env, stdout=log, stderr=log
    )


async def test_killing_a_worker_mid_stream_loses_no_windows(
    ingest_server: IngestServer, sync_redis: SyncRedis, tmp_path: Path
) -> None:
    # Slow workers (0.3 s per batch) so the victim is certain to hold unacked jobs.
    slow = {"WORKER_FAULT_DELAY_S": "0.3", "WORKER_BATCH_MAX": "4"}
    victim = spawn_worker("victim", tmp_path / "victim.log", **slow)
    survivor = spawn_worker("survivor", tmp_path / "survivor.log", **slow)
    try:
        await wait_until(
            lambda: sync_redis.exists(keys.worker_heartbeat("victim"))
            and sync_redis.exists(keys.worker_heartbeat("survivor")),
            30,
            "both workers up",
        )
        rec = synthetic_recording(n_channels=64, seconds=30.0)
        replay_task = asyncio.create_task(replay(ingest_server, rec, "s1", speed=6.0))

        def victim_holds_jobs() -> bool:
            consumers = sync_redis.xinfo_consumers(keys.WINDOWS_STREAM, keys.WORKER_GROUP)
            return any(c["name"] == b"victim" and c["pending"] > 0 for c in consumers)

        await wait_until(victim_holds_jobs, 20, "victim to hold pending jobs")
        victim.kill()  # TerminateProcess on Windows, SIGKILL elsewhere: no cleanup runs
        victim.wait(10)
        await replay_task
        n_emitted = len(emitted_window_ids(sync_redis))
        await wait_until(
            lambda: drained(sync_redis) and len(all_predictions(sync_redis)) >= n_emitted,
            60,
            "survivor to finish every window",
        )
    finally:
        for p in (victim, survivor):
            if p.poll() is None:
                p.kill()
                p.wait(10)

    preds = all_predictions(sync_redis)
    predicted = {p.window_id for p in preds}
    assert predicted == emitted_window_ids(sync_redis), "a window was lost"
    m = worker_metrics(sync_redis)
    assert m.get("reclaimed", 0) > 0, "the crash path was not exercised"
    assert m.get("lost_trimmed_pending", 0) == m.get("lost_trimmed_undelivered", 0) == 0
    # At-least-once: windows the victim published but never acked come out twice.
    duplicates = len(preds) - len(predicted)
    print(f"\nwindows {len(predicted)}, reclaimed {m['reclaimed']}, duplicates {duplicates}")
    assert_conserved(stream_accounting(sync_redis))


# ---- overload: the length cap trims unacknowledged jobs ---------------------------------


@pytest.mark.parametrize("ingest_settings", [{"windows_maxlen": 20}], indirect=True)
async def test_overload_trimming_is_counted_as_lost(
    ingest_server: IngestServer, sync_redis: SyncRedis
) -> None:
    """Three sessions at 40x real time into one deliberately slow worker (about 10 jobs/s
    against about 240/s arriving) and a cap of ~20 jobs: most jobs are trimmed. Every one
    of them must show up as lost, and the books must balance."""
    worker = make_worker(batch_max=2, fault_delay_s=0.2)
    with WorkerThread(worker):
        recs = [synthetic_recording(n_channels=64, seconds=20.0, seed=i) for i in range(3)]
        await asyncio.gather(*(replay(ingest_server, r, f"s{i}") for i, r in enumerate(recs)))
        await wait_until(lambda: drained(sync_redis), 60, "the worker to drain the stream")

    acc = stream_accounting(sync_redis)
    emitted = sum(
        int(sync_redis.hget(keys.ingest_metrics(f"s{i}"), "windows_emitted") or 0)  # type: ignore[arg-type]
        for i in range(3)
    )
    print(f"\n{acc}")
    assert acc["entries_added"] == emitted  # every job ingest produced reached the stream
    assert acc["lost_trimmed_undelivered"] > 0
    assert acc["acked"] == len(all_predictions(sync_redis))
    assert_conserved(acc)
