"""Worker against a real Redis: batching, ack-after-publish, reclaiming, dead-lettering, and
loss accounting when the ns:windows length cap trims jobs nobody acknowledged."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from redis import Redis

from neurostream_common import keys
from neurostream_common.channels import EEGMMIDB_CHANNELS
from neurostream_common.codec import WindowJob, decode_prediction, encode_window_job
from neurostream_common.model_card import ModelCardError, load_model_card
from neurostream_common.windowing import Window
from tests.integration.conftest import (
    MODEL_DIR,
    SERVED_MODEL,
    assert_conserved,
    make_worker,
    stream_accounting,
    worker_metrics,
)

pytestmark = pytest.mark.integration

PAD, CORE = 80, 320  # 0.5 s pad + 2 s window at 160 Hz, what the served model expects


def register_session(r: Redis, session_id: str = "s1") -> None:
    r.hset(keys.session_meta(session_id), mapping={"ch_names": json.dumps(EEGMMIDB_CHANNELS)})


def job_fields(end_sample: int, session_id: str = "s1", sfreq: float = 160.0) -> dict:  # type: ignore[type-arg]
    rng = np.random.default_rng(end_sample)
    span = PAD + CORE
    w = Window(
        start_sample=end_sample - span + 1,
        data=rng.normal(0, 10, (64, span)).astype(np.float32),
        labels=np.full(span, 1, np.int8),
        pad_samples=PAD,
    )
    now = time.time()
    return encode_window_job(WindowJob(session_id, w, sfreq, ingest_ts=now, emitted_ts=now))


def add_jobs(r: Redis, n: int, start: int = 0, maxlen: int | None = None) -> None:
    for i in range(start, start + n):
        r.xadd(
            keys.WINDOWS_STREAM,
            job_fields(1000 + 80 * i),  # type: ignore[arg-type]
            maxlen=maxlen,
            approximate=False,  # exact trimming, so the expected numbers are exact
        )


def predictions(r: Redis) -> list:  # type: ignore[type-arg]
    return [decode_prediction(f) for _, f in r.xrange(keys.PREDICTIONS_STREAM)]


def test_a_batch_is_whatever_is_available_and_never_waits(sync_redis: Redis) -> None:
    register_session(sync_redis)
    w = make_worker(batch_max=32, block_ms=2000)
    w.start()
    add_jobs(sync_redis, 3)
    t0 = time.perf_counter()
    assert w.step() == 3
    # Three jobs available, batch_max 32: processed at once, without waiting for more.
    assert time.perf_counter() - t0 < 1.0
    preds = predictions(sync_redis)
    assert [p.batch_size for p in preds] == [3, 3, 3]
    assert {p.window_id for p in preds} == {f"s1:{1000 + 80 * i}" for i in range(3)}


def test_predictions_are_published_then_acked(sync_redis: Redis) -> None:
    register_session(sync_redis)
    w = make_worker()
    w.start()
    add_jobs(sync_redis, 5)
    w.run_until_idle()
    preds = predictions(sync_redis)
    assert len(preds) == 5
    for p in preds:
        assert sum(p.probabilities) == pytest.approx(1.0, abs=1e-4)
        assert p.predicted_class == int(np.argmax(p.probabilities))
        assert p.true_label == 1 and p.model == SERVED_MODEL and not p.reclaimed
        assert min(p.queue_ms, p.preprocess_ms, p.inference_ms) >= 0
        assert p.latency_ms == pytest.approx(p.ingest_ms + p.queue_ms + p.worker_ms, abs=1e-6)
        assert p.worker_ms >= p.inference_ms
    acc = stream_accounting(sync_redis)
    assert acc["acked"] == 5 and acc["pending"] == 0 and acc["lag"] == 0
    assert_conserved(acc)


def test_batched_inference_equals_one_window_at_a_time(sync_redis: Redis) -> None:
    register_session(sync_redis)
    add_jobs(sync_redis, 6)
    batched = make_worker("wb", batch_max=32)
    batched.start()
    batched.run_until_idle()
    a = {p.window_id: p.probabilities for p in predictions(sync_redis)}
    sync_redis.delete(keys.WINDOWS_STREAM, keys.PREDICTIONS_STREAM)
    add_jobs(sync_redis, 6)
    single = make_worker("ws", batch_max=1)
    single.start()
    single.run_until_idle()
    b = {p.window_id: p.probabilities for p in predictions(sync_redis)}
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-6)


def test_jobs_of_a_crashed_consumer_are_reclaimed(sync_redis: Redis) -> None:
    register_session(sync_redis)
    w = make_worker(reclaim_min_idle_ms=0)
    w.start()
    add_jobs(sync_redis, 5)
    # A worker that reads and then dies leaves exactly this: jobs pending, never acked.
    sync_redis.xreadgroup(keys.WORKER_GROUP, "dead-worker", {keys.WINDOWS_STREAM: ">"}, count=5)
    w.run_until_idle()
    preds = predictions(sync_redis)
    assert len(preds) == 5 and all(p.reclaimed for p in preds)
    acc = stream_accounting(sync_redis)
    assert acc["pending"] == 0 and acc["lost_trimmed_pending"] == 0
    assert_conserved(acc)


def test_trimming_unacked_jobs_is_counted_as_lost(sync_redis: Redis) -> None:
    """The ns:windows cap under overload: 10 jobs, a consumer takes 3 and dies, then 30
    more arrive with a cap of 20. Redis trims the oldest 20: the 3 pending ones and 17 that
    nobody had read. All 20 must be counted as lost; the 20 survivors must be processed."""
    register_session(sync_redis)
    w = make_worker(reclaim_min_idle_ms=0)
    w.start()
    add_jobs(sync_redis, 10, maxlen=20)
    sync_redis.xreadgroup(keys.WORKER_GROUP, "dead-worker", {keys.WINDOWS_STREAM: ">"}, count=3)
    add_jobs(sync_redis, 30, start=10, maxlen=20)
    w.run_until_idle()

    acc = stream_accounting(sync_redis)
    assert acc["entries_added"] == 40
    assert acc["lost_trimmed_undelivered"] == 17
    assert acc["lost_trimmed_pending"] == 3
    assert acc["acked"] == 20 and acc["pending"] == 0 and acc["lag"] == 0
    assert_conserved(acc)
    # The survivors are exactly the 20 newest jobs.
    assert {p.window_end_sample for p in predictions(sync_redis)} == {
        1000 + 80 * i for i in range(20, 40)
    }


def test_loss_count_is_exact_with_several_workers_reading(sync_redis: Redis) -> None:
    """Interleaved reads by two workers while trimming: the atomic read script must not
    double-count or miss skipped jobs."""
    register_session(sync_redis)
    a, b = make_worker("wa", batch_max=4), make_worker("wb", batch_max=3)
    a.start()
    b.start()
    for round_ in range(10):
        add_jobs(sync_redis, 15, start=15 * round_, maxlen=10)
        a.step()
        b.step()
    a.run_until_idle()
    b.run_until_idle()
    acc = stream_accounting(sync_redis)
    assert acc["entries_added"] == 150
    assert acc["lost_trimmed_undelivered"] > 0
    assert acc["acked"] == len(predictions(sync_redis))
    assert_conserved(acc)


def test_a_job_the_model_cannot_take_is_dead_lettered_at_once(sync_redis: Redis) -> None:
    register_session(sync_redis)
    w = make_worker()
    w.start()
    sync_redis.xadd(keys.WINDOWS_STREAM, job_fields(1000, sfreq=128.0))  # type: ignore[arg-type]
    sync_redis.xadd(keys.WINDOWS_STREAM, job_fields(1080, session_id="unknown"))  # type: ignore[arg-type]
    add_jobs(sync_redis, 1, start=5)
    w.run_until_idle()
    dead = sync_redis.xrange(keys.WINDOWS_DEAD_STREAM)
    reasons = " | ".join(f[b"dead_reason"].decode() for _, f in dead)
    assert len(dead) == 2
    assert "128.0 Hz" in reasons and "no metadata for session unknown" in reasons
    assert len(predictions(sync_redis)) == 1  # the good job in the same batch still went through
    acc = stream_accounting(sync_redis)
    assert acc["dead_lettered"] == 2 and acc["pending"] == 0
    assert_conserved(acc)


def test_a_job_delivered_too_often_is_dead_lettered(sync_redis: Redis) -> None:
    register_session(sync_redis)
    w = make_worker(reclaim_min_idle_ms=0, max_deliveries=3)
    w.start()
    add_jobs(sync_redis, 1)
    (entry_id, _), *_ = sync_redis.xreadgroup(
        keys.WORKER_GROUP, "crashy", {keys.WINDOWS_STREAM: ">"}, count=1
    )[0][1]
    for _ in range(2):  # two more workers took it and died: delivered 3 times so far
        sync_redis.xclaim(keys.WINDOWS_STREAM, keys.WORKER_GROUP, "crashy", 0, [entry_id])
    w.run_until_idle()  # 4th delivery > max_deliveries: set aside instead of retried
    assert len(sync_redis.xrange(keys.WINDOWS_DEAD_STREAM)) == 1
    assert predictions(sync_redis) == []
    assert_conserved(stream_accounting(sync_redis))


def test_worker_metrics_and_heartbeat(sync_redis: Redis) -> None:
    register_session(sync_redis)
    w = make_worker("hb")
    w.start()
    add_jobs(sync_redis, 2)
    w.run_until_idle()
    w.heartbeat(force=True)
    hb = sync_redis.hgetall(keys.worker_heartbeat("hb"))
    assert hb[b"model"] == SERVED_MODEL.encode() and int(hb[b"predictions"]) == 2
    assert 0 < sync_redis.ttl(keys.worker_heartbeat("hb")) <= 5
    assert sync_redis.sismember(keys.WORKERS_SET, "hb")
    m = worker_metrics(sync_redis)
    assert m["delivered"] == 2 and m["predictions"] == 2 and m["acked"] == 2


def test_a_modified_model_file_is_refused(tmp_path: Path) -> None:
    d = tmp_path / SERVED_MODEL
    shutil.copytree(MODEL_DIR / SERVED_MODEL, d)
    load_model_card(d)  # intact copy loads
    data = bytearray((d / "model.onnx").read_bytes())
    data[-10] ^= 0xFF  # flip bits in the weights
    (d / "model.onnx").write_bytes(bytes(data))
    with pytest.raises(ModelCardError, match="sha256"):
        load_model_card(d)
