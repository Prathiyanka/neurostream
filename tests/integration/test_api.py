"""API end to end: replayer service -> ingest -> worker -> Redis -> persister -> Postgres,
driven only through the API, plus hot-swap, metrics and the live WebSocket."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from redis import Redis as SyncRedis
from sqlalchemy import create_engine, func, select
from websockets.asyncio.client import connect

from api.db import metadata, predictions, sessions, sync_url
from api.persister import Persister
from neurostream_common import keys
from neurostream_common.codec import Prediction, encode_prediction
from tests.integration.api_fixtures import (  # noqa: F401  (fixtures)
    Api,
    ServedApi,
    api,
    database,
    make_api,
    model_dir_copy,
)
from tests.integration.conftest import SERVED_MODEL, IngestServer, WorkerThread, make_worker

pytestmark = pytest.mark.integration


async def eventually(check: Callable[[], Awaitable[bool]], timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await check():
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def fake_prediction(window_end_sample: int, session_id: str = "ext1") -> Prediction:
    now = time.time()
    return Prediction(
        window_id=f"{session_id}:{window_end_sample}",
        session_id=session_id,
        window_end_sample=window_end_sample,
        window_end_ts=now,
        predicted_class=1,
        probabilities=(0.2, 0.5, 0.3),
        true_label=1,
        filled_samples=0,
        model=SERVED_MODEL,
        worker_id="w",
        reclaimed=False,
        batch_size=1,
        ingest_ms=0.1,
        queue_ms=1.0,
        preprocess_ms=0.5,
        inference_ms=1.0,
        worker_ms=2.0,
        latency_ms=3.1,
        publish_ts=now,
    )


def count_rows(db_url: str, table: object, **where: object) -> int:
    eng = create_engine(sync_url(db_url))
    with eng.connect() as conn:
        q = select(func.count()).select_from(table)  # type: ignore[arg-type]
        for col, val in where.items():
            q = q.where(getattr(table.c, col) == val)  # type: ignore[attr-defined]
        n = int(conn.execute(q).scalar_one())
    eng.dispose()
    return n


# ---- schema ----------------------------------------------------------------------------


def test_migrations_match_the_schema_in_code(database: str) -> None:  # noqa: F811
    eng = create_engine(sync_url(database))
    with eng.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), metadata)
    eng.dispose()
    assert diff == [], f"migration and api/db.py disagree: {diff}"


# ---- persistence -----------------------------------------------------------------------


async def test_persistence_is_idempotent_and_survives_a_restart(
    api: Api,  # noqa: F811
    sync_redis: SyncRedis,
) -> None:
    sync_redis.hset(keys.session_meta("ext1"), mapping={"subject": "7"})
    p0, p1 = fake_prediction(399), fake_prediction(479)
    for p in (p0, p1, p0):  # p0 twice, as after a worker crash
        sync_redis.xadd(keys.PREDICTIONS_STREAM, encode_prediction(p))  # type: ignore[arg-type]
    await eventually(
        lambda: asyncio.sleep(0, count_rows(api.db_url, predictions) == 2), 10, "rows stored"
    )
    await asyncio.sleep(0.5)
    assert count_rows(api.db_url, predictions) == 2  # the duplicate was not stored
    assert count_rows(api.db_url, sessions, id="ext1", source="external") == 1
    persisted = sync_redis.hgetall(keys.PERSISTER_METRICS)
    assert int(persisted[b"persisted"]) == 2 and int(persisted[b"duplicates"]) == 1


async def test_restarted_persister_stores_its_unacked_batch_once(
    database: str,  # noqa: F811
    sync_redis: SyncRedis,
) -> None:
    from redis.asyncio import Redis

    from api.db import make_engine
    from tests.integration.conftest import redis_test_url

    sync_redis.hset(keys.session_meta("ext2"), mapping={"subject": "3"})
    redis, engine = Redis.from_url(redis_test_url()), make_engine(database)
    try:
        first = Persister(redis, engine, consumer="api-x", batch=100)
        await first.ensure_group()
        for i in range(3):
            sync_redis.xadd(
                keys.PREDICTIONS_STREAM,
                encode_prediction(fake_prediction(400 + 80 * i, "ext2")),  # type: ignore[arg-type]
            )
        entries, _ = await first._read_new()  # read, then "crash" before storing
        assert len(entries) == 3
        restarted = Persister(redis, engine, consumer="api-x", batch=100)
        assert await restarted.step(block_ms=10) == 3  # re-read as pending, stored
        assert await restarted.step(block_ms=10) == 0
        assert await first.store(entries) == 0  # the old batch again: nothing new
        assert count_rows(database, predictions, session_id="ext2") == 3
        groups = sync_redis.xinfo_groups(keys.PREDICTIONS_STREAM)
        assert next(g for g in groups if g["name"] == b"api")["pending"] == 0
    finally:
        await engine.dispose()
        await redis.aclose()


# ---- sessions through the API ----------------------------------------------------------


async def test_a_session_started_through_the_api_is_streamed_predicted_and_stored(
    api: Api,  # noqa: F811
    sync_redis: SyncRedis,
) -> None:
    with WorkerThread(make_worker()):
        r = await api.client.post(
            "/sessions", json={"synthetic": True, "speed": 40, "max_seconds": 10}
        )
        assert r.status_code == 201, r.text
        sid = r.json()["id"]

        async def stored_and_finished() -> bool:
            s = (await api.client.get(f"/sessions/{sid}")).json()
            emitted = s["ingest"].get("windows_emitted", -1)
            return s["status"] == "finished" and s["summary"]["predictions"] == emitted

        await eventually(stored_and_finished, 30, "every window stored and session finished")
    s = (await api.client.get(f"/sessions/{sid}")).json()
    assert s["source"] == "api" and s["replay"]["status"] == "finished"
    assert s["summary"]["predictions"] == 16  # 10 s at 2 windows/s, minus the first 2.5 s
    assert s["summary"]["latency_ms"]["p50"] > 0
    assert s["summary"]["models"] == [SERVED_MODEL]
    assert 0 <= s["summary"]["balanced_accuracy"] <= 1
    preds = (await api.client.get(f"/sessions/{sid}/predictions?limit=5")).json()
    assert len(preds) == 5 and preds[0]["window_id"] == f"{sid}:399"
    listed = (await api.client.get("/sessions")).json()
    assert [x["id"] for x in listed] == [sid]


async def test_stopping_a_session_through_the_api(api: Api) -> None:  # noqa: F811
    r = await api.client.post("/sessions", json={"synthetic": True, "speed": 1})
    sid = r.json()["id"]
    await asyncio.sleep(0.5)
    stopped = await api.client.delete(f"/sessions/{sid}")
    assert stopped.status_code == 200 and stopped.json()["status"] == "stopped"
    s = (await api.client.get(f"/sessions/{sid}")).json()
    assert s["status"] == "stopped" and s["ended_at"] is not None


async def test_session_errors(api: Api, sync_redis: SyncRedis) -> None:  # noqa: F811
    bad = await api.client.post("/sessions", json={"synthetic": True, "subject": 2})
    assert bad.status_code == 422  # the replayer's validation, passed through
    assert (await api.client.get("/sessions/nope")).status_code == 404
    sync_redis.xadd(keys.PREDICTIONS_STREAM, encode_prediction(fake_prediction(399, "ext3")))  # type: ignore[arg-type]
    await eventually(
        lambda: asyncio.sleep(0, count_rows(api.db_url, sessions, id="ext3") == 1), 10, "row"
    )
    assert (await api.client.delete("/sessions/ext3")).status_code == 409  # not ours to stop


# ---- models ----------------------------------------------------------------------------


async def test_hot_swap_through_the_api(api: Api, sync_redis: SyncRedis) -> None:  # noqa: F811
    m = (await api.client.get("/models")).json()
    assert {x["id"] for x in m["available"]} >= {SERVED_MODEL, "eegnet-motor-1.0.0"}
    assert m["active"] == SERVED_MODEL
    with WorkerThread(make_worker(block_ms=50)) as w:
        r = await api.client.put("/models/active", json={"model": "eegnet-motor-1.0.0"})
        assert r.status_code == 200 and r.json()["previous"] == SERVED_MODEL

        async def swapped() -> bool:
            models = (await api.client.get("/models")).json()
            return bool(models["workers"].get("w1") == "eegnet-motor-1.0.0")

        await eventually(swapped, 15, "worker to report the new model")
        assert w.model.model_id == "eegnet-motor-1.0.0"
    assert (await api.client.put("/models/active", json={"model": "nope-1.0"})).status_code == 404
    assert (await api.client.put("/models/active", json={"model": "../x"})).status_code == 422


async def test_a_tampered_model_cannot_be_activated(
    ingest_server: IngestServer,
    database: str,  # noqa: F811
    tmp_path: Path,
) -> None:
    models = model_dir_copy(tmp_path)
    onnx = models / "eegnet-motor-1.0.0" / "model.onnx"
    data = bytearray(onnx.read_bytes())
    data[-10] ^= 0xFF
    onnx.write_bytes(bytes(data))
    async with make_api(ingest_server, database, model_dir=str(models)) as a:
        r = await a.client.put("/models/active", json={"model": "eegnet-motor-1.0.0"})
        assert r.status_code == 422 and "sha256" in r.json()["detail"]
        assert (await a.client.get("/models")).json()["active"] == SERVED_MODEL


# ---- metrics, health, live feed --------------------------------------------------------


async def test_metrics_health_and_history(api: Api, sync_redis: SyncRedis) -> None:  # noqa: F811
    with WorkerThread(make_worker()):
        # The first heartbeat comes after model warm-up, which is slower on CI runners.
        async def worker_up() -> bool:
            return bool((await api.client.get("/metrics")).json()["workers"]["count"] == 1)

        await eventually(worker_up, 20, "worker heartbeat")
        m = (await api.client.get("/metrics")).json()
        assert m["workers"]["count"] == 1 and m["workers"]["list"][0]["model"] == SERVED_MODEL
        assert set(m["jobs"]) >= {"acked", "stale", "lost", "dead_lettered"}

        async def snapshot_with_worker() -> bool:  # snapshots every 0.3 s in tests
            rows = (await api.client.get("/metrics/history?minutes=1")).json()
            return any(r["workers"] == 1 for r in rows)

        await eventually(snapshot_with_worker, 20, "a metrics snapshot")
    h = await api.client.get("/health")
    assert h.status_code == 200 and h.json() == {
        "status": "ok",
        "redis": True,
        "postgres": True,
        "replayer": True,
    }
    history = (await api.client.get("/metrics/history?minutes=1")).json()
    assert history and "stale" in history[-1]["counters"]


async def test_live_websocket_streams_predictions_traces_and_metrics(
    api: Api,  # noqa: F811
) -> None:
    async with ServedApi(api.app) as base:
        with WorkerThread(make_worker()):
            sid = (
                await api.client.post(
                    "/sessions", json={"synthetic": True, "speed": 4, "max_seconds": 20}
                )
            ).json()["id"]
            seen: dict[str, dict] = {}  # type: ignore[type-arg]
            async with connect(f"{base}/ws/live?session={sid}&channels=C3,Cz,C4") as ws:
                deadline = time.monotonic() + 20
                while len(seen) < 3 and time.monotonic() < deadline:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
                    seen.setdefault(msg["type"], msg)
            await api.client.delete(f"/sessions/{sid}")
    assert set(seen) == {"prediction", "trace", "metrics"}
    trace = seen["trace"]
    assert trace["session_id"] == sid and trace["channels"] == ["C3", "Cz", "C4"]
    assert trace["sfreq"] == 40.0 and len(trace["data"]) == 3
    assert len(trace["data"][0]) == 4  # one 16-sample packet -> 4 display samples
    assert seen["prediction"]["session_id"] == sid
    assert "stale" in seen["metrics"]["jobs"]
