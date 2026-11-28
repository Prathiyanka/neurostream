from __future__ import annotations

import asyncio
import os
import socket
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from redis import Redis as SyncRedis
from redis.asyncio import Redis

from ingest.app import create_app
from ingest.settings import Settings
from neurostream_common import keys
from neurostream_common.model_card import load_model_card
from worker.loop import Worker
from worker.model import InferenceModel
from worker.settings import Settings as WorkerSettings

# Integration tests use their own Redis database and wipe only that one.
TEST_REDIS_DB = 15


def redis_test_url() -> str:
    base = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    return base.rsplit("/", 1)[0] + f"/{TEST_REDIS_DB}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class IngestServer:
    ws_url: str
    http_url: str
    redis: Redis


@pytest.fixture
async def redis_client() -> AsyncIterator[Redis]:
    r = Redis.from_url(redis_test_url())
    await r.flushdb()
    yield r
    await r.flushdb()
    await r.aclose()


@pytest.fixture
def ingest_settings(request: pytest.FixtureRequest) -> Settings:
    """Ingest config; override fields per test with
    ``@pytest.mark.parametrize("ingest_settings", [{...}], indirect=True)``."""
    overrides: dict[str, Any] = getattr(request, "param", {})
    return Settings(redis_url=redis_test_url(), **overrides)


@pytest.fixture
async def ingest_server(
    redis_client: Redis, ingest_settings: Settings
) -> AsyncIterator[IngestServer]:
    port = free_port()
    app = create_app(ingest_settings)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", ws_max_size=4 << 20)
    )
    task = asyncio.create_task(server.serve())
    for _ in range(500):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started, "ingest server did not start"
    yield IngestServer(f"ws://127.0.0.1:{port}/ws/device", f"http://127.0.0.1:{port}", redis_client)
    server.should_exit = True
    await asyncio.wait_for(task, 10)


# ---- worker helpers -------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models"
SERVED_MODEL = "eegnet-motor-1.0.1"


@pytest.fixture
def sync_redis() -> Iterator[SyncRedis]:
    r = SyncRedis.from_url(redis_test_url())
    r.flushdb()
    yield r
    r.flushdb()
    r.close()


def make_worker(worker_id: str = "w1", **overrides: Any) -> Worker:
    """A worker on the test database with the served model and fast test timings."""
    defaults: dict[str, Any] = {
        "redis_url": redis_test_url(),
        "model_dir": str(MODEL_DIR),
        "model": SERVED_MODEL,
        "worker_id": worker_id,
        "block_ms": 100,
        "reclaim_interval_s": 0.0,
        "model_check_interval_s": 0.0,
    }
    s = WorkerSettings(**{**defaults, **overrides})
    card = load_model_card(MODEL_DIR / s.model)
    return Worker(s, SyncRedis.from_url(s.redis_url), InferenceModel(card))


class WorkerThread:
    """Runs ``Worker.run`` in a background thread (the worker is synchronous)."""

    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.thread = threading.Thread(target=worker.run, daemon=True)

    def __enter__(self) -> Worker:
        self.thread.start()
        return self.worker

    def __exit__(self, *exc: object) -> None:
        self.worker.stop()
        self.thread.join(10)
        assert not self.thread.is_alive(), "worker thread did not stop"


def worker_metrics(r: SyncRedis) -> dict[str, int]:
    return {k.decode(): int(v) for k, v in r.hgetall(keys.WORKER_METRICS).items()}  # type: ignore[union-attr]


def stream_accounting(r: SyncRedis) -> dict[str, int]:
    """Everything needed for the conservation check on ns:windows."""
    info = r.xinfo_stream(keys.WINDOWS_STREAM)
    group = next(g for g in r.xinfo_groups(keys.WINDOWS_STREAM) if g["name"] == b"workers")
    m = worker_metrics(r)
    return {
        "entries_added": int(info["entries-added"]),
        "acked": m.get("acked", 0),
        "stale": m.get("stale", 0),
        "lost_trimmed_undelivered": m.get("lost_trimmed_undelivered", 0),
        "lost_trimmed_pending": m.get("lost_trimmed_pending", 0),
        "dead_lettered": m.get("dead_lettered", 0),
        "pending": int(group["pending"]),
        "lag": int(group["lag"] or 0),
    }


def assert_conserved(acc: dict[str, int]) -> None:
    accounted = sum(v for k, v in acc.items() if k != "entries_added")
    assert acc["entries_added"] == accounted, acc
