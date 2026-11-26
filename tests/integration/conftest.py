from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import uvicorn
from redis.asyncio import Redis

from ingest.app import create_app
from ingest.settings import Settings

# Integration tests use their own Redis database and wipe only that one.
TEST_REDIS_DB = 15


def test_redis_url() -> str:
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
    r = Redis.from_url(test_redis_url())
    await r.flushdb()
    yield r
    await r.flushdb()
    await r.aclose()


@pytest.fixture
async def ingest_server(redis_client: Redis) -> AsyncIterator[IngestServer]:
    port = free_port()
    app = create_app(Settings(redis_url=test_redis_url()))
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
