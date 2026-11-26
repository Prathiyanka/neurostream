"""Bounded, batched writes to Redis, one writer per session.

Backpressure policy: the WebSocket reader never awaits Redis. It puts items in a
bounded per-session buffer; if the buffer is full the *oldest* item is dropped and
counted. For real-time data, stale samples are worth less than fresh ones, and
blocking the reader would just push the stall back onto the device. Per-session buffers
mean one session's burst can't starve another's.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from neurostream_common.codec import Fields

log = logging.getLogger("ingest.writer")


@dataclass(frozen=True)
class StreamItem:
    key: str
    fields: Fields
    maxlen: int


class DropOldestQueue:
    def __init__(self, maxsize: int) -> None:
        self._items: deque[StreamItem] = deque()
        self._maxsize = maxsize
        self._event = asyncio.Event()
        self.dropped = 0

    def put(self, item: StreamItem) -> None:
        if len(self._items) >= self._maxsize:
            self._items.popleft()
            self.dropped += 1
        self._items.append(item)
        self._event.set()

    def __len__(self) -> int:
        return len(self._items)

    async def get_batch(self, max_items: int) -> list[StreamItem]:
        while not self._items:
            self._event.clear()
            await self._event.wait()
        n = min(max_items, len(self._items))
        return [self._items.popleft() for _ in range(n)]


class SessionWriter:
    def __init__(
        self,
        redis: Redis,
        queue: DropOldestQueue,
        batch_max: int,
        metrics_key: str,
        metrics: Callable[[], dict[str, int]],
    ) -> None:
        self._redis = redis
        self.queue = queue
        self._batch_max = batch_max
        self._metrics_key = metrics_key
        self._metrics = metrics
        self.redis_errors = 0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        backoff = 0.05
        batch: list[StreamItem] = []
        while True:
            if not batch:
                batch = await self.queue.get_batch(self._batch_max)
            try:
                async with self._redis.pipeline(transaction=False) as pipe:
                    for item in batch:
                        pipe.xadd(item.key, item.fields, maxlen=item.maxlen, approximate=True)  # type: ignore[arg-type]
                    pipe.hset(self._metrics_key, mapping=self._metrics())
                    await pipe.execute()
                batch = []
                backoff = 0.05
            except (RedisError, OSError) as exc:
                # Keep the batch and retry; new items meanwhile go to the bounded queue.
                self.redis_errors += 1
                log.warning("redis write failed, retrying in %.2fs: %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 2.0)

    async def flush(self, timeout: float = 2.0) -> None:
        """Wait (bounded) for queued items to be written."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while len(self.queue) and loop.time() < deadline:
            await asyncio.sleep(0.01)
        # One more tick so an in-flight batch finishes.
        await asyncio.sleep(0.02)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
