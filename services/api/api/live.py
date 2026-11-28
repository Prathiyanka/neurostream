"""Live WebSocket feed: ``/ws/live``.

Messages to the client (JSON):
  {"type": "prediction", ...}  every prediction (all sessions, or the subscribed one)
  {"type": "metrics", ...}     system metrics, every second
  {"type": "trace", ...}       raw EEG for the subscribed session and channels, at 40 Hz

Messages from the client:
  {"type": "subscribe", "session": "<id>" | null, "channels": ["C3", "Cz", "C4"]}

Where the traces come from: ingest writes every accepted packet (after validation and
duplicate removal, before any filtering) to the ``ns:raw:{session}`` stream. One reader
task per watched session tails it with ``XREAD BLOCK`` and fans out to every client
watching that session. Display only: the signal is reduced from 160 Hz to 40 Hz by
averaging blocks of 4 samples, which is plenty for a scrolling trace and 4x less data.
The model never sees this path.

Fan-out is lossy on purpose: each client has a bounded queue that drops its oldest
message when full, so one slow browser cannot slow the others or the API.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from redis.asyncio import Redis
from redis.exceptions import RedisError

from neurostream_common import keys
from neurostream_common.codec import decode_raw_packet

log = logging.getLogger("api.live")

DECIMATE = 4  # 160 Hz -> 40 Hz for display
DEFAULT_CHANNELS = ("C3", "Cz", "C4")
CLIENT_QUEUE = 500


class Client:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.session: str | None = None
        self.channels: tuple[str, ...] = DEFAULT_CHANNELS
        self.queue: deque[str] = deque(maxlen=CLIENT_QUEUE)  # full -> oldest dropped
        self.dropped = 0
        self.wake = asyncio.Event()

    def send(self, message: str) -> None:
        if len(self.queue) == self.queue.maxlen:
            self.dropped += 1
        self.queue.append(message)
        self.wake.set()


class TraceReader:
    """Tails one session's raw stream and turns packets into 40 Hz trace messages."""

    def __init__(self, redis: Redis, session_id: str, hub: Hub) -> None:
        self.r = redis
        self.session_id = session_id
        self.hub = hub
        self.task: asyncio.Task[None] | None = None
        self._carry: np.ndarray | None = None  # samples not yet filling a block of 4
        self._next_sample: int | None = None  # absolute index expected next (gap check)
        self.ch_names: list[str] = []

    async def run(self) -> None:
        raw = await self.r.hget(keys.session_meta(self.session_id), "ch_names")  # type: ignore[misc]
        self.ch_names = json.loads(raw) if raw else []
        last = "$"
        while True:
            try:
                res = await self.r.xread({keys.raw_stream(self.session_id): last}, block=1000)
            except RedisError as exc:
                log.warning("trace read failed: %s", exc)
                await asyncio.sleep(1)
                continue
            for _, entries in res or []:
                for entry_id, fields in entries:
                    last = entry_id
                    self._packet(fields)

    def _packet(self, fields: dict[bytes, bytes]) -> None:
        p = decode_raw_packet(fields)
        if self._next_sample is not None and p.start_sample != self._next_sample:
            self._carry = None  # gap (lost packets): restart the block grid, never bridge it
        self._next_sample = p.start_sample + p.data.shape[1]
        data = p.data if self._carry is None else np.concatenate([self._carry, p.data], axis=1)
        start = p.start_sample - (0 if self._carry is None else self._carry.shape[1])
        n_blocks = data.shape[1] // DECIMATE
        self._carry = data[:, n_blocks * DECIMATE :] if data.shape[1] % DECIMATE else None
        if n_blocks == 0:
            return
        blocks = data[:, : n_blocks * DECIMATE].reshape(data.shape[0], n_blocks, DECIMATE)
        self.hub.trace(self.session_id, start, blocks.mean(axis=2), self.ch_names)


class Hub:
    def __init__(
        self,
        redis: Redis,
        metrics: Callable[[], Coroutine[Any, Any, dict[str, Any]]],
        metrics_interval_s: float = 1.0,
    ) -> None:
        self.r = redis
        self.metrics = metrics
        self.metrics_interval_s = metrics_interval_s
        self.clients: set[Client] = set()
        self.readers: dict[str, TraceReader] = {}
        self._tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._predictions()),
            asyncio.create_task(self._metrics()),
        ]

    async def stop(self) -> None:
        for t in self._tasks + [r.task for r in self.readers.values() if r.task]:
            t.cancel()
        for t in self._tasks + [r.task for r in self.readers.values() if r.task]:
            with contextlib.suppress(asyncio.CancelledError):
                await t

    # ---- fan-out ---------------------------------------------------------------------

    async def _predictions(self) -> None:
        while True:
            try:
                pubsub = self.r.pubsub()
                await pubsub.subscribe(keys.PREDICTIONS_CHANNEL)
                async for msg in pubsub.listen():
                    if msg["type"] != "message":
                        continue
                    data = msg["data"].decode()
                    session = json.loads(data)["session_id"]
                    out = '{"type":"prediction",' + data[1:]
                    for c in list(self.clients):
                        if c.session is None or c.session == session:
                            c.send(out)
            except RedisError as exc:
                log.warning("prediction subscription lost, resubscribing: %s", exc)
                await asyncio.sleep(1)

    async def _metrics(self) -> None:
        while True:
            await asyncio.sleep(self.metrics_interval_s)
            if not self.clients:
                continue
            try:
                m = await self.metrics()
            except RedisError as exc:
                log.warning("metrics collection failed: %s", exc)
                continue
            out = json.dumps({"type": "metrics", **m})
            for c in list(self.clients):
                c.send(out)

    def trace(self, session_id: str, start: int, data: np.ndarray, ch_names: list[str]) -> None:
        index = {name: i for i, name in enumerate(ch_names)}
        for c in list(self.clients):
            if c.session != session_id:
                continue
            chans = [ch for ch in c.channels if ch in index]
            c.send(
                json.dumps(
                    {
                        "type": "trace",
                        "session_id": session_id,
                        "sfreq": 160.0 / DECIMATE,
                        "start_sample": start,  # absolute index of the first input sample
                        "channels": chans,
                        "data": np.round(data[[index[ch] for ch in chans]], 2).tolist(),
                    }
                )
            )

    # ---- subscriptions ---------------------------------------------------------------

    def _watch(self, session_id: str | None) -> None:
        """Start a trace reader for a newly watched session; stop unwatched ones."""
        watched = {c.session for c in self.clients if c.session}
        if session_id and session_id not in self.readers:
            reader = TraceReader(self.r, session_id, self)
            reader.task = asyncio.create_task(reader.run())
            self.readers[session_id] = reader
        for sid in list(self.readers):
            if sid not in watched:
                task = self.readers.pop(sid).task
                if task:
                    task.cancel()

    async def serve(self, ws: WebSocket) -> None:
        await ws.accept()
        client = Client(ws)
        client.session = ws.query_params.get("session") or None
        if ws.query_params.get("channels"):
            client.channels = tuple(ws.query_params["channels"].split(","))
        self.clients.add(client)
        self._watch(client.session)
        sender = asyncio.create_task(self._send_loop(client))
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "subscribe":
                    client.session = msg.get("session") or None
                    if msg.get("channels"):
                        client.channels = tuple(str(c) for c in msg["channels"])
                    self._watch(client.session)
        except (WebSocketDisconnect, json.JSONDecodeError, RuntimeError):
            pass
        finally:
            self.clients.discard(client)
            sender.cancel()
            self._watch(None)

    async def _send_loop(self, client: Client) -> None:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            while True:
                await client.wake.wait()
                client.wake.clear()
                while client.queue:
                    await client.ws.send_text(client.queue.popleft())
