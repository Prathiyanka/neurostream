"""FastAPI app: device WebSocket endpoint plus health and metrics."""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ingest.session import SessionState
from ingest.settings import Settings
from ingest.writer import DropOldestQueue, SessionWriter, StreamItem
from neurostream_common import keys
from neurostream_common.codec import encode_raw_packet, encode_window_job
from neurostream_common.schemas import DataPacket, SessionEnd, SessionHello, parse_device_message

log = logging.getLogger("ingest")

# WebSocket close codes (4000-4999 are application-defined).
CLOSE_PROTOCOL = 4400
CLOSE_CONFLICT = 4409


class Ingest:
    """Holds per-session state. State is keyed by session, not by connection, so a
    device that reconnects continues its sequence and the gap is detected."""

    def __init__(self, settings: Settings, redis: Redis) -> None:
        self.settings = settings
        self.redis = redis
        self.sessions: dict[str, SessionState] = {}
        self.writers: dict[str, SessionWriter] = {}

    async def open_session(self, hello: SessionHello) -> SessionState:
        state = self.sessions.get(hello.session_id)
        if state is not None:
            if state.connected:
                raise SessionConflict(f"session {hello.session_id} is already connected")
            if not state.compatible_with(hello):
                raise SessionConflict(
                    f"session {hello.session_id} reconnected with a different shape"
                )
        else:
            s = self.settings
            state = SessionState(hello, s.window_s, s.step_s, s.pad_s, s.max_fill_s)
            self.sessions[hello.session_id] = state
            queue = DropOldestQueue(s.queue_max_items)
            writer = SessionWriter(
                self.redis,
                queue,
                s.write_batch_max,
                keys.ingest_metrics(hello.session_id),
                functools.partial(self._metrics, state, queue),
            )
            writer.start()
            self.writers[hello.session_id] = writer
        state.connected = True
        state.counters.connections += 1
        await self.redis.sadd(keys.SESSIONS_SET, hello.session_id)  # type: ignore[misc]
        await self.redis.hset(  # type: ignore[misc]
            keys.session_meta(hello.session_id),
            mapping={
                "session_id": hello.session_id,
                "subject": "" if hello.subject is None else hello.subject,
                "sfreq": hello.sfreq,
                "chunk_size": hello.chunk_size,
                "ch_names": json.dumps(hello.ch_names),
                "status": "streaming",
                "updated_at": time.time(),
            },
        )
        return state

    def _metrics(self, state: SessionState, queue: DropOldestQueue) -> dict[str, int]:
        state.counters.backpressure_drops = queue.dropped
        writer = self.writers.get(state.session_id)
        state.counters.redis_errors = writer.redis_errors if writer else 0
        m = state.metrics()
        m["queue_depth"] = len(queue)
        return m

    def handle_packet(
        self, state: SessionState, packet: DataPacket, ingest_ts: float
    ) -> str | None:
        result = state.process(packet, ingest_ts, now=time.time())
        if result.error is not None:
            return result.error
        writer = self.writers[state.session_id]
        s = self.settings
        if result.raw is not None:
            writer.queue.put(
                StreamItem(
                    keys.raw_stream(state.session_id), encode_raw_packet(result.raw), s.raw_maxlen
                )
            )
        for job in result.jobs:
            writer.queue.put(
                StreamItem(keys.WINDOWS_STREAM, encode_window_job(job), s.windows_maxlen)
            )
        return None

    async def disconnect(self, state: SessionState, ended: bool) -> None:
        state.connected = False
        writer = self.writers[state.session_id]
        await writer.flush()
        status = "ended" if ended else "disconnected"
        try:
            await self.redis.hset(  # type: ignore[misc]
                keys.session_meta(state.session_id),
                mapping={"status": status, "updated_at": time.time()},
            )
            await self.redis.hset(  # type: ignore[misc]
                keys.ingest_metrics(state.session_id), mapping=self._metrics(state, writer.queue)
            )
        except RedisError as exc:
            log.warning("could not record %s for %s: %s", status, state.session_id, exc)
        if ended:
            await writer.stop()
            del self.writers[state.session_id]
            del self.sessions[state.session_id]

    def all_metrics(self) -> dict[str, dict[str, int]]:
        return {
            sid: self._metrics(st, self.writers[sid].queue) for sid, st in self.sessions.items()
        }

    async def close(self) -> None:
        for writer in self.writers.values():
            await writer.flush()
            await writer.stop()


class SessionConflict(Exception):
    pass


def create_app(
    settings: Settings | None = None, redis_factory: Callable[[str], Redis] | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = redis_factory or (lambda url: Redis.from_url(url))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        redis = factory(settings.redis_url)
        app.state.ingest = Ingest(settings, redis)
        log.info("ingest started", extra={"redis_url": settings.redis_url})
        try:
            yield
        finally:
            await app.state.ingest.close()
            await redis.aclose()

    app = FastAPI(title="NeuroStream ingest", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        ingest: Ingest = app.state.ingest
        try:
            redis_ok = bool(await ingest.redis.ping())
        except (RedisError, OSError):
            redis_ok = False
        return {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
            "sessions": len(ingest.sessions),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, dict[str, int]]:
        return app.state.ingest.all_metrics()  # type: ignore[no-any-return]

    @app.websocket("/ws/device")
    async def device(ws: WebSocket) -> None:
        await ws.accept()
        ingest: Ingest = app.state.ingest
        state: SessionState | None = None
        ended = False
        try:
            first = parse_device_message(await ws.receive_text())
            if not isinstance(first, SessionHello):
                await ws.close(CLOSE_PROTOCOL, "first message must be hello")
                return
            try:
                state = await ingest.open_session(first)
            except SessionConflict as exc:
                await ws.close(CLOSE_CONFLICT, str(exc))
                return
            log.info("device connected", extra={"session_id": first.session_id})
            while True:
                raw = await ws.receive_text()
                ingest_ts = time.time()  # stamp arrival before any parsing work
                try:
                    msg = parse_device_message(raw)
                except ValidationError as exc:
                    state.counters.invalid += 1
                    await ws.send_json(
                        {"type": "error", "detail": exc.errors(include_url=False)[:3]}
                    )
                    continue
                if isinstance(msg, DataPacket):
                    error = ingest.handle_packet(state, msg, ingest_ts)
                    if error is not None:
                        await ws.send_json({"type": "error", "detail": error})
                elif isinstance(msg, SessionEnd):
                    ended = True
                    break
                else:
                    await ws.send_json({"type": "error", "detail": "unexpected hello"})
        except ValidationError:
            await ws.close(CLOSE_PROTOCOL, "invalid hello")
        except WebSocketDisconnect:
            pass
        finally:
            if state is not None:
                await ingest.disconnect(state, ended)
                log.info(
                    "device %s",
                    "ended" if ended else "disconnected",
                    extra={"session_id": state.session_id},
                )
        if ended:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await ws.close()  # the device may already have hung up

    return app
