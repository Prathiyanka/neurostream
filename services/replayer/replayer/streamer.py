"""Streams one recording to ingest over WebSocket at its true sampling rate.

Timing uses an *absolute schedule*: packet k is due at ``t0 + k * period``. Sleeping a
fixed ``period`` after each send would let every late wake-up (asyncio timers on Windows
have ~15 ms granularity) add up into drift. With an absolute schedule, lateness never
accumulates and jitter doesn't compound.

Like a real headset, the replayer doesn't buffer while disconnected: packets that came
due during an outage are lost, and the stream resumes at the current time. Ingest then
sees a sequence gap.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from replayer.faults import FaultConfig, FaultInjector
from replayer.recording import Recording

log = logging.getLogger("replayer")


class Connection(Protocol):
    async def send(self, message: str) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


Connector = Callable[[str], Awaitable[Connection]]


async def default_connector(url: str) -> Connection:
    return await ws_connect(url, max_size=4 * 1024 * 1024, open_timeout=5)


@dataclass(frozen=True)
class SessionConfig:
    url: str
    session_id: str
    chunk_size: int = 16
    speed: float = 1.0
    loop: bool = False  # restart the recording when it ends (seq keeps increasing)
    max_seconds: float | None = None  # stop after this much recording time
    send_truth: bool = True
    faults: FaultConfig = field(default_factory=FaultConfig)
    max_connect_attempts: int | None = None  # None == retry forever


@dataclass
class StreamStats:
    session_id: str
    packets_total: int = 0  # packets the recording was cut into (seq 0..total-1)
    sent: int = 0  # unique packets put on the wire
    dropped: int = 0  # dropped by fault injection
    duplicated: int = 0
    reordered: int = 0
    disconnects: int = 0  # injected disconnects
    connection_errors: int = 0  # unexpected send/connect failures
    lost_while_disconnected: int = 0
    errors_from_ingest: int = 0


def build_packet(
    rec: Recording, session_id: str, seq: int, chunk: int, start_wall: float, send_truth: bool
) -> dict[str, Any]:
    per_rec = rec.n_samples // chunk
    i = seq % per_rec
    sl = slice(i * chunk, (i + 1) * chunk)
    msg: dict[str, Any] = {
        "type": "data",
        "session_id": session_id,
        "seq": seq,
        "timestamp": start_wall + seq * chunk / rec.sfreq,
        # 3 decimals of a microvolt is far below the recording's resolution and roughly
        # halves the JSON size compared with full float repr.
        "samples": np.round(rec.data[:, sl].astype(np.float64), 3).tolist(),
    }
    if send_truth:
        msg["truth"] = {"labels": rec.labels[sl].tolist()}
    return msg


def hello_message(rec: Recording, cfg: SessionConfig) -> str:
    return json.dumps(
        {
            "type": "hello",
            "session_id": cfg.session_id,
            "sfreq": rec.sfreq,
            "ch_names": rec.ch_names,
            "chunk_size": cfg.chunk_size,
            "units": "uV",
            "subject": rec.subject,
        }
    )


async def _drain_replies(conn: Connection, stats: StreamStats) -> None:
    """Read (and log) error replies so they never pile up in the client buffer."""
    try:
        while True:
            reply = await conn.recv()
            stats.errors_from_ingest += 1
            log.warning(
                "ingest error", extra={"session_id": stats.session_id, "reply": str(reply)[:300]}
            )
    except (ConnectionClosed, OSError, asyncio.CancelledError):
        return


class SessionStreamer:
    def __init__(
        self, cfg: SessionConfig, rec: Recording, connector: Connector = default_connector
    ):
        if cfg.chunk_size < 1 or cfg.speed <= 0:
            raise ValueError("chunk_size must be >= 1 and speed > 0")
        self.cfg, self.rec, self.connector = cfg, rec, connector
        self.stats = StreamStats(cfg.session_id)
        self._injector = FaultInjector(cfg.faults)
        self._conn: Connection | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stop_requested = False

    def stop(self) -> None:
        """Ask a running session to end early. It finishes the packet in progress, sends
        `end` (so ingest closes the session cleanly, as for a normal finish) and returns."""
        self._stop_requested = True

    def _total_packets(self) -> int:
        per_rec = self.rec.n_samples // self.cfg.chunk_size
        if per_rec == 0:
            raise ValueError("recording shorter than one chunk")
        limit = math.inf if self.cfg.loop else per_rec
        if self.cfg.max_seconds is not None:
            limit = min(
                limit, math.ceil(self.cfg.max_seconds * self.rec.sfreq / self.cfg.chunk_size)
            )
        if limit == math.inf:
            return 2**62  # effectively forever
        return int(limit)

    async def _connect(self) -> None:
        attempts, backoff = 0, 0.1
        while True:
            attempts += 1
            try:
                conn = await self.connector(self.cfg.url)
                await conn.send(hello_message(self.rec, self.cfg))
                self._conn = conn
                self._reader = asyncio.create_task(_drain_replies(conn, self.stats))
                log.info("connected", extra={"session_id": self.cfg.session_id})
                return
            except (OSError, InvalidHandshake, ConnectionClosed, TimeoutError) as exc:
                self.stats.connection_errors += 1
                if self.cfg.max_connect_attempts and attempts >= self.cfg.max_connect_attempts:
                    raise ConnectionError(f"could not connect to {self.cfg.url}") from exc
                log.warning("connect failed, retrying in %.1fs: %s", backoff, exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 2.0)

    async def _close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._conn is not None:
            with contextlib.suppress(ConnectionClosed, OSError):
                await self._conn.close()
            self._conn = None

    async def _send(self, message: str) -> bool:
        assert self._conn is not None
        try:
            await self._conn.send(message)
            return True
        except (ConnectionClosed, OSError):
            self.stats.connection_errors += 1
            await self._close()
            return False

    async def run(self) -> StreamStats:
        cfg, rec, stats = self.cfg, self.rec, self.stats
        loop = asyncio.get_running_loop()
        total = self._total_packets()
        stats.packets_total = total
        period = cfg.chunk_size / rec.sfreq / cfg.speed
        disconnect_every = (
            round(cfg.faults.disconnect_every_s * rec.sfreq / cfg.chunk_size)
            if cfg.faults.disconnect_every_s > 0
            else 0
        )
        # The clock starts once the first connection is up, so a slow initial handshake
        # doesn't count as an outage. After that, outages do cost packets (see module doc).
        await self._connect()
        t0, start_wall = loop.time(), time.time()
        held: str | None = None  # packet delayed by an injected reorder
        last_disconnect_seq = -1
        seq = 0
        while seq < total and not self._stop_requested:
            if self._conn is None:
                await self._connect()
                due_now = int((loop.time() - t0) / period)
                if due_now > seq:  # packets that came due while we were offline are gone
                    stats.lost_while_disconnected += min(due_now, total) - seq
                    seq = due_now
                    held = None
                continue
            fate = self._injector.fate()
            delay = t0 + seq * period + fate.delay_s - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            if (
                disconnect_every
                and seq > 0
                and seq % disconnect_every == 0
                and seq != last_disconnect_seq
            ):
                last_disconnect_seq = seq
                stats.disconnects += 1
                await self._close()
                await asyncio.sleep(cfg.faults.reconnect_delay_s / cfg.speed)
                continue

            if fate.drop:
                stats.dropped += 1
            else:
                msg = json.dumps(
                    build_packet(
                        rec, cfg.session_id, seq, cfg.chunk_size, start_wall, cfg.send_truth
                    )
                )
                if fate.reorder and held is None:
                    held = msg
                    stats.reordered += 1
                else:
                    ok = await self._send(msg)
                    if ok:
                        stats.sent += 1
                        if fate.duplicate and await self._send(msg):
                            stats.duplicated += 1
                        if held is not None and await self._send(held):
                            stats.sent += 1
                    held = None
            seq += 1

        if self._conn is not None:
            if held is not None and await self._send(held):
                stats.sent += 1
            if self._conn is not None:
                await self._send(json.dumps({"type": "end", "session_id": cfg.session_id}))
                await asyncio.sleep(0.05)  # let the end message flush before closing
            await self._close()
        log.info("session finished", extra={"session_id": cfg.session_id, **stats.__dict__})
        return stats
