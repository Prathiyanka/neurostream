"""Persist predictions from Redis to Postgres, each exactly once.

Reads ``ns:predictions`` with its own consumer group (``api``), so persistence never
competes with the live pub/sub fan-out and several API instances could share the work.

Exactly once in the table, from at-least-once delivery:
  1. read a batch (atomic read script: entries trimmed before we read them are counted);
  2. ``INSERT ... ON CONFLICT (window_id) DO NOTHING`` in one transaction, so a
     prediction delivered twice (worker crash, or our own restart) is stored once;
  3. acknowledge only after the commit. A crash between 2 and 3 re-delivers the batch,
     and step 2 makes that harmless.

Trade-off (documented in the README): this runs inside the API process. If it ever falls
behind, it moves to its own service unchanged, since it already has its own group.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
from collections import deque
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from api.db import predictions, sessions
from neurostream_common import keys
from neurostream_common.codec import Prediction, decode_prediction
from neurostream_common.keys import PERSISTER_METRICS
from neurostream_common.redis_scripts import READ_SCRIPT

log = logging.getLogger("api.persister")

GROUP = "api"


def utc(ts: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, dt.UTC)


def prediction_row(p: Prediction) -> dict[str, Any]:
    return {
        "window_id": p.window_id,
        "session_id": p.session_id,
        "window_end_sample": p.window_end_sample,
        "window_end_at": utc(p.window_end_ts),
        "predicted_class": p.predicted_class,
        "probabilities": list(p.probabilities),
        "true_label": p.true_label,
        "filled_samples": p.filled_samples,
        "model": p.model,
        "worker_id": p.worker_id,
        "reclaimed": p.reclaimed,
        "batch_size": p.batch_size,
        "ingest_ms": p.ingest_ms,
        "queue_ms": p.queue_ms,
        "worker_ms": p.worker_ms,
        "preprocess_ms": p.preprocess_ms,
        "inference_ms": p.inference_ms,
        "latency_ms": p.latency_ms,
        "published_at": utc(p.publish_ts),
    }


class Persister:
    def __init__(self, redis: Redis, engine: AsyncEngine, consumer: str, batch: int) -> None:
        self.r = redis
        self.engine = engine
        self.consumer = consumer
        self.batch = batch
        self._read = redis.register_script(READ_SCRIPT)
        self._known_sessions: set[str] = set()
        self.recent_latencies: deque[float] = deque(maxlen=20_000)  # since last snapshot
        self.persisted_total = 0
        self.duplicates_total = 0

    async def ensure_group(self) -> None:
        try:
            await self.r.xgroup_create(keys.PREDICTIONS_STREAM, GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _read_own_pending(self) -> list[tuple[str, dict[bytes, bytes]]]:
        """Entries delivered to this consumer before a restart but never acknowledged."""
        res = await self.r.xreadgroup(
            GROUP, self.consumer, {keys.PREDICTIONS_STREAM: "0"}, count=self.batch
        )
        if not res:
            return []
        trimmed = [i for i, f in res[0][1] if not f]  # pending, but trimmed from the stream
        if trimmed:
            pipe = self.r.pipeline(transaction=False)
            pipe.xack(keys.PREDICTIONS_STREAM, GROUP, *trimmed)
            pipe.hincrby(PERSISTER_METRICS, "lost_trimmed_pending", len(trimmed))
            await pipe.execute()
        return [(i.decode(), f) for i, f in res[0][1] if f]

    async def _read_new(self) -> tuple[list[tuple[str, dict[bytes, bytes]]], str]:
        res, _skipped, last_id = await self._read(
            keys=[keys.PREDICTIONS_STREAM, PERSISTER_METRICS],
            args=[GROUP, self.consumer, self.batch],
        )
        entries = []
        if res:
            for entry_id, flat in res[0][1]:
                fields = {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}
                entries.append((entry_id.decode(), fields))
        return entries, last_id.decode() if isinstance(last_id, bytes) else str(last_id)

    async def _ensure_sessions(self, conn: Any, session_ids: set[str]) -> None:
        """Sessions not started through the API (CLI replayer, load tests) get a row the
        first time one of their predictions is stored, so the foreign key holds."""
        new = session_ids - self._known_sessions
        if not new:
            return
        rows = []
        for sid in sorted(new):
            meta = await self.r.hgetall(keys.session_meta(sid))  # type: ignore[misc]
            subject = meta.get(b"subject", b"").decode()
            rows.append(
                {
                    "id": sid,
                    "subject": int(subject) if subject.isdigit() else None,
                    "source": "external",
                    "status": "running",
                }
            )
        await conn.execute(insert(sessions).values(rows).on_conflict_do_nothing())
        self._known_sessions |= new

    async def store(self, entries: list[tuple[str, dict[bytes, bytes]]]) -> int:
        """Insert a batch idempotently, then ack it. Returns rows actually inserted."""
        if not entries:
            return 0
        preds = [decode_prediction(f) for _, f in entries]
        rows = [prediction_row(p) for p in preds]
        async with self.engine.begin() as conn:
            await self._ensure_sessions(conn, {p.session_id for p in preds})
            stmt = (
                insert(predictions)
                .values(rows)
                .on_conflict_do_nothing(index_elements=["window_id"])
                .returning(predictions.c.window_id)
            )
            inserted = len((await conn.execute(stmt)).all())
        duplicates = len(rows) - inserted
        pipe = self.r.pipeline(transaction=False)
        pipe.xack(keys.PREDICTIONS_STREAM, GROUP, *[i for i, _ in entries])
        pipe.hincrby(PERSISTER_METRICS, "persisted", inserted)
        if duplicates:
            pipe.hincrby(PERSISTER_METRICS, "duplicates", duplicates)
        await pipe.execute()
        self.persisted_total += inserted
        self.duplicates_total += duplicates
        self.recent_latencies.extend(p.latency_ms for p in preds)
        return inserted

    async def step(self, block_ms: int = 1000) -> int:
        """One round: pending first (after a restart), then new entries. Returns rows
        inserted. Waits up to ``block_ms`` when there is nothing to do."""
        entries = await self._read_own_pending()
        if not entries:
            entries, last_id = await self._read_new()
            if not entries:
                await self.r.xread({keys.PREDICTIONS_STREAM: last_id}, count=1, block=block_ms)
                return 0
        return await self.store(entries)

    async def run(self) -> None:
        await self.ensure_group()
        backoff = 0.2
        while True:
            try:
                await self.step()
                backoff = 0.2
            except asyncio.CancelledError:
                raise
            except (RedisError, SQLAlchemyError, OSError) as exc:
                # Nothing is acked on failure: the batch is re-read as pending and retried.
                log.warning("persist failed, retrying in %.1fs: %s", backoff, exc)
                with contextlib.suppress(RedisError):
                    await self.r.hincrby(PERSISTER_METRICS, "errors", 1)  # type: ignore[misc]
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    async def metrics(self) -> dict[str, int]:
        raw = await self.r.hgetall(PERSISTER_METRICS)  # type: ignore[misc]
        return {k.decode(): int(v) for k, v in raw.items()}


def jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        out[k] = v.isoformat() if isinstance(v, dt.datetime) else v
    result: dict[str, Any] = json.loads(json.dumps(out, default=str))
    return result
