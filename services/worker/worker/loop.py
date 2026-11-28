"""The inference worker: read window jobs, classify them in batches, publish predictions.

Stateless by design: every job carries its own padded window, so any worker can take any
job, and scaling out is just starting more workers in the same consumer group.

One iteration (``step``):
  1. every ``reclaim_interval_s``: take over jobs left pending by a dead worker;
  2. read whatever jobs are available right now (up to ``batch_max``, never waiting to
     fill a batch); if there are none, sleep on the stream until one arrives;
  3. preprocess each job (as its model card prescribes), run ONE batched ONNX call,
     publish the predictions and only then acknowledge the jobs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from redis import Redis
from redis.exceptions import RedisError

from neurostream_common import keys
from neurostream_common.codec import (
    Fields,
    Prediction,
    WindowJob,
    decode_window_job,
    encode_prediction,
)
from neurostream_common.labels import UNKNOWN
from worker.model import InferenceModel, JobError, Prepared
from worker.settings import Settings
from worker.streams import Entry, JobStreams

log = logging.getLogger("worker")


@dataclass
class Counters:
    """This worker's own totals (the shared ones live in ``ns:metrics:workers``)."""

    batches: int = 0
    predictions: int = 0
    reclaimed: int = 0
    dead_lettered: int = 0
    stale: int = 0
    errors: int = 0
    pad_short: int = 0
    started_at: float = field(default_factory=time.time)


class Worker:
    def __init__(self, settings: Settings, redis: Redis, model: InferenceModel) -> None:
        self.s = settings
        self.r = redis
        self.model = model
        self.streams = JobStreams(redis, settings.worker_id)
        self.counters = Counters()
        self._ch_names: dict[str, list[str]] = {}
        self._next_reclaim = 0.0
        self._next_heartbeat = 0.0
        self._stopping = False

    # ---- lifecycle -------------------------------------------------------------------

    def start(self) -> None:
        # Warm up before joining: the first real batches would otherwise be slow.
        warm_s = self.model.warm_up(self.s.batch_max)
        log.info("model warmed up", extra={"seconds": round(warm_s, 3)})
        self.streams.ensure_group()
        self.heartbeat(force=True)
        log.info(
            "worker started",
            extra={"worker_id": self.s.worker_id, "model": self.model.model_id},
        )

    def stop(self) -> None:
        self._stopping = True

    def run(self) -> None:
        self.start()
        backoff = 0.1
        while not self._stopping:
            try:
                self.step()
                backoff = 0.1
            except RedisError as exc:
                # Redis down or restarting: keep the process, retry with backoff. Jobs we
                # held unacked stay pending and will be reclaimed if we never come back.
                self.counters.errors += 1
                log.warning("redis error, retrying in %.1fs: %s", backoff, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
        try:
            self.r.delete(keys.worker_heartbeat(self.s.worker_id))
            self.r.srem(keys.WORKERS_SET, self.s.worker_id)
        except RedisError:
            pass
        log.info("worker stopped", extra={"worker_id": self.s.worker_id})

    def step(self) -> int:
        """One iteration. Returns the number of jobs handled (0 = was idle)."""
        handled = 0
        now = time.monotonic()
        if now >= self._next_reclaim:
            self._next_reclaim = now + self.s.reclaim_interval_s
            handled += self.reclaim()
        res = self.streams.read(self.s.batch_max)
        if res.skipped:
            log.warning(
                "jobs trimmed from %s before any worker read them",
                keys.WINDOWS_STREAM,
                extra={"lost": res.skipped},
            )
        if res.entries:
            handled += self.process([(e, 1) for e in res.entries], reclaimed=False)
        elif not handled:
            self.heartbeat()
            self.streams.wait(res.last_delivered_id, self.s.block_ms)
        self.heartbeat()
        return handled

    def run_until_idle(self, max_steps: int = 10_000) -> int:
        """Process until a step finds nothing to do (for tests and draining)."""
        total = 0
        for _ in range(max_steps):
            n = self.step()
            if n == 0:
                return total
            total += n
        return total

    # ---- reclaiming ------------------------------------------------------------------

    def reclaim(self) -> int:
        claimed, lost = self.streams.reclaim(self.s.reclaim_min_idle_ms, self.s.batch_max)
        if lost:
            log.warning("reclaimed jobs had been trimmed while pending", extra={"lost": lost})
        if not claimed:
            return lost
        retry: list[tuple[Entry, int]] = []
        for entry, deliveries in claimed:
            if deliveries > self.s.max_deliveries:
                self._dead_letter(entry, "max_deliveries", deliveries)
            else:
                retry.append((entry, deliveries))
        self.counters.reclaimed += len(retry)
        dead = len(claimed) - len(retry)
        log.info("reclaimed pending jobs", extra={"n": len(retry), "dead": dead})
        return lost + dead + self.process(retry, reclaimed=True)

    # ---- processing ------------------------------------------------------------------

    def _session_channels(self, session_id: str) -> list[str]:
        names = self._ch_names.get(session_id)
        if names is None:
            raw: bytes | None = self.r.hget(keys.session_meta(session_id), "ch_names")  # type: ignore[assignment]
            if raw is None:
                raise JobError(f"no metadata for session {session_id}")
            names = list(json.loads(raw))
            self._ch_names[session_id] = names
        return names

    def _dead_letter(self, entry: Entry, reason: str, deliveries: int) -> None:
        self.streams.dead_letter(entry, reason, deliveries, self.s.dead_maxlen)
        self.counters.dead_lettered += 1
        log.error(
            "job dead-lettered",
            extra={"stream_id": entry[0], "reason": reason, "deliveries": deliveries},
        )

    def process(self, items: Sequence[tuple[Entry, int]], reclaimed: bool) -> int:
        """Classify a batch and publish. Returns the number of jobs handled."""
        if not items:
            return 0
        read_ts = time.time()
        jobs: list[tuple[str, WindowJob, Prepared]] = []
        stale_ids: list[str] = []
        oldest_s = 0.0
        for entry, deliveries in items:
            try:
                job = decode_window_job(entry[1])
                age_s = read_ts - job.ingest_ts
                if self.s.stale_after_s and age_s > self.s.stale_after_s:
                    # Too late to be useful live: skip it (no preprocessing, no inference).
                    stale_ids.append(entry[0])
                    oldest_s = max(oldest_s, age_s)
                    continue
                prepared = self.model.prepare(job, self._session_channels(job.session_id))
            except (JobError, KeyError, ValueError) as exc:
                # Deterministic: retrying cannot help, so set it aside now.
                self._dead_letter(entry, f"{type(exc).__name__}: {exc}", deliveries)
                continue
            jobs.append((entry[0], job, prepared))
        if stale_ids:
            self.counters.stale += self.streams.ack_stale(stale_ids)
            log.warning(
                "skipped stale jobs",
                extra={"n": len(stale_ids), "oldest_s": round(oldest_s, 2)},
            )
        if not jobs:
            return len(items)

        t0 = time.perf_counter()
        probs = self.model.infer([p.x for _, _, p in jobs])
        inference_ms = (time.perf_counter() - t0) * 1000
        if self.s.fault_delay_s:
            time.sleep(self.s.fault_delay_s)

        publish_ts = time.time()
        out: list[tuple[Fields, str]] = []
        pad_short = 0
        for (_, job, prep), p in zip(jobs, probs, strict=True):
            w = job.window
            pad_short += prep.pad_short
            pred = Prediction(
                window_id=job.window_id,
                session_id=job.session_id,
                window_end_sample=w.end_sample,
                window_end_ts=job.ingest_ts,
                predicted_class=int(np.argmax(p)),
                probabilities=tuple(round(float(v), 6) for v in p),
                true_label=int(w.core_labels[0]) if w.label_pure else UNKNOWN,
                filled_samples=w.filled_samples,
                model=self.model.model_id,
                worker_id=self.s.worker_id,
                reclaimed=reclaimed,
                batch_size=len(jobs),
                ingest_ms=(job.emitted_ts - job.ingest_ts) * 1000,
                queue_ms=(read_ts - job.emitted_ts) * 1000,
                preprocess_ms=prep.preprocess_ms,
                inference_ms=inference_ms,
                worker_ms=(publish_ts - read_ts) * 1000,
                latency_ms=(publish_ts - job.ingest_ts) * 1000,
                publish_ts=publish_ts,
            )
            out.append((encode_prediction(pred), pred.to_json()))
        self.streams.publish_and_ack(
            out,
            [entry_id for entry_id, _, _ in jobs],
            self.s.predictions_maxlen,
            {
                "predictions": len(out),
                "reclaimed": len(out) if reclaimed else 0,
                "pad_short": pad_short,
            },
        )
        self.counters.batches += 1
        self.counters.predictions += len(out)
        self.counters.pad_short += pad_short
        return len(items)

    # ---- liveness --------------------------------------------------------------------

    def heartbeat(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._next_heartbeat:
            return
        self._next_heartbeat = now + self.s.heartbeat_interval_s
        c = self.counters
        key = keys.worker_heartbeat(self.s.worker_id)
        pipe = self.r.pipeline(transaction=False)
        pipe.hset(
            key,
            mapping={
                "worker_id": self.s.worker_id,
                "pid": os.getpid(),
                "model": self.model.model_id,
                "started_at": c.started_at,
                "last_seen": time.time(),
                "batches": c.batches,
                "predictions": c.predictions,
                "reclaimed": c.reclaimed,
                "dead_lettered": c.dead_lettered,
                "stale": c.stale,
                "errors": c.errors,
            },
        )
        pipe.expire(key, self.s.heartbeat_ttl_s)
        pipe.sadd(keys.WORKERS_SET, self.s.worker_id)
        pipe.execute()  # type: ignore[no-untyped-call]
