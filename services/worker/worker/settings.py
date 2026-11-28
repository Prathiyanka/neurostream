from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}"


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    model_dir: str = "models"
    model: str = "eegnet-motor-1.0.1"  # directory name under model_dir: <name>-<version>
    worker_id: str = field(default_factory=_default_worker_id)
    batch_max: int = 32  # upper bound only: a read returns whatever is available, never waits
    block_ms: int = 1000  # how long an idle worker sleeps on the stream before re-checking
    reclaim_interval_s: float = 0.5  # how often to look for jobs a dead worker left pending
    # A pending job idle this long is presumed abandoned. Well below stale_after_s, so a
    # dead worker's jobs are recovered while still fresh; ~100x a normal batch (~10 ms), so
    # a live worker's jobs are not taken from it.
    reclaim_min_idle_ms: int = 1000
    # Jobs whose window completed longer ago than this are skipped (acked, no prediction)
    # and counted as `stale`: under overload a late prediction is worth less than catching
    # up. 0 disables the check.
    stale_after_s: float = 2.0
    max_deliveries: int = 3  # a job delivered more often than this is dead-lettered
    ort_threads: int = 1  # one thread per worker; scale out with more workers
    predictions_maxlen: int = 100_000
    dead_maxlen: int = 1000
    heartbeat_interval_s: float = 1.0
    model_check_interval_s: float = 1.0  # how often to look for a new active model
    heartbeat_ttl_s: int = 5
    # Fault injection for tests and demos: hold every batch this long before publishing,
    # to simulate a slow or overloaded worker. Always 0 in normal operation.
    fault_delay_s: float = 0.0

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            redis_url=_env("REDIS_URL", cls.redis_url),
            model_dir=_env("MODEL_DIR", cls.model_dir),
            model=_env("MODEL", cls.model),
            worker_id=_env("WORKER_ID", "") or _default_worker_id(),
            batch_max=int(_env("WORKER_BATCH_MAX", str(cls.batch_max))),
            block_ms=int(_env("WORKER_BLOCK_MS", str(cls.block_ms))),
            reclaim_interval_s=float(_env("RECLAIM_INTERVAL_S", str(cls.reclaim_interval_s))),
            reclaim_min_idle_ms=int(_env("RECLAIM_MIN_IDLE_MS", str(cls.reclaim_min_idle_ms))),
            max_deliveries=int(_env("MAX_DELIVERIES", str(cls.max_deliveries))),
            ort_threads=int(_env("ORT_THREADS", str(cls.ort_threads))),
            predictions_maxlen=int(_env("PREDICTIONS_MAXLEN", str(cls.predictions_maxlen))),
            dead_maxlen=int(_env("DEAD_MAXLEN", str(cls.dead_maxlen))),
            heartbeat_interval_s=float(_env("HEARTBEAT_INTERVAL_S", str(cls.heartbeat_interval_s))),
            heartbeat_ttl_s=int(_env("HEARTBEAT_TTL_S", str(cls.heartbeat_ttl_s))),
            model_check_interval_s=float(
                _env("MODEL_CHECK_INTERVAL_S", str(cls.model_check_interval_s))
            ),
            fault_delay_s=float(_env("WORKER_FAULT_DELAY_S", str(cls.fault_delay_s))),
            stale_after_s=float(_env("STALE_AFTER_S", str(cls.stale_after_s))),
        )
