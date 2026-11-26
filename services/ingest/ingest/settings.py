from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    window_s: float = 2.0
    step_s: float = 0.5
    pad_s: float = 1.0
    max_fill_s: float = 0.25  # gaps up to this long are filled; longer ones reset windowing
    queue_max_items: int = 512  # per-session write buffer (items = raw packets or windows)
    write_batch_max: int = 64  # max items per Redis pipeline round-trip
    raw_maxlen: int = 1200  # ~2 min of raw packets per session at 10 packets/s
    windows_maxlen: int = 5000  # bound on the shared window-job stream (load shedding)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            redis_url=_env("REDIS_URL", cls.redis_url),
            window_s=float(_env("WINDOW_S", str(cls.window_s))),
            step_s=float(_env("STEP_S", str(cls.step_s))),
            pad_s=float(_env("PAD_S", str(cls.pad_s))),
            max_fill_s=float(_env("MAX_FILL_S", str(cls.max_fill_s))),
            queue_max_items=int(_env("INGEST_QUEUE_MAX_ITEMS", str(cls.queue_max_items))),
            write_batch_max=int(_env("INGEST_WRITE_BATCH_MAX", str(cls.write_batch_max))),
            raw_maxlen=int(_env("RAW_STREAM_MAXLEN", str(cls.raw_maxlen))),
            windows_maxlen=int(_env("WINDOWS_STREAM_MAXLEN", str(cls.windows_maxlen))),
        )
