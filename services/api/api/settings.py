from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://neurostream:change-me-locally@localhost:5432/neurostream"
    redis_url: str = "redis://localhost:6379/0"
    replayer_url: str = "http://localhost:8002"
    model_dir: str = "models"
    default_model: str = "eegnet-motor-1.0.1"  # served when ns:model:active was never set
    # Name of this API instance in the `api` consumer group on ns:predictions. Stable
    # across restarts, so a restarted instance re-reads its own unacknowledged entries.
    consumer: str = field(default_factory=socket.gethostname)
    persist_batch: int = 500
    snapshot_interval_s: float = 5.0
    live_metrics_interval_s: float = 1.0
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)  # Vite dev server (Phase 6)
    run_background: bool = True  # persister + snapshots; tests switch it off

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=_env("DATABASE_URL", cls.database_url),
            redis_url=_env("REDIS_URL", cls.redis_url),
            replayer_url=_env("REPLAYER_URL", cls.replayer_url),
            model_dir=_env("MODEL_DIR", cls.model_dir),
            default_model=_env("MODEL", cls.default_model),
            consumer=_env("API_CONSUMER", "") or socket.gethostname(),
            persist_batch=int(_env("PERSIST_BATCH", str(cls.persist_batch))),
            snapshot_interval_s=float(_env("SNAPSHOT_INTERVAL_S", str(cls.snapshot_interval_s))),
            cors_origins=tuple(
                o for o in _env("CORS_ORIGINS", ",".join(cls.cors_origins)).split(",") if o
            ),
        )
