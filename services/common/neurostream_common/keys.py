"""Redis key names, in one place so every service agrees on them."""

WINDOWS_STREAM = "ns:windows"  # shared stream of self-contained window jobs
WORKER_GROUP = "workers"  # consumer group the inference workers read with
SESSIONS_SET = "ns:sessions"
WINDOWS_DEAD_STREAM = "ns:windows:dead"  # jobs that failed repeatedly, kept for inspection
PREDICTIONS_STREAM = "ns:predictions"  # durable: the API persists from here (Phase 5)
PREDICTIONS_CHANNEL = "ns:predictions:live"  # pub/sub fan-out for live views (lossy)
WORKER_METRICS = "ns:metrics:workers"  # counters summed over all workers
WORKERS_SET = "ns:workers"


def worker_heartbeat(worker_id: str) -> str:
    """Per-worker hash with a short TTL: it disappears when the worker dies."""
    return f"ns:worker:{worker_id}"


def raw_stream(session_id: str) -> str:
    """Per-session stream of raw packets (traces for the dashboard, debugging)."""
    return f"ns:raw:{session_id}"


def session_meta(session_id: str) -> str:
    return f"ns:session:{session_id}"


def ingest_metrics(session_id: str) -> str:
    return f"ns:metrics:ingest:{session_id}"
