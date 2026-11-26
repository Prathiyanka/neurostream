"""Redis key names, in one place so every service agrees on them."""

WINDOWS_STREAM = "ns:windows"  # shared stream of self-contained window jobs
WORKER_GROUP = "workers"  # consumer group the inference workers read with
SESSIONS_SET = "ns:sessions"


def raw_stream(session_id: str) -> str:
    """Per-session stream of raw packets (traces for the dashboard, debugging)."""
    return f"ns:raw:{session_id}"


def session_meta(session_id: str) -> str:
    return f"ns:session:{session_id}"


def ingest_metrics(session_id: str) -> str:
    return f"ns:metrics:ingest:{session_id}"
