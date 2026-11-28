"""Postgres schema (SQLAlchemy Core) and engine.

Core, not the ORM: the hot path is bulk ``INSERT ... ON CONFLICT DO NOTHING`` of
predictions, which Core expresses directly, and the API returns plain dicts anyway.
The migration in ``migrations/versions`` creates exactly these tables; a test checks
that the two never drift apart.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, REAL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

metadata = MetaData()

sessions = Table(
    "sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("subject", SmallInteger, nullable=True),
    # "api": started through POST /sessions; "external": seen in predictions only (e.g. a
    # replayer run from the command line, or a load test).
    Column("source", String(16), nullable=False),
    Column("status", String(16), nullable=False),  # running|finished|stopped|failed|interrupted
    Column("speed", Float, nullable=True),
    Column("config", JSONB, nullable=True),  # replay request (faults, max_seconds, ...)
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("ended_at", DateTime(timezone=True), nullable=True),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    # Deterministic "<session>:<end sample>". UNIQUE makes persistence idempotent: a
    # prediction duplicated by at-least-once delivery (worker crash) is stored once.
    Column("window_id", String(96), nullable=False, unique=True),
    Column(
        "session_id",
        String(64),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("window_end_sample", Integer, nullable=False),
    Column("window_end_at", DateTime(timezone=True), nullable=False),
    Column("predicted_class", SmallInteger, nullable=False),
    Column("probabilities", ARRAY(REAL), nullable=False),
    Column("true_label", SmallInteger, nullable=False),  # -1 = unknown / mixed
    Column("filled_samples", Integer, nullable=False),
    Column("model", String(64), nullable=False),
    Column("worker_id", String(128), nullable=False),
    Column("reclaimed", Boolean, nullable=False),
    Column("batch_size", SmallInteger, nullable=False),
    Column("ingest_ms", REAL, nullable=False),
    Column("queue_ms", REAL, nullable=False),
    Column("worker_ms", REAL, nullable=False),
    Column("preprocess_ms", REAL, nullable=False),
    Column("inference_ms", REAL, nullable=False),
    Column("latency_ms", REAL, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=False),
    Column("persisted_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("ix_predictions_session_sample", "session_id", "window_end_sample"),
)

metrics_snapshots = Table(
    "metrics_snapshots",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("taken_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("workers", Integer, nullable=False),
    Column("queue_lag", Integer, nullable=False),  # jobs not yet delivered to a worker
    Column("queue_pending", Integer, nullable=False),  # delivered, not yet acknowledged
    Column("predictions_per_s", Float, nullable=False),
    Column("latency_p50_ms", Float, nullable=True),  # over predictions since last snapshot
    Column("latency_p95_ms", Float, nullable=True),
    Column("counters", JSONB, nullable=False),  # worker totals: acked, stale, lost, dead, ...
    Index("ix_metrics_snapshots_taken_at", "taken_at"),
)

model_events = Table(
    "model_events",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("model", String(64), nullable=False),
    Column("previous", String(64), nullable=True),
    Column("source", String(32), nullable=False),
    Column("note", Text, nullable=True),
)


def async_url(url: str) -> str:
    """Any postgres URL -> the asyncpg form the API uses. (psycopg's async mode needs a
    selector event loop, which Windows does not use by default; asyncpg works with both.)"""
    for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix) :]
    raise ValueError(f"not a postgres URL: {url}")


def sync_url(url: str) -> str:
    """Any postgres URL -> the psycopg (sync) form Alembic uses."""
    for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    raise ValueError(f"not a postgres URL: {url}")


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(async_url(url), pool_size=5, max_overflow=5, pool_pre_ping=True)
