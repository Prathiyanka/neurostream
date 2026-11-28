"""initial schema

Revision ID: 0001
Revises: (none)
Create Date: 2026-11-27 22:15:40.836386
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "metrics_snapshots",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "taken_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("workers", sa.Integer(), nullable=False),
        sa.Column("queue_lag", sa.Integer(), nullable=False),
        sa.Column("queue_pending", sa.Integer(), nullable=False),
        sa.Column("predictions_per_s", sa.Float(), nullable=False),
        sa.Column("latency_p50_ms", sa.Float(), nullable=True),
        sa.Column("latency_p95_ms", sa.Float(), nullable=True),
        sa.Column("counters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_metrics_snapshots_taken_at", "metrics_snapshots", ["taken_at"], unique=False
    )
    op.create_table(
        "model_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("previous", sa.String(length=64), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("subject", sa.SmallInteger(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("speed", sa.Float(), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("window_id", sa.String(length=96), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("window_end_sample", sa.Integer(), nullable=False),
        sa.Column("window_end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_class", sa.SmallInteger(), nullable=False),
        sa.Column("probabilities", postgresql.ARRAY(sa.REAL()), nullable=False),
        sa.Column("true_label", sa.SmallInteger(), nullable=False),
        sa.Column("filled_samples", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("worker_id", sa.String(length=128), nullable=False),
        sa.Column("reclaimed", sa.Boolean(), nullable=False),
        sa.Column("batch_size", sa.SmallInteger(), nullable=False),
        sa.Column("ingest_ms", sa.REAL(), nullable=False),
        sa.Column("queue_ms", sa.REAL(), nullable=False),
        sa.Column("worker_ms", sa.REAL(), nullable=False),
        sa.Column("preprocess_ms", sa.REAL(), nullable=False),
        sa.Column("inference_ms", sa.REAL(), nullable=False),
        sa.Column("latency_ms", sa.REAL(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "persisted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("window_id"),
    )
    op.create_index(
        "ix_predictions_session_sample",
        "predictions",
        ["session_id", "window_end_sample"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_session_sample", table_name="predictions")
    op.drop_table("predictions")
    op.drop_table("sessions")
    op.drop_table("model_events")
    op.drop_index("ix_metrics_snapshots_taken_at", table_name="metrics_snapshots")
    op.drop_table("metrics_snapshots")
