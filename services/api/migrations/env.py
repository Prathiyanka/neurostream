"""Alembic environment: synchronous psycopg connection, URL from DATABASE_URL."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from api.db import metadata, sync_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Tests pass the URL in the Alembic config; everything else uses DATABASE_URL.
url = sync_url(
    config.get_main_option("sqlalchemy.url")
    or os.environ.get("DATABASE_URL")
    or "postgresql://neurostream:change-me-locally@localhost:5432/neurostream"
)


def run_migrations_online() -> None:
    engine = create_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run_migrations_online()
