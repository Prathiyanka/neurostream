"""Fixtures for API tests: a throwaway Postgres database migrated with the real Alembic
migrations, and the API wired to an in-process replayer and ingest."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
import uvicorn
from alembic import command
from alembic.config import Config
from fastapi import FastAPI

from api.app import create_app
from api.db import sync_url
from api.settings import Settings
from replayer.server import ServerSettings
from replayer.server import create_app as create_replayer
from tests.integration.conftest import MODEL_DIR, REPO_ROOT, IngestServer, free_port, redis_test_url

TEST_DB = "neurostream_test"


def base_database_url() -> str:
    """DATABASE_URL from the environment (CI) or the local .env file."""
    url = os.environ.get("DATABASE_URL")
    env_file = REPO_ROOT / ".env"
    if not url and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                url = line.split("=", 1)[1].strip()
    if not url:
        pytest.skip("DATABASE_URL not set")
    return url


def api_database_url() -> str:
    base = base_database_url()
    return base.rsplit("/", 1)[0] + f"/{TEST_DB}"


def _plain(url: str) -> str:
    return sync_url(url).replace("postgresql+psycopg://", "postgresql://")


def alembic_config(url: str) -> Config:
    cfg = Config(str(REPO_ROOT / "services" / "api" / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "services" / "api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url(url))
    return cfg


@pytest.fixture
def database() -> Iterator[str]:
    """A clean, migrated test database (dropped and re-migrated for every test)."""
    with psycopg.connect(_plain(base_database_url()), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    url = api_database_url()
    with psycopg.connect(_plain(url), autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    command.upgrade(alembic_config(url), "head")
    yield url


@dataclass
class Api:
    client: httpx.AsyncClient
    app: FastAPI
    db_url: str


def api_settings(db_url: str, **overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "database_url": db_url,
        "redis_url": redis_test_url(),
        "replayer_url": "http://replayer",
        "model_dir": str(MODEL_DIR),
        "consumer": "api-test",
        "snapshot_interval_s": 0.3,
        "run_background": True,
    }
    return Settings(**{**defaults, **overrides})


@pytest.fixture
async def api(ingest_server: IngestServer, database: str) -> AsyncIterator[Api]:
    async with make_api(ingest_server, database) as a:
        yield a


class make_api:
    def __init__(self, ingest: IngestServer, db_url: str, **overrides: Any) -> None:
        self.ingest, self.db_url, self.overrides = ingest, db_url, overrides

    async def __aenter__(self) -> Api:
        replayer = create_replayer(
            ServerSettings(ingest_url=self.ingest.ws_url, data_dir=REPO_ROOT / "data")
        )
        self._replayer_cm = replayer.router.lifespan_context(replayer)
        await self._replayer_cm.__aenter__()
        app = create_app(
            api_settings(self.db_url, **self.overrides),
            replayer_transport=httpx.ASGITransport(app=replayer),
        )
        self._app_cm = app.router.lifespan_context(app)
        await self._app_cm.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://api", timeout=30
        )
        return Api(self.client, app, self.db_url)

    async def __aexit__(self, *exc: object) -> None:
        await self.client.aclose()
        await self._app_cm.__aexit__(None, None, None)
        await self._replayer_cm.__aexit__(None, None, None)


class ServedApi:
    """The API on a real port (WebSocket tests need a real server)."""

    def __init__(self, app: FastAPI) -> None:
        self.port = free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="off"
            )
        )

    async def __aenter__(self) -> str:
        self.task = asyncio.create_task(self.server.serve())
        for _ in range(500):
            if self.server.started:
                return f"ws://127.0.0.1:{self.port}"
            await asyncio.sleep(0.01)
        raise AssertionError("API server did not start")

    async def __aexit__(self, *exc: object) -> None:
        self.server.should_exit = True
        await asyncio.wait_for(self.task, 10)


def model_dir_copy(tmp_path: Path) -> Path:
    import shutil

    d = tmp_path / "models"
    shutil.copytree(MODEL_DIR, d)
    return d
