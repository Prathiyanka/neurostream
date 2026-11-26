"""Smoke tests for the Compose infrastructure. Run after `docker compose up -d`."""

import os

import pytest

pytestmark = pytest.mark.integration


def test_redis_is_reachable_and_supports_streams() -> None:
    import redis

    r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    assert r.ping()
    key = "neurostream:test:smoke"
    r.delete(key)
    entry_id = r.xadd(key, {"seq": 1})
    assert r.xlen(key) == 1
    r.xgroup_create(key, "smoke-group", id="0")
    got = r.xreadgroup("smoke-group", "c1", {key: ">"}, count=1)
    assert got[0][1][0][0] == entry_id
    r.delete(key)


def test_postgres_is_reachable() -> None:
    import psycopg

    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://neurostream:change-me-locally@localhost:5432/neurostream",
    ).replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(url, connect_timeout=5) as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
