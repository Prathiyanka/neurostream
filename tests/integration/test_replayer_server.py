"""The replayer HTTP service driving real sessions into ingest."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from neurostream_common import keys
from replayer.server import ServerSettings, create_app
from tests.integration.conftest import REPO_ROOT, IngestServer

pytestmark = pytest.mark.integration


@pytest.fixture
async def replayer(ingest_server: IngestServer) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(ServerSettings(ingest_url=ingest_server.ws_url, data_dir=REPO_ROOT / "data"))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://replayer") as c:
            yield c


async def wait_status(c: httpx.AsyncClient, sid: str, status: str, timeout: float = 10) -> dict:  # type: ignore[type-arg]
    for _ in range(int(timeout / 0.05)):
        r = (await c.get(f"/replays/{sid}")).json()
        if r["status"] == status:
            return r  # type: ignore[no-any-return]
        await asyncio.sleep(0.05)
    raise AssertionError(f"{sid} never reached {status}: {r}")


async def test_start_runs_a_session_into_ingest(
    replayer: httpx.AsyncClient, ingest_server: IngestServer
) -> None:
    r = await replayer.post(
        "/replays", json={"synthetic": True, "speed": 40, "max_seconds": 5, "session_id": "rs1"}
    )
    assert r.status_code == 201, r.text
    done = await wait_status(replayer, "rs1", "finished")
    assert done["stats"]["sent"] == done["stats"]["packets_total"] == 50
    for _ in range(100):  # ingest marks the session ended once flushed
        if await ingest_server.redis.hget(keys.session_meta("rs1"), "status") == b"ended":  # type: ignore[misc]
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError("ingest never saw the session end")


async def test_stop_ends_a_running_replay(replayer: httpx.AsyncClient) -> None:
    r = await replayer.post("/replays", json={"synthetic": True, "speed": 1, "session_id": "rs2"})
    assert r.status_code == 201
    await asyncio.sleep(0.5)
    stopped = (await replayer.delete("/replays/rs2")).json()
    assert stopped["status"] == "stopped"
    assert 0 < stopped["stats"]["sent"] < stopped["stats"]["packets_total"]
    listed = (await replayer.get("/replays")).json()
    assert [x["session_id"] for x in listed] == ["rs2"]


async def test_bad_requests_are_refused_immediately(replayer: httpx.AsyncClient) -> None:
    both = await replayer.post("/replays", json={"synthetic": True, "subject": 3})
    assert both.status_code == 422
    assert (await replayer.post("/replays", json={"subject": 88})).status_code in (404, 422)
    assert (
        await replayer.post("/replays", json={"synthetic": True, "speed": 0})
    ).status_code == 422
    assert (await replayer.delete("/replays/nope")).status_code == 404
