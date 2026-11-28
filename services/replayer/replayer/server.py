"""The replayer as a long-running "device" service: ``python -m replayer serve``.

The API starts and stops replays over HTTP (see README: why HTTP, not a Redis command
channel). Each replay is one ``SessionStreamer`` task streaming to ingest, exactly as the
command-line mode does, so the API never runs device simulation in its own process.

    POST   /replays            start one (subject or synthetic, speed, faults)
    GET    /replays            all replays this process knows (running and recent)
    GET    /replays/{id}       one replay, with live counters
    DELETE /replays/{id}       stop early (sends `end`, so ingest closes it cleanly)
    GET    /health
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from neurostream_common.schemas import SessionId
from replayer.faults import FaultConfig
from replayer.recording import Recording, edf_path, excluded_subjects, load_subject
from replayer.recording import synthetic_recording as make_synthetic
from replayer.streamer import SessionConfig, SessionStreamer

log = logging.getLogger("replayer.server")

Status = Literal["running", "finished", "stopped", "failed"]


@dataclass(frozen=True)
class ServerSettings:
    ingest_url: str = "ws://localhost:8001/ws/device"
    data_dir: Path = Path("data")
    max_running: int = 64  # refuse new replays beyond this (each is a live stream)
    keep_finished: int = 200  # finished replays remembered for GET /replays

    @classmethod
    def from_env(cls) -> ServerSettings:
        return cls(
            ingest_url=os.environ.get("INGEST_URL", cls.ingest_url),
            data_dir=Path(os.environ.get("DATA_DIR", str(cls.data_dir))),
            max_running=int(os.environ.get("REPLAYER_MAX_RUNNING", str(cls.max_running))),
        )


class FaultsIn(BaseModel):
    drop_prob: float = Field(default=0.0, ge=0, le=1)
    duplicate_prob: float = Field(default=0.0, ge=0, le=1)
    reorder_prob: float = Field(default=0.0, ge=0, le=1)
    jitter_ms: float = Field(default=0.0, ge=0, le=1000)
    disconnect_every_s: float = Field(default=0.0, ge=0)
    reconnect_delay_s: float = Field(default=1.0, ge=0, le=60)


class ReplayIn(BaseModel):
    subject: int | None = Field(default=None, ge=1, le=109)
    synthetic: bool = False  # 64-channel synthetic signal instead of a real subject
    speed: float = Field(default=1.0, gt=0, le=100)
    max_seconds: float | None = Field(default=None, gt=0)
    loop: bool = False
    chunk_size: int = Field(default=16, ge=1, le=160)
    faults: FaultsIn = Field(default_factory=FaultsIn)
    seed: int = 0
    session_id: SessionId | None = None  # generated if omitted

    @model_validator(mode="after")
    def _one_source(self) -> ReplayIn:
        if (self.subject is None) == (not self.synthetic):
            raise ValueError("give exactly one of `subject` or `synthetic: true`")
        return self


@dataclass
class Replay:
    session_id: str
    request: ReplayIn
    streamer: SessionStreamer
    task: asyncio.Task[Any] | None = None
    status: Status = "running"
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    stop_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "subject": self.request.subject,
            "synthetic": self.request.synthetic,
            "speed": self.request.speed,
            "max_seconds": self.request.max_seconds,
            "loop": self.request.loop,
            "faults": self.request.faults.model_dump(),
            "status": self.status,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "stats": asdict(self.streamer.stats),
        }


class ReplayManager:
    def __init__(self, settings: ServerSettings) -> None:
        self.s = settings
        self.replays: OrderedDict[str, Replay] = OrderedDict()

    def running(self) -> int:
        return sum(r.status == "running" for r in self.replays.values())

    async def _recording(self, req: ReplayIn) -> Recording:
        if req.synthetic:
            seconds = req.max_seconds or 120.0
            return make_synthetic(n_channels=64, seconds=min(seconds, 600.0), seed=req.seed)
        assert req.subject is not None
        if req.subject in excluded_subjects(self.s.data_dir):
            raise HTTPException(422, f"subject {req.subject} is excluded (see docs/DATA.md)")
        if not edf_path(self.s.data_dir, req.subject, 4).exists():
            raise HTTPException(404, f"no data for subject {req.subject}: run download_data.py")
        # MNE reads the EDF files synchronously: keep the event loop free for live streams.
        return await asyncio.to_thread(load_subject, req.subject, self.s.data_dir)

    async def start(self, req: ReplayIn) -> Replay:
        if self.running() >= self.s.max_running:
            raise HTTPException(429, f"{self.s.max_running} replays already running")
        session_id = req.session_id or (
            f"sub{req.subject:03d}-{secrets.token_hex(3)}"
            if req.subject is not None
            else f"synth-{secrets.token_hex(3)}"
        )
        if session_id in self.replays and self.replays[session_id].status == "running":
            raise HTTPException(409, f"replay {session_id} is already running")
        rec = await self._recording(req)
        f = req.faults
        cfg = SessionConfig(
            url=self.s.ingest_url,
            session_id=session_id,
            chunk_size=req.chunk_size,
            speed=req.speed,
            loop=req.loop,
            max_seconds=req.max_seconds,
            faults=FaultConfig(**f.model_dump(), seed=req.seed),
            max_connect_attempts=10,
        )
        replay = Replay(session_id, req, SessionStreamer(cfg, rec))
        replay.task = asyncio.create_task(self._run(replay))
        self.replays[session_id] = replay
        self._forget_old()
        log.info("replay started", extra={"session_id": session_id, "subject": req.subject})
        return replay

    async def _run(self, replay: Replay) -> None:
        try:
            await replay.streamer.run()
            replay.status = "stopped" if replay.stop_requested else "finished"
        except Exception as exc:
            replay.status, replay.error = "failed", f"{type(exc).__name__}: {exc}"
            log.error("replay failed", extra={"session_id": replay.session_id, "err": str(exc)})
        finally:
            replay.ended_at = time.time()

    async def stop(self, session_id: str, timeout: float = 5.0) -> Replay:
        replay = self.get(session_id)
        if replay.status == "running" and replay.task is not None:
            replay.stop_requested = True
            replay.streamer.stop()
            try:
                await asyncio.wait_for(asyncio.shield(replay.task), timeout)
            except TimeoutError:  # e.g. stuck reconnecting: cancel instead
                replay.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await replay.task
                replay.status, replay.ended_at = "stopped", time.time()
        return replay

    def get(self, session_id: str) -> Replay:
        replay = self.replays.get(session_id)
        if replay is None:
            raise HTTPException(404, f"no replay {session_id}")
        return replay

    def _forget_old(self) -> None:
        done = [k for k, r in self.replays.items() if r.status != "running"]
        for k in done[: max(0, len(done) - self.s.keep_finished)]:
            del self.replays[k]

    async def shutdown(self) -> None:
        for r in list(self.replays.values()):
            if r.status == "running":
                await self.stop(r.session_id, timeout=2.0)


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    settings = settings or ServerSettings.from_env()
    manager = ReplayManager(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.shutdown()

    app = FastAPI(title="NeuroStream replayer", lifespan=lifespan)
    app.state.manager = manager

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "running": manager.running(), "ingest_url": settings.ingest_url}

    @app.post("/replays", status_code=201)
    async def start(req: ReplayIn) -> dict[str, Any]:
        return (await manager.start(req)).to_dict()

    @app.get("/replays")
    async def list_replays() -> list[dict[str, Any]]:
        return [r.to_dict() for r in reversed(manager.replays.values())]

    @app.get("/replays/{session_id}")
    async def get_replay(session_id: str) -> dict[str, Any]:
        return manager.get(session_id).to_dict()

    @app.delete("/replays/{session_id}")
    async def stop(session_id: str) -> dict[str, Any]:
        return (await manager.stop(session_id)).to_dict()

    return app


def serve(host: str = "0.0.0.0", port: int = 8002) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_config=None)
