"""FastAPI app: REST + WebSocket.

    GET    /health
    GET    /sessions                    recent sessions (Postgres)
    POST   /sessions                    start a replay (sent to the replayer service)
    GET    /sessions/{id}               summary: accuracy, latency, ingest counters
    DELETE /sessions/{id}               stop a replay the API started
    GET    /sessions/{id}/predictions   stored predictions
    GET    /models                      available models, active one, per-worker model
    PUT    /models/active               hot-swap the served model
    GET    /metrics                     system metrics now
    GET    /metrics/history             snapshots stored every 5 s
    WS     /ws/live                     predictions, metrics, EEG traces

No authentication: this is a local research/portfolio system (see README).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from api.db import make_engine, metrics_snapshots, model_events, predictions, sessions
from api.live import Hub
from api.metrics import active_model, collect
from api.models_registry import list_models, verify_model
from api.persister import Persister, jsonable
from api.settings import Settings
from neurostream_common import keys
from neurostream_common.model_card import ModelCardError

log = logging.getLogger("api")


# ---- request bodies --------------------------------------------------------------------


class Faults(BaseModel):
    drop_prob: float = Field(default=0.0, ge=0, le=1)
    duplicate_prob: float = Field(default=0.0, ge=0, le=1)
    reorder_prob: float = Field(default=0.0, ge=0, le=1)
    jitter_ms: float = Field(default=0.0, ge=0, le=1000)
    disconnect_every_s: float = Field(default=0.0, ge=0)
    reconnect_delay_s: float = Field(default=1.0, ge=0, le=60)


class SessionCreate(BaseModel):
    subject: int | None = Field(default=None, ge=1, le=109)
    synthetic: bool = False
    speed: float = Field(default=1.0, gt=0, le=100)
    max_seconds: float | None = Field(default=None, gt=0)
    loop: bool = False
    faults: Faults = Field(default_factory=Faults)
    seed: int = 0


class ActiveModel(BaseModel):
    model: str


# ---- replayer client -------------------------------------------------------------------


class ReplayerClient:
    """The replayer is an external device service; the API only sends it commands."""

    def __init__(self, base_url: str, transport: httpx.AsyncBaseTransport | None = None):
        self.http = httpx.AsyncClient(base_url=base_url, timeout=15.0, transport=transport)

    async def call(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = await self.http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"replayer unreachable: {exc}") from exc
        if r.status_code >= 500:
            raise HTTPException(502, f"replayer error {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:  # pass the replayer's own explanation through
            raise HTTPException(r.status_code, r.json().get("detail", r.text))
        return r.json()

    async def healthy(self) -> bool:
        try:
            return (await self.http.get("/health")).status_code == 200
        except httpx.HTTPError:
            return False


# ---- queries ---------------------------------------------------------------------------


async def session_summary(engine: AsyncEngine, session_id: str) -> dict[str, Any]:
    p = predictions.c
    async with engine.connect() as conn:
        stats = (
            await conn.execute(
                select(
                    func.count().label("predictions"),
                    func.percentile_cont(0.5).within_group(p.latency_ms).label("p50"),
                    func.percentile_cont(0.95).within_group(p.latency_ms).label("p95"),
                    func.percentile_cont(0.99).within_group(p.latency_ms).label("p99"),
                    func.avg(p.batch_size).label("mean_batch"),
                    func.count().filter(p.reclaimed).label("reclaimed"),
                    func.array_agg(func.distinct(p.model)).label("models"),
                ).where(p.session_id == session_id)
            )
        ).one()
        per_class = (
            await conn.execute(
                select(
                    p.true_label,
                    func.count(),
                    func.count().filter(p.predicted_class == p.true_label),
                )
                .where(p.session_id == session_id, p.true_label >= 0)
                .group_by(p.true_label)
            )
        ).all()
    counts = [(cast(int, row[1]), cast(int, row[2])) for row in per_class]
    recalls = [correct / n for n, correct in counts if n]
    return {
        "predictions": stats.predictions,
        "labelled_windows": sum(n for n, _ in counts),
        # Mean per-class recall on windows with one known label (chance = 33.3%).
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "latency_ms": {"p50": stats.p50, "p95": stats.p95, "p99": stats.p99},
        "mean_batch_size": float(stats.mean_batch) if stats.mean_batch is not None else None,
        "reclaimed": stats.reclaimed,
        "models": [m for m in (stats.models or []) if m],
    }


# ---- background ------------------------------------------------------------------------


async def snapshot_loop(
    s: Settings, redis: Redis, engine: AsyncEngine, persister: Persister
) -> None:
    """Every few seconds: store a metrics snapshot, and mark sessions ingest has ended."""
    last_persisted, last_t = persister.persisted_total, time.monotonic()
    while True:
        await asyncio.sleep(s.snapshot_interval_s)
        try:
            m = await collect(redis, s.default_model)
            now = time.monotonic()
            lat = list(persister.recent_latencies)
            persister.recent_latencies.clear()
            rate = (persister.persisted_total - last_persisted) / max(now - last_t, 1e-9)
            last_persisted, last_t = persister.persisted_total, now
            async with engine.begin() as conn:
                await conn.execute(
                    insert(metrics_snapshots).values(
                        workers=m["workers"]["count"],
                        queue_lag=m["queue"]["lag"],
                        queue_pending=m["queue"]["pending"],
                        predictions_per_s=rate,
                        latency_p50_ms=float(np.percentile(lat, 50)) if lat else None,
                        latency_p95_ms=float(np.percentile(lat, 95)) if lat else None,
                        counters={**m["jobs"], "persister": m["persister"]},
                    )
                )
                await sync_session_status(conn, redis)
        except (RedisError, SQLAlchemyError, OSError) as exc:
            log.warning("snapshot failed: %s", exc)


ORPHAN_AFTER_S = 30.0  # a device gone this long without ending its session


async def sync_session_status(conn: Any, redis: Redis, now: float | None = None) -> None:
    """Close sessions Postgres still lists as running:

    * ingest saw the device send `end` -> ``finished``;
    * ingest saw the device disconnect and it has not come back for ``ORPHAN_AFTER_S``
      (e.g. the replayer was restarted and lost its in-memory replays) -> ``interrupted``.
      Without this, such a session would show as running forever.
    """
    now = time.time() if now is None else now
    running = (
        await conn.execute(select(sessions.c.id).where(sessions.c.status == "running"))
    ).scalars()
    for sid in list(running):
        meta = await redis.hgetall(keys.session_meta(sid))  # type: ignore[misc]
        status = meta.get(b"status")
        new = None
        if status == b"ended":
            new = "finished"
        elif (
            status == b"disconnected"
            and now - float(meta.get(b"updated_at", b"0")) > ORPHAN_AFTER_S
        ):
            new = "interrupted"
        if new:
            await conn.execute(
                update(sessions)
                .where(sessions.c.id == sid, sessions.c.status == "running")
                .values(status=new, ended_at=func.now())
            )


# ---- app -------------------------------------------------------------------------------


def create_app(
    settings: Settings | None = None, replayer_transport: httpx.AsyncBaseTransport | None = None
) -> FastAPI:
    s = settings or Settings.from_env()
    model_dir = Path(s.model_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        redis = Redis.from_url(s.redis_url)
        engine = make_engine(s.database_url)
        replayer = ReplayerClient(s.replayer_url, replayer_transport)
        persister = Persister(redis, engine, s.consumer, s.persist_batch)
        hub = Hub(redis, lambda: collect(redis, s.default_model), s.live_metrics_interval_s)
        app.state.redis, app.state.engine = redis, engine
        app.state.replayer, app.state.persister, app.state.hub = replayer, persister, hub
        await persister.ensure_group()
        hub.start()
        tasks = []
        if s.run_background:
            tasks = [
                asyncio.create_task(persister.run()),
                asyncio.create_task(snapshot_loop(s, redis, engine, persister)),
            ]
        yield
        for t in tasks:
            t.cancel()
        for t in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await hub.stop()
        await replayer.http.aclose()
        await engine.dispose()
        await redis.aclose()

    app = FastAPI(title="NeuroStream API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(s.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def st(name: str) -> Any:
        return getattr(app.state, name)

    @app.get("/health")
    async def health() -> Any:
        ok = {"redis": False, "postgres": False, "replayer": False}
        with contextlib.suppress(RedisError, OSError):
            ok["redis"] = bool(await st("redis").ping())
        with contextlib.suppress(SQLAlchemyError, OSError):
            async with st("engine").connect() as conn:
                ok["postgres"] = (await conn.execute(text("SELECT 1"))).scalar() == 1
        ok["replayer"] = await st("replayer").healthy()
        status = 200 if ok["redis"] and ok["postgres"] else 503
        return JSONResponse({"status": "ok" if status == 200 else "degraded", **ok}, status)

    # ---- sessions ----

    @app.get("/sessions")
    async def list_sessions(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        async with st("engine").connect() as conn:
            rows = await conn.execute(
                select(sessions).order_by(sessions.c.created_at.desc()).limit(limit)
            )
            return [jsonable(dict(r._mapping)) for r in rows]

    @app.post("/sessions", status_code=201)
    async def start_session(req: SessionCreate) -> dict[str, Any]:
        replay = await st("replayer").call("POST", "/replays", json=req.model_dump())
        row = {
            "id": replay["session_id"],
            "subject": req.subject,
            "source": "api",
            "status": "running",
            "speed": req.speed,
            "config": req.model_dump(),
        }
        async with st("engine").begin() as conn:
            await conn.execute(
                insert(sessions)
                .values(row)
                .on_conflict_do_update(
                    index_elements=["id"],
                    set_={"source": "api", "status": "running", "config": row["config"]},
                )
            )
        return {**row, "replay": replay}

    async def _session_row(session_id: str) -> dict[str, Any]:
        async with st("engine").connect() as conn:
            row = (await conn.execute(select(sessions).where(sessions.c.id == session_id))).first()
        if row is None:
            raise HTTPException(404, f"no session {session_id}")
        return dict(row._mapping)

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        row = await _session_row(session_id)
        redis: Redis = st("redis")
        ingest = await redis.hgetall(keys.ingest_metrics(session_id))  # type: ignore[misc]
        out = jsonable(row)
        out["summary"] = await session_summary(st("engine"), session_id)
        out["ingest"] = {k.decode(): int(v) for k, v in ingest.items()}
        if row["source"] == "api":
            with contextlib.suppress(HTTPException):
                out["replay"] = await st("replayer").call("GET", f"/replays/{session_id}")
        return out

    @app.delete("/sessions/{session_id}")
    async def stop_session(session_id: str) -> dict[str, Any]:
        row = await _session_row(session_id)
        if row["source"] != "api":
            raise HTTPException(409, "session was not started through the API")
        try:
            replay = await st("replayer").call("DELETE", f"/replays/{session_id}")
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            # The replayer no longer knows it (restarted): the stream is gone either way.
            replay = {"session_id": session_id, "status": "stopped", "note": "unknown to replayer"}
        status = replay["status"] if replay["status"] != "running" else "stopped"
        async with st("engine").begin() as conn:
            await conn.execute(
                update(sessions)
                .where(sessions.c.id == session_id)
                .values(status=status, ended_at=func.now())
            )
        return {**jsonable(row), "status": status, "replay": replay}

    @app.get("/sessions/{session_id}/predictions")
    async def session_predictions(
        session_id: str,
        limit: int = Query(500, ge=1, le=5000),
        after_sample: int = Query(-1),
    ) -> list[dict[str, Any]]:
        await _session_row(session_id)
        p = predictions.c
        async with st("engine").connect() as conn:
            rows = await conn.execute(
                select(predictions)
                .where(p.session_id == session_id, p.window_end_sample > after_sample)
                .order_by(p.window_end_sample)
                .limit(limit)
            )
            return [jsonable(dict(r._mapping)) for r in rows]

    # ---- models ----

    @app.get("/models")
    async def models() -> dict[str, Any]:
        m = await collect(st("redis"), s.default_model)
        return {
            "available": list_models(model_dir),
            "active": m["model"]["active"],
            "workers": m["model"]["workers"],
            "errors": {
                w["worker_id"]: w["model_error"] for w in m["workers"]["list"] if w["model_error"]
            },
        }

    @app.put("/models/active")
    async def set_active_model(body: ActiveModel) -> dict[str, Any]:
        try:
            verify_model(model_dir, body.model)
        except ModelCardError as exc:
            code = 404 if "not found" in str(exc) else 422
            raise HTTPException(code, str(exc)) from exc
        redis: Redis = st("redis")
        previous = await active_model(redis, s.default_model)
        await redis.set(keys.MODEL_ACTIVE, body.model)
        async with st("engine").begin() as conn:
            await conn.execute(
                insert(model_events).values(model=body.model, previous=previous, source="api")
            )
        log.info("active model set", extra={"model": body.model, "previous": previous})
        return {"active": body.model, "previous": previous}

    # ---- metrics ----

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return await collect(st("redis"), s.default_model)

    @app.get("/metrics/history")
    async def metrics_history(minutes: float = Query(10, gt=0, le=24 * 60)) -> list[dict[str, Any]]:
        since = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=minutes)
        async with st("engine").connect() as conn:
            rows = await conn.execute(
                select(metrics_snapshots)
                .where(metrics_snapshots.c.taken_at >= since)
                .order_by(metrics_snapshots.c.taken_at)
            )
            return [jsonable(dict(r._mapping)) for r in rows]

    @app.websocket("/ws/live")
    async def live(ws: WebSocket) -> None:
        await st("hub").serve(ws)

    return app
