"""System metrics, collected from Redis. One function feeds GET /metrics, the live
WebSocket and the Postgres snapshots, so all three always agree."""

from __future__ import annotations

import time
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from neurostream_common import keys
from neurostream_common.keys import PERSISTER_METRICS

INGEST_FIELDS = (
    "accepted",
    "duplicates",
    "late",
    "gaps",
    "missing_packets",
    "invalid",
    "windows_emitted",
    "backpressure_drops",
)


def _ints(raw: dict[bytes, bytes]) -> dict[str, int]:
    out = {}
    for k, v in raw.items():
        try:
            out[k.decode()] = int(v)
        except ValueError:
            continue
    return out


def _text(raw: dict[bytes, bytes]) -> dict[str, str]:
    return {k.decode(): v.decode() for k, v in raw.items()}


async def live_workers(r: Redis) -> list[dict[str, str]]:
    """Workers with a live heartbeat. Entries whose heartbeat expired are dropped from the
    set, so a crashed worker disappears from the count within its TTL (5 s)."""
    ids = [w.decode() for w in await r.smembers(keys.WORKERS_SET)]  # type: ignore[misc]
    if not ids:
        return []
    pipe = r.pipeline(transaction=False)
    for w in ids:
        pipe.hgetall(keys.worker_heartbeat(w))
    beats = await pipe.execute()
    alive: list[dict[str, str]] = [_text(hb) for hb in beats if hb]
    dead = [w for w, hb in zip(ids, beats, strict=True) if not hb]
    if dead:
        await r.srem(keys.WORKERS_SET, *dead)  # type: ignore[misc]
    return sorted(alive, key=lambda h: h.get("worker_id", ""))


async def queue_state(r: Redis) -> dict[str, int]:
    try:
        info = await r.xinfo_stream(keys.WINDOWS_STREAM)
        groups = await r.xinfo_groups(keys.WINDOWS_STREAM)
    except ResponseError:  # stream not created yet
        return {"length": 0, "added": 0, "lag": 0, "pending": 0}
    g = next((g for g in groups if g["name"] == keys.WORKER_GROUP.encode()), None)
    return {
        "length": int(info["length"]),
        "added": int(info["entries-added"]),
        "lag": int((g or {}).get("lag") or 0),
        "pending": int((g or {}).get("pending") or 0),
    }


async def active_model(r: Redis, default: str) -> str:
    raw = await r.get(keys.MODEL_ACTIVE)
    return raw.decode() if raw else default


async def ingest_totals(r: Redis) -> dict[str, int]:
    """Ingest counters summed over every session ingest has seen."""
    ids = [s.decode() for s in await r.smembers(keys.SESSIONS_SET)]  # type: ignore[misc]
    totals = dict.fromkeys(INGEST_FIELDS, 0)
    connected = 0
    if ids:
        pipe = r.pipeline(transaction=False)
        for sid in ids:
            pipe.hgetall(keys.ingest_metrics(sid))
            pipe.hget(keys.session_meta(sid), "status")
        res = await pipe.execute()
        for m, status in zip(res[::2], res[1::2], strict=True):
            for k, v in _ints(m).items():
                if k in totals:
                    totals[k] += v
            connected += status == b"streaming"  # ingest's status while a device is on
    return {"sessions": len(ids), "sessions_connected": connected, **totals}


async def collect(r: Redis, default_model: str) -> dict[str, Any]:
    workers = await live_workers(r)
    jobs = _ints(await r.hgetall(keys.WORKER_METRICS))  # type: ignore[misc]
    lost = jobs.get("lost_trimmed_undelivered", 0) + jobs.get("lost_trimmed_pending", 0)
    return {
        "ts": time.time(),
        "model": {
            "active": await active_model(r, default_model),
            "workers": {w["worker_id"]: w.get("model", "") for w in workers},
        },
        "workers": {
            "count": len(workers),
            "list": [
                {
                    k: w.get(k)
                    for k in (
                        "worker_id",
                        "model",
                        "model_error",
                        "model_loading",
                        "predictions",
                        "reclaimed",
                        "stale",
                        "dead_lettered",
                        "errors",
                        "last_seen",
                    )
                }
                for w in workers
            ],
        },
        "queue": await queue_state(r),
        # Every window job ends in exactly one of: acked (predicted), stale (skipped as too
        # old), lost (trimmed by the stream cap before being acknowledged), dead-lettered,
        # or it is still pending / waiting (see the conservation identity in the README).
        "jobs": {
            "acked": jobs.get("acked", 0),
            "stale": jobs.get("stale", 0),
            "lost": lost,
            "lost_trimmed_undelivered": jobs.get("lost_trimmed_undelivered", 0),
            "lost_trimmed_pending": jobs.get("lost_trimmed_pending", 0),
            "dead_lettered": jobs.get("dead_lettered", 0),
            "reclaimed": jobs.get("reclaimed", 0),
            "pad_short": jobs.get("pad_short", 0),
        },
        "persister": _ints(await r.hgetall(PERSISTER_METRICS)),  # type: ignore[misc]
        "ingest": await ingest_totals(r),
    }
