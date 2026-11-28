"""Replayer -> ingest -> Redis, end to end, against a real Redis."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest
from websockets.asyncio.client import connect

from ingest.settings import Settings
from neurostream_common import keys
from neurostream_common.codec import decode_raw_packet, decode_window_job
from neurostream_common.windowing import WindowSpec, extract_windows
from replayer.faults import FaultConfig
from replayer.recording import Recording, edf_path, load_subject, synthetic_recording
from replayer.streamer import SessionConfig, SessionStreamer, StreamStats
from tests.integration.conftest import IngestServer

pytestmark = pytest.mark.integration
_S = Settings()
SPEC = WindowSpec(window_s=_S.window_s, step_s=_S.step_s, pad_s=_S.pad_s)  # what ingest uses


def n_windows(seconds: float, sfreq: float = 160.0) -> int:
    """Windows a clean recording of this length yields on ingest's grid."""
    return (round(seconds * sfreq) - SPEC.span_samples) // SPEC.step_samples + 1


async def metrics(server: IngestServer, session_id: str) -> dict[str, int]:
    raw = await server.redis.hgetall(keys.ingest_metrics(session_id))  # type: ignore[misc]
    return {k.decode(): int(v) for k, v in raw.items()}


async def windows_for(server: IngestServer, session_id: str) -> list:  # type: ignore[type-arg]
    entries = await server.redis.xrange(keys.WINDOWS_STREAM)
    jobs = [decode_window_job(f) for _, f in entries]
    return [j for j in jobs if j.session_id == session_id]


async def stream(
    server: IngestServer, rec: Recording, session_id: str, **kw: object
) -> StreamStats:
    cfg = SessionConfig(
        url=server.ws_url,
        session_id=session_id,
        speed=kw.pop("speed", 40.0),  # type: ignore[arg-type]
        max_connect_attempts=5,
        **kw,
    )  # type: ignore[arg-type]
    stats = await SessionStreamer(cfg, rec).run()
    # Ingest flushes asynchronously after `end`; it marks the session ended once flushed.
    for _ in range(200):
        status = await server.redis.hget(keys.session_meta(session_id), "status")  # type: ignore[misc]
        if status == b"ended":
            break
        await asyncio.sleep(0.02)
    else:
        raise AssertionError(f"session {session_id} never reached 'ended'")
    return stats


async def test_clean_stream_produces_exactly_the_offline_windows(
    ingest_server: IngestServer,
) -> None:
    rec = synthetic_recording(n_channels=8, seconds=10.0)
    stats = await stream(ingest_server, rec, "clean")
    jobs = await windows_for(ingest_server, "clean")
    offline = list(extract_windows(rec.data, rec.labels, SPEC))

    # Every window arrives, byte-identical (up to the 3-decimal wire rounding) to training's.
    assert len(jobs) == len(offline) == n_windows(10.0)
    for job, ref in zip(jobs, offline, strict=True):
        assert job.window.start_sample == ref.start_sample
        np.testing.assert_allclose(job.window.data, ref.data, atol=6e-4)
        np.testing.assert_array_equal(job.window.labels, ref.labels)
        assert job.ingest_ts <= job.emitted_ts

    raw = await ingest_server.redis.xrange(keys.raw_stream("clean"))
    assert len(raw) == stats.sent == 100
    assert decode_raw_packet(raw[0][1]).seq == 0

    m = await metrics(ingest_server, "clean")
    assert m["accepted"] == 100 and m["duplicates"] == m["gaps"] == m["invalid"] == 0
    assert m["windows_emitted"] == n_windows(10.0)

    meta = await ingest_server.redis.hgetall(keys.session_meta("clean"))  # type: ignore[misc]
    assert meta[b"status"] == b"ended"
    assert json.loads(meta[b"ch_names"]) == rec.ch_names


async def test_injected_faults_are_detected_exactly(ingest_server: IngestServer) -> None:
    rec = synthetic_recording(n_channels=8, seconds=30.0)
    faults = FaultConfig(drop_prob=0.05, duplicate_prob=0.05, reorder_prob=0.05, seed=7)
    stats = await stream(ingest_server, rec, "faulty", faults=faults)
    m = await metrics(ingest_server, "faulty")

    assert stats.dropped and stats.duplicated and stats.reordered
    assert m["duplicates"] == stats.duplicated
    assert m["late"] == stats.reordered
    assert m["accepted"] + m["late"] == stats.sent  # every unique packet is accounted for
    assert m["missing_packets"] == stats.dropped + stats.reordered  # reorders look like gaps first
    assert m["windows_with_fill"] > 0
    # Single lost packets are filled, so the window grid is never reset.
    assert m["windower_resets"] == 0
    assert len(await windows_for(ingest_server, "faulty")) == m["windows_emitted"]


async def test_disconnect_and_reconnect_continue_the_session(ingest_server: IngestServer) -> None:
    rec = synthetic_recording(n_channels=8, seconds=20.0)
    faults = FaultConfig(disconnect_every_s=5.0, reconnect_delay_s=1.0)
    stats = await stream(ingest_server, rec, "flaky", faults=faults, speed=10.0)
    m = await metrics(ingest_server, "flaky")

    assert stats.disconnects == 3
    assert m["connections"] == 4
    assert m["missing_packets"] == stats.lost_while_disconnected > 0
    assert m["gaps"] == stats.disconnects
    # A 1 s outage is too long to fill, so windowing restarts after each one.
    assert m["windower_resets"] == stats.disconnects


async def test_concurrent_sessions_are_isolated(ingest_server: IngestServer) -> None:
    recs = [synthetic_recording(n_channels=8, seconds=10.0, seed=i) for i in range(5)]
    await asyncio.gather(*(stream(ingest_server, r, f"multi{i}") for i, r in enumerate(recs)))
    for i in range(5):
        jobs = await windows_for(ingest_server, f"multi{i}")
        assert len(jobs) == n_windows(10.0)
        ref = next(iter(extract_windows(recs[i].data, recs[i].labels, SPEC)))
        np.testing.assert_allclose(jobs[0].window.data, ref.data, atol=6e-4)


async def test_protocol_violations(ingest_server: IngestServer) -> None:
    # Data before hello -> connection closed with the protocol-error code.
    async with connect(ingest_server.ws_url) as ws:
        await ws.send(json.dumps({"type": "end", "session_id": "x"}))
        await ws.wait_closed()
        assert ws.close_code == 4400

    hello = {
        "type": "hello",
        "session_id": "bad",
        "sfreq": 160,
        "ch_names": ["A", "B"],
        "chunk_size": 2,
    }
    async with connect(ingest_server.ws_url) as ws:
        await ws.send(json.dumps(hello))
        await ws.send("{not json")
        assert json.loads(await ws.recv())["type"] == "error"
        wrong_shape = {
            "type": "data",
            "session_id": "bad",
            "seq": 0,
            "timestamp": 0,
            "samples": [[1, 2, 3], [1, 2, 3]],
        }
        await ws.send(json.dumps(wrong_shape))
        assert "chunk_size" in json.loads(await ws.recv())["detail"]

        # A second live connection for the same session is refused.
        async with connect(ingest_server.ws_url) as ws2:
            await ws2.send(json.dumps(hello))
            await ws2.wait_closed()
            assert ws2.close_code == 4409

        await ws.send(json.dumps({"type": "end", "session_id": "bad"}))
        await ws.wait_closed()
    assert (await metrics(ingest_server, "bad"))["invalid"] == 2


DATA_DIR = Path(os.environ.get("EEG_DATA_DIR", "data"))


@pytest.mark.skipif(not edf_path(DATA_DIR, 1, 4).exists(), reason="EEGMMIDB not downloaded")
async def test_real_eeg_subject(ingest_server: IngestServer) -> None:
    rec = load_subject(1, DATA_DIR)
    assert rec.data.shape[0] == 64 and rec.sfreq == 160.0
    assert {"C3", "Cz", "C4"} <= set(rec.ch_names)
    await stream(ingest_server, rec, "real1", max_seconds=10.0, speed=40.0)
    jobs = await windows_for(ingest_server, "real1")
    assert len(jobs) == n_windows(10.0)
    assert jobs[0].window.data.shape == (64, SPEC.span_samples)
    ref = next(iter(extract_windows(rec.data, rec.labels, SPEC)))
    np.testing.assert_allclose(jobs[0].window.data, ref.data, atol=6e-4)
