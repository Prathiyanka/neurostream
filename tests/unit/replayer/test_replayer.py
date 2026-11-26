import asyncio
import json
import time

import numpy as np
import pytest

from replayer.faults import FaultConfig, FaultInjector
from replayer.recording import synthetic_recording
from replayer.streamer import SessionConfig, SessionStreamer, build_packet


class FakeConnection:
    def __init__(self, log: list[dict[str, object]], fail_after: int | None = None) -> None:
        self.log = log
        self.closed = False
        self._fail_after = fail_after
        self._sent = 0

    async def send(self, message: str) -> None:
        if self.closed:
            raise OSError("closed")
        self._sent += 1
        if self._fail_after is not None and self._sent > self._fail_after:
            self.closed = True
            raise OSError("connection reset")
        self.log.append(json.loads(message))

    async def recv(self) -> str:
        await asyncio.Event().wait()
        raise AssertionError

    async def close(self) -> None:
        self.closed = True


class FakeDevice:
    """Records every message across (re)connections."""

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.connections = 0

    async def connect(self, url: str) -> FakeConnection:
        self.connections += 1
        return FakeConnection(self.messages)

    def data_seqs(self) -> list[int]:
        return [int(m["seq"]) for m in self.messages if m["type"] == "data"]  # type: ignore[call-overload]


REC = synthetic_recording(n_channels=4, seconds=5.0)  # 800 samples -> 50 packets of 16


def cfg(**kw: object) -> SessionConfig:
    base: dict[str, object] = {"url": "ws://x", "session_id": "t1", "speed": 1000.0}
    base.update(kw)
    return SessionConfig(**base)  # type: ignore[arg-type]


def test_fault_config_validation() -> None:
    with pytest.raises(ValueError):
        FaultConfig(drop_prob=1.5)
    with pytest.raises(ValueError):
        FaultConfig(jitter_ms=-1)


def test_fault_injection_is_deterministic_and_close_to_rate() -> None:
    c = FaultConfig(drop_prob=0.1, duplicate_prob=0.2, seed=42)
    a = [FaultInjector(c).fate() for _ in range(1)]
    inj1, inj2 = FaultInjector(c), FaultInjector(c)
    f1 = [inj1.fate() for _ in range(5000)]
    f2 = [inj2.fate() for _ in range(5000)]
    assert f1 == f2 and a[0] == f1[0]
    assert 0.08 < sum(f.drop for f in f1) / 5000 < 0.12
    assert not any(f.drop and f.duplicate for f in f1)  # a dropped packet can't be duplicated


def test_build_packet_slices_and_loops() -> None:
    p0 = build_packet(REC, "s", 0, 16, start_wall=100.0, send_truth=True)
    assert np.allclose(p0["samples"], REC.data[:, :16], atol=1e-3)
    assert p0["truth"]["labels"] == REC.labels[:16].tolist()
    p50 = build_packet(REC, "s", 50, 16, start_wall=100.0, send_truth=False)  # wraps to start
    assert np.allclose(p50["samples"], REC.data[:, :16], atol=1e-3)
    assert "truth" not in p50
    assert p50["timestamp"] == pytest.approx(100.0 + 50 * 16 / 160)


async def test_clean_stream_sends_every_packet_in_order() -> None:
    dev = FakeDevice()
    stats = await SessionStreamer(cfg(), REC, dev.connect).run()
    assert dev.messages[0]["type"] == "hello" and dev.messages[-1]["type"] == "end"
    assert dev.data_seqs() == list(range(50))
    assert stats.sent == stats.packets_total == 50


async def test_stream_runs_at_real_time_rate() -> None:
    dev = FakeDevice()
    start = time.perf_counter()
    await SessionStreamer(cfg(speed=10.0, max_seconds=2.0), REC, dev.connect).run()
    elapsed = time.perf_counter() - start
    # 20 packets at 10 ms each (10x speed); the last one is due at 190 ms.
    assert len(dev.data_seqs()) == 20
    assert 0.17 < elapsed < 0.6


async def test_faults_match_reported_stats() -> None:
    dev = FakeDevice()
    faults = FaultConfig(drop_prob=0.1, duplicate_prob=0.1, reorder_prob=0.1, seed=3)
    stats = await SessionStreamer(
        cfg(faults=faults, loop=True, max_seconds=60.0), REC, dev.connect
    ).run()
    seqs = dev.data_seqs()
    assert len(seqs) == stats.sent + stats.duplicated
    assert len(set(seqs)) == stats.sent == stats.packets_total - stats.dropped
    assert stats.dropped > 0 and stats.duplicated > 0 and stats.reordered > 0
    assert seqs != sorted(seqs)  # reordering actually happened


async def test_injected_disconnects_lose_packets_and_reconnect() -> None:
    dev = FakeDevice()
    faults = FaultConfig(disconnect_every_s=1.0, reconnect_delay_s=0.5)
    stats = await SessionStreamer(cfg(speed=20.0, faults=faults), REC, dev.connect).run()
    assert stats.disconnects >= 3 and dev.connections == stats.disconnects + 1
    hellos = [m for m in dev.messages if m["type"] == "hello"]
    assert len(hellos) == dev.connections
    assert stats.lost_while_disconnected > 0
    assert stats.sent + stats.lost_while_disconnected == stats.packets_total


async def test_unexpected_connection_loss_reconnects() -> None:
    messages: list[dict[str, object]] = []
    calls = 0

    async def connect(url: str) -> FakeConnection:
        nonlocal calls
        calls += 1
        return FakeConnection(messages, fail_after=10 if calls == 1 else None)

    stats = await SessionStreamer(cfg(), REC, connect).run()
    assert calls == 2 and stats.connection_errors == 1
    assert messages[-1]["type"] == "end"


async def test_gives_up_after_max_connect_attempts() -> None:
    async def refuse(url: str) -> FakeConnection:
        raise OSError("refused")

    with pytest.raises(ConnectionError):
        await SessionStreamer(cfg(max_connect_attempts=2), REC, refuse).run()


async def test_slow_initial_connect_does_not_skip_packets() -> None:
    dev = FakeDevice()

    async def slow_connect(url: str) -> FakeConnection:
        await asyncio.sleep(0.1)  # = 10 packet periods at speed 100
        return await dev.connect(url)

    stats = await SessionStreamer(cfg(speed=100.0), REC, slow_connect).run()
    assert stats.lost_while_disconnected == 0
    assert dev.data_seqs()[:3] == [0, 1, 2]
