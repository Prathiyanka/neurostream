"""Per-session ingest logic. Pure (no I/O) so it is unit-testable without Redis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from neurostream_common.codec import RawPacket, WindowJob
from neurostream_common.labels import UNKNOWN
from neurostream_common.schemas import DataPacket, SessionHello, check_packet_against_hello
from neurostream_common.sequence import SeqStatus, SequenceTracker
from neurostream_common.windowing import StreamingWindower, WindowSpec


@dataclass
class IngestCounters:
    invalid: int = 0
    windows_emitted: int = 0
    windows_with_fill: int = 0
    backpressure_drops: int = 0
    redis_errors: int = 0
    connections: int = 0


@dataclass
class ProcessResult:
    status: SeqStatus | None  # None == rejected as invalid
    error: str | None = None
    raw: RawPacket | None = None
    jobs: list[WindowJob] = field(default_factory=list)


class SessionState:
    def __init__(
        self, hello: SessionHello, window_s: float, step_s: float, pad_s: float, max_fill_s: float
    ) -> None:
        self.hello = hello
        self.spec = WindowSpec(sfreq=hello.sfreq, window_s=window_s, step_s=step_s, pad_s=pad_s)
        self.tracker = SequenceTracker()
        self.windower = StreamingWindower(
            self.spec,
            n_channels=len(hello.ch_names),
            max_fill_samples=round(max_fill_s * hello.sfreq),
        )
        self.counters = IngestCounters()
        self.connected = False

    @property
    def session_id(self) -> str:
        return self.hello.session_id

    def compatible_with(self, hello: SessionHello) -> bool:
        """A reconnecting device must describe the same stream as before."""
        h = self.hello
        return (h.sfreq, h.ch_names, h.chunk_size) == (
            hello.sfreq,
            hello.ch_names,
            hello.chunk_size,
        )

    def process(self, packet: DataPacket, ingest_ts: float, now: float) -> ProcessResult:
        error = check_packet_against_hello(packet, self.hello)
        if error is not None:
            self.counters.invalid += 1
            return ProcessResult(status=None, error=error)

        seq = self.tracker.observe(packet.seq)
        if not seq.accept:
            return ProcessResult(status=seq.status)

        n = packet.n_samples
        start_sample = packet.seq * self.hello.chunk_size
        data = np.asarray(packet.samples, dtype=np.float32)
        labels = (
            np.asarray(packet.truth.labels, dtype=np.int8)
            if packet.truth is not None
            else np.full(n, UNKNOWN, dtype=np.int8)
        )
        raw = RawPacket(
            session_id=self.session_id,
            seq=packet.seq,
            start_sample=start_sample,
            timestamp=packet.timestamp,
            ingest_ts=ingest_ts,
            data=data,
            labels=labels,
        )
        windows = self.windower.push(start_sample, data, labels)
        jobs = [
            WindowJob(self.session_id, w, self.hello.sfreq, ingest_ts=ingest_ts, emitted_ts=now)
            for w in windows
        ]
        self.counters.windows_emitted += len(jobs)
        self.counters.windows_with_fill += sum(1 for w in windows if w.filled_samples)
        return ProcessResult(status=seq.status, raw=raw, jobs=jobs)

    def metrics(self) -> dict[str, int]:
        out = {**asdict(self.tracker.counters), **asdict(self.counters)}
        out["windower_resets"] = self.windower.resets
        out["filled_samples"] = self.windower.filled_total
        return out
