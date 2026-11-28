"""Encoding of window jobs and raw packets as Redis Stream fields.

Arrays travel as raw little-endian float32/int8 bytes, not JSON: a 64 x 480 window is
123 KB as bytes but several times that as JSON text, and decoding is a zero-copy view.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np

from neurostream_common.windowing import FloatArray, LabelArray, Window

Fields = dict[str, bytes | str | int | float]
RawFields = Mapping[bytes, bytes] | Mapping[str, bytes]

F32 = np.dtype("<f4")
I8 = np.dtype("i1")


@dataclass(frozen=True)
class WindowJob:
    session_id: str
    window: Window
    sfreq: float
    ingest_ts: float  # wall time the packet holding the window's last sample reached ingest
    emitted_ts: float  # wall time ingest produced the job

    @property
    def window_id(self) -> str:
        """Deterministic id, so a window reprocessed after a crash can be de-duplicated."""
        return f"{self.session_id}:{self.window.end_sample}"


class _Fields:
    """Field access that works whether redis-py returned bytes or str keys."""

    def __init__(self, fields: RawFields) -> None:
        self._f: dict[str, bytes] = {
            (k.decode() if isinstance(k, bytes) else k): v for k, v in fields.items()
        }

    def raw(self, key: str) -> bytes:
        value = self._f[key]
        return value if isinstance(value, bytes) else str(value).encode()

    def text(self, key: str) -> str:
        return self.raw(key).decode()

    def integer(self, key: str) -> int:
        return int(self.raw(key))

    def number(self, key: str) -> float:
        return float(self.raw(key))


def encode_window_job(job: WindowJob) -> Fields:
    w = job.window
    return {
        "window_id": job.window_id,
        "session_id": job.session_id,
        "start_sample": w.start_sample,
        "end_sample": w.end_sample,
        "n_channels": w.data.shape[0],
        "n_samples": w.data.shape[1],
        "pad_samples": w.pad_samples,
        "sfreq": job.sfreq,
        "data": np.ascontiguousarray(w.data, dtype=F32).tobytes(),
        "labels": np.ascontiguousarray(w.labels, dtype=I8).tobytes(),
        "filled_samples": w.filled_samples,
        "ingest_ts": job.ingest_ts,
        "emitted_ts": job.emitted_ts,
    }


def decode_window_job(fields: RawFields) -> WindowJob:
    f = _Fields(fields)
    n_ch, n = f.integer("n_channels"), f.integer("n_samples")
    data: FloatArray = np.frombuffer(f.raw("data"), dtype=F32).reshape(n_ch, n)
    labels: LabelArray = np.frombuffer(f.raw("labels"), dtype=I8)
    window = Window(
        start_sample=f.integer("start_sample"),
        data=data,
        labels=labels,
        pad_samples=f.integer("pad_samples"),
        filled_samples=f.integer("filled_samples"),
    )
    return WindowJob(
        session_id=f.text("session_id"),
        window=window,
        sfreq=f.number("sfreq"),
        ingest_ts=f.number("ingest_ts"),
        emitted_ts=f.number("emitted_ts"),
    )


@dataclass(frozen=True)
class RawPacket:
    session_id: str
    seq: int
    start_sample: int
    timestamp: float
    ingest_ts: float
    data: FloatArray  # (n_channels, n_samples)
    labels: LabelArray  # (n_samples,), UNKNOWN if the device sent no truth


def encode_raw_packet(p: RawPacket) -> Fields:
    return {
        "session_id": p.session_id,
        "seq": p.seq,
        "start_sample": p.start_sample,
        "timestamp": p.timestamp,
        "ingest_ts": p.ingest_ts,
        "n_channels": p.data.shape[0],
        "n_samples": p.data.shape[1],
        "data": np.ascontiguousarray(p.data, dtype=F32).tobytes(),
        "labels": np.ascontiguousarray(p.labels, dtype=I8).tobytes(),
    }


def decode_raw_packet(fields: RawFields) -> RawPacket:
    f = _Fields(fields)
    n_ch, n = f.integer("n_channels"), f.integer("n_samples")
    return RawPacket(
        session_id=f.text("session_id"),
        seq=f.integer("seq"),
        start_sample=f.integer("start_sample"),
        timestamp=f.number("timestamp"),
        ingest_ts=f.number("ingest_ts"),
        data=np.frombuffer(f.raw("data"), dtype=F32).reshape(n_ch, n),
        labels=np.frombuffer(f.raw("labels"), dtype=I8),
    )


@dataclass(frozen=True)
class Prediction:
    """One classified window, as published by a worker.

    Timestamps are wall-clock seconds on the shared host clock. ``window_end_ts`` is when
    the packet holding the window's last sample reached ingest; ``latency_ms`` runs from
    there to ``publish_ts`` (just before the worker's publishing round trip).

    Needs a precise wall clock: on Linux (and in the Docker containers) ``time.time()``
    resolves well under a microsecond; on a Windows host with Python 3.11 it ticks in ~1 to
    16 ms steps, so latencies measured there are coarse. All reported numbers come from
    the containers.
    """

    window_id: str
    session_id: str
    window_end_sample: int
    window_end_ts: float
    predicted_class: int
    probabilities: tuple[float, ...]
    true_label: int  # label shared by every classified sample, or UNKNOWN (-1) if mixed
    filled_samples: int  # samples in the span synthesized to cover a packet gap
    model: str
    worker_id: str
    reclaimed: bool  # processed after being reclaimed from another (dead or slow) worker
    batch_size: int  # windows in the inference call this one was part of
    ingest_ms: float  # window complete at ingest -> job produced
    queue_ms: float  # job produced -> read by the worker (ingest buffer, Redis, waiting)
    preprocess_ms: float  # this window's filter + normalization
    inference_ms: float  # the whole batched ONNX call this window was part of
    worker_ms: float  # read by the worker -> publish_ts (whole batch's preprocessing included)
    latency_ms: float  # end to end: window_end_ts -> publish_ts = ingest + queue + worker
    publish_ts: float

    def to_json(self) -> str:
        d = asdict(self)
        d["probabilities"] = list(self.probabilities)
        return json.dumps(d, separators=(",", ":"))


_PRED_INT = ("window_end_sample", "predicted_class", "true_label", "filled_samples", "batch_size")
_PRED_FLOAT = (
    "window_end_ts",
    "ingest_ms",
    "queue_ms",
    "preprocess_ms",
    "inference_ms",
    "worker_ms",
    "latency_ms",
    "publish_ts",
)


def encode_prediction(p: Prediction) -> Fields:
    d = asdict(p)
    d["probabilities"] = json.dumps(list(p.probabilities))
    d["reclaimed"] = int(p.reclaimed)
    return d


def decode_prediction(fields: RawFields) -> Prediction:
    f = _Fields(fields)
    kw: dict[str, object] = {k: f.integer(k) for k in _PRED_INT}
    kw.update({k: f.number(k) for k in _PRED_FLOAT})
    return Prediction(
        window_id=f.text("window_id"),
        session_id=f.text("session_id"),
        model=f.text("model"),
        worker_id=f.text("worker_id"),
        reclaimed=bool(f.integer("reclaimed")),
        probabilities=tuple(float(x) for x in json.loads(f.text("probabilities"))),
        **kw,  # type: ignore[arg-type]
    )
