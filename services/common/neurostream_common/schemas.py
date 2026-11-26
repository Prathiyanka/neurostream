"""Device -> ingest wire protocol (JSON over WebSocket), validated with Pydantic.

A connection sends one ``hello``, then any number of ``data`` packets, then optionally
``end``. Shape checks that depend on the hello (channel count, chunk size) are done by
:func:`check_packet_against_hello`, because a single packet can't know them.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

PROTOCOL_VERSION = 1
SessionId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionHello(_Strict):
    type: Literal["hello"] = "hello"
    protocol_version: Literal[1] = 1
    session_id: SessionId
    sfreq: Annotated[float, Field(gt=0, le=10_000)]
    ch_names: Annotated[list[str], Field(min_length=1, max_length=512)]
    chunk_size: Annotated[int, Field(ge=1, le=10_000)]
    units: Literal["uV"] = "uV"
    subject: int | None = None  # metadata only (which recording is being replayed)

    @model_validator(mode="after")
    def _unique_channels(self) -> SessionHello:
        if len(set(self.ch_names)) != len(self.ch_names):
            raise ValueError("ch_names must be unique")
        return self


class Truth(_Strict):
    """Ground truth, deliberately outside ``samples`` so it can never reach the model."""

    labels: list[int]


class DataPacket(_Strict):
    type: Literal["data"] = "data"
    session_id: SessionId
    seq: Annotated[int, Field(ge=0)]
    timestamp: FiniteFloat  # device time of the first sample, seconds since epoch
    samples: list[list[FiniteFloat]]  # channels x n_samples, microvolts
    truth: Truth | None = None

    @model_validator(mode="after")
    def _rectangular(self) -> DataPacket:
        if not self.samples:
            raise ValueError("samples must not be empty")
        n = len(self.samples[0])
        if n == 0 or any(len(ch) != n for ch in self.samples):
            raise ValueError("samples must be a non-empty channels x n rectangle")
        if self.truth is not None and len(self.truth.labels) != n:
            raise ValueError("truth.labels must have one label per sample")
        return self

    @property
    def n_samples(self) -> int:
        return len(self.samples[0])


class SessionEnd(_Strict):
    type: Literal["end"] = "end"
    session_id: SessionId


DeviceMessage = Annotated[SessionHello | DataPacket | SessionEnd, Field(discriminator="type")]
device_message_adapter: TypeAdapter[SessionHello | DataPacket | SessionEnd] = TypeAdapter(
    DeviceMessage
)


def parse_device_message(raw: str | bytes) -> SessionHello | DataPacket | SessionEnd:
    """Parse and validate one JSON message. Raises ``pydantic.ValidationError``."""
    return device_message_adapter.validate_json(raw)


def check_packet_against_hello(packet: DataPacket, hello: SessionHello) -> str | None:
    """Return an error message if the packet doesn't match its session's hello."""
    if packet.session_id != hello.session_id:
        return f"session_id {packet.session_id!r} != hello {hello.session_id!r}"
    if len(packet.samples) != len(hello.ch_names):
        return f"{len(packet.samples)} channels, hello declared {len(hello.ch_names)}"
    if packet.n_samples != hello.chunk_size:
        return f"{packet.n_samples} samples, hello declared chunk_size {hello.chunk_size}"
    return None
