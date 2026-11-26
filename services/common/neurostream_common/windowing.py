"""Sliding-window construction, shared by training (offline) and ingest (streaming).

Each window is self-contained: ``pad`` samples of history followed by the ``window``
samples that are actually classified. A causal filter run over the whole span has
settled by the time it reaches the core, so a worker can process any window without
per-session state. That is what lets workers be stateless and scale horizontally.

The window grid lives in *absolute sample indices* of the session: window k's span
starts at ``origin + k * step``. ``StreamingWindower`` is tested to produce exactly the
windows ``extract_windows`` produces offline, so training sees the same inputs as serving.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from neurostream_common.labels import UNKNOWN

FloatArray = npt.NDArray[np.float32]
LabelArray = npt.NDArray[np.int8]


@dataclass(frozen=True)
class WindowSpec:
    sfreq: float = 160.0
    window_s: float = 2.0
    step_s: float = 0.5
    pad_s: float = 1.0

    def __post_init__(self) -> None:
        if self.window_samples < 1 or self.step_samples < 1 or self.pad_samples < 0:
            raise ValueError(f"invalid window spec {self}")

    @property
    def window_samples(self) -> int:
        return round(self.window_s * self.sfreq)

    @property
    def step_samples(self) -> int:
        return round(self.step_s * self.sfreq)

    @property
    def pad_samples(self) -> int:
        return round(self.pad_s * self.sfreq)

    @property
    def span_samples(self) -> int:
        return self.pad_samples + self.window_samples


@dataclass(frozen=True)
class Window:
    start_sample: int  # absolute index of the first sample of the padded span
    data: FloatArray  # (n_channels, span_samples)
    labels: LabelArray  # (span_samples,)
    pad_samples: int
    filled_samples: int = 0  # samples in the span that were synthesized to cover a gap

    @property
    def end_sample(self) -> int:
        """Absolute index of the last sample: the window's identity and its timestamp."""
        return self.start_sample + int(self.data.shape[1]) - 1

    @property
    def core_labels(self) -> LabelArray:
        return self.labels[self.pad_samples :]

    @property
    def label_end(self) -> int:
        return int(self.labels[-1])

    @property
    def label_pure(self) -> bool:
        """True if every classified sample has the same, known label."""
        core = self.core_labels
        return bool(core[0] != UNKNOWN and np.all(core == core[0]))


def extract_windows(data: FloatArray, labels: LabelArray, spec: WindowSpec) -> Iterator[Window]:
    """Offline windowing of one contiguous recording (training path)."""
    if data.ndim != 2 or labels.shape != (data.shape[1],):
        raise ValueError("expected data (n_channels, n) and labels (n,)")
    span, step = spec.span_samples, spec.step_samples
    for start in range(0, data.shape[1] - span + 1, step):
        yield Window(
            start_sample=start,
            data=np.ascontiguousarray(data[:, start : start + span], dtype=np.float32),
            labels=labels[start : start + span].copy(),
            pad_samples=spec.pad_samples,
        )


class StreamingWindower:
    """Incremental windowing of one session's packets (serving path).

    Gaps up to ``max_fill_samples`` are filled by holding the last value (labels
    ``UNKNOWN``, counted in ``Window.filled_samples``), so a single lost packet doesn't
    cost seconds of windows. Longer gaps reset the buffer and restart the grid at the
    new data, because inventing a lot of signal would make predictions meaningless.
    """

    def __init__(self, spec: WindowSpec, n_channels: int, max_fill_samples: int = 40) -> None:
        self.spec = spec
        self.n_channels = n_channels
        self.max_fill_samples = max_fill_samples
        self.resets = 0
        self.filled_total = 0
        self._clear()

    def _clear(self) -> None:
        self._data: FloatArray = np.empty((self.n_channels, 0), dtype=np.float32)
        self._labels: LabelArray = np.empty(0, dtype=np.int8)
        self._filled = np.empty(0, dtype=bool)
        self._buf_start: int | None = None  # absolute index of _data[:, 0]
        self._next_start = 0  # absolute start of the next window to emit

    @property
    def next_expected_sample(self) -> int | None:
        if self._buf_start is None:
            return None
        return self._buf_start + int(self._data.shape[1])

    def push(self, start_sample: int, data: FloatArray, labels: LabelArray) -> list[Window]:
        if data.shape[0] != self.n_channels or labels.shape != (data.shape[1],):
            raise ValueError("data/labels shape mismatch")
        expected = self.next_expected_sample
        if expected is None:
            self._buf_start = self._next_start = start_sample
        elif start_sample < expected:
            raise ValueError(f"sample {start_sample} already received (expected {expected})")
        elif start_sample > expected:
            gap = start_sample - expected
            if gap <= self.max_fill_samples and self._data.shape[1] > 0:
                self._append(
                    np.repeat(self._data[:, -1:], gap, axis=1),
                    np.full(gap, UNKNOWN, dtype=np.int8),
                    filled=True,
                )
                self.filled_total += gap
            else:
                self.resets += 1
                self._clear()
                self._buf_start = self._next_start = start_sample
        self._append(data.astype(np.float32, copy=False), labels.astype(np.int8), filled=False)
        return self._emit()

    def _append(self, data: FloatArray, labels: LabelArray, *, filled: bool) -> None:
        self._data = np.concatenate([self._data, data], axis=1)
        self._labels = np.concatenate([self._labels, labels])
        self._filled = np.concatenate([self._filled, np.full(labels.shape[0], filled)])

    def _emit(self) -> list[Window]:
        assert self._buf_start is not None
        span, step = self.spec.span_samples, self.spec.step_samples
        out: list[Window] = []
        while self._next_start + span <= self._buf_start + self._data.shape[1]:
            i = self._next_start - self._buf_start
            out.append(
                Window(
                    start_sample=self._next_start,
                    data=self._data[:, i : i + span].copy(),
                    labels=self._labels[i : i + span].copy(),
                    pad_samples=self.spec.pad_samples,
                    filled_samples=int(self._filled[i : i + span].sum()),
                )
            )
            self._next_start += step
        # Drop samples no future window can use.
        drop = self._next_start - self._buf_start
        if drop > 0:
            self._data = self._data[:, drop:]
            self._labels = self._labels[drop:]
            self._filled = self._filled[drop:]
            self._buf_start = self._next_start
        return out
