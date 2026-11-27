"""Per-window signal processing, shared *verbatim* by training and the inference workers.

Pipeline for one window job (padded span, all channels):
    select channels -> causal band-pass over the whole span -> drop the pad -> normalize

* **Causal**: ``sosfilt`` only uses past samples, so a sample's filtered value never
  depends on the future. That holds in a live stream. (MNE's default ``filtfilt`` is
  zero-phase, which means non-causal, and would make training inputs differ from serving.)
* **Second-order sections** (SOS) instead of (b, a) coefficients: a 4th-order band-pass
  in (b, a) form is numerically fragile at low cutoffs. SOS is the robust form.
* **Steady-state initial conditions**: each window is filtered from scratch, so the filter
  starts "cold". Scaling ``sosfilt_zi`` by each channel's first sample makes the filter act
  as if the signal had been sitting at that value forever. That removes the large
  start-up transient caused by the EEG's DC offset, so far less padding is needed.
* **Normalization**: subtract each channel's mean, then divide the whole window by *one*
  scalar (its overall std). One scalar keeps the relative power between channels (e.g. C3
  vs. C4, the signal motor imagery lives in), unlike per-channel z-scoring, which would
  erase it. No statistics are fitted, so nothing can leak from test data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, sosfilt, sosfilt_zi

from neurostream_common.channels import channel_indices
from neurostream_common.windowing import FloatArray

EPS = 1e-6


@dataclass(frozen=True)
class PreprocessConfig:
    sfreq: float = 160.0
    l_freq: float = 8.0
    h_freq: float = 30.0
    order: int = 4  # per band edge, so the band-pass has 2 * order poles
    init: str = "steady"  # "steady" | "zero"
    normalize: bool = True
    channels: tuple[str, ...] | None = None  # None = all channels, in recording order

    @cached_property
    def sos(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            butter(
                self.order,
                [self.l_freq, self.h_freq],
                btype="bandpass",
                fs=self.sfreq,
                output="sos",
            ),
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sfreq": self.sfreq,
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "order": self.order,
            "init": self.init,
            "normalize": self.normalize,
            "channels": list(self.channels) if self.channels is not None else None,
        }


def causal_filter(
    data: npt.NDArray[np.floating], sos: npt.NDArray[np.float64], init: str = "steady"
) -> npt.NDArray[np.float64]:
    """Causally filter ``data`` (n_channels, n) along time, from a cold start."""
    x = np.asarray(data, dtype=np.float64)
    if init == "zero":
        return np.asarray(sosfilt(sos, x, axis=-1))
    if init != "steady":
        raise ValueError(f"unknown init {init!r}")
    zi = sosfilt_zi(sos)[:, None, :] * x[:, 0][None, :, None]  # (sections, channels, 2)
    y, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return np.asarray(y)


def filter_and_crop(
    span: npt.NDArray[np.floating],
    pad_samples: int,
    cfg: PreprocessConfig,
    ch_idx: Sequence[int] | None = None,
) -> FloatArray:
    """Band-pass the padded span and return only the classified core (float32)."""
    x = span if ch_idx is None else span[list(ch_idx)]
    y = causal_filter(x, cfg.sos, cfg.init)
    return y[:, pad_samples:].astype(np.float32)


def normalize_window(core: npt.NDArray[np.floating]) -> FloatArray:
    x = np.asarray(core, dtype=np.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    scale = np.float32(1.0 / (float(x.std()) + EPS))
    return np.asarray(x * scale, dtype=np.float32)


class WindowPreprocessor:
    """Bound to one recording's channel layout. Workers build one per session layout."""

    def __init__(self, cfg: PreprocessConfig, available_channels: Sequence[str]) -> None:
        self.cfg = cfg
        self.ch_idx = channel_indices(available_channels, cfg.channels)

    @property
    def n_channels(self) -> int:
        return len(self.ch_idx)

    def __call__(self, span: npt.NDArray[np.floating], pad_samples: int) -> FloatArray:
        core = filter_and_crop(span, pad_samples, self.cfg, self.ch_idx)
        out: FloatArray = normalize_window(core) if self.cfg.normalize else core
        return out
