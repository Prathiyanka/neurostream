"""The served model: ONNX Runtime session + the preprocessing its model card prescribes."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from neurostream_common.codec import WindowJob
from neurostream_common.model_card import ModelCard
from neurostream_common.preprocessing import WindowPreprocessor


class JobError(Exception):
    """A job this model can never process (wrong rate, shape or channels). Retrying is
    pointless, so such jobs are dead-lettered at once instead of being retried."""


@dataclass(frozen=True)
class Prepared:
    x: npt.NDArray[np.float32]  # (n_channels, n_times), model input for one window
    preprocess_ms: float
    pad_short: bool  # the job carried less settling pad than the model card asks for


class InferenceModel:
    def __init__(self, card: ModelCard, threads: int = 1) -> None:
        self.card = card
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(card.onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self._cfg = card.preprocess_config()
        self._pad_samples = round(card.pad_s * card.sfreq)
        self._preps: dict[tuple[str, ...], WindowPreprocessor] = {}

    @property
    def model_id(self) -> str:
        return self.card.model_id

    def _preprocessor(self, ch_names: Sequence[str]) -> WindowPreprocessor:
        key = tuple(ch_names)
        prep = self._preps.get(key)
        if prep is None:
            try:
                prep = WindowPreprocessor(self._cfg, key)
            except (KeyError, ValueError) as exc:
                raise JobError(f"session layout lacks the model's channels: {exc}") from exc
            self._preps[key] = prep
        return prep

    def prepare(self, job: WindowJob, ch_names: Sequence[str]) -> Prepared:
        w = job.window
        if abs(job.sfreq - self.card.sfreq) > 1e-6:
            raise JobError(f"sampling rate {job.sfreq} Hz, model expects {self.card.sfreq} Hz")
        if w.data.shape[0] != len(ch_names):
            raise JobError(f"job has {w.data.shape[0]} channels, session has {len(ch_names)}")
        core = w.data.shape[1] - w.pad_samples
        if core != self.card.n_times:
            raise JobError(f"window core is {core} samples, model expects {self.card.n_times}")
        t0 = time.perf_counter()
        x = self._preprocessor(ch_names)(w.data, w.pad_samples)
        ms = (time.perf_counter() - t0) * 1000
        return Prepared(np.asarray(x, dtype=np.float32), ms, w.pad_samples < self._pad_samples)

    def warm_up(self, batch_max: int, rounds: int = 3) -> float:
        """Run the model on dummy input for every batch size up to ``batch_max``.

        Measured: without this, a freshly started worker's first seconds of real traffic
        had inference calls of 17 to 100 ms (vs. ~2 ms when warm), as ONNX Runtime grows
        its memory arena and caches are cold. Returns the seconds spent."""
        t0 = time.perf_counter()
        n_ch = int(self.card.raw["input"]["shape"][2])
        x = np.zeros((n_ch, self.card.n_times), np.float32)
        for _ in range(rounds):
            for b in range(1, batch_max + 1):
                self.infer([x] * b)
        return time.perf_counter() - t0

    def infer(self, xs: Sequence[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        """Class probabilities for a batch of prepared windows, in one ONNX call."""
        batch = np.stack(xs)[:, None, :, :]  # (batch, 1, channels, times)
        out = self.session.run([self.card.output_name], {self.card.input_name: batch})[0]
        return np.asarray(out, dtype=np.float32)
