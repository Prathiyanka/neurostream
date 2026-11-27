"""Defaults shared by all experiments, so every result uses the same configuration."""

from __future__ import annotations

import logging
from pathlib import Path

from neurostream_common.eegmmidb import usable_subjects
from neurostream_common.preprocessing import PreprocessConfig
from neurostream_common.windowing import WindowSpec

DATA_DIR = Path("data")
RESULTS_DIR = Path("ml/results")
SPEC = WindowSpec(sfreq=160.0, window_s=2.0, step_s=0.5, pad_s=1.0)
PREPROCESS = PreprocessConfig(sfreq=160.0, l_freq=8.0, h_freq=30.0, order=4, init="steady")
SEED = 0


def subjects() -> list[int]:
    return usable_subjects(DATA_DIR)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
