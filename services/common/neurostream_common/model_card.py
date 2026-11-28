"""Loading an exported model directory: ``model.onnx`` + ``model_card.json``.

The model card is the train/serve contract: it records the channels, filter, pad and
normalization the model was trained with, and the SHA-256 of ``model.onnx``. Training
writes it (``neurostream_ml.export``); workers rebuild their input from it, so the two
sides cannot silently disagree. No ONNX Runtime import here: this module only reads files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurostream_common.preprocessing import PreprocessConfig


class ModelCardError(Exception):
    """The model directory is missing a file or fails its integrity check."""


@dataclass(frozen=True)
class ModelCard:
    directory: Path
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def version(self) -> str:
        return str(self.raw["version"])

    @property
    def model_id(self) -> str:
        return f"{self.name}-{self.version}"

    @property
    def onnx_path(self) -> Path:
        return self.directory / str(self.raw["onnx"]["file"])

    @property
    def sfreq(self) -> float:
        return float(self.raw["preprocessing"]["sfreq"])

    @property
    def pad_s(self) -> float:
        """Pad the model should be served with (at least this much signal before a window)."""
        return float(self.raw["preprocessing"]["pad_s"])

    @property
    def classes(self) -> list[str]:
        return list(self.raw["output"]["classes"])

    @property
    def input_name(self) -> str:
        return str(self.raw["input"]["name"])

    @property
    def output_name(self) -> str:
        return str(self.raw["output"]["name"])

    @property
    def n_times(self) -> int:
        return int(self.raw["input"]["shape"][3])

    def preprocess_config(self) -> PreprocessConfig:
        p = self.raw["preprocessing"]
        return PreprocessConfig(
            sfreq=float(p["sfreq"]),
            l_freq=float(p["l_freq"]),
            h_freq=float(p["h_freq"]),
            order=int(p["order"]),
            init=str(p["init"]),
            normalize=bool(p["normalize"]),
            channels=tuple(p["channels"]),
        )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model_card(directory: Path, verify: bool = True) -> ModelCard:
    """Read ``directory/model_card.json``; with ``verify``, check the ONNX file's hash."""
    card_path = directory / "model_card.json"
    if not card_path.is_file():
        raise ModelCardError(f"{card_path} not found")
    card = ModelCard(directory, json.loads(card_path.read_text(encoding="utf-8")))
    if not card.onnx_path.is_file():
        raise ModelCardError(f"{card.onnx_path} not found")
    if verify:
        expected = str(card.raw["onnx"]["sha256"])
        actual = sha256_file(card.onnx_path)
        if actual != expected:
            raise ModelCardError(
                f"{card.onnx_path} sha256 {actual[:12]}... does not match the model card "
                f"({expected[:12]}...): refusing to load a modified or corrupted model"
            )
    return card
