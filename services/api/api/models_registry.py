"""Available models (``models/<name>-<version>/``) and hot-swapping the active one.

Hot-swap protocol: the API verifies the requested model (card present, ONNX hash matches),
then sets ``ns:model:active``. Each worker checks that key about once a second, loads and
warms up the new model on a background thread while still serving with the old one, and
swaps between two batches. Nothing restarts; every prediction records which model made
it, and each worker's heartbeat says what it is serving, so a rollout is observable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neurostream_common.model_card import ModelCardError, load_model_card


def model_summary(directory: Path) -> dict[str, Any] | None:
    try:
        card = load_model_card(directory, verify=False)
    except (ModelCardError, KeyError, ValueError):
        return None
    raw = card.raw
    test = raw.get("metrics") or {}  # fold-0 held-out subjects
    return {
        "id": card.model_id,
        "name": card.name,
        "version": card.version,
        "created_at": raw.get("created_at"),
        "n_channels": raw["input"]["shape"][2],
        "serving_pad_s": card.pad_s,
        "sha256": raw["onnx"]["sha256"],
        "test_balanced_accuracy": test.get("balanced_accuracy"),
    }


def list_models(model_dir: Path) -> list[dict[str, Any]]:
    if not model_dir.is_dir():
        return []
    out = [model_summary(d) for d in sorted(model_dir.iterdir()) if d.is_dir()]
    return [m for m in out if m is not None]


def verify_model(model_dir: Path, model_id: str) -> None:
    """Raise ``ModelCardError`` unless ``model_id`` exists and its hash matches."""
    if "/" in model_id or "\\" in model_id or model_id.startswith("."):
        raise ModelCardError(f"invalid model id {model_id!r}")
    load_model_card(model_dir / model_id, verify=True)
