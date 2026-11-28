"""Export a trained EEGNet checkpoint to ONNX with a model card.

    python -m neurostream_ml.export --channels motor --fold 0 --version 1.0.0

Output goes to ``models/<name>-<version>/{model.onnx, model_card.json}``. Each export is
checked against PyTorch with ONNX Runtime before it is written, and the model card
records the preprocessing a worker must apply, so the worker can rebuild exactly the
training-time input.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from neurostream_common.labels import CLASS_NAMES
from neurostream_ml.common import PREPROCESS, RESULTS_DIR, SPEC
from neurostream_ml.eegnet import EEGNet, WithSoftmax

MODELS_DIR = Path("models")
INPUT_NAME, OUTPUT_NAME = "input", "probabilities"


def export_onnx(model: EEGNet, path: Path, opset: int = 18) -> float:
    """Write ``path`` and return the max |prob difference| between PyTorch and ORT."""
    cfg = model.config
    wrapped = WithSoftmax(model).eval()
    example = torch.randn(4, 1, cfg["n_channels"], cfg["n_times"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # torch.export-based exporter (the TorchScript one is deprecated). verbose=False also
    # avoids its emoji progress output, which crashes on Windows' cp1252 console.
    torch.onnx.export(
        wrapped,
        (example,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=4096)}},
        opset_version=opset,
        dynamo=True,
        verbose=False,
    )
    # The exporter puts the weights in a side file (model.onnx.data). Fold them back in:
    # the model is ~10 KB, and one file means the model card's sha256 covers the weights,
    # so a swapped or corrupted weight file cannot pass the integrity check.
    side = path.with_name(path.name + ".data")
    onnx.save(onnx.load(str(path)), str(path), save_as_external_data=False)
    side.unlink(missing_ok=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = torch.randn(32, 1, cfg["n_channels"], cfg["n_times"])
    with torch.no_grad():
        ref = wrapped(x).numpy()
    got = sess.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})[0]
    return float(np.abs(ref - got).max())


def build_card(
    name: str,
    version: str,
    ckpt: dict[str, Any],
    fold_result: dict[str, Any] | None,
    onnx_path: Path,
    parity: float,
    serving_pad_s: float,
) -> dict[str, Any]:
    cfg = ckpt["config"]
    card: dict[str, Any] = {
        "name": name,
        "version": version,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "architecture": {"type": "EEGNet", **cfg},
        "input": {
            "name": INPUT_NAME,
            "shape": ["batch", 1, cfg["n_channels"], cfg["n_times"]],
            "dtype": "float32",
            "channels": ckpt["ch_names"],
            "sfreq": SPEC.sfreq,
        },
        "output": {"name": OUTPUT_NAME, "classes": list(CLASS_NAMES), "kind": "softmax"},
        "preprocessing": {
            **PREPROCESS.to_dict(),
            "channels": ckpt["ch_names"],
            "window_s": SPEC.window_s,
            "step_s": SPEC.step_s,
            "pad_s": serving_pad_s,
            "trained_with_pad_s": SPEC.pad_s,
            "normalization": "per-window: subtract channel means, divide by overall std",
        },
        "onnx": {
            "file": onnx_path.name,
            "sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "opset": 18,
            "max_abs_diff_vs_pytorch": parity,
        },
        "disclaimer": "Research/portfolio model. Not a medical device.",
    }
    if fold_result is not None:
        t = fold_result["test"]
        card["training"] = {
            "dataset": "PhysioNet EEGMMIDB, runs 4/8/12 (motor imagery)",
            "split": "cross-subject (test subjects never seen in training)",
            "fold": fold_result["fold"],
            "train_subjects": fold_result["train_subjects"],
            "val_subjects": fold_result["val_subjects"],
            "test_subjects": fold_result["test_subjects"],
            "best_epoch": fold_result["best_epoch"],
        }
        card["metrics"] = {
            k: t[k]
            for k in (
                "n",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "majority_baseline",
                "confusion",
                "subject_bal_acc_mean",
                "subject_bal_acc_ci95",
            )
        }
    return card


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--channels", choices=["all", "motor"], required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--serving-pad-s", type=float, default=SPEC.pad_s)
    args = p.parse_args(argv)
    ckpt = torch.load(
        RESULTS_DIR / "checkpoints" / f"eegnet_{args.channels}_fold{args.fold}.pt",
        weights_only=False,
    )
    model = EEGNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    name = f"eegnet-{args.channels}"
    out_dir = MODELS_DIR / f"{name}-{args.version}"
    parity = export_onnx(model, out_dir / "model.onnx")
    if parity > 1e-4:
        raise RuntimeError(f"ONNX output differs from PyTorch by {parity}")
    results = json.loads((RESULTS_DIR / f"eegnet_cross_{args.channels}.json").read_text())
    fold = next(f for f in results["folds"] if f["fold"] == args.fold)
    card = build_card(
        name, args.version, ckpt, fold, out_dir / "model.onnx", parity, args.serving_pad_s
    )
    (out_dir / "model_card.json").write_text(json.dumps(card, indent=1))
    print(f"exported {out_dir} (parity {parity:.2e})")


if __name__ == "__main__":
    main()
