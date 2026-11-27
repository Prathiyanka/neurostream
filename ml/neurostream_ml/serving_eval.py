"""Checks run through the exact serving code path (float32, raw window -> preprocessor -> ONNX).

1. pad sweep: accuracy of an exported model on its held-out test subjects when windows
   are cut with a shorter pad than training used (decides whether the pad can shrink).
2. latency: per-window preprocessing + ONNX inference time and window-job size, per
   channel set and pad.

    python -m neurostream_ml.serving_eval pad-sweep --model models/eegnet-motor-1.0.0
    python -m neurostream_ml.serving_eval latency
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from neurostream_common.eegmmidb import MOTOR_IMAGERY_RUNS, load_run
from neurostream_common.preprocessing import PreprocessConfig, WindowPreprocessor
from neurostream_common.windowing import WindowSpec, extract_windows
from neurostream_ml.common import DATA_DIR, RESULTS_DIR, setup_logging
from neurostream_ml.metrics import summarize

PADS = (0.25, 0.375, 0.5, 0.75, 1.0)


def session(path: Path, threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def preprocessor_from_card(card: dict[str, Any], ch_names: list[str]) -> WindowPreprocessor:
    p = card["preprocessing"]
    cfg = PreprocessConfig(
        sfreq=p["sfreq"],
        l_freq=p["l_freq"],
        h_freq=p["h_freq"],
        order=p["order"],
        init=p["init"],
        normalize=p["normalize"],
        channels=tuple(p["channels"]),
    )
    return WindowPreprocessor(cfg, ch_names)


def pad_sweep(model_dir: Path) -> dict[str, Any]:
    card = json.loads((model_dir / "model_card.json").read_text())
    sess = session(model_dir / "model.onnx", threads=4)
    test_subjects = card["training"]["test_subjects"]
    rows = []
    for pad in PADS:
        spec = WindowSpec(sfreq=160.0, window_s=2.0, step_s=0.5, pad_s=pad)
        ys, ps = [], []
        for s in test_subjects:
            for run in MOTOR_IMAGERY_RUNS:
                rec = load_run(s, run, DATA_DIR)
                prep = preprocessor_from_card(card, rec.ch_names)
                # Same grid as training: core windows aligned to the 1 s-pad grid, so every
                # pad value is evaluated on exactly the same classified samples.
                offset = round((1.0 - pad) * 160)
                wins = [
                    w
                    for w in extract_windows(rec.data[:, offset:], rec.labels[offset:], spec)
                    if w.label_pure
                ]
                if not wins:
                    continue
                X = np.stack([prep(w.data, w.pad_samples) for w in wins])[:, None]
                ps.append(sess.run(None, {"input": X})[0].argmax(1))
                ys.append(np.array([w.label_end for w in wins]))
        m = summarize(np.concatenate(ys), np.concatenate(ps))
        rows.append(
            {"pad_s": pad, **{k: m[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1")}}
        )
        print(rows[-1])
    return {
        "model": card["name"],
        "version": card["version"],
        "test_subjects": test_subjects,
        "rows": rows,
    }


def latency(models: list[Path], n: int = 2000) -> dict[str, Any]:
    rec = load_run(1, 4, DATA_DIR)
    rows = []
    for model_dir in models:
        card = json.loads((model_dir / "model_card.json").read_text())
        sess = session(model_dir / "model.onnx", threads=1)
        n_sel = len(card["preprocessing"]["channels"])
        for pad in (0.5, 1.0):
            spec = WindowSpec(sfreq=160.0, window_s=2.0, step_s=0.5, pad_s=pad)
            prep = preprocessor_from_card(card, rec.ch_names)
            wins = list(extract_windows(rec.data, rec.labels, spec))
            pre_ms, inf_ms = [], []
            for i in range(n + 50):
                w = wins[i % len(wins)]
                t0 = time.perf_counter()
                x = prep(w.data, w.pad_samples)[None, None]
                t1 = time.perf_counter()
                sess.run(None, {"input": x})
                t2 = time.perf_counter()
                if i >= 50:  # warm-up excluded
                    pre_ms.append((t1 - t0) * 1e3)
                    inf_ms.append((t2 - t1) * 1e3)
            tot = np.add(pre_ms, inf_ms)
            span = spec.span_samples
            rows.append(
                {
                    "model": card["name"],
                    "n_channels": n_sel,
                    "pad_s": pad,
                    # What ingest would write per window job if it sent only the model's channels
                    # vs. all 64 channels.
                    "job_bytes_model_channels": n_sel * span * 4,
                    "job_bytes_all_channels": 64 * span * 4,
                    "redis_kbps_per_session_model_channels": n_sel * span * 4 * 2 / 1000,
                    "preprocess_ms_p50": float(np.percentile(pre_ms, 50)),
                    "preprocess_ms_p95": float(np.percentile(pre_ms, 95)),
                    "inference_ms_p50": float(np.percentile(inf_ms, 50)),
                    "inference_ms_p95": float(np.percentile(inf_ms, 95)),
                    "total_ms_p50": float(np.percentile(tot, 50)),
                    "total_ms_p95": float(np.percentile(tot, 95)),
                    "total_ms_p99": float(np.percentile(tot, 99)),
                }
            )
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()})
    return {
        "n_windows": n,
        "threads": 1,
        "note": "single process, 1 ORT thread, this laptop CPU",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["pad-sweep", "latency"])
    p.add_argument("--model", type=Path, action="append")
    args = p.parse_args(argv)
    setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "pad-sweep":
        for m in args.model:
            res = pad_sweep(m)
            (RESULTS_DIR / f"pad_sweep_{res['model']}.json").write_text(json.dumps(res, indent=1))
    else:
        models = args.model or sorted(Path("models").glob("eegnet-*"))
        (RESULTS_DIR / "latency.json").write_text(json.dumps(latency(models), indent=1))


if __name__ == "__main__":
    main()
