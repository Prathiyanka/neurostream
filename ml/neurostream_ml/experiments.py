"""Experiment runners. Every number in docs/MODEL_RESULTS.md comes from ml/results/*.json
written here.

    python -m neurostream_ml.experiments csp                       # both channel sets
    python -m neurostream_ml.experiments eegnet-cross --channels motor [--folds 0 1 2 3 4]
    python -m neurostream_ml.experiments eegnet-within --channels motor

EEGNet runs save after every fold (cross-subject) or subject (within-subject) and resume
when re-run with the same arguments, so an interrupted run loses at most one unit of work.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from neurostream_common.channels import CHANNEL_SETS
from neurostream_ml.common import (
    DATA_DIR,
    PREPROCESS,
    RESULTS_DIR,
    SEED,
    SPEC,
    setup_logging,
    subjects,
)
from neurostream_ml.csp import CSPLDA, window_covariances
from neurostream_ml.dataset import WindowSet, build_cache, load_window_set
from neurostream_ml.eegnet import EEGNet
from neurostream_ml.metrics import bootstrap_ci, per_subject, summarize
from neurostream_ml.splits import WITHIN_TEST_RUNS, WITHIN_TRAIN_RUNS, subject_folds
from neurostream_ml.train import TrainConfig, predict_proba, train_eegnet

log = logging.getLogger("ml.experiments")


def load(channels: str) -> WindowSet:
    cache = build_cache(subjects(), DATA_DIR, SPEC, PREPROCESS)
    return load_window_set(cache, CHANNEL_SETS[channels])


def aggregate(y: np.ndarray, pred: np.ndarray, subj: np.ndarray) -> dict[str, Any]:
    ps = per_subject(y, pred, subj)
    vals = list(ps.values())
    lo, hi = bootstrap_ci(vals)
    return {
        **summarize(y, pred),
        "per_subject_bal_acc": {str(k): v for k, v in ps.items()},
        "subject_bal_acc_mean": float(np.mean(vals)),
        "subject_bal_acc_std": float(np.std(vals)),
        "subject_bal_acc_ci95": [lo, hi],
        "subjects_above_chance": int(sum(v > 1 / 3 for v in vals)),
        "n_subjects": len(vals),
    }


def save(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    log.info("wrote %s", path)
    return path


def run_csp(channels: str) -> dict[str, Any]:
    ws = load(channels)
    covs = window_covariances(ws.X)
    folds = subject_folds(subjects(), k=5, seed=SEED)
    # Cross-subject: 5 folds of held-out subjects (CSP has no early stopping; val unused).
    preds = np.empty(len(ws), dtype=np.int64)
    tested = np.zeros(len(ws), dtype=bool)
    fold_scores = []
    for f in folds:
        tr, te = ws.mask(f.train + f.val), ws.mask(f.test)
        model = CSPLDA().fit_covs(covs[tr], ws.y[tr])
        preds[te] = model.predict_covs(covs[te])
        tested |= te
        fold_scores.append(summarize(ws.y[te], preds[te]))
    cross = aggregate(ws.y[tested], preds[tested], ws.subject[tested])
    cross["per_fold"] = fold_scores
    # Within-subject: train on runs 4+8, test on run 12, one model per subject.
    yt, yp, ss = [], [], []
    for s in subjects():
        tr, te = ws.mask([s], WITHIN_TRAIN_RUNS), ws.mask([s], WITHIN_TEST_RUNS)
        if len(np.unique(ws.y[tr])) < 3 or not te.any():
            continue
        model = CSPLDA().fit_covs(covs[tr], ws.y[tr])
        yt.append(ws.y[te])
        yp.append(model.predict_covs(covs[te]))
        ss.append(ws.subject[te])
    within = aggregate(np.concatenate(yt), np.concatenate(yp), np.concatenate(ss))
    return {
        "model": "csp_lda",
        "channels": channels,
        "n_channels": len(ws.ch_names),
        "cross_subject": cross,
        "within_subject": within,
    }


def checkpoint_path(channels: str, fold: int) -> Path:
    return RESULTS_DIR / "checkpoints" / f"eegnet_{channels}_fold{fold}.pt"


def parse_history_log(path: Path) -> dict[int, list[dict[str, float]]]:
    """Per-fold epoch history from an ``eegnet-cross`` log, for checkpoints that predate
    history being stored in the checkpoint. Only folds whose final "fold N test" line was
    logged are returned (a fold cut off mid-training is not)."""
    epoch_re = re.compile(r"epoch (\d+) loss ([\d.]+) val_bal_acc ([\d.]+) \((\d+)s\)")
    fold_re = re.compile(r"fold (\d+) test")
    out: dict[int, list[dict[str, float]]] = {}
    current: list[dict[str, float]] = []
    for line in path.read_text().splitlines():
        if m := epoch_re.search(line):
            current.append(
                {
                    "epoch": int(m[1]),
                    "train_loss": float(m[2]),
                    "val_bal_acc": float(m[3]),
                    "elapsed_s": float(m[4]),
                }
            )
        elif m := fold_re.search(line):
            out[int(m[1])] = current
            current = []
    return out


def _load_partial(path: Path, header: dict[str, Any]) -> dict[str, Any]:
    """Resume from a results file written by an interrupted run, if it is compatible."""
    if not path.exists():
        return {**header, "folds": []}
    prev: dict[str, Any] = json.loads(path.read_text())
    for k, v in header.items():
        if prev.get(k) != v:
            raise SystemExit(
                f"{path} was written with {k}={prev.get(k)!r}, not {v!r}. "
                "Move it away to start over."
            )
    return prev


def _fold_record(
    i: int, f: Any, ws: WindowSet, model: torch.nn.Module, extra: dict[str, Any]
) -> dict[str, Any]:
    te = ws.mask(f.test)
    pred = predict_proba(model, ws.X, te).argmax(1)
    return {
        "fold": i,
        "train_subjects": f.train,
        "val_subjects": f.val,
        "test_subjects": f.test,
        **extra,
        "test": aggregate(ws.y[te], pred, ws.subject[te]),
    }


def _rescore_checkpoint(
    i: int, f: Any, ws: WindowSet, ckpt: dict[str, Any], log_history: list[dict[str, float]]
) -> dict[str, Any]:
    """Rebuild a fold's record from a saved model, without retraining."""
    if ckpt["ch_names"] != ws.ch_names:
        raise SystemExit(f"checkpoint for fold {i} has a different channel layout")
    model = EEGNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    va = ws.mask(f.val)
    val_bal = float(balanced_accuracy_score(ws.y[va], predict_proba(model, ws.X, va).argmax(1)))
    history = ckpt.get("history") or log_history
    if not history:
        raise SystemExit(f"no training history for fold {i}: pass --history-log")
    best = max(history, key=lambda h: h["val_bal_acc"])  # first max, like the trainer
    if abs(best["val_bal_acc"] - val_bal) > 1e-3:  # the log rounds to 4 decimals
        raise SystemExit(
            f"fold {i}: checkpoint scores {val_bal:.4f} on validation but the log's best "
            f"epoch says {best['val_bal_acc']:.4f}; checkpoint and log do not match"
        )
    log.info("fold %d rebuilt from checkpoint (val bal acc %.4f matches log)", i, val_bal)
    return _fold_record(
        i,
        f,
        ws,
        model,
        {
            "best_epoch": int(best["epoch"]),
            "val_bal_acc": val_bal,
            "train_seconds": history[-1].get("elapsed_s"),
            "history": history,
            "rescored_from_checkpoint": True,
        },
    )


def run_eegnet_cross(
    channels: str, fold_ids: list[int], cfg: TrainConfig, history_log: Path | None = None
) -> Path:
    """Train (or resume) cross-subject folds, saving the results file after every fold.

    A fold is skipped if it is already in the results file. If its checkpoint exists but
    its results do not (the run was killed after saving the model), it is re-scored from
    the checkpoint instead of retrained.
    """
    ws = load(channels)
    folds = subject_folds(subjects(), k=5, seed=SEED)
    name = f"eegnet_cross_{channels}"
    path = RESULTS_DIR / f"{name}.json"
    header = {
        "model": "eegnet",
        "channels": channels,
        "n_channels": len(ws.ch_names),
        "train_config": cfg.__dict__,
    }
    results = _load_partial(path, header)
    log_hist = parse_history_log(history_log) if history_log else {}
    for i in fold_ids:
        if any(r["fold"] == i for r in results["folds"]):
            log.info("fold %d already in %s, skipping", i, path)
            continue
        f = folds[i]
        ckpt_file = checkpoint_path(channels, i)
        if ckpt_file.exists():
            ckpt = torch.load(ckpt_file, weights_only=True)
            rec = _rescore_checkpoint(i, f, ws, ckpt, log_hist.get(i, []))
        else:
            resume = ckpt_file.with_suffix(".resume.pt")
            res = train_eegnet(ws.X, ws.y, ws.mask(f.train), ws.mask(f.val), cfg, resume)
            extra = {
                "best_epoch": res.best_epoch,
                "val_bal_acc": res.best_val_bal_acc,
                "train_seconds": res.seconds,
                "history": res.history,
            }
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": res.model.state_dict(),
                    "config": res.model.config,
                    "ch_names": ws.ch_names,
                    "fold": i,
                    "train_config": cfg.__dict__,
                    "history": res.history,
                },
                ckpt_file,
            )
            resume.unlink(missing_ok=True)  # final model saved: per-epoch state not needed
            rec = _fold_record(i, f, ws, res.model, extra)
        results["folds"] = sorted([*results["folds"], rec], key=lambda r: r["fold"])
        save(name, results)  # after every fold: a kill loses at most the fold in progress
        log.info(
            "fold %d test %s",
            i,
            {k: round(rec["test"][k], 4) for k in ("accuracy", "balanced_accuracy", "macro_f1")},
        )
    return path


def run_eegnet_within(channels: str, cfg: TrainConfig) -> Path:
    """One EEGNet per subject; validation = run 8 for early stopping, then test on run 12.

    Per-subject predictions are saved after every subject (to a gitignored partial file),
    so an interrupted run resumes where it stopped. The aggregate is written at the end.
    """
    ws = load(channels)
    name = f"eegnet_within_{channels}"
    partial = RESULTS_DIR / "checkpoints" / f"{name}.partial.json"
    header = {
        "model": "eegnet",
        "channels": channels,
        "n_channels": len(ws.ch_names),
        "train_config": cfg.__dict__,
    }
    state = _load_partial(partial, header)
    done: dict[str, Any] = state.setdefault("subjects", {})
    state.pop("folds", None)
    start = time.perf_counter()
    for s in subjects():
        if str(s) in done:
            continue
        tr, va, te = ws.mask([s], (4,)), ws.mask([s], (8,)), ws.mask([s], WITHIN_TEST_RUNS)
        if len(np.unique(ws.y[tr])) < 3 or not te.any() or not va.any():
            done[str(s)] = None  # recorded as skipped, so it is not retried
            continue
        res = train_eegnet(ws.X, ws.y, tr, va, cfg)
        done[str(s)] = {
            "y": ws.y[te].tolist(),
            "pred": predict_proba(res.model, ws.X, te).argmax(1).tolist(),
            "best_epoch": res.best_epoch,
        }
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(state))
        log.info("subject %d done (%.0fs this session)", s, time.perf_counter() - start)
    kept = {int(k): v for k, v in done.items() if v is not None}
    yt = np.concatenate([np.asarray(v["y"]) for v in kept.values()])
    yp = np.concatenate([np.asarray(v["pred"]) for v in kept.values()])
    ss = np.concatenate([np.full(len(v["y"]), k) for k, v in kept.items()])
    return save(
        name,
        {
            **header,
            "note": "train run 4, early-stop on run 8, test run 12",
            "skipped_subjects": sorted(int(k) for k, v in done.items() if v is None),
            "best_epochs": {str(k): v["best_epoch"] for k, v in kept.items()},
            "within_subject": aggregate(yt, yp, ss),
        },
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["csp", "eegnet-cross", "eegnet-within"])
    p.add_argument("--channels", choices=list(CHANNEL_SETS), default="motor")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--threads", type=int, default=TrainConfig.threads)
    p.add_argument(
        "--history-log",
        type=Path,
        help="log of an earlier eegnet-cross run, for checkpoints saved without history",
    )
    args = p.parse_args(argv)
    setup_logging()
    cfg = TrainConfig(epochs=args.epochs, patience=args.patience, threads=args.threads)
    if args.command == "csp":
        for ch in CHANNEL_SETS:
            save(f"csp_lda_{ch}", run_csp(ch))
    elif args.command == "eegnet-cross":
        run_eegnet_cross(args.channels, args.folds, cfg, args.history_log)
    else:
        run_eegnet_within(args.channels, cfg)


if __name__ == "__main__":
    main()
