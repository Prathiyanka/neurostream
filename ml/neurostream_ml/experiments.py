"""Experiment runners. Every number in docs/MODEL_RESULTS.md comes from ml/results/*.json
written here.

    python -m neurostream_ml.experiments csp                       # both channel sets
    python -m neurostream_ml.experiments eegnet-cross --channels motor [--folds 0 1 2 3 4]
    python -m neurostream_ml.experiments eegnet-within --channels motor
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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


def run_eegnet_cross(channels: str, fold_ids: list[int], cfg: TrainConfig) -> dict[str, Any]:
    ws = load(channels)
    folds = subject_folds(subjects(), k=5, seed=SEED)
    out: list[dict[str, Any]] = []
    for i in fold_ids:
        f = folds[i]
        tr, va, te = ws.mask(f.train), ws.mask(f.val), ws.mask(f.test)
        res = train_eegnet(ws.X[tr], ws.y[tr], ws.X[va], ws.y[va], cfg)
        proba = predict_proba(res.model, ws.X[te])
        pred = proba.argmax(1)
        out.append(
            {
                "fold": i,
                "train_subjects": f.train,
                "val_subjects": f.val,
                "test_subjects": f.test,
                "best_epoch": res.best_epoch,
                "val_bal_acc": res.best_val_bal_acc,
                "train_seconds": res.seconds,
                "history": res.history,
                "test": aggregate(ws.y[te], pred, ws.subject[te]),
            }
        )
        ckpt = RESULTS_DIR / "checkpoints" / f"eegnet_{channels}_fold{i}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": res.model.state_dict(),
                "config": res.model.config,
                "ch_names": ws.ch_names,
                "fold": i,
            },
            ckpt,
        )
        log.info(
            "fold %d test %s",
            i,
            {
                k: round(out[-1]["test"][k], 4)
                for k in ("accuracy", "balanced_accuracy", "macro_f1")
            },
        )
    return {
        "model": "eegnet",
        "channels": channels,
        "n_channels": len(ws.ch_names),
        "train_config": cfg.__dict__,
        "folds": out,
    }


def run_eegnet_within(channels: str, cfg: TrainConfig) -> dict[str, Any]:
    """One EEGNet per subject; validation = run 8 for early stopping, then test on run 12."""
    ws = load(channels)
    yt, yp, ss = [], [], []
    start = time.perf_counter()
    for s in subjects():
        tr, va, te = ws.mask([s], (4,)), ws.mask([s], (8,)), ws.mask([s], WITHIN_TEST_RUNS)
        if len(np.unique(ws.y[tr])) < 3 or not te.any() or not va.any():
            continue
        res = train_eegnet(ws.X[tr], ws.y[tr], ws.X[va], ws.y[va], cfg)
        yt.append(ws.y[te])
        yp.append(predict_proba(res.model, ws.X[te]).argmax(1))
        ss.append(ws.subject[te])
        log.info("subject %d done (%.0fs total)", s, time.perf_counter() - start)
    return {
        "model": "eegnet",
        "channels": channels,
        "n_channels": len(ws.ch_names),
        "train_config": cfg.__dict__,
        "note": "train run 4, early-stop on run 8, test run 12",
        "within_subject": aggregate(np.concatenate(yt), np.concatenate(yp), np.concatenate(ss)),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["csp", "eegnet-cross", "eegnet-within"])
    p.add_argument("--channels", choices=list(CHANNEL_SETS), default="motor")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--patience", type=int, default=TrainConfig.patience)
    p.add_argument("--threads", type=int, default=TrainConfig.threads)
    args = p.parse_args(argv)
    setup_logging()
    cfg = TrainConfig(epochs=args.epochs, patience=args.patience, threads=args.threads)
    if args.command == "csp":
        for ch in CHANNEL_SETS:
            save(f"csp_lda_{ch}", run_csp(ch))
    elif args.command == "eegnet-cross":
        save(f"eegnet_cross_{args.channels}", run_eegnet_cross(args.channels, args.folds, cfg))
    else:
        save(f"eegnet_within_{args.channels}", run_eegnet_within(args.channels, cfg))


if __name__ == "__main__":
    main()
