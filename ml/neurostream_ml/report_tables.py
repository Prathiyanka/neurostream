"""Turns ml/results/*.json into markdown tables, so the docs contain only computed numbers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurostream_ml.metrics import bootstrap_ci

CLASSES = ("rest", "left", "right")


def load(results: Path, name: str) -> dict[str, Any] | None:
    p = results / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def from_confusion(conf: np.ndarray) -> dict[str, float]:
    recall = np.diag(conf) / np.maximum(conf.sum(axis=1), 1)
    precision = np.diag(conf) / np.maximum(conf.sum(axis=0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    counts = conf.sum(axis=1)
    return {
        "accuracy": float(np.trace(conf) / conf.sum()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "majority_baseline": float(counts.max() / counts.sum()),
        "n": int(conf.sum()),
    }


def pool_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the 5 cross-subject folds (each subject is tested exactly once)."""
    conf = np.sum([np.asarray(f["test"]["confusion"]) for f in folds], axis=0)
    per_subj: dict[str, float] = {}
    for f in folds:
        per_subj.update(f["test"]["per_subject_bal_acc"])
    vals = list(per_subj.values())
    return {
        **from_confusion(conf),
        "confusion": conf.tolist(),
        "per_subject_bal_acc": per_subj,
        "subject_bal_acc_mean": float(np.mean(vals)),
        "subject_bal_acc_std": float(np.std(vals)),
        "subject_bal_acc_ci95": list(bootstrap_ci(vals)),
        "subjects_above_chance": int(sum(v > 1 / 3 for v in vals)),
        "n_subjects": len(vals),
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def row(label: str, r: dict[str, Any]) -> str:
    lo, hi = r["subject_bal_acc_ci95"]
    return (
        f"| {label} | **{pct(r['balanced_accuracy'])}** | {pct(lo)} to {pct(hi)} | "
        f"{pct(r['subject_bal_acc_std'])} | {pct(r['macro_f1'])} | {pct(r['accuracy'])} | "
        f"{pct(r['majority_baseline'])} | {r['subjects_above_chance']}/{r['n_subjects']} |"
    )


HEADER = (
    "| Model / channels | Balanced acc. | 95% CI (subjects) | Subject std | Macro-F1 | "
    "Accuracy | Majority baseline | Subjects > 33.3% |\n|---|---|---|---|---|---|---|---|"
)


def confusion_table(label: str, conf: list[list[int]]) -> str:
    c = np.asarray(conf, dtype=float)
    norm = c / c.sum(axis=1, keepdims=True)
    lines = [
        f"**{label}** (rows = true, columns = predicted, row-normalized)",
        "",
        "| | " + " | ".join(CLASSES) + " |",
        "|---|---|---|---|",
    ]
    for i, name in enumerate(CLASSES):
        lines.append(f"| **{name}** | " + " | ".join(pct(v) for v in norm[i]) + " |")
    return "\n".join(lines)
