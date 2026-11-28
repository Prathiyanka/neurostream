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


def paired_diff(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, float, int]:
    """Mean per-subject difference a - b with a 95% bootstrap CI, over shared subjects.

    Paired, because the same subject is hard or easy for every model: comparing two
    independent CIs would hide a consistent small difference behind subject variance.
    """
    keys = sorted(set(a) & set(b), key=int)
    d = [a[k] - b[k] for k in keys]
    lo, hi = bootstrap_ci(d)
    return float(np.mean(d)), lo, hi, len(keys)


def _cross(results: Path, name: str) -> dict[str, Any] | None:
    r = load(results, name)
    if r is None:
        return None
    if "cross_subject" in r:  # CSP+LDA: already pooled over folds
        return dict(r["cross_subject"])
    if len(r["folds"]) < 5:
        return None
    return pool_folds(r["folds"])


def build_report(results: Path) -> str:
    out: list[str] = []
    cross = {
        label: _cross(results, name)
        for label, name in [
            ("CSP+LDA, 64 ch", "csp_lda_all"),
            ("CSP+LDA, 21 motor ch", "csp_lda_motor"),
            ("EEGNet, 64 ch", "eegnet_cross_all"),
            ("EEGNet, 21 motor ch", "eegnet_cross_motor"),
        ]
    }
    out += ["## Cross-subject (5 folds, each subject tested once)", "", HEADER]
    out += [row(k, v) for k, v in cross.items() if v is not None]

    within: dict[str, dict[str, Any]] = {}
    for label, name, key in [
        ("CSP+LDA, 64 ch", "csp_lda_all", "within_subject"),
        ("CSP+LDA, 21 motor ch", "csp_lda_motor", "within_subject"),
        ("EEGNet, 21 motor ch", "eegnet_within_motor", "within_subject"),
    ]:
        r = load(results, name)
        if r is not None:
            within[label] = r[key]
    out += [
        "",
        "## Within-subject (test run 12)",
        "",
        "CSP+LDA trains on runs 4 and 8. EEGNet trains on run 4 only and uses run 8 for"
        " early stopping, so it sees half the training data: the EEGNet row is a lower bound.",
        "",
        HEADER,
    ]
    out += [row(k, v) for k, v in within.items()]

    out += ["", "## Paired per-subject differences (balanced accuracy, 95% bootstrap CI)", ""]
    out += ["| Comparison | Mean difference | 95% CI | Subjects |", "|---|---|---|---|"]
    pairs = [
        ("cross", "EEGNet, 64 ch", "EEGNet, 21 motor ch"),
        ("cross", "CSP+LDA, 64 ch", "CSP+LDA, 21 motor ch"),
        ("cross", "EEGNet, 21 motor ch", "CSP+LDA, 21 motor ch"),
        ("cross", "EEGNet, 64 ch", "CSP+LDA, 64 ch"),
        ("within", "CSP+LDA, 64 ch", "CSP+LDA, 21 motor ch"),
        ("within", "EEGNet, 21 motor ch", "CSP+LDA, 21 motor ch"),
    ]
    for split, a, b in pairs:
        src: dict[str, Any] = cross if split == "cross" else within
        if src.get(a) is None or src.get(b) is None:
            continue
        m, lo, hi, n = paired_diff(src[a]["per_subject_bal_acc"], src[b]["per_subject_bal_acc"])
        out.append(
            f"| {split}: {a} minus {b} | {100 * m:+.1f} pt | "
            f"{100 * lo:+.1f} to {100 * hi:+.1f} | {n} |"
        )

    out += ["", "## EEGNet per fold (cross-subject)", ""]
    out += [
        "| Channels | Fold | Balanced acc. | Macro-F1 | Best epoch | Val bal. acc. | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    for ch in ("all", "motor"):
        r = load(results, f"eegnet_cross_{ch}")
        for f in (r or {}).get("folds", []):
            t = f["test"]
            secs = f.get("train_seconds")
            out.append(
                f"| {ch} | {f['fold']} | {pct(t['balanced_accuracy'])} | {pct(t['macro_f1'])} | "
                f"{f['best_epoch']} | {pct(f['val_bal_acc'])} | "
                f"{'' if secs is None else f'{secs / 60:.1f} min'} |"
            )

    out += ["", "## Confusion matrices (cross-subject, pooled)", ""]
    for k, v in cross.items():
        if v is not None:
            out += [confusion_table(k, v["confusion"]), ""]

    s = load(results, "settling")
    if s is not None:
        imp = s["impulse"]
        out += [
            "## Filter settling",
            "",
            "Impulse response: amplitude < 1% of peak after "
            f"{imp['amplitude_below_0.01_of_peak_s']:.3f} s, < 0.1% after "
            f"{imp['amplitude_below_0.001_of_peak_s']:.3f} s; 99% of its energy"
            f" within {imp['energy_0.99_s']:.3f} s, 99.9% within {imp['energy_0.999_s']:.3f} s.",
            "",
            "Relative RMS error of the filtered window versus a long-settled reference "
            f"({s['empirical_steady_init'][0]['n_windows']} windows):",
            "",
            "| Pad | Init | Core median | Core p95 | First 100 ms median | First 100 ms p95 |",
            "|---|---|---|---|---|---|",
        ]
        for init in ("steady", "zero"):
            for e in s[f"empirical_{init}_init"]:
                out.append(
                    f"| {e['pad_s']:g} s | {init} | {100 * e['core_rel_rms_median']:.2f}% | "
                    f"{100 * e['core_rel_rms_p95']:.2f}% | "
                    f"{100 * e['first100ms_rel_rms_median']:.2f}% | "
                    f"{100 * e['first100ms_rel_rms_p95']:.2f}% |"
                )
        out.append("")

    for p in sorted(results.glob("pad_sweep_*.json")):
        sweep: dict[str, Any] = json.loads(p.read_text())
        out += [
            f"## Pad sweep: {sweep['model']} {sweep['version']} (exact float32 serving path, "
            f"{len(sweep['test_subjects'])} held-out subjects)",
            "",
            "| Serving pad | Windows | Balanced acc. | Macro-F1 | Accuracy |",
            "|---|---|---|---|---|",
        ]
        out += [
            f"| {x['pad_s']:g} s | {x['n']} | {pct(x['balanced_accuracy'])} | "
            f"{pct(x['macro_f1'])} | {pct(x['accuracy'])} |"
            for x in sweep["rows"]
        ]
        out.append("")

    lat = load(results, "latency")
    if lat is not None:
        out += [
            f"## Latency and data volume ({lat['note']}, {lat['n_windows']} windows)",
            "",
            "| Model | Pad | Job size (model ch.) | Job size (64 ch.) | Redis per session "
            "| Preprocess p50 | Inference p50 | Total p50 | Total p95 | Total p99 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for x in lat["rows"]:
            out.append(
                f"| {x['model']} | {x['pad_s']:g} s | {x['job_bytes_model_channels'] / 1000:.1f} KB"
                f" | {x['job_bytes_all_channels'] / 1000:.1f} KB"
                f" | {x['redis_kbps_per_session_model_channels']:.1f} KB/s"
                f" | {x['preprocess_ms_p50']:.2f} ms | {x['inference_ms_p50']:.2f} ms"
                f" | {x['total_ms_p50']:.2f} ms | {x['total_ms_p95']:.2f} ms"
                f" | {x['total_ms_p99']:.2f} ms |"
            )
        out.append("")
    return "\n".join(out)


def main() -> None:
    import sys

    from neurostream_ml.common import RESULTS_DIR

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(build_report(RESULTS_DIR))


if __name__ == "__main__":
    main()
