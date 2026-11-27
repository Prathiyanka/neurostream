"""How much padding does the causal filter need before the classified core is accurate?

Two measurements:
1. Impulse response: how long until the filter's response to a click has died away
   (theory, signal-independent).
2. Empirical, on real EEG: for each pad length P, filter each window's padded span from
   a cold start (exactly as a worker does) and compare its core with the same samples
   filtered with the *whole run* as history (the 'infinitely padded' reference).
   Reported as relative RMS error over the core and over the core's first 100 ms.
   That's where any leftover transient sits.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np

from neurostream_common.eegmmidb import load_run
from neurostream_common.preprocessing import PreprocessConfig, causal_filter
from neurostream_ml.common import DATA_DIR, PREPROCESS, RESULTS_DIR, SPEC, setup_logging, subjects

PADS_S = (0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0)


def impulse_settling(cfg: PreprocessConfig, n: int = 1600) -> dict[str, float]:
    x = np.zeros((1, n))
    x[0, 0] = 1.0
    h = causal_filter(x, cfg.sos, init="zero")[0]
    peak = np.abs(h).max()
    energy = np.cumsum(h**2) / np.sum(h**2)
    out: dict[str, float] = {}
    for frac in (0.01, 0.001):
        above = np.nonzero(np.abs(h) > frac * peak)[0]
        out[f"amplitude_below_{frac:g}_of_peak_s"] = float((above[-1] + 1) / cfg.sfreq)
    for frac in (0.99, 0.999):
        out[f"energy_{frac:g}_s"] = float((np.searchsorted(energy, frac) + 1) / cfg.sfreq)
    return out


def empirical_error(
    cfg: PreprocessConfig, n_subjects: int = 20, max_windows_per_run: int = 60
) -> list[dict[str, Any]]:
    subs = subjects()[:: max(1, len(subjects()) // n_subjects)][:n_subjects]
    win, step = SPEC.window_samples, SPEC.step_samples
    first = round(0.1 * cfg.sfreq)
    rows = []
    errs: dict[float, dict[str, list[float]]] = {p: {"core": [], "first100ms": []} for p in PADS_S}
    for s in subs:
        rec = load_run(s, 4, DATA_DIR)
        ref = causal_filter(rec.data, cfg.sos, cfg.init)  # whole-run history
        max_pad = round(max(PADS_S) * cfg.sfreq)
        starts = range(max_pad, rec.n_samples - win, step)[:max_windows_per_run]
        for p_s in PADS_S:
            pad = round(p_s * cfg.sfreq)
            for c0 in starts:  # c0 = first sample of the core
                y = causal_filter(rec.data[:, c0 - pad : c0 + win], cfg.sos, cfg.init)[:, pad:]
                r = ref[:, c0 : c0 + win]
                errs[p_s]["core"].append(float(np.linalg.norm(y - r) / np.linalg.norm(r)))
                errs[p_s]["first100ms"].append(
                    float(
                        np.linalg.norm(y[:, :first] - r[:, :first]) / np.linalg.norm(r[:, :first])
                    )
                )
    for p_s in PADS_S:
        core, head = np.asarray(errs[p_s]["core"]), np.asarray(errs[p_s]["first100ms"])
        rows.append(
            {
                "pad_s": p_s,
                "n_windows": int(core.size),
                "core_rel_rms_median": float(np.median(core)),
                "core_rel_rms_p95": float(np.percentile(core, 95)),
                "first100ms_rel_rms_median": float(np.median(head)),
                "first100ms_rel_rms_p95": float(np.percentile(head, 95)),
            }
        )
    return rows


def main() -> None:
    setup_logging()
    result: dict[str, Any] = {
        "filter": PREPROCESS.to_dict(),
        "impulse": impulse_settling(PREPROCESS),
        "empirical_steady_init": empirical_error(PREPROCESS),
        "empirical_zero_init": empirical_error(dataclasses.replace(PREPROCESS, init="zero")),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "settling.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["impulse"], indent=1))
    for key in ("empirical_steady_init", "empirical_zero_init"):
        print(key)
        for r in result[key]:
            print(
                f"  pad {r['pad_s']:.4f}s"
                f" core med {r['core_rel_rms_median']:.4f} p95 {r['core_rel_rms_p95']:.4f}"
                f" | first100ms med {r['first100ms_rel_rms_median']:.4f}"
                f" p95 {r['first100ms_rel_rms_p95']:.4f}"
            )


if __name__ == "__main__":
    main()
