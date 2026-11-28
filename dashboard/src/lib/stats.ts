import type { Prediction } from "./types";

/** Linear-interpolation percentile (same definition as numpy's default). */
export function percentile(values: number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * (p / 100);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const a = sorted[lo] as number;
  const b = sorted[hi] as number;
  return a + (b - a) * (pos - lo);
}

/** Mean per-class recall over windows with a known label (chance = 1/3). */
export function balancedAccuracy(preds: Prediction[]): number | null {
  const recalls: number[] = [];
  for (const c of [0, 1, 2]) {
    const ofClass = preds.filter((p) => p.true_label === c);
    if (ofClass.length) {
      recalls.push(ofClass.filter((p) => p.predicted_class === c).length / ofClass.length);
    }
  }
  return recalls.length ? recalls.reduce((a, b) => a + b, 0) / recalls.length : null;
}

export interface Point {
  t: number; // seconds (wall clock)
  v: number;
}

export interface Marker {
  t: number;
  label: string;
}

/** Rolling balanced accuracy: one point per labelled prediction, over the last `window`. */
export function rollingAccuracy(preds: Prediction[], window = 60): Point[] {
  const labelled = preds.filter((p) => p.true_label >= 0);
  const out: Point[] = [];
  for (let i = 0; i < labelled.length; i++) {
    const slice = labelled.slice(Math.max(0, i + 1 - window), i + 1);
    const acc = balancedAccuracy(slice);
    if (acc !== null && slice.length >= 10) {
      out.push({ t: (labelled[i] as Prediction).publish_ts, v: acc });
    }
  }
  return out;
}

/** p50 / p95 latency per time bucket (default 1 s), so the chart is readable at any rate. */
export function latencySeries(
  preds: Prediction[],
  bucketS = 1,
): { p50: Point[]; p95: Point[] } {
  const buckets = new Map<number, number[]>();
  for (const p of preds) {
    const k = Math.floor(p.publish_ts / bucketS) * bucketS;
    const list = buckets.get(k) ?? [];
    list.push(p.latency_ms);
    buckets.set(k, list);
  }
  const keys = [...buckets.keys()].sort((a, b) => a - b);
  return {
    p50: keys.map((k) => ({ t: k + bucketS / 2, v: percentile(buckets.get(k)!, 50)! })),
    p95: keys.map((k) => ({ t: k + bucketS / 2, v: percentile(buckets.get(k)!, 95)! })),
  };
}

/**
 * Where the served model changed, from the predictions themselves: the first prediction
 * made by a different model than the one before it. (Predictions record their model, so
 * this marks the moment a hot-swap actually took effect, not when it was requested.)
 */
export function modelChanges(preds: Prediction[]): Marker[] {
  const out: Marker[] = [];
  let previous: string | null = null;
  for (const p of [...preds].sort((a, b) => a.publish_ts - b.publish_ts)) {
    if (previous !== null && p.model !== previous) {
      out.push({ t: p.publish_ts, label: shortModel(p.model) });
    }
    previous = p.model;
  }
  return out;
}

/** "eegnet-motor-1.0.1" -> "motor 1.0.1" for compact labels. */
export function shortModel(id: string): string {
  const m = /^eegnet-(.+)-(\d+\.\d+\.\d+)$/.exec(id);
  return m ? `${m[1]} ${m[2]}` : id;
}
