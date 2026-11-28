import type { Prediction } from "../lib/types";

/** A prediction with sensible defaults, for tests. */
export function pred(over: Partial<Prediction> = {}): Prediction {
  return {
    window_id: "s:1",
    session_id: "s",
    window_end_sample: 399,
    window_end_ts: 1000,
    predicted_class: 0,
    probabilities: [0.6, 0.2, 0.2],
    true_label: 0,
    filled_samples: 0,
    model: "eegnet-motor-1.0.1",
    worker_id: "w1",
    reclaimed: false,
    batch_size: 1,
    ingest_ms: 0.2,
    queue_ms: 3,
    preprocess_ms: 1,
    inference_ms: 1,
    worker_ms: 2,
    latency_ms: 5,
    publish_ts: 1000,
    ...over,
  };
}
