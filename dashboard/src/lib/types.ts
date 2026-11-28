// Shapes of what the API sends. Kept in one place so a backend change shows up as a
// type error, not a silent `undefined` on screen.

export const CLASSES = ["rest", "left", "right"] as const;
export type ClassName = (typeof CLASSES)[number];

export interface Prediction {
  window_id: string;
  session_id: string;
  window_end_sample: number;
  window_end_ts: number; // seconds (wall clock)
  predicted_class: number;
  probabilities: number[];
  true_label: number; // -1 = unknown / mixed labels in the window
  filled_samples: number;
  model: string;
  worker_id: string;
  reclaimed: boolean;
  batch_size: number;
  ingest_ms: number;
  queue_ms: number;
  preprocess_ms: number;
  inference_ms: number;
  worker_ms: number;
  latency_ms: number;
  publish_ts: number;
}

export interface WorkerInfo {
  worker_id: string;
  model: string | null;
  model_error: string | null;
  model_loading: string | null; // "1" while a hot-swap is loading
  predictions: string | null;
  reclaimed: string | null;
  stale: string | null;
  dead_lettered: string | null;
  errors: string | null;
  last_seen: string | null; // seconds, as text (Redis hash)
}

export interface Metrics {
  ts: number;
  model: { active: string; workers: Record<string, string> };
  workers: { count: number; list: WorkerInfo[] };
  queue: { length: number; added: number; lag: number; pending: number };
  jobs: {
    acked: number;
    stale: number;
    lost: number;
    lost_trimmed_undelivered: number;
    lost_trimmed_pending: number;
    dead_lettered: number;
    reclaimed: number;
    pad_short: number;
  };
  persister: Record<string, number>;
  ingest: {
    sessions: number;
    sessions_connected: number;
    accepted: number;
    duplicates: number;
    late: number;
    gaps: number;
    missing_packets: number;
    invalid: number;
    windows_emitted: number;
    backpressure_drops: number;
  };
}

export interface Trace {
  session_id: string;
  sfreq: number;
  start_sample: number; // absolute index (160 Hz) of the first input sample
  channels: string[];
  data: number[][]; // [channel][sample], 40 Hz
}

export type LiveMessage =
  | ({ type: "prediction" } & Prediction)
  | ({ type: "metrics" } & Metrics)
  | ({ type: "trace" } & Trace);

export interface Session {
  id: string;
  subject: number | null;
  source: "api" | "external";
  status: "running" | "finished" | "stopped" | "failed";
  speed: number | null;
  created_at: string;
  ended_at: string | null;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  n_channels: number;
  serving_pad_s: number;
  test_balanced_accuracy: number | null;
}

export interface ModelsResponse {
  available: ModelInfo[];
  active: string;
  workers: Record<string, string>;
  errors: Record<string, string>;
}

export interface SessionRequest {
  subject?: number;
  synthetic?: boolean;
  speed: number;
  max_seconds?: number;
  faults: {
    drop_prob: number;
    duplicate_prob: number;
    reorder_prob: number;
    jitter_ms: number;
    disconnect_every_s: number;
  };
}
