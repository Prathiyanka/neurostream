import type { Trace } from "./types";

/**
 * Last few seconds of each channel, for the scrolling trace canvas.
 *
 * Fixed-size ring buffers of (time, value), written by WebSocket messages and read by the
 * canvas on every animation frame. Kept outside React state on purpose: re-rendering React
 * 40 times a second per channel would be wasteful, while the canvas only needs to read.
 */
export class TraceBuffer {
  readonly capacity: number;
  private times: Float64Array;
  private values = new Map<string, Float32Array>();
  private head = 0; // next write position
  private count = 0;
  sfreq = 40;
  sessionId: string | null = null;
  channels: string[] = [];

  constructor(seconds = 10, sfreq = 40) {
    this.capacity = Math.ceil(seconds * sfreq) + 1;
    this.times = new Float64Array(this.capacity);
    this.sfreq = sfreq;
  }

  reset(sessionId: string | null = null): void {
    this.head = 0;
    this.count = 0;
    this.values.clear();
    this.channels = [];
    this.sessionId = sessionId;
  }

  push(trace: Trace): void {
    if (trace.session_id !== this.sessionId || trace.channels.join() !== this.channels.join()) {
      this.reset(trace.session_id);
      this.channels = [...trace.channels];
      for (const ch of this.channels) this.values.set(ch, new Float32Array(this.capacity));
    }
    this.sfreq = trace.sfreq;
    const n = trace.data[0]?.length ?? 0;
    const t0 = trace.start_sample / 160; // seconds of recording time
    for (let j = 0; j < n; j++) {
      this.times[this.head] = t0 + j / trace.sfreq;
      trace.channels.forEach((ch, c) => {
        this.values.get(ch)![this.head] = trace.data[c]?.[j] ?? Number.NaN;
      });
      this.head = (this.head + 1) % this.capacity;
      this.count = Math.min(this.count + 1, this.capacity);
    }
  }

  get length(): number {
    return this.count;
  }

  /** Oldest-first copy of the buffered samples: times and one array per channel. */
  snapshot(): { times: number[]; series: Map<string, number[]> } {
    const start = (this.head - this.count + this.capacity) % this.capacity;
    const times: number[] = [];
    const series = new Map<string, number[]>(this.channels.map((ch) => [ch, []]));
    for (let i = 0; i < this.count; i++) {
      const k = (start + i) % this.capacity;
      times.push(this.times[k] as number);
      for (const ch of this.channels) series.get(ch)!.push(this.values.get(ch)![k] as number);
    }
    return { times, series };
  }
}

/** Symmetric display range for one channel: robust to a single spike (98th percentile). */
export function robustRange(values: number[]): number {
  const abs = values.filter(Number.isFinite).map((v) => Math.abs(v - median(values)));
  if (abs.length === 0) return 1;
  abs.sort((a, b) => a - b);
  const r = abs[Math.floor(0.98 * (abs.length - 1))] as number;
  return r > 0 ? r * 1.1 : 1;
}

export function median(values: number[]): number {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length === 0) return 0;
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? (v[mid] as number) : ((v[mid - 1] as number) + (v[mid] as number)) / 2;
}

/** True where consecutive samples are further apart than ~1.5 sample periods (a gap). */
export function isGap(prevT: number, t: number, sfreq: number): boolean {
  return t - prevT > 1.5 / sfreq;
}
