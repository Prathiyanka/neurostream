import { type FormEvent, useState } from "react";

import { api, ApiError } from "../lib/api";
import type { Session, SessionRequest } from "../lib/types";

// A mild but visible fault mix: the counters move, predictions keep flowing.
export const FAULT_PRESET = {
  drop_prob: 0.02,
  duplicate_prob: 0.02,
  reorder_prob: 0.01,
  jitter_ms: 20,
  disconnect_every_s: 0,
};
const NO_FAULTS = { drop_prob: 0, duplicate_prob: 0, reorder_prob: 0, jitter_ms: 0, disconnect_every_s: 0 };
const SPEEDS = [1, 2, 5, 10];

export function buildRequest(form: {
  source: "subject" | "synthetic";
  subject: number;
  speed: number;
  minutes: number | null;
  faults: boolean;
}): SessionRequest {
  return {
    ...(form.source === "synthetic" ? { synthetic: true } : { subject: form.subject }),
    speed: form.speed,
    ...(form.minutes ? { max_seconds: form.minutes * 60 } : {}),
    faults: form.faults ? FAULT_PRESET : NO_FAULTS,
  };
}

export function SessionControls({ onStarted }: { onStarted: (s: Session) => void }) {
  const [source, setSource] = useState<"subject" | "synthetic">("subject");
  const [subject, setSubject] = useState(1);
  const [speed, setSpeed] = useState(1);
  const [minutes, setMinutes] = useState<number | null>(5);
  const [faults, setFaults] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onStarted(await api.startSession(buildRequest({ source, subject, speed, minutes, faults })));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reach the API");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="panel" onSubmit={submit} aria-label="Start a session">
      <div className="panel-head">
        <h2>New session</h2>
      </div>
      <div className="field">
        <span className="label">Source</span>
        <div className="segmented" role="group" aria-label="Source">
          <button type="button" aria-pressed={source === "subject"} onClick={() => setSource("subject")}>
            Recorded EEG
          </button>
          <button type="button" aria-pressed={source === "synthetic"} onClick={() => setSource("synthetic")}>
            Synthetic
          </button>
        </div>
      </div>
      {source === "subject" && (
        <div className="field">
          <label htmlFor="subject">Subject (1–109)</label>
          <input
            id="subject"
            className="input"
            type="number"
            min={1}
            max={109}
            value={subject}
            onChange={(e) => setSubject(Number(e.target.value))}
          />
          <span className="hint">Fold-0 held-out subjects (never trained on): 1, 5, 11, 16, 17, …</span>
        </div>
      )}
      <div className="inline">
        <div className="field" style={{ flex: 1 }}>
          <label htmlFor="speed">Speed</label>
          <select id="speed" className="select" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}× {s === 1 ? "(real time)" : ""}
              </option>
            ))}
          </select>
        </div>
        <div className="field" style={{ flex: 1 }}>
          <label htmlFor="minutes">Duration</label>
          <select
            id="minutes"
            className="select"
            value={minutes ?? 0}
            onChange={(e) => setMinutes(Number(e.target.value) || null)}
          >
            <option value={1}>1 min</option>
            <option value={5}>5 min</option>
            <option value={0}>Full recording</option>
          </select>
        </div>
      </div>
      <label className="check">
        <input type="checkbox" checked={faults} onChange={(e) => setFaults(e.target.checked)} />
        <span>
          Fault injection
          <span className="hint" style={{ display: "block" }}>
            2% drop · 2% dup · 1% reorder · 20 ms jitter
          </span>
        </span>
      </label>
      <button className="btn primary" type="submit" disabled={busy}>
        {busy ? "Starting…" : "Start session"}
      </button>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </form>
  );
}
