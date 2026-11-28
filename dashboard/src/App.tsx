import { useCallback, useEffect, useMemo, useState } from "react";

import { ConnectionBadge } from "./components/ConnectionBadge";
import { LineChart } from "./components/LineChart";
import { MetricTiles } from "./components/MetricTiles";
import { ModelSwitcher } from "./components/ModelSwitcher";
import { PredictionPanel } from "./components/PredictionPanel";
import { SessionControls } from "./components/SessionControls";
import { SessionList } from "./components/SessionList";
import { TraceCanvas } from "./components/TraceCanvas";
import { WorkerPanel } from "./components/WorkerPanel";
import { useLive } from "./hooks/useLive";
import { api } from "./lib/api";
import { DEFAULT_CHANNELS } from "./lib/channels";
import { latencySeries, modelChanges, rollingAccuracy, shortModel } from "./lib/stats";
import type { ModelsResponse, Session } from "./lib/types";

const POLL_MS = 3000;

export function App() {
  const live = useLive(DEFAULT_CHANNELS);
  const [channels, setChannels] = useState(DEFAULT_CHANNELS);
  const [watched, setWatched] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [models, setModels] = useState<ModelsResponse | null>(null);

  const refresh = useCallback(async () => {
    const [s, m] = await Promise.allSettled([api.sessions(), api.models()]);
    if (s.status === "fulfilled") setSessions(s.value);
    if (m.status === "fulfilled") setModels(m.value);
  }, []);

  useEffect(() => {
    void refresh();
    const id = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  // Follow the newest running session until the user picks one.
  useEffect(() => {
    if (watched === null) {
      const running = sessions.find((s) => s.status === "running");
      if (running) setWatched(running.id);
    }
  }, [sessions, watched]);

  const { watch } = live;
  useEffect(() => watch(watched, channels), [watch, watched, channels]);

  const preds = live.predictions;
  const markers = useMemo(() => modelChanges(preds), [preds]);
  const accuracy = useMemo(() => rollingAccuracy(preds), [preds]);
  const latency = useMemo(() => latencySeries(preds), [preds]);
  const latest = preds[preds.length - 1] ?? null;
  const activeModel = live.metrics?.model.active ?? models?.active ?? null;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>NeuroStream</h1>
          <span className="tagline">Real-time EEG motor-imagery decoding · research demo, not a medical device</span>
        </div>
        <div className="spacer" />
        {activeModel && (
          <span className="chip" title={activeModel}>
            serving <strong>{shortModel(activeModel)}</strong>
          </span>
        )}
        {watched && (
          <span className="chip">
            watching <strong className="mono">{watched}</strong>
          </span>
        )}
        <ConnectionBadge state={live.connection} />
      </header>

      <div className="layout">
        <aside className="sidebar">
          <SessionControls
            onStarted={(s) => {
              setWatched(s.id);
              void refresh();
            }}
          />
          <SessionList
            sessions={sessions}
            watched={watched}
            onWatch={setWatched}
            onStop={(id) => void api.stopSession(id).then(refresh, refresh)}
          />
          <ModelSwitcher models={models} onChanged={() => void refresh()} />
        </aside>

        <main className="main">
          <div className="row live">
            <TraceCanvas buffer={live.traces} channels={channels} onChannels={setChannels} />
            <PredictionPanel prediction={latest} />
          </div>
          <div className="row charts">
            <LineChart
              title="Rolling balanced accuracy"
              subtitle="last 60 labelled windows"
              testId="accuracy-chart"
              series={[{ label: "accuracy", color: "--accent", points: accuracy }]}
              markers={markers}
              yMin={0}
              yMax={1}
              reference={{ value: 1 / 3, label: "chance 33%" }}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <LineChart
              title="End-to-end latency"
              subtitle="per second"
              testId="latency-chart"
              series={[
                { label: "p50", color: "--ok", points: latency.p50 },
                { label: "p95", color: "--right", points: latency.p95 },
              ]}
              markers={markers}
              yMin={0}
              yMax="auto"
              reference={{ value: 50, label: "50 ms target (p95)" }}
              format={(v) => `${v.toFixed(0)} ms`}
            />
          </div>
          <MetricTiles metrics={live.metrics} />
          <WorkerPanel workers={live.workers} activeModel={activeModel} />
        </main>
      </div>
      <footer className="footer">
        PhysioNet EEGMMIDB · EEGNet served with ONNX Runtime · balanced accuracy on held-out people ≈ 46%
      </footer>
    </div>
  );
}
