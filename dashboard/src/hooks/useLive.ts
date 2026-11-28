import { useCallback, useEffect, useRef, useState } from "react";

import { liveUrl } from "../lib/api";
import { type ConnectionState, LiveClient } from "../lib/live";
import { TraceBuffer } from "../lib/traces";
import type { LiveMessage, Metrics, Prediction } from "../lib/types";
import { type TrackedWorker, trackWorkers } from "../lib/workers";

const MAX_PREDICTIONS = 1200; // 10 minutes of one session at 2 windows/s

export interface LiveState {
  connection: ConnectionState;
  metrics: Metrics | null;
  predictions: Prediction[]; // watched session, oldest first
  workers: Map<string, TrackedWorker>;
  traces: TraceBuffer; // mutable, read by the canvas each frame
  watch: (session: string | null, channels: string[]) => void;
}

export function useLive(initialChannels: string[]): LiveState {
  const [connection, setConnection] = useState<ConnectionState>({
    status: "connecting",
    attempt: 1,
  });
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [workers, setWorkers] = useState<Map<string, TrackedWorker>>(new Map());
  const traces = useRef(new TraceBuffer()).current;
  const client = useRef<LiveClient | null>(null);
  const watched = useRef<{ session: string | null; channels: string[] }>({
    session: null,
    channels: initialChannels,
  });

  useEffect(() => {
    const c = new LiveClient({
      url: liveUrl(),
      onState: setConnection,
      onMessage: (raw) => {
        const msg = raw as LiveMessage;
        if (msg.type === "metrics") {
          setMetrics(msg);
          setWorkers((prev) => trackWorkers(prev, msg));
        } else if (msg.type === "prediction") {
          if (msg.session_id !== watched.current.session) return;
          setPredictions((prev) => {
            const next = prev.length >= MAX_PREDICTIONS ? prev.slice(1) : prev.slice();
            next.push(msg);
            return next;
          });
        } else if (msg.type === "trace") {
          if (msg.session_id === watched.current.session) traces.push(msg);
        }
      },
    });
    client.current = c;
    c.subscribe(watched.current);
    c.start();
    return () => c.stop();
  }, [traces]);

  const watch = useCallback(
    (session: string | null, channels: string[]) => {
      if (session !== watched.current.session) {
        setPredictions([]);
        traces.reset(session);
      }
      watched.current = { session, channels };
      client.current?.subscribe({ session, channels });
    },
    [traces],
  );

  return { connection, metrics, predictions, workers, traces, watch };
}
