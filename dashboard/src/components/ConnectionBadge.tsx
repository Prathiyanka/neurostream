import { useEffect, useState } from "react";

import type { ConnectionState } from "../lib/live";

/** Re-render every `ms` (for countdowns). */
export function useNow(ms = 500): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}

export function describeConnection(
  state: ConnectionState,
  now: number,
): { text: string; tone: "ok" | "warn" | "bad"; title: string } {
  switch (state.status) {
    case "live":
      return { text: "Live", tone: "ok", title: "Connected to the live feed" };
    case "connecting":
      return {
        text: state.attempt > 1 ? `Connecting (attempt ${state.attempt})` : "Connecting",
        tone: "warn",
        title: "Opening the live feed",
      };
    case "reconnecting": {
      const s = Math.max(0, Math.ceil((state.retryAt - now) / 1000));
      return {
        text: s > 0 ? `Reconnecting in ${s} s` : "Reconnecting",
        tone: "bad",
        title: `Connection lost: ${state.reason}. Attempt ${state.attempt}.`,
      };
    }
    case "closed":
      return { text: "Disconnected", tone: "bad", title: "Live feed closed" };
  }
}

export function ConnectionBadge({ state }: { state: ConnectionState }) {
  const now = useNow();
  const d = describeConnection(state, now);
  const pulse = state.status !== "live" && state.status !== "closed";
  return (
    <span className="badge" role="status" aria-live="polite" title={d.title} data-testid="connection">
      <span className={`dot ${d.tone}${pulse ? " pulse" : ""}`} />
      {d.text}
    </span>
  );
}
