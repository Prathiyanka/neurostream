import type { Metrics } from "../lib/types";

interface Tile {
  name: string;
  value: number | string;
  tone?: "warn" | "bad";
  title: string;
}

export function metricTiles(m: Metrics): Tile[] {
  const queue = m.queue.lag + m.queue.pending;
  return [
    { name: "Workers", value: m.workers.count, tone: m.workers.count === 0 ? "bad" : undefined,
      title: "Workers with a live heartbeat" },
    { name: "Queue depth", value: queue, tone: queue > 50 ? "warn" : undefined,
      title: "Window jobs waiting (lag) plus being processed (pending)" },
    { name: "Packets dropped", value: m.ingest.missing_packets,
      tone: m.ingest.missing_packets ? "warn" : undefined,
      title: "Packets ingest never received (sequence gaps), all sessions" },
    { name: "Duplicates", value: m.ingest.duplicates,
      title: "Duplicate packets ingest detected and discarded" },
    { name: "Stale", value: m.jobs.stale, tone: m.jobs.stale ? "warn" : undefined,
      title: "Jobs skipped because they were already older than the stale threshold (2 s)" },
    { name: "Lost", value: m.jobs.lost, tone: m.jobs.lost ? "bad" : undefined,
      title: "Jobs trimmed by the stream length cap before being acknowledged" },
    { name: "Dead-lettered", value: m.jobs.dead_lettered,
      tone: m.jobs.dead_lettered ? "bad" : undefined,
      title: "Jobs set aside after failing (ns:windows:dead)" },
    { name: "Reclaimed", value: m.jobs.reclaimed,
      title: "Jobs taken over from a worker that stopped responding" },
  ];
}

export function MetricTiles({ metrics }: { metrics: Metrics | null }) {
  return (
    <section className="panel" aria-label="System metrics">
      <div className="panel-head">
        <h2>System</h2>
        {metrics && (
          <span className="hint">
            {metrics.ingest.sessions_connected} streaming · {metrics.jobs.acked.toLocaleString()}{" "}
            windows predicted
          </span>
        )}
      </div>
      {metrics ? (
        <div className="tiles">
          {metricTiles(metrics).map((t) => (
            <div className="tile" key={t.name} title={t.title} data-testid={`tile-${t.name}`}>
              <div className={`value ${t.tone ?? ""}`}>
                {typeof t.value === "number" ? t.value.toLocaleString() : t.value}
              </div>
              <div className="name">{t.name}</div>
            </div>
          ))}
        </div>
      ) : (
        <p className="empty">Waiting for metrics…</p>
      )}
    </section>
  );
}
