import { shortModel } from "../lib/stats";
import { sortWorkers, type TrackedWorker, type WorkerStatus } from "../lib/workers";

const TONE: Record<WorkerStatus, "ok" | "warn" | "bad"> = {
  healthy: "ok",
  loading: "warn",
  late: "warn",
  error: "bad",
  lost: "bad",
};

const LABEL: Record<WorkerStatus, string> = {
  healthy: "healthy",
  loading: "loading model",
  late: "heartbeat late",
  error: "model error",
  lost: "lost",
};

export function WorkerPanel({
  workers,
  activeModel,
}: {
  workers: Map<string, TrackedWorker>;
  activeModel: string | null;
}) {
  const rows = sortWorkers(workers);
  return (
    <section className="panel" aria-label="Workers" data-testid="workers">
      <div className="panel-head">
        <h2>Workers</h2>
        <span className="hint">heartbeat every ~1 s · lost after 5 s without one</span>
      </div>
      {rows.length === 0 ? (
        <p className="empty">No workers yet</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Worker</th>
              <th>Status</th>
              <th>Model</th>
              <th className="num">Last heartbeat</th>
              <th className="num">Predictions</th>
              <th className="num">Reclaimed</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((w) => (
              <tr key={w.id} className={w.status === "lost" ? "lost" : ""} data-status={w.status}>
                <td className="mono" title={w.id}>
                  {w.id}
                </td>
                <td>
                  <span className="status" title={w.info.model_error || undefined}>
                    <span className={`dot ${TONE[w.status]}`} />
                    {LABEL[w.status]}
                  </span>
                </td>
                <td>
                  <span
                    className="mono"
                    style={{
                      color:
                        activeModel && w.info.model && w.info.model !== activeModel
                          ? "var(--warn)"
                          : undefined,
                    }}
                    title={
                      activeModel && w.info.model !== activeModel
                        ? `active model is ${activeModel}`
                        : undefined
                    }
                  >
                    {w.info.model ? shortModel(w.info.model) : "–"}
                  </span>
                </td>
                <td className="num mono">
                  {w.heartbeatAgeS === null ? "–" : `${w.heartbeatAgeS.toFixed(1)} s ago`}
                </td>
                <td className="num">{Number(w.info.predictions ?? 0).toLocaleString()}</td>
                <td className="num">{Number(w.info.reclaimed ?? 0).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
