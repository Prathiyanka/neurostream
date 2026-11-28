import type { Session } from "../lib/types";

const TONE: Record<Session["status"], string> = {
  running: "ok",
  finished: "",
  stopped: "warn",
  failed: "bad",
};

export function SessionList({
  sessions,
  watched,
  onWatch,
  onStop,
}: {
  sessions: Session[];
  watched: string | null;
  onWatch: (id: string) => void;
  onStop: (id: string) => void;
}) {
  return (
    <section className="panel" aria-label="Sessions">
      <div className="panel-head">
        <h2>Sessions</h2>
      </div>
      {sessions.length === 0 ? (
        <p className="empty">No sessions yet</p>
      ) : (
        <ul className="sessions">
          {sessions.map((s) => (
            <li
              key={s.id}
              className={`session${s.id === watched ? " active" : ""}`}
              onClick={() => onWatch(s.id)}
              aria-current={s.id === watched}
            >
              <span className={`dot ${TONE[s.status]}`} title={s.status} />
              <span className="name mono" title={s.id}>
                {s.id}
              </span>
              {s.status === "running" && s.source === "api" ? (
                <button
                  className="btn small"
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onStop(s.id);
                  }}
                >
                  Stop
                </button>
              ) : (
                <span className="hint">{s.status}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
