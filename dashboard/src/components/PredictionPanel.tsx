import { shortModel } from "../lib/stats";
import { CLASSES, type Prediction } from "../lib/types";

export const CLASS_COLORS = ["var(--rest)", "var(--left)", "var(--right)"] as const;

export function PredictionPanel({ prediction }: { prediction: Prediction | null }) {
  return (
    <section className="panel" aria-label="Current prediction" data-testid="prediction">
      <div className="panel-head">
        <h2>Prediction</h2>
        {prediction && (
          <span className="chip" title={prediction.model} data-testid="prediction-model">
            model <strong>{shortModel(prediction.model)}</strong>
          </span>
        )}
      </div>
      {prediction ? <Body p={prediction} /> : <p className="empty">Waiting for predictions…</p>}
    </section>
  );
}

function Body({ p }: { p: Prediction }) {
  const predicted = CLASSES[p.predicted_class] ?? "?";
  const truth = p.true_label >= 0 ? CLASSES[p.true_label] : null;
  const correct = truth !== null && p.true_label === p.predicted_class;
  return (
    <>
      <div
        className="prediction-class"
        style={{ color: CLASS_COLORS[p.predicted_class] }}
        data-testid="predicted-class"
      >
        {predicted}
      </div>
      <div>
        {CLASSES.map((name, i) => {
          const prob = p.probabilities[i] ?? 0;
          return (
            <div className="prob" key={name}>
              <span>{name}</span>
              <div
                className="track"
                role="meter"
                aria-label={`${name} probability`}
                aria-valuemin={0}
                aria-valuemax={1}
                aria-valuenow={prob}
              >
                <div
                  className="bar"
                  style={{ width: `${(prob * 100).toFixed(1)}%`, background: CLASS_COLORS[i] }}
                />
              </div>
              <span className="pct">{(prob * 100).toFixed(0)}%</span>
            </div>
          );
        })}
      </div>
      <div className="truth" data-testid="truth">
        <span className="label">True label</span>
        {truth ? (
          <>
            <span className="chip" style={{ color: CLASS_COLORS[p.true_label] }}>
              {truth}
            </span>
            <span style={{ color: correct ? "var(--ok)" : "var(--bad)" }}>
              {correct ? "✓ correct" : "✗ wrong"}
            </span>
          </>
        ) : (
          <span className="hint" title="The window spans a label change or a gap">
            mixed / unknown
          </span>
        )}
      </div>
      <dl className="meta">
        <dt>Latency</dt>
        <dd className="mono">{p.latency_ms.toFixed(1)} ms</dd>
        <dt>Batch</dt>
        <dd className="mono">{p.batch_size}</dd>
        <dt>Worker</dt>
        <dd className="mono" title={p.worker_id}>
          {p.worker_id.slice(0, 14)}
          {p.reclaimed ? " (reclaimed)" : ""}
        </dd>
      </dl>
    </>
  );
}
