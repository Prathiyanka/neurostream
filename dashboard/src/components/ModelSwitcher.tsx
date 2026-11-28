import { useEffect, useState } from "react";

import { api, ApiError } from "../lib/api";
import { shortModel } from "../lib/stats";
import type { ModelsResponse } from "../lib/types";

export function ModelSwitcher({
  models,
  onChanged,
}: {
  models: ModelsResponse | null;
  onChanged: () => void;
}) {
  const [choice, setChoice] = useState<string>("");
  const [touched, setTouched] = useState(false); // user picked something: stop following
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Follow the active model (it can change from elsewhere, e.g. the API) until the user
  // picks a different one.
  useEffect(() => {
    if (models && !touched) setChoice(models.active);
  }, [models, touched]);

  if (!models) return null;
  const selected = models.available.find((m) => m.id === choice);

  const activate = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.activateModel(choice);
      setTouched(false);
      onChanged();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reach the API");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel" aria-label="Model">
      <div className="panel-head">
        <h2>Model</h2>
        <span className="chip" title="What workers are told to serve">
          active <strong>{shortModel(models.active)}</strong>
        </span>
      </div>
      <div className="inline">
        <select
          className="select"
          aria-label="Model to serve"
          value={choice}
          onChange={(e) => {
            setChoice(e.target.value);
            setTouched(true);
          }}
        >
          {models.available.map((m) => (
            <option key={m.id} value={m.id}>
              {shortModel(m.id)} · {m.n_channels} ch
            </option>
          ))}
        </select>
        <button
          className="btn"
          type="button"
          disabled={busy || choice === models.active}
          onClick={activate}
        >
          Hot-swap
        </button>
      </div>
      {selected && (
        <p className="hint" style={{ margin: "8px 0 0" }}>
          {selected.test_balanced_accuracy !== null
            ? `Held-out balanced accuracy ${(selected.test_balanced_accuracy * 100).toFixed(1)}% (fold 0) · `
            : ""}
          serving pad {selected.serving_pad_s} s
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
