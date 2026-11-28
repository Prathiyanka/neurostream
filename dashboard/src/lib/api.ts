import type { Metrics, ModelsResponse, Session, SessionRequest } from "./types";

// Same origin: nginx (container) or the Vite dev server proxies /api to the API service.
const BASE = "/api";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  const data: unknown = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg?: string }) => d.msg ?? "invalid").join("; ")
          : `HTTP ${res.status}`;
    throw new ApiError(res.status, message);
  }
  return data as T;
}

export const api = {
  sessions: () => request<Session[]>("GET", "/sessions?limit=20"),
  startSession: (req: SessionRequest) => request<Session>("POST", "/sessions", req),
  stopSession: (id: string) => request<Session>("DELETE", `/sessions/${encodeURIComponent(id)}`),
  models: () => request<ModelsResponse>("GET", "/models"),
  activateModel: (model: string) =>
    request<{ active: string; previous: string }>("PUT", "/models/active", { model }),
  metrics: () => request<Metrics>("GET", "/metrics"),
};

export function liveUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/live`;
}
