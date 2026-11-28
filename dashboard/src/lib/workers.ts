import type { Metrics, WorkerInfo } from "./types";

export type WorkerStatus = "healthy" | "late" | "loading" | "error" | "lost";

export interface TrackedWorker {
  id: string;
  info: WorkerInfo;
  status: WorkerStatus;
  heartbeatAgeS: number | null; // vs. the server's clock (metrics.ts), not the browser's
  goneSince: number | null; // server time when its heartbeat disappeared
}

// Workers heartbeat about every second; the key expires after 5 s.
export const LATE_AFTER_S = 3;
export const FORGET_LOST_AFTER_S = 60;

/**
 * Merge a metrics message into the known workers.
 *
 * The API only lists workers whose heartbeat key still exists, so a killed worker simply
 * vanishes from `metrics.workers`. Remembering previously seen workers is what lets the
 * panel show it as "lost" (red, with how long ago) instead of silently dropping the row.
 */
export function trackWorkers(
  previous: Map<string, TrackedWorker>,
  metrics: Metrics,
): Map<string, TrackedWorker> {
  const now = metrics.ts;
  const next = new Map<string, TrackedWorker>();
  for (const info of metrics.workers.list) {
    const lastSeen = info.last_seen ? Number(info.last_seen) : null;
    const age = lastSeen === null ? null : Math.max(0, now - lastSeen);
    next.set(info.worker_id, {
      id: info.worker_id,
      info,
      status: statusOf(info, age),
      heartbeatAgeS: age,
      goneSince: null,
    });
  }
  for (const [id, w] of previous) {
    if (next.has(id)) continue;
    const goneSince = w.goneSince ?? now;
    if (now - goneSince > FORGET_LOST_AFTER_S) continue;
    const lastSeen = w.info.last_seen ? Number(w.info.last_seen) : null;
    next.set(id, {
      ...w,
      status: "lost",
      goneSince,
      heartbeatAgeS: lastSeen === null ? null : Math.max(0, now - lastSeen),
    });
  }
  return next;
}

function statusOf(info: WorkerInfo, ageS: number | null): WorkerStatus {
  if (info.model_error) return "error";
  if (ageS !== null && ageS > LATE_AFTER_S) return "late";
  if (info.model_loading === "1") return "loading";
  return "healthy";
}

/** Live workers first (by id), then lost ones (most recently lost first). */
export function sortWorkers(workers: Map<string, TrackedWorker>): TrackedWorker[] {
  return [...workers.values()].sort((a, b) => {
    const lostA = a.status === "lost" ? 1 : 0;
    const lostB = b.status === "lost" ? 1 : 0;
    if (lostA !== lostB) return lostA - lostB;
    if (lostA) return (b.goneSince ?? 0) - (a.goneSince ?? 0);
    return a.id.localeCompare(b.id);
  });
}
