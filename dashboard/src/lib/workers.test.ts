import { describe, expect, it } from "vitest";

import type { Metrics, WorkerInfo } from "./types";
import { FORGET_LOST_AFTER_S, sortWorkers, trackWorkers } from "./workers";

function worker(id: string, over: Partial<WorkerInfo> = {}): WorkerInfo {
  return {
    worker_id: id,
    model: "eegnet-motor-1.0.1",
    model_error: "",
    model_loading: "0",
    predictions: "10",
    reclaimed: "0",
    stale: "0",
    dead_lettered: "0",
    errors: "0",
    last_seen: "100",
    ...over,
  };
}

function metrics(ts: number, list: WorkerInfo[]): Metrics {
  return { ts, workers: { count: list.length, list } } as unknown as Metrics;
}

describe("trackWorkers", () => {
  it("classifies live workers from their heartbeat and model state", () => {
    const m = metrics(105, [
      worker("ok", { last_seen: "104.5" }),
      worker("slow", { last_seen: "100" }), // 5 s without a heartbeat
      worker("loading", { last_seen: "105", model_loading: "1" }),
      worker("broken", { last_seen: "105", model_error: "ModelCardError: sha256 ..." }),
    ]);
    const w = trackWorkers(new Map(), m);
    expect(w.get("ok")!.status).toBe("healthy");
    expect(w.get("ok")!.heartbeatAgeS).toBeCloseTo(0.5);
    expect(w.get("slow")!.status).toBe("late");
    expect(w.get("loading")!.status).toBe("loading");
    expect(w.get("broken")!.status).toBe("error");
  });

  it("keeps a worker that vanished (killed) as lost, then forgets it", () => {
    let w = trackWorkers(new Map(), metrics(100, [worker("a"), worker("b")]));
    w = trackWorkers(w, metrics(106, [worker("a", { last_seen: "106" })])); // b's key expired
    expect(w.get("b")!.status).toBe("lost");
    expect(w.get("b")!.goneSince).toBe(106);
    expect(w.get("b")!.heartbeatAgeS).toBe(6);
    w = trackWorkers(w, metrics(120, [worker("a", { last_seen: "120" })]));
    expect(w.get("b")!.goneSince).toBe(106); // still counted from when it disappeared
    w = trackWorkers(w, metrics(106 + FORGET_LOST_AFTER_S + 1, [worker("a")]));
    expect(w.has("b")).toBe(false);
  });

  it("a lost worker that comes back is live again", () => {
    let w = trackWorkers(new Map(), metrics(100, [worker("a")]));
    w = trackWorkers(w, metrics(110, []));
    expect(w.get("a")!.status).toBe("lost");
    w = trackWorkers(w, metrics(111, [worker("a", { last_seen: "111" })]));
    expect(w.get("a")!.status).toBe("healthy");
  });
});

describe("sortWorkers", () => {
  it("lists live workers first, lost ones last", () => {
    let w = trackWorkers(new Map(), metrics(100, [worker("c"), worker("a"), worker("b")]));
    w = trackWorkers(w, metrics(101, [worker("c"), worker("b")]));
    expect(sortWorkers(w).map((x) => x.id)).toEqual(["b", "c", "a"]);
  });
});
