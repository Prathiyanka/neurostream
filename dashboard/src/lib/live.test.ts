import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { backoffDelay, type ConnectionState, LiveClient } from "./live";

class FakeSocket {
  static instances: FakeSocket[] = [];
  readyState = 0;
  sent: string[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.closed = true;
  }
  // test helpers
  open() {
    this.readyState = 1;
    this.onopen?.();
  }
  message(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  drop(code = 1006) {
    this.readyState = 3;
    this.onclose?.({ code, reason: "" });
  }
}

const last = () => FakeSocket.instances[FakeSocket.instances.length - 1]!;

function makeClient(overrides: Partial<ConstructorParameters<typeof LiveClient>[0]> = {}) {
  const states: ConnectionState[] = [];
  const messages: unknown[] = [];
  const client = new LiveClient({
    url: "ws://test/ws/live",
    onState: (s) => states.push(s),
    onMessage: (m) => messages.push(m),
    random: () => 1, // deterministic: always the full backoff ceiling
    now: () => Date.now(),
    WebSocketImpl: FakeSocket as unknown as typeof WebSocket,
    ...overrides,
  });
  return { client, states, messages };
}

beforeEach(() => {
  vi.useFakeTimers();
  FakeSocket.instances = [];
});
afterEach(() => vi.useRealTimers());

describe("backoffDelay", () => {
  it("doubles per attempt up to the cap, with full jitter", () => {
    expect([1, 2, 3, 4, 5, 6].map((a) => backoffDelay(a, 500, 10_000, () => 1))).toEqual([
      500, 1000, 2000, 4000, 8000, 10_000,
    ]);
    expect(backoffDelay(3, 500, 10_000, () => 0)).toBe(0);
    expect(backoffDelay(3, 500, 10_000, () => 0.5)).toBe(1000);
  });
});

describe("LiveClient", () => {
  it("goes live, subscribes, and delivers messages", () => {
    const { client, states, messages } = makeClient();
    client.subscribe({ session: "s1", channels: ["C3"] });
    client.start();
    last().open();
    expect(states.map((s) => s.status)).toEqual(["connecting", "live"]);
    expect(JSON.parse(last().sent[0]!)).toEqual({ type: "subscribe", session: "s1", channels: ["C3"] });
    last().message({ type: "metrics", ts: 1 });
    expect(messages).toEqual([{ type: "metrics", ts: 1 }]);
  });

  it("reconnects with growing backoff and re-subscribes after reconnecting", () => {
    const { client, states } = makeClient();
    client.start();
    client.subscribe({ session: "s2", channels: ["Cz"] });
    last().drop(); // never opened: attempt 1 failed
    const r1 = states.at(-1)!;
    expect(r1.status).toBe("reconnecting");
    vi.advanceTimersByTime(499);
    expect(FakeSocket.instances).toHaveLength(1); // still waiting
    vi.advanceTimersByTime(1);
    expect(FakeSocket.instances).toHaveLength(2);
    last().drop(); // attempt 2 failed: next wait doubles
    vi.advanceTimersByTime(999);
    expect(FakeSocket.instances).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(FakeSocket.instances).toHaveLength(3);
    last().open();
    expect(states.at(-1)!.status).toBe("live");
    expect(JSON.parse(last().sent[0]!)).toMatchObject({ session: "s2", channels: ["Cz"] });
  });

  it("treats a silent connection as dead (watchdog) and reconnects", () => {
    const { client, states } = makeClient({ silenceTimeoutMs: 5000 });
    client.start();
    const first = last();
    first.open();
    vi.advanceTimersByTime(4000);
    first.message({ type: "metrics" }); // resets the watchdog
    vi.advanceTimersByTime(4999);
    expect(states.at(-1)!.status).toBe("live");
    vi.advanceTimersByTime(1);
    expect(first.closed).toBe(true);
    const s = states.at(-1)!;
    expect(s.status === "reconnecting" && s.reason).toContain("no data");
  });

  it("ignores a late close event from a socket it already gave up on", () => {
    const { client, states } = makeClient({ silenceTimeoutMs: 1000 });
    client.start();
    const stale = last();
    stale.open();
    vi.advanceTimersByTime(1000); // watchdog gives up on it
    vi.advanceTimersByTime(500); // backoff -> new socket
    const fresh = last();
    expect(fresh).not.toBe(stale);
    fresh.open();
    const sockets = FakeSocket.instances.length;
    stale.drop(); // the old socket's close finally arrives
    vi.advanceTimersByTime(900); // less than the watchdog, so only the stale close could act
    expect(FakeSocket.instances).toHaveLength(sockets); // no extra reconnect
    expect(states.at(-1)!.status).toBe("live");
    expect(fresh.closed).toBe(false);
  });

  it("starts the backoff from scratch after a healthy connection", () => {
    const { client, states } = makeClient({ healthyAfterMs: 5000 });
    client.start();
    last().drop();
    vi.advanceTimersByTime(500);
    last().drop();
    vi.advanceTimersByTime(1000); // attempt 3
    last().open();
    for (let i = 0; i < 6; i++) {
      vi.advanceTimersByTime(1000); // healthy for 6 s: a message every second
      last().message({});
    }
    last().drop();
    const s = states.at(-1)!;
    expect(s.status).toBe("reconnecting");
    expect(s.status === "reconnecting" && s.attempt).toBe(1);
  });

  it("stop() closes and does not reconnect", () => {
    const { client, states } = makeClient();
    client.start();
    last().open();
    client.stop();
    last().drop();
    vi.advanceTimersByTime(60_000);
    expect(FakeSocket.instances).toHaveLength(1);
    expect(states.at(-1)!.status).toBe("closed");
  });
});
