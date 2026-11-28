// WebSocket client for /ws/live with automatic reconnect.
//
// Reconnect policy: exponential backoff with full jitter (random delay in [0, base*2^n],
// capped), reset after a connection has been healthy for a while. Jitter matters when
// the API restarts: without it every open dashboard would reconnect in the same instant.
//
// A watchdog treats "connected but silent" as dead: the API sends metrics every second,
// so 5 s without any message means the connection is broken even if the socket has not
// noticed (e.g. a laptop waking from sleep). It closes the socket and reconnects.

export type ConnectionState =
  | { status: "connecting"; attempt: number }
  | { status: "live"; since: number }
  | { status: "reconnecting"; attempt: number; retryAt: number; reason: string }
  | { status: "closed" };

export interface Subscription {
  session: string | null;
  channels: string[];
}

export interface LiveClientOptions {
  url: string;
  onMessage: (msg: unknown) => void;
  onState: (state: ConnectionState) => void;
  baseDelayMs?: number;
  maxDelayMs?: number;
  silenceTimeoutMs?: number;
  healthyAfterMs?: number; // connection must last this long to reset the backoff
  random?: () => number;
  now?: () => number;
  WebSocketImpl?: typeof WebSocket;
}

export function backoffDelay(
  attempt: number,
  baseMs: number,
  maxMs: number,
  random: () => number = Math.random,
): number {
  const ceiling = Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt - 1));
  return Math.round(random() * ceiling);
}

export class LiveClient {
  private ws: WebSocket | null = null;
  private attempt = 0;
  private openedAt = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private watchdog: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private subscription: Subscription = { session: null, channels: ["C3", "Cz", "C4"] };
  private readonly o: Required<Omit<LiveClientOptions, "WebSocketImpl">> & {
    WebSocketImpl: typeof WebSocket;
  };

  constructor(options: LiveClientOptions) {
    this.o = {
      baseDelayMs: 500,
      maxDelayMs: 10_000,
      silenceTimeoutMs: 5_000,
      healthyAfterMs: 5_000,
      random: Math.random,
      now: () => Date.now(),
      WebSocketImpl: WebSocket,
      ...options,
    };
  }

  start(): void {
    this.stopped = false;
    this.connect();
  }

  stop(): void {
    this.stopped = true;
    this.clearTimers();
    this.ws?.close();
    this.ws = null;
    this.o.onState({ status: "closed" });
  }

  /** Change what this client watches; re-sent automatically after every reconnect. */
  subscribe(sub: Subscription): void {
    this.subscription = sub;
    this.send({ type: "subscribe", ...sub });
  }

  private send(msg: object): void {
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify(msg));
  }

  private connect(): void {
    this.attempt += 1;
    this.o.onState({ status: "connecting", attempt: this.attempt });
    const ws = new this.o.WebSocketImpl(this.o.url);
    this.ws = ws;
    // Every handler ignores a socket that is no longer current: after the watchdog gives
    // up on a silent socket, its late close event must not start a second reconnect.
    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.openedAt = this.o.now();
      this.send({ type: "subscribe", ...this.subscription });
      this.o.onState({ status: "live", since: this.openedAt });
      this.armWatchdog();
    };
    ws.onmessage = (ev: MessageEvent) => {
      if (this.ws !== ws) return;
      this.armWatchdog();
      try {
        this.o.onMessage(JSON.parse(String(ev.data)));
      } catch {
        // ignore a malformed message rather than drop the connection
      }
    };
    ws.onclose = (ev: CloseEvent) => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.scheduleReconnect(ev.reason || `connection closed (${ev.code})`);
    };
    ws.onerror = () => {
      // onclose follows and does the reconnecting
    };
  }

  private armWatchdog(): void {
    if (this.watchdog) clearTimeout(this.watchdog);
    this.watchdog = setTimeout(() => {
      const ws = this.ws;
      this.ws = null;
      ws?.close();
      this.scheduleReconnect("no data for " + this.o.silenceTimeoutMs / 1000 + " s");
    }, this.o.silenceTimeoutMs);
  }

  private scheduleReconnect(reason: string): void {
    if (this.stopped || this.retryTimer) return;
    this.clearTimers();
    if (this.openedAt && this.o.now() - this.openedAt >= this.o.healthyAfterMs) {
      this.attempt = 0; // it was a good connection: start the backoff from scratch
    }
    this.openedAt = 0;
    // `attempt` failed attempts so far (0 after a healthy connection): the first retry
    // waits up to the base delay, each further failure doubles it.
    const delay = backoffDelay(
      Math.max(1, this.attempt),
      this.o.baseDelayMs,
      this.o.maxDelayMs,
      this.o.random,
    );
    this.o.onState({
      status: "reconnecting",
      attempt: this.attempt + 1,
      retryAt: this.o.now() + delay,
      reason,
    });
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }

  private clearTimers(): void {
    if (this.retryTimer) clearTimeout(this.retryTimer);
    if (this.watchdog) clearTimeout(this.watchdog);
    this.retryTimer = null;
    this.watchdog = null;
  }
}
