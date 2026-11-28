import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Metrics, WorkerInfo } from "../lib/types";
import { trackWorkers } from "../lib/workers";
import { pred } from "../test/fixtures";
import { ConnectionBadge, describeConnection } from "./ConnectionBadge";
import { MetricTiles } from "./MetricTiles";
import { PredictionPanel } from "./PredictionPanel";
import { buildRequest, FAULT_PRESET, SessionControls } from "./SessionControls";
import { WorkerPanel } from "./WorkerPanel";

afterEach(() => vi.restoreAllMocks());

describe("ConnectionBadge", () => {
  it("shows live, connecting and a reconnect countdown", () => {
    expect(describeConnection({ status: "live", since: 0 }, 0)).toMatchObject({
      text: "Live",
      tone: "ok",
    });
    expect(describeConnection({ status: "connecting", attempt: 3 }, 0).text).toBe(
      "Connecting (attempt 3)",
    );
    const r = describeConnection(
      { status: "reconnecting", attempt: 2, retryAt: 10_000, reason: "connection closed (1006)" },
      7_200,
    );
    expect(r).toMatchObject({ text: "Reconnecting in 3 s", tone: "bad" });
    expect(r.title).toContain("1006");
  });

  it("renders as an accessible status", () => {
    render(<ConnectionBadge state={{ status: "closed" }} />);
    expect(screen.getByRole("status")).toHaveTextContent("Disconnected");
  });
});

describe("PredictionPanel", () => {
  it("waits politely before the first prediction", () => {
    render(<PredictionPanel prediction={null} />);
    expect(screen.getByText(/Waiting for predictions/)).toBeInTheDocument();
  });

  it("shows the class, probabilities, true label and the model version", () => {
    render(
      <PredictionPanel
        prediction={pred({
          predicted_class: 2,
          probabilities: [0.1, 0.25, 0.65],
          true_label: 1,
          model: "eegnet-motor-1.0.0",
          latency_ms: 8.24,
        })}
      />,
    );
    expect(screen.getByTestId("predicted-class")).toHaveTextContent("right");
    expect(screen.getByRole("meter", { name: "right probability" })).toHaveAttribute(
      "aria-valuenow",
      "0.65",
    );
    expect(screen.getByText("65%")).toBeInTheDocument();
    const truth = screen.getByTestId("truth");
    expect(truth).toHaveTextContent("left");
    expect(truth).toHaveTextContent("wrong");
    expect(screen.getByTestId("prediction-model")).toHaveTextContent("motor 1.0.0");
    expect(screen.getByText("8.2 ms")).toBeInTheDocument();
  });

  it("says so when the window has no single true label", () => {
    render(<PredictionPanel prediction={pred({ true_label: -1 })} />);
    expect(screen.getByTestId("truth")).toHaveTextContent("mixed / unknown");
  });
});

describe("WorkerPanel", () => {
  const info = (id: string, over: Partial<WorkerInfo> = {}): WorkerInfo => ({
    worker_id: id,
    model: "eegnet-motor-1.0.1",
    model_error: "",
    model_loading: "0",
    predictions: "1234",
    reclaimed: "3",
    stale: "0",
    dead_lettered: "0",
    errors: "0",
    last_seen: "100",
    ...over,
  });
  const m = (ts: number, list: WorkerInfo[]) =>
    ({ ts, workers: { count: list.length, list } }) as unknown as Metrics;

  it("shows a killed worker as lost, next to the live ones", () => {
    let w = trackWorkers(new Map(), m(100, [info("w-a"), info("w-b")]));
    w = trackWorkers(w, m(108, [info("w-a", { last_seen: "107.6" })]));
    render(<WorkerPanel workers={w} activeModel="eegnet-motor-1.0.1" />);
    const rows = screen.getAllByRole("row").slice(1); // skip header
    expect(rows.map((r) => r.getAttribute("data-status"))).toEqual(["healthy", "lost"]);
    expect(within(rows[0]!).getByText("0.4 s ago")).toBeInTheDocument();
    expect(within(rows[1]!).getByText("lost")).toBeInTheDocument();
    expect(within(rows[1]!).getByText("8.0 s ago")).toBeInTheDocument();
    expect(within(rows[0]!).getByText("1,234")).toBeInTheDocument();
  });

  it("flags a worker still on an old model during a hot-swap", () => {
    const w = trackWorkers(
      new Map(),
      m(100, [info("w-a", { model_loading: "1", last_seen: "100" })]),
    );
    render(<WorkerPanel workers={w} activeModel="eegnet-motor-1.0.0" />);
    expect(screen.getByText("loading model")).toBeInTheDocument();
    expect(screen.getByText("motor 1.0.1")).toHaveAttribute(
      "title",
      "active model is eegnet-motor-1.0.0",
    );
  });
});

describe("MetricTiles", () => {
  it("highlights stale, lost and dead-lettered jobs only when non-zero", () => {
    const metrics = {
      ts: 1,
      model: { active: "x", workers: {} },
      workers: { count: 2, list: [] },
      queue: { length: 0, added: 0, lag: 3, pending: 1 },
      jobs: {
        acked: 10,
        stale: 4,
        lost: 0,
        lost_trimmed_undelivered: 0,
        lost_trimmed_pending: 0,
        dead_lettered: 0,
        reclaimed: 2,
        pad_short: 0,
      },
      persister: {},
      ingest: {
        sessions: 1,
        sessions_connected: 1,
        accepted: 100,
        duplicates: 1,
        late: 0,
        gaps: 1,
        missing_packets: 5,
        invalid: 0,
        windows_emitted: 10,
        backpressure_drops: 0,
      },
    } satisfies Metrics;
    render(<MetricTiles metrics={metrics} />);
    const value = (name: string) => screen.getByTestId(`tile-${name}`).querySelector(".value")!;
    expect(value("Queue depth")).toHaveTextContent("4");
    expect(value("Stale")).toHaveTextContent("4");
    expect(value("Stale")).toHaveClass("warn");
    expect(value("Lost")).not.toHaveClass("bad");
    expect(value("Packets dropped")).toHaveTextContent("5");
  });
});

describe("SessionControls", () => {
  it("builds the request the API expects", () => {
    expect(
      buildRequest({ source: "subject", subject: 7, speed: 2, minutes: 5, faults: true }),
    ).toEqual({ subject: 7, speed: 2, max_seconds: 300, faults: FAULT_PRESET });
    const synth = buildRequest({
      source: "synthetic",
      subject: 7,
      speed: 1,
      minutes: null,
      faults: false,
    });
    expect(synth).toMatchObject({ synthetic: true, speed: 1 });
    expect(synth).not.toHaveProperty("subject");
    expect(synth).not.toHaveProperty("max_seconds");
  });

  it("starts a session and reports API errors", async () => {
    const user = userEvent.setup();
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: "subject 88 is excluded (see docs/DATA.md)" }), {
          status: 422,
        }),
      )
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "sub007-abc" }), { status: 201 }));
    const onStarted = vi.fn();
    render(<SessionControls onStarted={onStarted} />);

    await user.click(screen.getByRole("button", { name: "Start session" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("subject 88 is excluded");
    expect(onStarted).not.toHaveBeenCalled();

    await user.clear(screen.getByLabelText("Subject (1–109)"));
    await user.type(screen.getByLabelText("Subject (1–109)"), "7");
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Start session" }));
    await vi.waitFor(() => expect(onStarted).toHaveBeenCalledWith({ id: "sub007-abc" }));
    const [url, init] = fetchMock.mock.calls[1]!;
    expect(url).toBe("/api/sessions");
    expect(JSON.parse(String(init!.body))).toMatchObject({ subject: 7, faults: FAULT_PRESET });
  });
});
