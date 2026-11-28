import { describe, expect, it } from "vitest";

import { isGap, robustRange, TraceBuffer } from "./traces";

const trace = (start: number, values: number[], session = "s1", channels = ["C3"]) => ({
  session_id: session,
  sfreq: 40,
  start_sample: start,
  channels,
  data: channels.map(() => values),
});

describe("TraceBuffer", () => {
  it("keeps the newest samples in order, dropping the oldest", () => {
    const b = new TraceBuffer(1, 4); // capacity 5 samples
    b.push(trace(0, [1, 2, 3]));
    b.push(trace(12, [4, 5, 6])); // 160 Hz index 12 = 3 display samples later
    const { times, series } = b.snapshot();
    expect(series.get("C3")).toEqual([2, 3, 4, 5, 6]);
    expect(times[0]).toBeCloseTo(1 / 40);
    expect(times[2]).toBeCloseTo(12 / 160);
  });

  it("starts over when the session or the channel set changes", () => {
    const b = new TraceBuffer();
    b.push(trace(0, [1, 2]));
    b.push(trace(8, [3, 4], "s2"));
    expect(b.sessionId).toBe("s2");
    expect(b.snapshot().series.get("C3")).toEqual([3, 4]);
    b.push(trace(16, [5], "s2", ["C3", "C4"]));
    expect(b.channels).toEqual(["C3", "C4"]);
    expect(b.length).toBe(1);
  });

  it("lets the canvas see gaps from lost packets", () => {
    const b = new TraceBuffer();
    b.push(trace(0, [1, 2, 3, 4]));
    b.push(trace(160, [5])); // 1 s later: packets were lost in between
    const { times } = b.snapshot();
    expect(isGap(times[2]!, times[3]!, 40)).toBe(false);
    expect(isGap(times[3]!, times[4]!, 40)).toBe(true);
  });
});

describe("robustRange", () => {
  it("ignores a single spike when scaling a lane", () => {
    const values = Array.from({ length: 200 }, (_, i) => Math.sin(i / 5) * 10);
    values[100] = 5000;
    const r = robustRange(values);
    expect(r).toBeGreaterThan(9);
    expect(r).toBeLessThan(20);
  });
});
