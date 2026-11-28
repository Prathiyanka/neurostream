import { describe, expect, it } from "vitest";

import {
  balancedAccuracy,
  latencySeries,
  modelChanges,
  percentile,
  rollingAccuracy,
  shortModel,
} from "./stats";
import { pred } from "../test/fixtures";

describe("percentile", () => {
  it("matches numpy's linear interpolation", () => {
    expect(percentile([], 50)).toBeNull();
    expect(percentile([3, 1, 2], 50)).toBe(2);
    expect(percentile([1, 2, 3, 4], 50)).toBe(2.5);
    expect(percentile([0, 10], 95)).toBeCloseTo(9.5);
  });
});

describe("balancedAccuracy", () => {
  it("averages per-class recall and ignores unknown labels", () => {
    const preds = [
      pred({ true_label: 0, predicted_class: 0 }),
      pred({ true_label: 0, predicted_class: 0 }),
      pred({ true_label: 0, predicted_class: 0 }),
      pred({ true_label: 0, predicted_class: 1 }), // rest recall 3/4
      pred({ true_label: 1, predicted_class: 1 }), // left recall 1/1
      pred({ true_label: -1, predicted_class: 2 }), // unknown: ignored
    ];
    expect(balancedAccuracy(preds)).toBeCloseTo((0.75 + 1) / 2);
    expect(balancedAccuracy([pred({ true_label: -1 })])).toBeNull();
  });

  it("is not fooled by always predicting the majority class", () => {
    const preds = [
      ...Array.from({ length: 8 }, () => pred({ true_label: 0, predicted_class: 0 })),
      pred({ true_label: 1, predicted_class: 0 }),
      pred({ true_label: 2, predicted_class: 0 }),
    ];
    expect(balancedAccuracy(preds)).toBeCloseTo(1 / 3); // plain accuracy would be 80%
  });
});

describe("rollingAccuracy", () => {
  it("needs 10 labelled windows before the first point", () => {
    const preds = Array.from({ length: 12 }, (_, i) => pred({ publish_ts: 1000 + i }));
    const pts = rollingAccuracy(preds, 60);
    expect(pts).toHaveLength(3);
    expect(pts[0]).toEqual({ t: 1009, v: 1 });
  });
});

describe("latencySeries", () => {
  it("gives p50 and p95 per one-second bucket", () => {
    const preds = [1, 2, 3, 4, 100].map((ms, i) =>
      pred({ latency_ms: ms, publish_ts: 2000 + i * 0.1 }),
    );
    preds.push(pred({ latency_ms: 7, publish_ts: 2001.5 }));
    const { p50, p95 } = latencySeries(preds);
    expect(p50).toEqual([
      { t: 2000.5, v: 3 },
      { t: 2001.5, v: 7 },
    ]);
    expect(p95[0]!.v).toBeCloseTo(80.8);
  });
});

describe("modelChanges", () => {
  it("marks the first prediction made by a new model", () => {
    const preds = [
      pred({ publish_ts: 1, model: "eegnet-motor-1.0.1" }),
      pred({ publish_ts: 2, model: "eegnet-motor-1.0.1" }),
      pred({ publish_ts: 3, model: "eegnet-motor-1.0.0" }),
      pred({ publish_ts: 4, model: "eegnet-motor-1.0.0" }),
      pred({ publish_ts: 5, model: "eegnet-all-1.0.0" }),
    ];
    expect(modelChanges(preds)).toEqual([
      { t: 3, label: "motor 1.0.0" },
      { t: 5, label: "all 1.0.0" },
    ]);
    expect(modelChanges(preds.slice(0, 2))).toEqual([]);
  });

  it("uses publish order even if predictions arrive out of order", () => {
    const preds = [
      pred({ publish_ts: 3, model: "b-model" }),
      pred({ publish_ts: 1, model: "a-model" }),
    ];
    expect(modelChanges(preds)).toEqual([{ t: 3, label: "b-model" }]);
  });
});

describe("shortModel", () => {
  it("shortens model ids for labels", () => {
    expect(shortModel("eegnet-motor-1.0.1")).toBe("motor 1.0.1");
    expect(shortModel("custom")).toBe("custom");
  });
});
