import { type Marker, percentile, type Point } from "../lib/stats";
import { cssVar, useCanvas } from "./canvas";

export interface Series {
  label: string;
  color: string; // CSS variable name, e.g. "--accent"
  points: Point[];
}

interface Props {
  title: string;
  subtitle?: string;
  series: Series[];
  markers: Marker[]; // model changes: dashed vertical line + label
  yMin: number;
  yMax: number | "auto";
  format: (v: number) => string;
  reference?: { value: number; label: string }; // e.g. chance level
  windowS?: number;
  height?: number;
  testId?: string;
}

const PAD = { left: 44, right: 10, top: 18, bottom: 20 };

/**
 * Small canvas line chart. The time axis ends at the newest data point rather than the
 * browser's clock: timestamps come from the server, and the two clocks need not agree.
 */
export function LineChart(props: Props) {
  const { title, series, markers, windowS = 120, height = 170, testId } = props;
  const ref = useCanvas((ctx, w, h) => drawChart(ctx, w, h, props), [series, markers], {
    height,
  });
  const latest = series.map((s) => s.points[s.points.length - 1]?.v);
  return (
    <section className="panel" aria-label={title} data-testid={testId}>
      <div className="panel-head">
        <h2>{title}</h2>
        <div className="legend">
          {series.map((s, i) => (
            <span key={s.label}>
              <i style={{ background: `var(${s.color})` }} />
              {s.label}
              {latest[i] !== undefined && (
                <strong className="mono" style={{ marginLeft: 6 }}>
                  {props.format(latest[i] as number)}
                </strong>
              )}
            </span>
          ))}
          {markers.length > 0 && (
            <span title="Model hot-swaps (first prediction made by the new model)">
              <i style={{ background: "var(--warn)" }} />
              model change
            </span>
          )}
        </div>
      </div>
      <div className="canvas-wrap" style={{ height }}>
        <canvas ref={ref} style={{ height }} />
      </div>
      <p className="hint" style={{ margin: "4px 0 0" }}>
        Last {windowS / 60} min{props.subtitle ? ` · ${props.subtitle}` : ""}
      </p>
    </section>
  );
}

export function autoMax(values: number[], floor = 0): number {
  const bulk = percentile(values, 95) ?? 0;
  return niceMax(Math.max(floor * 1.2, bulk * 1.5, 1));
}

export function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function drawChart(ctx: CanvasRenderingContext2D, w: number, h: number, p: Props) {
  const windowS = p.windowS ?? 120;
  const all = p.series.flatMap((s) => s.points);
  const tEnd = Math.max(0, ...all.map((pt) => pt.t), ...p.markers.map((m) => m.t));
  const tStart = tEnd - windowS;
  const inView = all.filter((pt) => pt.t >= tStart);
  // "auto" scales to the bulk of the data (95th percentile x 1.5, never below the reference
  // line) and clips rare spikes at the top, so one slow window cannot flatten the chart.
  const yMax = p.yMax === "auto" ? autoMax(inView.map((pt) => pt.v), p.reference?.value) : p.yMax;
  const plotW = w - PAD.left - PAD.right;
  const plotH = h - PAD.top - PAD.bottom;
  const x = (t: number) => PAD.left + ((t - tStart) / windowS) * plotW;
  const y = (v: number) => PAD.top + plotH - ((v - p.yMin) / (yMax - p.yMin)) * plotH;

  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;
  // horizontal grid + y labels
  ctx.strokeStyle = cssVar("--border");
  ctx.fillStyle = cssVar("--muted");
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = p.yMin + ((yMax - p.yMin) * i) / 4;
    const yy = Math.round(y(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(PAD.left, yy);
    ctx.lineTo(w - PAD.right, yy);
    ctx.stroke();
    ctx.fillText(p.format(v), PAD.left - 6, yy + 4);
  }
  if (all.length === 0) {
    ctx.textAlign = "center";
    ctx.fillText("No data yet", PAD.left + plotW / 2, PAD.top + plotH / 2);
    return;
  }
  // reference line (e.g. chance)
  if (p.reference) {
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = cssVar("--faint");
    const yy = y(p.reference.value);
    ctx.beginPath();
    ctx.moveTo(PAD.left, yy);
    ctx.lineTo(w - PAD.right, yy);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.textAlign = "left";
    ctx.fillText(p.reference.label, PAD.left + 4, yy - 4);
  }
  // model-change markers
  ctx.strokeStyle = cssVar("--warn");
  ctx.fillStyle = cssVar("--warn");
  for (const m of p.markers) {
    if (m.t < tStart) continue;
    const xx = Math.round(x(m.t)) + 0.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(xx, PAD.top);
    ctx.lineTo(xx, PAD.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.textAlign = xx > w - 90 ? "right" : "left";
    ctx.fillText(`▸ ${m.label}`, xx + (xx > w - 90 ? -4 : 4), PAD.top - 5);
  }
  // series
  for (const s of p.series) {
    ctx.strokeStyle = cssVar(s.color);
    ctx.lineWidth = 1.75;
    ctx.beginPath();
    let first = true;
    for (const pt of s.points) {
      if (pt.t < tStart) continue;
      const yy = y(Math.min(pt.v, yMax));
      if (first) ctx.moveTo(x(pt.t), yy);
      else ctx.lineTo(x(pt.t), yy);
      first = false;
    }
    ctx.stroke();
  }
  // time labels
  ctx.fillStyle = cssVar("--muted");
  ctx.textAlign = "center";
  for (let s = 0; s <= windowS; s += windowS / 4) {
    ctx.textAlign = s === 0 ? "right" : "center";
    ctx.fillText(s === 0 ? "now" : `−${s} s`, s === 0 ? w - PAD.right : x(tEnd - s), h - 5);
  }
}
