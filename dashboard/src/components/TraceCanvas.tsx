import { MOTOR_CHANNELS } from "../lib/channels";
import { isGap, median, robustRange, type TraceBuffer } from "../lib/traces";
import { cssVar, useCanvas } from "./canvas";

const LANE_COLORS = ["--left", "--accent", "--right", "--ok", "--rest", "--warn"];
const WINDOW_S = 10;
const HEIGHT_PER_LANE = 70;
const AXIS_H = 20;
const LABEL_W = 44;

export function TraceCanvas({
  buffer,
  channels,
  onChannels,
}: {
  buffer: TraceBuffer;
  channels: string[];
  onChannels: (channels: string[]) => void;
}) {
  const height = Math.max(1, channels.length) * HEIGHT_PER_LANE + AXIS_H;
  const ref = useCanvas(
    (ctx, w, h) => drawTraces(ctx, w, h, buffer),
    [buffer, channels.join()],
    { animate: true, height },
  );

  const toggle = (ch: string) => {
    const next = channels.includes(ch)
      ? channels.filter((c) => c !== ch)
      : [...channels, ch].sort((a, b) => MOTOR_CHANNELS.indexOf(a) - MOTOR_CHANNELS.indexOf(b));
    if (next.length > 0 && next.length <= 6) onChannels(next);
  };

  return (
    <section className="panel" aria-label="EEG traces">
      <div className="panel-head">
        <h2>EEG</h2>
        <span className="hint">raw signal as ingested · shown at 40 Hz · pick up to 6 channels</span>
      </div>
      <div className="channel-picker" role="group" aria-label="Channels (up to 6)">
          {MOTOR_CHANNELS.map((ch) => (
            <button
              key={ch}
              type="button"
              aria-pressed={channels.includes(ch)}
              onClick={() => toggle(ch)}
            >
              {ch}
            </button>
          ))}
      </div>
      <div className="canvas-wrap" style={{ height, marginTop: 10 }}>
        <canvas ref={ref} style={{ height }} data-testid="trace-canvas" />
      </div>
    </section>
  );
}

function drawTraces(ctx: CanvasRenderingContext2D, w: number, h: number, buffer: TraceBuffer) {
  const muted = cssVar("--muted");
  const faint = cssVar("--border");
  ctx.font = "11px system-ui, sans-serif";
  const lanes = buffer.channels;
  const plotW = w - LABEL_W;
  const laneH = (h - AXIS_H) / Math.max(1, lanes.length);

  // time grid: one line per second, labelled relative to the newest sample
  ctx.strokeStyle = faint;
  ctx.fillStyle = muted;
  ctx.lineWidth = 1;
  for (let s = 0; s <= WINDOW_S; s += 1) {
    const x = LABEL_W + (plotW * (WINDOW_S - s)) / WINDOW_S;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, h - AXIS_H);
    ctx.stroke();
    if (s % 2 === 0) {
      ctx.textAlign = s === 0 ? "right" : "center";
      ctx.fillText(s === 0 ? "now" : `−${s} s`, x, h - 5);
    }
  }

  if (buffer.length < 2 || lanes.length === 0) {
    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    ctx.fillText("Select a running session to see its EEG", LABEL_W + plotW / 2, (h - AXIS_H) / 2);
    return;
  }

  const { times, series } = buffer.snapshot();
  const tEnd = times[times.length - 1] as number;
  const tStart = tEnd - WINDOW_S;
  lanes.forEach((ch, lane) => {
    const values = series.get(ch) ?? [];
    const visible = values.filter((_, i) => (times[i] as number) >= tStart);
    const center = median(visible);
    const range = robustRange(visible);
    const mid = lane * laneH + laneH / 2;
    const scale = (laneH / 2 - 4) / range;

    ctx.fillStyle = cssVar("--text");
    ctx.textAlign = "left";
    ctx.fillText(ch, 4, mid - 2);
    ctx.fillStyle = muted;
    ctx.fillText(`±${range.toFixed(0)} µV`, 4, mid + 12);

    ctx.strokeStyle = cssVar(LANE_COLORS[lane % LANE_COLORS.length] as string);
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < times.length; i++) {
      const t = times[i] as number;
      const v = values[i] as number;
      if (t < tStart || !Number.isFinite(v)) {
        pen = false;
        continue;
      }
      if (i > 0 && isGap(times[i - 1] as number, t, buffer.sfreq)) pen = false; // lost packets
      const x = LABEL_W + ((t - tStart) / WINDOW_S) * plotW;
      const y = mid - (v - center) * scale;
      if (pen) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();
  });
}
