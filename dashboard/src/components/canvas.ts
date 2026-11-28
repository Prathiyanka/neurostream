import { type RefObject, useEffect, useRef } from "react";

/** Read a CSS custom property (canvas cannot use `var(--x)` directly). */
export function cssVar(name: string, fallback = "#888"): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export type Draw = (ctx: CanvasRenderingContext2D, width: number, height: number) => void;

/**
 * Keep a canvas sized to its CSS box at device-pixel resolution and call `draw`:
 * on resize, whenever `deps` change, and on every animation frame if `animate`.
 * jsdom (tests) has no 2D context: then nothing is drawn and nothing throws.
 */
export function useCanvas(
  draw: Draw,
  deps: unknown[],
  { animate = false, height }: { animate?: boolean; height: number },
): RefObject<HTMLCanvasElement | null> {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext?.("2d") ?? null;
    if (!canvas || !ctx) return;
    let frame = 0;
    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(height * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, height);
      drawRef.current(ctx, w, height);
      if (animate) frame = requestAnimationFrame(render);
    };
    render();
    const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => render()) : null;
    ro?.observe(canvas);
    return () => {
      cancelAnimationFrame(frame);
      ro?.disconnect();
    };
  }, [animate, height, ...deps]);

  return ref;
}
