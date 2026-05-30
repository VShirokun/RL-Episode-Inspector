// Pure, side-effect-free helpers shared by the playback loop and the charts.
// Kept free of React/DOM so they can be unit-tested directly (see
// frameSync.test.ts) — this is the "one source of truth" math for seeking.

export function clampFrame(frame: number, numFrames: number): number {
  if (numFrames <= 0) return 0;
  const f = Math.round(frame);
  if (f < 0) return 0;
  if (f > numFrames - 1) return numFrames - 1;
  return f;
}

/** Map a frame index to an x pixel within a plot area of [padLeft, width-padRight]. */
export function frameToX(
  frame: number,
  numFrames: number,
  width: number,
  padLeft: number,
  padRight: number,
): number {
  const inner = Math.max(1, width - padLeft - padRight);
  if (numFrames <= 1) return padLeft;
  return padLeft + (frame / (numFrames - 1)) * inner;
}

/** Inverse of frameToX: map an x pixel back to the nearest frame index. */
export function xToFrame(
  x: number,
  numFrames: number,
  width: number,
  padLeft: number,
  padRight: number,
): number {
  const inner = Math.max(1, width - padLeft - padRight);
  const t = (x - padLeft) / inner;
  return clampFrame(t * (numFrames - 1), numFrames);
}

/**
 * Advance the current frame during playback.
 *
 * Given the elapsed wall-clock seconds since the last tick, the episode's dt
 * and a playback speed multiplier, returns the next frame and whether playback
 * should stop (because the end was reached — playback does not loop).
 */
export function advanceFrame(
  currentFrame: number,
  numFrames: number,
  dtSeconds: number,
  speed: number,
  elapsedSeconds: number,
): { frame: number; reachedEnd: boolean } {
  if (numFrames <= 0 || dtSeconds <= 0) {
    return { frame: 0, reachedEnd: true };
  }
  const framesAdvanced = (elapsedSeconds * speed) / dtSeconds;
  const next = currentFrame + framesAdvanced;
  if (next >= numFrames - 1) {
    return { frame: numFrames - 1, reachedEnd: true };
  }
  return { frame: next, reachedEnd: false };
}

/** Min/max over a numeric series, with a tiny pad so flat lines still render. */
export function seriesExtent(values: number[]): { min: number; max: number } {
  if (values.length === 0) return { min: 0, max: 1 };
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
  if (min === max) {
    const pad = Math.abs(min) > 1e-9 ? Math.abs(min) * 0.1 : 1;
    return { min: min - pad, max: max + pad };
  }
  return { min, max };
}
