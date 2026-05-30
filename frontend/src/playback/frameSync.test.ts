import { describe, expect, it } from "vitest";
import {
  advanceFrame,
  clampFrame,
  frameToX,
  seriesExtent,
  xToFrame,
} from "./frameSync";

describe("clampFrame", () => {
  it("clamps to [0, n-1] and rounds", () => {
    expect(clampFrame(-5, 10)).toBe(0);
    expect(clampFrame(99, 10)).toBe(9);
    expect(clampFrame(3.6, 10)).toBe(4);
    expect(clampFrame(5, 0)).toBe(0);
  });
});

describe("frameToX / xToFrame round-trip", () => {
  const n = 100;
  const width = 500;
  const padL = 40;
  const padR = 10;
  it("maps endpoints to the plot edges", () => {
    expect(frameToX(0, n, width, padL, padR)).toBeCloseTo(padL);
    expect(frameToX(n - 1, n, width, padL, padR)).toBeCloseTo(width - padR);
  });
  it("inverts back to the same frame", () => {
    for (const f of [0, 17, 50, 99]) {
      const x = frameToX(f, n, width, padL, padR);
      expect(xToFrame(x, n, width, padL, padR)).toBe(f);
    }
  });
  it("clamps clicks outside the plot area", () => {
    expect(xToFrame(0, n, width, padL, padR)).toBe(0);
    expect(xToFrame(width + 100, n, width, padL, padR)).toBe(n - 1);
  });
});

describe("advanceFrame", () => {
  it("advances by elapsed*speed/dt", () => {
    // dt=1/60, 1s elapsed at 1x -> ~60 frames
    const r = advanceFrame(0, 600, 1 / 60, 1, 1);
    expect(r.frame).toBeCloseTo(60, 0);
    expect(r.reachedEnd).toBe(false);
  });
  it("respects speed multiplier", () => {
    const slow = advanceFrame(0, 600, 1 / 60, 0.5, 1).frame;
    const fast = advanceFrame(0, 600, 1 / 60, 2, 1).frame;
    expect(fast).toBeGreaterThan(slow);
  });
  it("stops at the end without looping", () => {
    const r = advanceFrame(595, 600, 1 / 60, 4, 1);
    expect(r.frame).toBe(599);
    expect(r.reachedEnd).toBe(true);
  });
});

describe("seriesExtent", () => {
  it("pads a flat series so it still renders", () => {
    const { min, max } = seriesExtent([5, 5, 5]);
    expect(min).toBeLessThan(5);
    expect(max).toBeGreaterThan(5);
  });
  it("ignores non-finite values", () => {
    const { min, max } = seriesExtent([0, NaN, 10, Infinity]);
    expect(min).toBe(0);
    expect(max).toBe(10);
  });
});
