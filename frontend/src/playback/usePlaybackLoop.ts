// Drives playback with requestAnimationFrame. Reads/writes the store via
// getState so the rAF loop never needs to re-subscribe on every frame.

import { useEffect } from "react";
import { usePlaybackStore } from "./playbackStore";

export function usePlaybackLoop(): void {
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const elapsed = (now - last) / 1000;
      last = now;
      // Clamp huge gaps (e.g. tab was backgrounded) so we don't jump frames.
      usePlaybackStore.getState().tick(Math.min(elapsed, 0.25));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
}
