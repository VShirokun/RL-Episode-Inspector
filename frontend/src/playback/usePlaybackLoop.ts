// Drives playback with requestAnimationFrame — but ONLY while actually playing.
// When paused/finished the loop is fully torn down so the app goes idle (no
// per-frame main-thread wakeups). It starts/stops by watching isPlaying.

import { useEffect } from "react";
import { usePlaybackStore } from "./playbackStore";

export function usePlaybackLoop(): void {
  useEffect(() => {
    let raf = 0;
    let last = 0;
    let running = false;

    const loop = (now: number) => {
      const elapsed = (now - last) / 1000;
      last = now;
      // Clamp huge gaps (e.g. tab was backgrounded) so we don't jump frames.
      usePlaybackStore.getState().tick(Math.min(elapsed, 0.25));
      // tick() sets isPlaying=false at the end of the episode; stop then.
      if (usePlaybackStore.getState().isPlaying) {
        raf = requestAnimationFrame(loop);
      } else {
        running = false;
      }
    };

    const start = () => {
      if (running) return;
      running = true;
      last = performance.now();
      raf = requestAnimationFrame(loop);
    };
    const stop = () => {
      running = false;
      cancelAnimationFrame(raf);
    };

    let wasPlaying = usePlaybackStore.getState().isPlaying;
    if (wasPlaying) start();
    const unsubscribe = usePlaybackStore.subscribe((state) => {
      if (state.isPlaying !== wasPlaying) {
        wasPlaying = state.isPlaying;
        if (wasPlaying) start();
        else stop();
      }
    });

    return () => {
      unsubscribe();
      stop();
    };
  }, []);
}
