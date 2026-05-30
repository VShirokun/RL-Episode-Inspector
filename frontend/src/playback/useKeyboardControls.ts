// Global keyboard shortcuts (see docs/§Interactive Timeline):
//   Space            play/pause
//   ArrowRight/Left  +/- 1 frame
//   Shift+Arrow      +/- 10 frames
//   Home / End       first / last frame

import { useEffect } from "react";
import { usePlaybackStore } from "./playbackStore";

export function useKeyboardControls(): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Don't hijack typing in form fields.
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;

      const store = usePlaybackStore.getState();
      switch (e.key) {
        case " ":
          e.preventDefault();
          store.togglePlay();
          break;
        case "ArrowRight":
          e.preventDefault();
          store.stepFrames(e.shiftKey ? 10 : 1);
          break;
        case "ArrowLeft":
          e.preventDefault();
          store.stepFrames(e.shiftKey ? -10 : -1);
          break;
        case "Home":
          e.preventDefault();
          store.first();
          break;
        case "End":
          e.preventDefault();
          store.last();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}
