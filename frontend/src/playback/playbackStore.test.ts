// Tests the store's pure-ish playback logic by injecting a loaded episode
// directly (no network). Verifies stepping, clamping, seek-pauses-playback and
// speed-respecting advancement.

import { beforeEach, describe, expect, it } from "vitest";
import { usePlaybackStore } from "./playbackStore";
import type { LoadedEpisode } from "../types/episode";

function fakeEpisode(n = 100): LoadedEpisode {
  return {
    metadata: {
      schema_version: "0.1.0", episode_id: "e", run_id: null, task_name: "T",
      task_source: "test", env_id: 0, episode_index: 0, created_at: "",
      num_frames: n, dt: 1 / 60, fps: 60, duration_seconds: n / 60,
      global_step_start: 0, global_step_end: n - 1, terminated: false,
      truncated: true, reset_reason: null, episode_return: 1, policy_checkpoint: null,
      seed: null, signals: [], viewer: { type: "cartpole", state_mapping: {} },
    },
    columns: { frame_index: Array.from({ length: n }, (_, i) => i) },
  };
}

beforeEach(() => {
  usePlaybackStore.setState({
    loaded: fakeEpisode(), currentFrame: 0, isPlaying: false, speed: 1, loop: true,
  });
});

describe("stepping", () => {
  it("steps forward and backward, clamped", () => {
    const s = usePlaybackStore.getState();
    s.stepFrames(1);
    expect(usePlaybackStore.getState().currentFrame).toBe(1);
    s.stepFrames(-5);
    expect(usePlaybackStore.getState().currentFrame).toBe(0);
    s.last();
    expect(usePlaybackStore.getState().currentFrame).toBe(99);
    s.stepFrames(10);
    expect(usePlaybackStore.getState().currentFrame).toBe(99);
  });
});

describe("seek pauses playback", () => {
  it("stops playing when seeking", () => {
    usePlaybackStore.getState().play();
    expect(usePlaybackStore.getState().isPlaying).toBe(true);
    usePlaybackStore.getState().seek(42);
    expect(usePlaybackStore.getState().isPlaying).toBe(false);
    expect(usePlaybackStore.getState().currentFrame).toBe(42);
  });
});

describe("play restarts at end", () => {
  it("rewinds to 0 if parked at the last frame", () => {
    usePlaybackStore.getState().last();
    usePlaybackStore.getState().play();
    expect(usePlaybackStore.getState().currentFrame).toBe(0);
    expect(usePlaybackStore.getState().isPlaying).toBe(true);
  });
});

describe("tick advances and stops (loop off)", () => {
  it("advances while playing and halts at the end", () => {
    usePlaybackStore.setState({ loop: false });
    usePlaybackStore.getState().play();
    usePlaybackStore.getState().tick(0.1); // 0.1s @ 60fps ~ +6 frames
    expect(usePlaybackStore.getState().currentFrame).toBeGreaterThan(0);
    usePlaybackStore.getState().setSpeed(4);
    usePlaybackStore.getState().tick(100); // huge jump -> end
    expect(usePlaybackStore.getState().currentFrame).toBe(99);
    expect(usePlaybackStore.getState().isPlaying).toBe(false);
  });
});

describe("looping", () => {
  it("restarts at 0 and keeps playing when loop is on", () => {
    usePlaybackStore.setState({ loop: true });
    usePlaybackStore.getState().last(); // park near the end (pauses)
    usePlaybackStore.getState().play();
    usePlaybackStore.getState().setSpeed(4);
    usePlaybackStore.getState().tick(100); // overshoot the end
    expect(usePlaybackStore.getState().isPlaying).toBe(true); // still playing
    expect(usePlaybackStore.getState().currentFrame).toBe(0); // wrapped to start
  });

  it("toggleLoop flips the flag", () => {
    usePlaybackStore.setState({ loop: true });
    usePlaybackStore.getState().toggleLoop();
    expect(usePlaybackStore.getState().loop).toBe(false);
  });
});
