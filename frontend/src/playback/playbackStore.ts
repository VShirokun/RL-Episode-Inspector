// Single source of truth for playback. The 3D viewer, charts, timeline and
// value panels all read currentFrame from here — no component owns frame state.

import { create } from "zustand";
import { api, ApiError } from "../api/client";
import type {
  EpisodeSummary,
  LoadedEpisode,
  PlaybackSpeed,
} from "../types/episode";
import { advanceFrame, clampFrame } from "./frameSync";

interface PlaybackState {
  // episode list
  episodes: EpisodeSummary[];
  listLoading: boolean;
  listError: string | null;

  // selected / loaded episode
  selectedId: string | null;
  loaded: LoadedEpisode | null;
  episodeLoading: boolean;
  episodeError: string | null;

  // playback
  currentFrame: number; // may be fractional during playback
  isPlaying: boolean;
  speed: PlaybackSpeed;
  loop: boolean; // when true, playback restarts at frame 0 instead of stopping
  renderMode: "models" | "cubes"; // 3D bodies: real meshes (default) or proxy cubes

  // derived
  numFrames: () => number;

  // actions
  fetchEpisodes: () => Promise<void>;
  selectEpisode: (id: string) => Promise<void>;
  selectRanked: (mode: "best" | "worst" | "median") => Promise<void>;
  setFrame: (frame: number) => void;
  seek: (frame: number) => void; // pause + set (used by chart click/drag)
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  stepFrames: (delta: number) => void;
  first: () => void;
  last: () => void;
  setSpeed: (speed: PlaybackSpeed) => void;
  toggleLoop: () => void;
  setRenderMode: (mode: "models" | "cubes") => void;
  tick: (elapsedSeconds: number) => void;
}

function errMsg(e: unknown): string {
  if (e instanceof ApiError) return e.status ? `${e.message} (HTTP ${e.status})` : e.message;
  return String(e);
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  episodes: [],
  listLoading: false,
  listError: null,

  selectedId: null,
  loaded: null,
  episodeLoading: false,
  episodeError: null,

  currentFrame: 0,
  isPlaying: false,
  speed: 1,
  loop: true,
  renderMode: "models",

  numFrames: () => get().loaded?.metadata.num_frames ?? 0,

  fetchEpisodes: async () => {
    set({ listLoading: true, listError: null });
    try {
      const episodes = await api.listEpisodes();
      set({ episodes, listLoading: false });
      // Auto-select the best episode on first load for a friendly landing state.
      if (episodes.length > 0 && get().selectedId === null) {
        await get().selectEpisode(episodes[0].episode_id);
      }
    } catch (e) {
      set({ listLoading: false, listError: errMsg(e) });
    }
  },

  selectEpisode: async (id: string) => {
    if (get().episodeLoading) return;
    set({ episodeLoading: true, episodeError: null, isPlaying: false, selectedId: id });
    try {
      const [metadata, frames] = await Promise.all([
        api.getMetadata(id),
        api.getFrames(id),
      ]);
      const loaded: LoadedEpisode = { metadata, columns: frames.columns };
      set({ loaded, currentFrame: 0, episodeLoading: false });
    } catch (e) {
      set({ episodeLoading: false, episodeError: errMsg(e), loaded: null });
    }
  },

  selectRanked: async (mode) => {
    try {
      const summary = await api.getRanked(mode);
      await get().selectEpisode(summary.episode_id);
    } catch (e) {
      set({ episodeError: errMsg(e) });
    }
  },

  setFrame: (frame) => set({ currentFrame: clampFrame(frame, get().numFrames()) }),

  seek: (frame) =>
    set({ currentFrame: clampFrame(frame, get().numFrames()), isPlaying: false }),

  play: () => {
    const n = get().numFrames();
    if (n === 0) return;
    // Restart from the beginning if we're parked at the end.
    const atEnd = get().currentFrame >= n - 1;
    set({ isPlaying: true, currentFrame: atEnd ? 0 : get().currentFrame });
  },

  pause: () => set({ isPlaying: false }),

  togglePlay: () => (get().isPlaying ? get().pause() : get().play()),

  stepFrames: (delta) =>
    set({
      isPlaying: false,
      currentFrame: clampFrame(Math.round(get().currentFrame) + delta, get().numFrames()),
    }),

  first: () => set({ isPlaying: false, currentFrame: 0 }),

  last: () => set({ isPlaying: false, currentFrame: Math.max(0, get().numFrames() - 1) }),

  setSpeed: (speed) => set({ speed }),

  toggleLoop: () => set({ loop: !get().loop }),

  setRenderMode: (mode) => set({ renderMode: mode }),

  tick: (elapsedSeconds) => {
    const { isPlaying, loaded, currentFrame, speed, loop } = get();
    if (!isPlaying || !loaded) return;
    const { frame, reachedEnd } = advanceFrame(
      currentFrame,
      loaded.metadata.num_frames,
      loaded.metadata.dt,
      speed,
      elapsedSeconds,
    );
    if (reachedEnd && loop) {
      // Restart from the beginning and keep playing.
      set({ currentFrame: 0, isPlaying: true });
    } else {
      set({ currentFrame: frame, isPlaying: !reachedEnd });
    }
  },
}));

/** Read a numeric frame column safely at an integer frame index. */
export function columnValue(
  loaded: LoadedEpisode | null,
  column: string,
  frame: number,
): number | undefined {
  if (!loaded) return undefined;
  const col = loaded.columns[column];
  if (!col) return undefined;
  const idx = clampFrame(frame, loaded.metadata.num_frames);
  const v = col[idx];
  return typeof v === "number" ? v : v ? 1 : 0;
}
