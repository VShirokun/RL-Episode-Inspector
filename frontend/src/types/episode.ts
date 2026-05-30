// Mirrors python/rl_episode_inspector/storage/schemas.py

import type { SignalSpec, ViewerSpec } from "./signal";

export interface EpisodeSummary {
  episode_id: string;
  task_name: string;
  episode_return: number;
  num_frames: number;
  duration_seconds: number;
  created_at: string;
  terminated: boolean;
  truncated: boolean;
  reset_reason: string | null;
}

export interface EpisodeMetadata {
  schema_version: string;
  episode_id: string;
  run_id: string | null;
  task_name: string;
  task_source: string;
  env_id: number;
  episode_index: number;
  created_at: string;
  num_frames: number;
  dt: number;
  fps: number;
  duration_seconds: number;
  global_step_start: number;
  global_step_end: number;
  terminated: boolean;
  truncated: boolean;
  reset_reason: string | null;
  episode_return: number;
  policy_checkpoint: string | null;
  seed: number | null;
  signals: SignalSpec[];
  viewer: ViewerSpec;
}

/** Frame columns as returned by GET /api/episodes/{id}/frames. */
export type FrameColumns = Record<string, number[] | boolean[]>;

export interface FramesResponse {
  episode_id: string;
  start: number;
  count: number;
  columns: FrameColumns;
}

/** Everything the viewer/charts need for one loaded episode. */
export interface LoadedEpisode {
  metadata: EpisodeMetadata;
  columns: FrameColumns;
}

export const PLAYBACK_SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4] as const;
export type PlaybackSpeed = (typeof PLAYBACK_SPEEDS)[number];
