// Mirrors python/rl_episode_inspector/storage/signal_schema.py

export type SignalKind =
  | "state"
  | "reward_raw"
  | "reward_weighted"
  | "reward_total"
  | "action"
  | "observation"
  | "debug"
  | "event";

export interface SignalSpec {
  name: string;
  kind: SignalKind;
  dtype: string;
  shape: number[];
  unit: string | null;
  description: string | null;
  display?: Record<string, unknown> | null;
}

export interface ViewerSpec {
  type: string;
  state_mapping: Record<string, string>;
}
