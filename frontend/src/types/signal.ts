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

export interface BodySpec {
  name: string;
  parent: number; // index into bodies[], -1 = root
  pos: [string, string, string]; // px,py,pz column names
  quat: [string, string, string, string]; // qw,qx,qy,qz column names
  mesh?: string | null; // GLB path under /assets (models mode); null => cube
}

export interface MarkerSpec {
  name: string;
  pos: [string, string, string];
  color: string | null;
}

export interface ViewerSpec {
  type: string;
  state_mapping: Record<string, string>;
  bodies?: BodySpec[];
  markers?: MarkerSpec[];
  up_axis?: "z" | "y";
  orient_mode?: "quaternion" | "bone";
}
