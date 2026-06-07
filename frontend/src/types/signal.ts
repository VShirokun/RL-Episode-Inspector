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
  agent?: string | null; // multi-agent: which agent this signal belongs to
  display?: Record<string, unknown> | null;
}

export interface AgentSpec {
  id: string;
  label?: string | null;
  team?: string | null;
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

export interface LightSpec {
  name: string;
  kind: "directional" | "point" | "ambient" | "hemisphere";
  color: [number, number, number]; // linear RGB 0..1
  intensity: number; // normalized for real-time rendering
  direction?: [number, number, number] | null; // sim frame, for directional
  position?: [number, number, number] | null; // sim frame, for point
}

export interface ViewerSpec {
  type: string;
  state_mapping: Record<string, string>;
  bodies?: BodySpec[];
  markers?: MarkerSpec[];
  lights?: LightSpec[]; // captured from the sim; empty => default rig
  up_axis?: "z" | "y";
}
